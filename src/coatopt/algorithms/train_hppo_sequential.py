#!/usr/bin/env python3
"""
Single-agent PPO with hybrid action space and sequential constraint scheduling.

Uses discrete material selection + continuous thickness (TruncatedNormal) with
constraint-based multi-objective training. Alternates between objectives during
warmup, then applies gradually tightening constraints.

Config section: [hppo_sequential]
  total_episodes           = 10000
  warmup_episodes          = 500            # Per objective
  warmup_block_episodes    = 0              # 0 = one block per objective; smaller = interleave targets every N episodes
  episodes_per_step          = 200            # Episodes per phase
  steps_per_objective      = 10             # Constraint levels per objective
  episodes_per_update      = 10             # Episodes before PPO update
  n_epochs                 = 5              # SGD epochs per update
  batch_size               = 64
  constraint_penalty       = 3.0
  constraint_level_schedule = cycle        # "cycle": repeat the ramp; "ramp": climb once, then hold at the top
  update_constraint_bounds = false         # keep the per-objective bests anchoring the thresholds rising after warmup
  constraint_anchor        = warmup        # "warmup": scale thresholds by this run's warmup best; "absolute": by the normalised 1.0, rising if beaten
  constraint_extend_low    = 0.2           # widen the low end of the constraint range by this fraction of the front's spread
  constraint_extend_high   = 0.2           # widen the high end likewise; this is what demands more than the run has reached
  constraint_source        = box           # "box": draw each threshold from its objective's range; "reference": take them all from one archived design
  constraint_ref_extend    = 1.0           # reference only: how far past that design to ask, in units of its local spacing on the front
  pareto_bonus             = 0.0            # Hypervolume improvement bonus
  bc_weight                = 0.1            # Behavior cloning weight from Pareto episodes (0.0 = disabled)
  bc_selection             = target         # "target": imitate archive best for current target+constraints; "all": whole front
  bc_every                 = 0              # BC on every k-th minibatch; 0 = first minibatch of each epoch
  lockstep_rollouts        = false          # run each update's episodes in lockstep with batched policy calls
  lr                       = 3e-4
  lr_final                 = 3e-5           # Final LR (annealing target)
  lr_decay_episodes        = 10000          # Anneal over this many episodes per phase
  restart_decay_on_phase   = false          # Restart LR/entropy decay each constraint phase (like warm restarts)
  gamma                    = 0.99
  gae_lambda               = 0.95
  clip_range               = 0.2
  ent_coef                 = 0.01
  ent_coef_final           = 0.001          # Final entropy coefficient (annealing target)
  ent_coef_thickness       =                # Thickness-head entropy coefficient; unset tracks ent_coef
  ent_coef_thickness_final =                # Its annealing target; unset = the value above, i.e. no annealing
  ent_decay_episodes       = 10000          # Anneal over this many episodes per phase
  vf_coef                  = 0.5
  max_grad_norm            = 0.5
  min_layers_before_air    = 4
  mask_consecutive_materials = true
  pre_model_type           = linear         # Layer-stack encoder: linear | lstm | attention
  pre_model_params         = {"hidden": 32, "layers": 2}   # encoder kwargs; attention also takes heads, ff_mult
                                            # (legacy use_lstm/lstm_hidden/lstm_layers still work)
  hidden                   = [256, 256]     # MLP layers after the encoder/before policy heads
  torch_threads            = 1              # intra-op threads; match request_cpus (0 = leave torch's default)
  seed                     = 42
  verbose                  = 1
  plot_freq                = 500
  hypervolume_freq         = 500            # archive hypervolume cadence (defaults to plot_freq; it is O(N^2) in front size)
"""

import ast
import configparser
import math
import time
from pathlib import Path
from typing import List

import gymnasium as gym
import mlflow
import numpy as np
import torch
import torch.nn as nn

from coatopt.environments.environment import CoatingEnvironment
from coatopt.utils.checkpoint import load_checkpoint, save_checkpoint
from coatopt.utils.configs import Config, load_config
from coatopt.utils.utils import load_materials_from_parser


class CoatOptHybridEnv(gym.Env):
    """Hybrid discrete+continuous action space environment.

    Action space: Dict with discrete material and continuous thickness.
    Identical to the one in train_ppo_multiagent.py.
    """

    def __init__(
        self,
        config: Config,
        materials: dict,
        warmup_episodes: int = 500,
        episodes_per_step: int = 200,
        steps_per_objective: int = 10,
        constraint_penalty: float = 3.0,
        mask_consecutive_materials: bool = True,
        min_layers_before_air: int = 4,
        randomise_constraints: bool = False,
        warmup_block_episodes: int = None,
        constraint_level_schedule: str = "cycle",
        update_constraint_bounds: bool = False,
        constraint_anchor: str = "warmup",
        constraint_extend_low: float = 0.2,
        constraint_extend_high: float = 0.2,
        constraint_source: str = "box",
        constraint_ref_extend: float = 1.0,
    ):
        super().__init__()
        self.env = CoatingEnvironment(config, materials)
        self.mask_consecutive = mask_consecutive_materials
        self.min_layers_before_air = min_layers_before_air

        # Scheduling parameters (like train_sb3_discrete)
        self.objectives = list(config.data.optimise_parameters)
        self.warmup_episodes_per_objective = (
            warmup_episodes  # Episodes per objective during warmup
        )
        # Cycle the warmup target every warmup_block_episodes; None gives
        # one block per objective.
        self.warmup_block_episodes = warmup_block_episodes or warmup_episodes
        self.total_warmup_episodes = warmup_episodes * len(
            self.objectives
        )  # Total warmup
        self.episodes_per_step = episodes_per_step
        self.steps_per_objective = steps_per_objective
        self.randomise_constraints = randomise_constraints
        if constraint_level_schedule not in ("cycle", "ramp"):
            raise ValueError(
                f"constraint_level_schedule must be 'cycle' or 'ramp', "
                f"got {constraint_level_schedule!r}"
            )
        self.constraint_level_schedule = constraint_level_schedule
        self.episode_count = 0
        self.is_warmup = True

        # Enable constrained training in base environment
        self.env.enable_constrained_training(
            warmup_episodes_per_objective=warmup_episodes,
            steps_per_objective=steps_per_objective,
            episodes_per_step=episodes_per_step,
            constraint_penalty=constraint_penalty,
            update_constraint_bounds=update_constraint_bounds,
            constraint_anchor_mode=constraint_anchor,
            constraint_extend_low=constraint_extend_low,
            constraint_extend_high=constraint_extend_high,
            constraint_source=constraint_source,
            constraint_ref_extend=constraint_ref_extend,
        )

        # Action space: Dict with discrete material + continuous thickness
        self.action_space = gym.spaces.Dict(
            {
                "material": gym.spaces.Discrete(self.env.n_materials),
                "thickness": gym.spaces.Box(
                    low=np.array([self.env.min_thickness], dtype=np.float32),
                    high=np.array([self.env.max_thickness], dtype=np.float32),
                    dtype=np.float32,
                ),
            }
        )

        # Observation space (includes objective weights and constraint thresholds)
        n_features = 1 + self.env.n_materials + 2
        n_objectives = len(self.env.optimise_parameters)
        n_constraints = len(self.env.optimise_parameters)
        obs_size = self.env.max_layers * n_features + 1 + n_objectives + n_constraints
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

        self.prev_material = None
        self.current_layer = 0

    def get_action_mask(self) -> np.ndarray:
        """Return mask for valid material choices (1=valid, 0=invalid)."""
        mask = np.ones(self.env.n_materials, dtype=np.float32)
        if self.mask_consecutive and self.prev_material is not None:
            mask[self.prev_material] = 0.0
        if self.current_layer < self.min_layers_before_air:
            mask[0] = 0.0  # air is index 0
        if mask.sum() == 0:
            mask[:] = 1.0
        return mask

    def _get_obs(self, state) -> np.ndarray:
        """Convert state to observation with objective weights and constraint thresholds."""
        base = state.observation_array(self.env.max_thickness).ravel()
        n_obj = len(self.objectives)
        obs = np.empty(len(base) + 2 * n_obj, dtype=np.float32)
        obs[: len(base)] = base
        for i, obj in enumerate(self.objectives):
            obs[len(base) + i] = 1.0 if obj == self.env.target_objective else 0.0
        for i, obj in enumerate(self.env.optimise_parameters):
            obs[len(base) + n_obj + i] = self.env.constraints.get(obj, 0.0)
        return obs

    def reset(self, seed=None, options=None):
        """Reset with scheduling (warmup then constrained) like train_sb3_discrete."""
        super().reset(seed=seed)
        state = self.env.reset()
        self.prev_material = None
        self.current_layer = 0
        self.episode_count += 1

        # Warmup phase
        if self.episode_count <= self.total_warmup_episodes:
            self.is_warmup = True
            # Alternate objectives every warmup_block_episodes
            obj_idx = ((self.episode_count - 1) // self.warmup_block_episodes) % len(
                self.objectives
            )
            self.env.target_objective = self.objectives[obj_idx]
            self.env.constraints = {}
            self.env.is_warmup = True
        else:
            # Constrained phase
            if self.is_warmup:
                self.is_warmup = False
                self.env.is_warmup = False
                print(f"\nWarmup complete at episode {self.episode_count}")
                print(f"Best warmup rewards: {self.env.warmup_best_rewards}")

            constrained_episode = self.episode_count - self.total_warmup_episodes
            phase = (constrained_episode - 1) // self.episodes_per_step

            # Alternate objectives
            obj_idx = phase % len(self.objectives)
            target_obj = self.objectives[obj_idx]

            # Constraint level. "cycle" restarts the ramp once it tops out;
            # "ramp" climbs once and holds there.
            sweep = phase // len(self.objectives)
            if self.constraint_level_schedule == "ramp":
                level = min(sweep, self.steps_per_objective - 1)
            else:
                level = sweep % self.steps_per_objective

            # Constrain the other objectives: "reference" takes them all from
            # one archived design, "box" from the range the front spans.
            max_frac = (level + 1) / self.steps_per_objective
            constraints = None
            if self.env.constraint_source == "reference":
                constraints = self.env.constraint_reference(
                    target_obj,
                    level_frac=max_frac,
                    randomise=self.randomise_constraints,
                )
            if constraints is None:
                # Also the fallback while the front is too small to measure on
                constraints = {}
                for i, obj in enumerate(self.objectives):
                    if i != obj_idx:
                        low, high = self.env.constraint_range(obj)
                        top = low + max_frac * (high - low)
                        constraints[obj] = (
                            np.random.uniform(low, top)
                            if self.randomise_constraints
                            else top
                        )

            self.env.target_objective = target_obj
            self.env.constraints = constraints

        obs = self._get_obs(state)
        info = {"mask": self.get_action_mask()}
        return obs, info

    def step(self, action, **kwargs):
        """Execute action (dict with material and thickness)."""
        material_idx = int(action["material"])
        thickness = float(action["thickness"][0])

        # CoatingEnvironment.step returns: state, rewards, terminated, finished, total_reward, full_action, vals
        # Pass through any kwargs (e.g., episode_data for Pareto tracking)
        state, rewards, terminated, finished, total_reward, full_action, vals = (
            self.env.step([material_idx, thickness], **kwargs)
        )
        self.prev_material = material_idx
        self.current_layer += 1

        done = terminated or finished

        # Use reward from base environment (already has target objective and constraints applied)
        # The environment handles intermediate rewards via use_intermediate_reward flag
        reward = total_reward

        obs = self._get_obs(state)
        info = {"mask": self.get_action_mask()}
        if done:
            info["vals"] = vals

        return obs, reward, done, False, info


class RolloutBuffer:
    """Simple rollout buffer for on-policy learning."""

    def __init__(self):
        self.clear()

    def clear(self):
        self.observations = []
        self.materials = []
        self.thicknesses = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.masks = []
        # Target objective of each step's episode. Read by the advantage
        # normalisation, since a rollout can span several targets.
        self.target_idxs = []
        self.ptr = 0

    def add(
        self, obs, material, thickness, reward, value, log_prob, done, mask, target_idx
    ):
        self.observations.append(obs)
        self.materials.append(material)
        self.thicknesses.append(thickness)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        self.masks.append(mask)
        self.target_idxs.append(target_idx)
        self.ptr += 1

    def finalize(
        self, last_value: float, gamma: float = 0.99, gae_lambda: float = 0.95
    ):
        """Compute returns and advantages using GAE."""
        self.returns = np.zeros(len(self.rewards), dtype=np.float32)
        self.advantages = np.zeros(len(self.rewards), dtype=np.float32)

        gae = 0.0
        for t in reversed(range(len(self.rewards))):
            next_value = (
                last_value if t == len(self.rewards) - 1 else self.values[t + 1]
            )
            next_not_done = 0.0 if self.dones[t] else 1.0
            delta = (
                self.rewards[t] + gamma * next_value * next_not_done - self.values[t]
            )
            gae = delta + gamma * gae_lambda * next_not_done * gae
            self.advantages[t] = gae
            self.returns[t] = gae + self.values[t]

    def get(self):
        """Return all data as tensors, with episode grouping recovered from dones."""
        dones = np.asarray(self.dones)
        ends = np.flatnonzero(dones)
        episode_idx = np.zeros(len(dones), dtype=np.int64)
        episode_idx[ends[:-1] + 1] = 1
        episode_idx = np.cumsum(episode_idx)
        starts = np.concatenate(([0], ends[:-1] + 1))
        step_idx = np.arange(len(dones)) - starts[episode_idx]
        observations = np.array(self.observations)
        return {
            "observations": torch.FloatTensor(observations),
            "materials": torch.LongTensor(self.materials),
            "thicknesses": torch.FloatTensor(self.thicknesses),
            "log_probs": torch.FloatTensor(self.log_probs),
            "returns": torch.FloatTensor(self.returns),
            "advantages": torch.FloatTensor(self.advantages),
            "masks": torch.FloatTensor(np.array(self.masks)),
            "episode_last_obs": torch.FloatTensor(observations[ends]),
            "episode_idx": torch.LongTensor(episode_idx),
            "step_idx": torch.LongTensor(step_idx),
            "target_idxs": torch.LongTensor(self.target_idxs),
        }


class LSTMEncoder(nn.Module):
    """Recurrent encoder of the layer stack.

    encode_step carries (h, c) so a rollout step feeds only the new layer.
    """

    def __init__(self, in_dim, hidden=32, layers=1, **_):
        super().__init__()
        self.net = nn.LSTM(in_dim, hidden, layers, batch_first=True)
        self.out_dim = hidden

    def encode_all(self, seq):
        return self.net(seq)[0]

    def encode_step(self, prefix, new_row, state):
        out, state = self.net(prefix if state is None else new_row, state)
        return out[:, -1], state


class _AttentionBlock(nn.Module):
    """Pre-norm causal self-attention + feedforward."""

    def __init__(self, dim, heads, ff_mult):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_mult * dim), nn.ReLU(), nn.Linear(ff_mult * dim, dim)
        )

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)

        def split(z):
            return z.view(b, t, self.heads, d // self.heads).transpose(1, 2)

        attended = torch.nn.functional.scaled_dot_product_attention(
            split(q), split(k), split(v), is_causal=True
        )
        x = x + self.proj(attended.transpose(1, 2).reshape(b, t, d))
        return x + self.ff(self.norm2(x))


class AttentionEncoder(nn.Module):
    """Causal self-attention encoder of the layer stack.

    Causal and dropout-free so position t sees only the first t layers and
    rollout and update encode the same state identically.
    """

    def __init__(self, in_dim, hidden=32, layers=2, heads=2, ff_mult=2, max_len=50):
        super().__init__()
        self.embed = nn.Linear(in_dim, hidden)
        self.pos = nn.Parameter(torch.randn(1, max_len, hidden) * 0.02)
        self.blocks = nn.ModuleList(
            [_AttentionBlock(hidden, heads, ff_mult) for _ in range(layers)]
        )
        self.out_dim = hidden

    def encode_all(self, seq):
        x = self.embed(seq) + self.pos[:, : seq.shape[1]]
        for block in self.blocks:
            x = block(x)
        return x

    def encode_step(self, prefix, new_row, state):
        return self.encode_all(prefix)[:, -1], None


PRE_MODELS = {"lstm": LSTMEncoder, "attention": AttentionEncoder}

# Thickness sampling width as a fraction of (max_t - min_t), so the clamp
# means the same thing at any thickness range.
LOG_STD_MIN, LOG_STD_MAX = -5.0, -1.2  # 0.7% .. 30% of the thickness range
# Where the width starts, strictly inside the clamp (8% of the range).
LOG_STD_INIT = -2.5
INV_SQRT2 = 1.0 / math.sqrt(2.0)
INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)
LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)
LOG_SQRT_2PI_E = 0.5 * math.log(2.0 * math.pi * math.e)
EPS = float(torch.finfo(torch.float32).eps)


class HybridActorCritic(nn.Module):
    """Actor-Critic with hybrid discrete+continuous actions.

    Heads: masked categorical material, TruncatedNormal thickness, value.
    pre_model_type ("linear", "lstm", "attention") sets the stack encoder.
    """

    def __init__(
        self,
        obs_dim: int,
        n_materials: int,
        min_thickness: float,
        max_thickness: float,
        hidden_dims: List[int] = [256, 256],
        pre_model_type: str = "linear",
        pre_model_params: dict = None,
        max_layers: int = None,
        n_constraints: int = None,
    ):
        super().__init__()
        self.n_materials = n_materials
        self.min_t = min_thickness
        self.max_t = max_thickness
        self.pre_model_type = pre_model_type
        self.use_sequence = pre_model_type in PRE_MODELS
        if pre_model_type != "linear" and not self.use_sequence:
            raise ValueError(
                f"Unknown pre_model_type {pre_model_type!r}; "
                f"expected 'linear' or one of {sorted(PRE_MODELS)}"
            )

        if self.use_sequence:
            assert max_layers is not None and n_constraints is not None, (
                f"max_layers and n_constraints required for {pre_model_type}"
            )

            # Observation structure: [layer_sequence (flattened), current_layer, constraints]
            n_features_per_layer = 1 + n_materials + 2  # thickness + one-hot + 2
            self.max_layers = max_layers
            self.n_features_per_layer = n_features_per_layer
            self.n_constraints = n_constraints

            self.encoder = PRE_MODELS[pre_model_type](
                n_features_per_layer, max_len=max_layers, **(pre_model_params or {})
            )

            # After the encoder: [encoded stack, target one-hot, constraints]
            combined_dim = (
                self.encoder.out_dim + n_constraints + n_constraints
            )  # one objective per constraint
            prev_dim = combined_dim
        else:
            # Standard MLP trunk, straight off the flattened observation
            prev_dim = obs_dim

        # Trunk MLP
        layers = []
        for h in hidden_dims:
            layers.extend([nn.Linear(prev_dim, h), nn.ReLU()])
            prev_dim = h
        self.trunk = nn.Sequential(*layers)

        # Discrete head (material)
        self.material_head = nn.Linear(prev_dim, n_materials)

        # Continuous head (thickness) - conditioned on material choice
        # Takes concatenated features + one-hot material
        self.thickness_mean = nn.Linear(prev_dim + n_materials, 1)
        self.thickness_logstd = nn.Linear(prev_dim + n_materials, 1)

        # One value head, not one per objective: the trunk input already
        # carries the target one-hot and the constraint thresholds.
        self.n_objectives = n_constraints  # width of the observation tail
        self.value_head = nn.Linear(prev_dim, 1)

    def _obs_layers(self, obs):
        """Layer-stack rows of an observation, without the objective tail."""
        tail = self.n_objectives + self.n_constraints
        return obs[:, :-tail].view(-1, self.max_layers, self.n_features_per_layer)

    def _seq_features(self, obs, episode_idx=None, step_idx=None):
        """Encoded layer stack at each step.

        With episode_idx/step_idx, obs holds one final observation per episode
        and each step's feature is gathered from a single causal pass.
        """
        layers = self._obs_layers(obs)
        # Prepend a padding row (row -1 is always padding) so position t
        # means "t layers placed", including t=0.
        seqs = torch.cat([layers[:, -1:], layers[:, :-1]], dim=1)
        encoded = self.encoder.encode_all(seqs)
        if episode_idx is None:
            step_idx = (layers[:, :, 0] > 0).sum(dim=1)
            episode_idx = torch.arange(len(obs))
        return encoded[episode_idx, step_idx]

    def _seq_step(self, obs, state=None):
        """Encoded layer stack during rollout, advancing by the new layer.

        A recurrent encoder carries state between steps; a first step or a
        missing state rebuilds from the layers placed so far.
        """
        layers = self._obs_layers(obs)
        n_placed = int((layers[0, :, 0] > 0).sum())
        if state is None or n_placed == 0:
            # Padding row (row -1 is always padding) then the layers so far.
            prefix = torch.cat([layers[:, -1:], layers[:, :n_placed]], dim=1)
            return self.encoder.encode_step(prefix, None, None)
        return self.encoder.encode_step(None, layers[:, n_placed - 1 : n_placed], state)

    def _trunk_features(self, obs, seq_features=None):
        if not self.use_sequence:
            return self.trunk(obs)
        tail = obs[:, -(self.n_objectives + self.n_constraints) :]
        return self.trunk(torch.cat([seq_features, tail], dim=1))

    def _material_log_probs(self, features, mask):
        logits = self.material_head(features) + (1.0 - mask) * -1e8
        return torch.log_softmax(logits, dim=-1)

    def _thickness_params(self, features, material):
        """Mean and scale of the thickness normal for the chosen material, its
        truncation bounds in standard units, the lower bound's CDF and the
        truncated mass. Shared by act and evaluate so log probs stay comparable.
        """
        material_onehot = torch.nn.functional.one_hot(
            material, num_classes=self.n_materials
        ).to(features.dtype)
        thickness_input = torch.cat([features, material_onehot], dim=-1)
        width = self.max_t - self.min_t
        mean = self.min_t + width * torch.sigmoid(
            self.thickness_mean(thickness_input).squeeze(-1)
        )
        log_std = torch.clamp(
            self.thickness_logstd(thickness_input).squeeze(-1),
            LOG_STD_MIN,
            LOG_STD_MAX,
        )
        scale = width * torch.exp(log_std)
        a = (self.min_t - mean) / scale
        b = (self.max_t - mean) / scale
        cdf_a = 0.5 * (1.0 + torch.erf(a * INV_SQRT2))
        cdf_b = 0.5 * (1.0 + torch.erf(b * INV_SQRT2))
        mass = (cdf_b - cdf_a).clamp_min(EPS)
        return mean, scale, a, b, cdf_a, mass

    @staticmethod
    def _thickness_log_prob(mean, scale, mass, thickness):
        u = (thickness - mean) / scale
        return -0.5 * u * u - LOG_SQRT_2PI - torch.log(mass) - torch.log(scale)

    @staticmethod
    def _thickness_std(scale, a, b, mass):
        """Realised spread of the truncated normal, which sets the search width."""
        pdf_a = torch.exp(-0.5 * a * a) * INV_SQRT_2PI
        pdf_b = torch.exp(-0.5 * b * b) * INV_SQRT_2PI
        var = 1.0 - (b * pdf_b - a * pdf_a) / mass - ((pdf_b - pdf_a) / mass) ** 2
        return scale * var.clamp_min(0.0).sqrt()

    @staticmethod
    def _thickness_entropy(scale, a, b, mass):
        pdf_a = torch.exp(-0.5 * a * a) * INV_SQRT_2PI
        pdf_b = torch.exp(-0.5 * b * b) * INV_SQRT_2PI
        return (
            LOG_SQRT_2PI_E
            + torch.log(mass)
            - 0.5 * (b * pdf_b - a * pdf_a) / mass
            + torch.log(scale)
        )

    def forward(self, obs, mask, deterministic=False, lstm_state=None):
        """Forward pass returning actions, log_probs, and value.

        Args:
            obs: observation tensor
            mask: action mask
            deterministic: whether to sample or take argmax
            lstm_state: (h, c) carried from this episode's previous step, or
                None at its first step

        Returns:
            material: sampled material index
            thickness: sampled thickness value
            log_prob: total log probability (discrete + continuous)
            value: state value V(s)
            lstm_state: updated (h, c) to pass to the next step (None without LSTM)
        """
        seq_features = None
        if self.use_sequence:
            seq_features, lstm_state = self._seq_step(obs, lstm_state)
        features = self._trunk_features(obs, seq_features)

        log_probs_d = self._material_log_probs(features, mask)
        if deterministic:
            material = log_probs_d.argmax(dim=-1)
        else:
            material = torch.multinomial(log_probs_d.exp(), 1).squeeze(-1)

        mean, scale, a, b, cdf_a, mass = self._thickness_params(features, material)
        if deterministic:
            thickness = mean
        else:
            # Inverse-CDF sample of the truncated normal
            u = torch.rand_like(mean).clamp(EPS, 1.0 - EPS)
            z = math.sqrt(2.0) * torch.erfinv(2.0 * (cdf_a + u * mass) - 1.0)
            thickness = mean + scale * torch.minimum(torch.maximum(z, a), b)

        log_prob = log_probs_d.gather(-1, material.unsqueeze(-1)).squeeze(
            -1
        ) + self._thickness_log_prob(mean, scale, mass, thickness)
        value = self.value_head(features).squeeze(-1)
        std = self._thickness_std(scale, a, b, mass)
        return material, thickness, log_prob, value, std, lstm_state

    def evaluate_actions(
        self,
        obs,
        mask,
        materials,
        thicknesses,
        episode_last_obs=None,
        episode_idx=None,
        step_idx=None,
    ):
        """Evaluate log_probs and values for given actions.

        Args:
            obs: observation tensor
            mask: action mask
            materials: material actions
            thicknesses: thickness actions
            episode_last_obs: final observation per episode; if given with
                episode_idx/step_idx, the LSTM runs once per episode
                instead of once per step
            episode_idx: episode index (into episode_last_obs) of each step
            step_idx: position of each step within its episode
        """
        seq_features = None
        if self.use_sequence:
            seq_features = (
                self._seq_features(episode_last_obs, episode_idx, step_idx)
                if episode_last_obs is not None
                else self._seq_features(obs)
            )
        features = self._trunk_features(obs, seq_features)

        log_probs_d = self._material_log_probs(features, mask)
        mean, scale, a, b, _, mass = self._thickness_params(features, materials)

        log_prob = log_probs_d.gather(-1, materials.unsqueeze(-1)).squeeze(
            -1
        ) + self._thickness_log_prob(mean, scale, mass, thicknesses)
        value = self.value_head(features).squeeze(-1)

        # Separate entropies: the two heads carry their own coefficients.
        entropy_d = -(log_probs_d.exp() * log_probs_d).sum(dim=-1)
        entropy_c = self._thickness_entropy(scale, a, b, mass)
        return log_prob, value, entropy_d, entropy_c


def select_bc_episodes(
    front_rewards,
    front_episodes,
    objectives,
    target_objective,
    constraints,
    top_k: int = 8,
):
    """Pick archive episodes matching the current constrained subproblem.

    Ranks Pareto designs by (constraint violation, target reward) and returns
    the top_k, falling back to the least-violating when none are feasible.
    """
    if target_objective not in objectives:
        return front_episodes
    t_idx = objectives.index(target_objective)
    scored = []
    for (reward_vec, _), episode in zip(front_rewards, front_episodes):
        if episode is None:
            continue
        r = np.asarray(reward_vec, dtype=float)
        violation = 0.0
        for j, obj in enumerate(objectives):
            if j == t_idx:
                continue
            thr = float(constraints.get(obj, 0.0))
            violation = max(violation, thr - float(r[j]))
        scored.append((max(0.0, violation), -float(r[t_idx]), episode))
    if not scored:
        return None
    scored.sort(key=lambda s: (s[0], s[1]))
    return [s[2] for s in scored[:top_k]]


def compute_bc_loss_from_pareto(
    policy, pareto_episodes, env_wrapper, bc_weight=0.1, batch_size=32
):
    """Behaviour cloning loss on transitions drawn from archived Pareto episodes.

    Each episode's flat layer-stack observations and actions are cached on it
    the first time it is drawn. The objective tail is filled in per call, as
    the target and thresholds change every episode.
    """
    valid = [ep for ep in (pareto_episodes or []) if ep is not None]
    if not valid:
        return 0.0
    picks = torch.randperm(len(valid))[: min(batch_size, len(valid))].tolist()
    max_thickness = env_wrapper.env.max_thickness
    cached = []
    for i in picks:
        episode = valid[i]
        entry = episode.get("_bc_cache")
        if entry is None:
            states = episode.get("states")
            materials = episode.get("discrete_actions")
            thicknesses = episode.get("continuous_actions")
            if not states or not materials or not thicknesses:
                continue
            entry = (
                np.stack([s.observation_array(max_thickness).ravel() for s in states]),
                np.array([int(m) for m in materials], dtype=np.int64),
                np.array([float(t) for t in thicknesses], dtype=np.float32),
            )
            episode["_bc_cache"] = entry
        cached.append(entry)
    if not cached:
        return 0.0

    base = np.concatenate([c[0] for c in cached])
    materials = np.concatenate([c[1] for c in cached])
    thicknesses = np.concatenate([c[2] for c in cached])
    n = min(batch_size, len(base))
    idx = torch.randperm(len(base))[:n].numpy()

    n_obj = len(env_wrapper.objectives)
    width = base.shape[1]
    obs = np.empty((n, width + 2 * n_obj), dtype=np.float32)
    obs[:, :width] = base[idx]
    target = env_wrapper.env.target_objective
    constraints = env_wrapper.env.constraints
    for j, objective in enumerate(env_wrapper.objectives):
        obs[:, width + j] = 1.0 if objective == target else 0.0
    for j, objective in enumerate(env_wrapper.env.optimise_parameters):
        obs[:, width + n_obj + j] = constraints.get(objective, 0.0)

    log_probs, *_ = policy.evaluate_actions(
        torch.from_numpy(obs),
        torch.ones(n, policy.n_materials),
        torch.from_numpy(materials[idx]),
        torch.from_numpy(thicknesses[idx]),
    )
    return -bc_weight * log_probs.mean()


class FlatAdam:
    """Adam over one flat parameter buffer, so a step is a few vector ops
    instead of one per parameter tensor. Parameters and their gradients are
    views into the buffer.
    """

    def __init__(self, params, lr, betas=(0.9, 0.999), eps=1e-8):
        self.params = list(params)
        n = sum(p.numel() for p in self.params)
        self.flat = torch.zeros(n)
        self.flat_grad = torch.zeros(n)
        self.exp_avg = torch.zeros(n)
        self.exp_avg_sq = torch.zeros(n)
        self.step_count = 0
        self.param_groups = [{"lr": lr, "betas": betas, "eps": eps}]
        offset = 0
        for p in self.params:
            k = p.numel()
            self.flat[offset : offset + k].copy_(p.data.reshape(-1))
            p.data = self.flat[offset : offset + k].view_as(p)
            p.grad = self.flat_grad[offset : offset + k].view_as(p)
            offset += k

    def zero_grad(self):
        self.flat_grad.zero_()

    @torch.no_grad()
    def clip_grad_norm_(self, max_norm):
        norm = self.flat_grad.norm()
        self.flat_grad.mul_(torch.clamp(max_norm / (norm + 1e-6), max=1.0))
        return norm

    @torch.no_grad()
    def step(self):
        group = self.param_groups[0]
        lr, (beta1, beta2), eps = group["lr"], group["betas"], group["eps"]
        self.step_count += 1
        self.exp_avg.lerp_(self.flat_grad, 1.0 - beta1)
        self.exp_avg_sq.mul_(beta2).addcmul_(
            self.flat_grad, self.flat_grad, value=1.0 - beta2
        )
        bias1 = 1.0 - beta1**self.step_count
        bias2 = 1.0 - beta2**self.step_count
        denom = (self.exp_avg_sq.sqrt() / math.sqrt(bias2)).add_(eps)
        self.flat.addcdiv_(self.exp_avg, denom, value=-lr / bias1)

    def state_dict(self):
        return {
            "flat": self.flat.clone(),
            "exp_avg": self.exp_avg.clone(),
            "exp_avg_sq": self.exp_avg_sq.clone(),
            "step_count": self.step_count,
            "param_groups": [dict(g) for g in self.param_groups],
        }

    def load_state_dict(self, state):
        self.flat.copy_(state["flat"])
        self.exp_avg.copy_(state["exp_avg"])
        self.exp_avg_sq.copy_(state["exp_avg_sq"])
        self.step_count = int(state["step_count"])
        self.param_groups = [dict(g) for g in state["param_groups"]]


class PPOAgent:
    """PPO agent with hybrid actions."""

    def __init__(
        self,
        policy: HybridActorCritic,
        lr: float = 3e-4,
        gamma: float = 0.99,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        ent_coef_thickness: float = None,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
    ):
        self.policy = policy
        self.optimizer = FlatAdam(policy.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        # Thickness-head entropy coefficient; None tracks ent_coef
        self.ent_coef_thickness = (
            ent_coef if ent_coef_thickness is None else ent_coef_thickness
        )
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.lstm_state = None  # carried across rollout steps of an episode
        self.last_thickness_std = float("nan")  # sampling width of the last act()

    def act(self, obs, mask, deterministic=False):
        """Sample action from policy.

        The LSTM state is carried between calls so each step feeds the LSTM
        only the layer just placed. _seq_step restarts it whenever the
        observation has no layers placed, so a reset needs no bookkeeping here.

        Args:
            obs: observation
            mask: action mask
            deterministic: whether to sample or take argmax
        """
        with torch.inference_mode():
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            mask_t = torch.as_tensor(mask, dtype=torch.float32).unsqueeze(0)
            material, thickness, log_prob, value, std, self.lstm_state = self.policy(
                obs_t, mask_t, deterministic, self.lstm_state
            )
            out = torch.cat(
                [material.float(), thickness, log_prob, value, std]
            ).tolist()
        self.last_thickness_std = out[4]
        return int(out[0]), out[1], out[2], out[3]

    def act_batch(self, obs, mask):
        """Sample actions for episodes at the same layer count in one policy call."""
        with torch.inference_mode():
            material, thickness, log_prob, value, std, _ = self.policy(
                torch.as_tensor(obs, dtype=torch.float32),
                torch.as_tensor(mask, dtype=torch.float32),
            )
        return (
            material.tolist(),
            thickness.tolist(),
            log_prob.tolist(),
            value.tolist(),
            std.tolist(),
        )

    def update(
        self,
        rollout_data: dict,
        n_epochs: int,
        batch_size: int,
        pareto_episodes=None,
        bc_weight=0.1,
        env_wrapper=None,
        bc_every: int = 0,
    ):
        """PPO update using rollout data with optional BC loss from Pareto episodes.

        Args:
            rollout_data: dict with observations, actions, returns, etc.
            n_epochs: number of optimization epochs
            batch_size: minibatch size
            pareto_episodes: optional list of Pareto front episodes for BC loss
            bc_weight: weight for behavior cloning loss
            env_wrapper: environment wrapper (for BC loss)
            bc_every: add the BC loss on every k-th minibatch; 0 keeps it on
                the first minibatch of each epoch
        """
        obs = rollout_data["observations"]
        materials = rollout_data["materials"]
        thicknesses = rollout_data["thicknesses"]
        old_log_probs = rollout_data["log_probs"]
        returns = rollout_data["returns"]
        advantages = rollout_data["advantages"]
        masks = rollout_data["masks"]
        target_idxs = rollout_data["target_idxs"]

        # Normalise advantages within each target objective: a rollout spans
        # several targets, and they sit at different reward levels.
        advantages = advantages.clone()
        for t in torch.unique(target_idxs):
            sel = target_idxs == t
            group = advantages[sel]
            advantages[sel] = (
                (group - group.mean()) / (group.std() + 1e-8)
                if group.numel() > 1
                else torch.zeros_like(group)
            )

        n_samples = len(obs)
        logs = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "entropy_material": 0.0,
            "entropy_thickness": 0.0,
            "clip_frac": 0.0,
            "bc_loss": 0.0,
        }
        n_updates = 0

        # With an LSTM, minibatch whole episodes: one LSTM pass over the
        # episode's final observation covers all of its steps.
        episode_last_obs = rollout_data.get("episode_last_obs")
        episode_idx = rollout_data.get("episode_idx")
        step_idx = rollout_data.get("step_idx")
        use_episode_batches = self.policy.use_sequence and episode_last_obs is not None
        if use_episode_batches:
            n_episodes = len(episode_last_obs)
            episode_steps = [
                torch.nonzero(episode_idx == e, as_tuple=True)[0]
                for e in range(n_episodes)
            ]
            # keep roughly batch_size steps per minibatch
            episodes_per_batch = max(1, round(batch_size * n_episodes / n_samples))

        use_bc = (
            pareto_episodes is not None and bc_weight > 0 and env_wrapper is not None
        )

        for epoch in range(n_epochs):
            if use_episode_batches:
                ep_order = torch.randperm(n_episodes)
                batches = [
                    ep_order[start : start + episodes_per_batch]
                    for start in range(0, n_episodes, episodes_per_batch)
                ]
            else:
                order = torch.randperm(n_samples)
                batches = [
                    order[start : start + batch_size]
                    for start in range(0, n_samples, batch_size)
                ]

            for i_batch, batch in enumerate(batches):
                if use_episode_batches:
                    batch_idx = torch.cat([episode_steps[int(e)] for e in batch])
                    episode_kwargs = {
                        "episode_last_obs": episode_last_obs[batch],
                        "episode_idx": torch.cat(
                            [
                                torch.full(
                                    (len(episode_steps[int(e)]),), i, dtype=torch.long
                                )
                                for i, e in enumerate(batch)
                            ]
                        ),
                        "step_idx": step_idx[batch_idx],
                    }
                else:
                    batch_idx = batch
                    episode_kwargs = {}

                log_probs, values, entropy_d, entropy_c = self.policy.evaluate_actions(
                    obs[batch_idx],
                    masks[batch_idx],
                    materials[batch_idx],
                    thicknesses[batch_idx],
                    **episode_kwargs,
                )

                # Policy loss (clipped surrogate)
                log_prob_diff = log_probs - old_log_probs[batch_idx]
                # Clamp to prevent overflow
                log_prob_diff = torch.clamp(log_prob_diff, -10, 10)
                ratio = torch.exp(log_prob_diff)
                adv = advantages[batch_idx]
                surr1 = ratio * adv
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
                    * adv
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = ((values - returns[batch_idx]) ** 2).mean()

                # Total loss
                loss = (
                    policy_loss
                    + self.vf_coef * value_loss
                    - self.ent_coef * entropy_d.mean()
                    - self.ent_coef_thickness * entropy_c.mean()
                )

                # Behaviour cloning rides a PPO minibatch so its gradient joins
                # the PPO gradient in one clipped step.
                bc_turn = i_batch % bc_every == 0 if bc_every else i_batch == 0
                if use_bc and bc_turn:
                    bc_loss = compute_bc_loss_from_pareto(
                        self.policy,
                        pareto_episodes,
                        env_wrapper,
                        bc_weight,
                        batch_size=32,
                    )
                    if torch.is_tensor(bc_loss):
                        logs["bc_loss"] += bc_loss.item()
                        loss = loss + bc_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.clip_grad_norm_(self.max_grad_norm)
                self.optimizer.step()

                # Logging
                logs["policy_loss"] += policy_loss.item()
                logs["value_loss"] += value_loss.item()
                # Kept as the sum so ppo.entropy still means what it did
                entropy_d_mean = entropy_d.mean().item()
                entropy_c_mean = entropy_c.mean().item()
                logs["entropy"] += entropy_d_mean + entropy_c_mean
                logs["entropy_material"] += entropy_d_mean
                logs["entropy_thickness"] += entropy_c_mean
                logs["clip_frac"] += (
                    ((ratio - 1.0).abs() > self.clip_range).float().mean().item()
                )
                n_updates += 1

        # Average over updates
        for k in logs:
            logs[k] /= max(n_updates, 1)

        return logs


def collect_lockstep(
    env, agent, buffer, n_episodes, objective_to_idx, track_bc, record
):
    """Run n_episodes episodes in lockstep: one batched policy call per layer,
    then one env step per live episode.

    The env's per-episode fields are swapped in and out around each step, so
    the schedule, constraint sampling, reward and Pareto staging are those of
    the sequential loop. The env must already be reset for the first episode
    and is left reset for the next update, so episode_count advances by
    n_episodes as before. Returns the next (obs, mask) and the step count.
    """
    base = env.env
    keys = ("current_state", "current_index", "done", "target_objective", "constraints")

    episodes = []
    for i in range(n_episodes):
        if i == 0:
            obs, mask = env._get_obs(base.current_state), env.get_action_mask()
        else:
            obs, info = env.reset()
            mask = info["mask"]
        episodes.append(
            {
                "env": tuple(getattr(base, k) for k in keys),
                "wrapper": (env.prev_material, env.current_layer),
                "obs": obs,
                "mask": mask,
                "states": [],
                "discrete_actions": [],
                "continuous_actions": [],
                "stds": [],
                "thicknesses": [],
                "transitions": [],
            }
        )

    live = list(range(n_episodes))
    n_steps = 0
    while live:
        obs_batch = np.stack([episodes[i]["obs"] for i in live])
        mask_batch = np.stack([episodes[i]["mask"] for i in live])
        materials, thicknesses, log_probs, values, stds = agent.act_batch(
            obs_batch, mask_batch
        )
        still_live = []
        for j, i in enumerate(live):
            ep = episodes[i]
            for k, v in zip(keys, ep["env"]):
                setattr(base, k, v)
            env.prev_material, env.current_layer = ep["wrapper"]

            target_idx = objective_to_idx[base.target_objective]
            material, thickness = materials[j], thicknesses[j]
            action = {
                "material": material,
                "thickness": np.array([thickness], dtype=np.float32),
            }
            kwargs = {}
            if track_bc:
                ep["states"].append(base.current_state.copy())
                ep["discrete_actions"].append(torch.tensor(material))
                ep["continuous_actions"].append(torch.tensor(thickness))
                kwargs["episode_data"] = {
                    k: ep[k]
                    for k in ("states", "discrete_actions", "continuous_actions")
                }
            ep["stds"].append(stds[j])
            ep["thicknesses"].append(thickness)

            next_obs, reward, done, _, info = env.step(action, **kwargs)
            ep["transitions"].append(
                (
                    ep["obs"],
                    material,
                    thickness,
                    reward,
                    values[j],
                    log_probs[j],
                    done,
                    ep["mask"],
                    target_idx,
                )
            )
            n_steps += 1
            if done:
                if "vals" in info:
                    record(info["vals"], reward, ep["stds"], ep["thicknesses"])
            else:
                ep["obs"], ep["mask"] = next_obs, info["mask"]
                still_live.append(i)
            ep["env"] = tuple(getattr(base, k) for k in keys)
            ep["wrapper"] = (env.prev_material, env.current_layer)
        live = still_live

    for ep in episodes:
        for transition in ep["transitions"]:
            buffer.add(*transition)
    obs, info = env.reset()
    return obs, info["mask"], n_steps


def train(config_path: str, save_dir: str):
    """Train single-agent PPO with sequential constraints."""
    parser = configparser.ConfigParser()
    parser.read(config_path)
    section = "hppo_sequential"

    def _get(key, fallback, cast=str):
        return cast(parser.get(section, key, fallback=str(fallback)))

    # Load config and materials
    config = load_config(config_path)
    materials = load_materials_from_parser(parser, config_path)

    # Read hyperparameters
    total_episodes = _get("total_episodes", 10000, int)
    warmup_episodes = _get("warmup_episodes", 500, int)
    # 0 / unset = one warmup block per objective; smaller value = interleaved
    warmup_block_episodes = _get("warmup_block_episodes", 0, int) or None
    bc_selection = _get("bc_selection", "target", str)
    episodes_per_step = _get("episodes_per_step", 200, int)
    steps_per_objective = _get("steps_per_objective", 10, int)
    episodes_per_update = _get("episodes_per_update", 10, int)
    n_epochs = _get("n_epochs", 5, int)
    batch_size = _get("batch_size", 64, int)
    constraint_penalty = _get("constraint_penalty", 3.0, float)
    pareto_bonus = _get("pareto_bonus", 0.0, float)
    bc_weight = _get("bc_weight", 0.0, float)
    lr = _get("lr", 3e-4, float)
    lr_final = _get("lr_final", lr, float)
    lr_decay_episodes = _get("lr_decay_episodes", total_episodes, int)
    restart_decay_on_phase = _get(
        "restart_decay_on_phase", False, lambda x: x.lower() == "true"
    )
    gamma = _get("gamma", 0.99, float)
    gae_lambda = _get("gae_lambda", 0.95, float)
    clip_range = _get("clip_range", 0.2, float)
    ent_coef = _get("ent_coef", 0.01, float)
    ent_coef_final = _get("ent_coef_final", ent_coef, float)
    # The thickness head can carry its own entropy coefficient. Unset, both
    # track ent_coef; set alone, the final defaults to itself (no annealing).
    _ent_thickness = _get("ent_coef_thickness", "", str).strip()
    if _ent_thickness:
        ent_coef_thickness = float(_ent_thickness)
        ent_coef_thickness_final = _get(
            "ent_coef_thickness_final", ent_coef_thickness, float
        )
    else:
        ent_coef_thickness = ent_coef
        ent_coef_thickness_final = ent_coef_final
    vf_coef = _get("vf_coef", 0.5, float)
    max_grad_norm = _get("max_grad_norm", 0.5, float)
    min_layers_before_air = _get("min_layers_before_air", 4, int)
    mask_consecutive = _get(
        "mask_consecutive_materials", True, lambda x: x.lower() == "true"
    )
    # Layer-stack encoder. Older configs said use_lstm/lstm_hidden/lstm_layers;
    # those still work and map onto the same setting.
    pre_model_type = _get("pre_model_type", "", str).strip().lower()
    pre_model_params = ast.literal_eval(_get("pre_model_params", "{}") or "{}")
    if not pre_model_type:
        use_lstm = _get("use_lstm", False, lambda x: x.lower() == "true")
        pre_model_type = "lstm" if use_lstm else "linear"
        pre_model_params = {
            "hidden": _get("lstm_hidden", 128, int),
            "layers": _get("lstm_layers", 1, int),
        }
    use_sequence = pre_model_type != "linear"
    verbose = _get("verbose", 1, int)
    seed = _get("seed", 42, int)

    # Read logging frequencies
    mlflow_log_freq = parser.getint("general", "mlflow_log_freq", fallback=50)
    plot_freq = _get("plot_freq", 500, int)
    # Hypervolume is quadratic in archive size; keep it off the per-log path
    hypervolume_freq = _get("hypervolume_freq", plot_freq, int)
    randomise_constraints = _get(
        "randomise_constraints", False, lambda x: x.lower() == "true"
    )
    # "cycle" repeats the constraint ramp (the original behaviour); "ramp"
    # climbs once and holds at the top level for the rest of the run.
    constraint_level_schedule = _get("constraint_level_schedule", "cycle", str).strip()
    # Keep the per-objective bests that anchor the thresholds rising during the
    # constrained phase, instead of freezing them at whatever warmup reached.
    update_constraint_bounds = _get(
        "update_constraint_bounds", False, lambda x: x.lower() == "true"
    )
    # "warmup" scales thresholds by this run's warmup best; "absolute" by the
    # normalised reward's own 1.0, rising only if a run beats it.
    constraint_anchor = _get("constraint_anchor", "warmup", str).strip().lower()
    # Fraction of the front's own spread to widen each end of that range by
    constraint_extend_low = _get("constraint_extend_low", 0.2, float)
    constraint_extend_high = _get("constraint_extend_high", 0.2, float)
    # "box": each threshold drawn from its objective's range. "reference":
    # all taken from one archived design, which answers the subproblem.
    constraint_source = _get("constraint_source", "box", str).strip().lower()
    # How far past the reference point to ask, in units of that point's local
    # neighbour spacing on the front. 0.0 asks only to match it.
    constraint_ref_extend = _get("constraint_ref_extend", 1.0, float)

    # Parse hidden layers
    hidden_str = _get("hidden", "[256, 256]")
    hidden = eval(hidden_str)

    # These tensors are too small to gain from intra-op threading, and extra
    # threads thrash a single-CPU slot. 0 leaves torch's default alone.
    torch_threads = _get("torch_threads", 1, int)
    lockstep_rollouts = _get("lockstep_rollouts", False, lambda x: x.lower() == "true")
    bc_every = _get("bc_every", 0, int)
    if torch_threads > 0:
        torch.set_num_threads(torch_threads)

    # Set seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Log hyperparameters
    # guarded: mlflow.log_params starts a run of its own if none is active
    if mlflow.active_run():
        mlflow.log_params(
            {
                "total_episodes": total_episodes,
                "warmup_episodes": warmup_episodes,
                "episodes_per_step": episodes_per_step,
                "steps_per_objective": steps_per_objective,
                "episodes_per_update": episodes_per_update,
                "n_epochs": n_epochs,
                "batch_size": batch_size,
                "constraint_penalty": constraint_penalty,
                "constraint_level_schedule": constraint_level_schedule,
                "update_constraint_bounds": update_constraint_bounds,
                "constraint_anchor": constraint_anchor,
                "constraint_extend_low": constraint_extend_low,
                "constraint_extend_high": constraint_extend_high,
                "constraint_source": constraint_source,
                "constraint_ref_extend": constraint_ref_extend,
                "pareto_bonus": pareto_bonus,
                "bc_weight": bc_weight,
                "lr": lr,
                "gamma": gamma,
                "gae_lambda": gae_lambda,
                "clip_range": clip_range,
                "ent_coef": ent_coef,
                "vf_coef": vf_coef,
                "hidden": str(hidden),
                "pre_model_type": pre_model_type,
                "pre_model_params": str(pre_model_params),
                "seed": seed,
            }
        )

    # Create environment with scheduling (like train_sb3_discrete)
    env = CoatOptHybridEnv(
        config=config,
        materials=materials,
        warmup_episodes=warmup_episodes,
        episodes_per_step=episodes_per_step,
        steps_per_objective=steps_per_objective,
        constraint_penalty=constraint_penalty,
        mask_consecutive_materials=mask_consecutive,
        min_layers_before_air=min_layers_before_air,
        randomise_constraints=randomise_constraints,
        warmup_block_episodes=warmup_block_episodes,
        constraint_level_schedule=constraint_level_schedule,
        update_constraint_bounds=update_constraint_bounds,
        constraint_anchor=constraint_anchor,
        constraint_extend_low=constraint_extend_low,
        constraint_extend_high=constraint_extend_high,
        constraint_source=constraint_source,
        constraint_ref_extend=constraint_ref_extend,
    )

    # Enable Pareto bonus (hypervolume improvement reward)
    if pareto_bonus > 0:
        env.env.enable_pareto_bonus(bonus=pareto_bonus)
        print(f"Enabled Pareto bonus: {pareto_bonus}")

    # Get actual observation size from environment
    test_obs, _ = env.reset()
    obs_dim = len(test_obs)
    n_materials = env.action_space["material"].n
    min_thickness = env.env.min_thickness
    max_thickness = env.env.max_thickness

    # Get number of constraints for the sequence encoder
    n_constraints = len(config.data.optimise_parameters)
    max_layers = config.data.n_layers

    policy = HybridActorCritic(
        obs_dim=obs_dim,
        n_materials=n_materials,
        min_thickness=min_thickness,
        max_thickness=max_thickness,
        hidden_dims=hidden,
        pre_model_type=pre_model_type,
        pre_model_params=pre_model_params,
        # n_constraints sizes the objective/constraint tail; max_layers is
        # only read by the sequence encoders
        max_layers=max_layers,
        n_constraints=n_constraints,
    )

    # Strong negative bias on air (index 0), so its probability cannot build up
    # while masked and fire the moment it is unmasked
    with torch.no_grad():
        initial_air_bias = -3.0
        policy.material_head.bias[0] = initial_air_bias

        # Start the width strictly inside the clamp: filling the bias with
        # LOG_STD_MAX sits on the boundary, where clamp passes no gradient.
        policy.thickness_logstd.weight.mul_(0.01)
        policy.thickness_logstd.bias.fill_(LOG_STD_INIT)

    if verbose:
        print(
            f"Initialized air material bias: {initial_air_bias:.2f} (low probability)"
        )

    if verbose:
        if use_sequence:
            print(f"Layer-stack encoder: {pre_model_type} {pre_model_params}")
        if lockstep_rollouts:
            print(f"Lockstep rollouts: {episodes_per_update} episodes per batch")
        if restart_decay_on_phase:
            print(
                f"LR/entropy decay will restart every {lr_decay_episodes} episodes (warm restarts enabled)"
            )
        if bc_weight > 0:
            print(f"Behavior cloning from Pareto episodes enabled: weight={bc_weight}")

    # Create agent and buffer
    agent = PPOAgent(
        policy=policy,
        lr=lr,
        gamma=gamma,
        clip_range=clip_range,
        ent_coef=ent_coef,
        ent_coef_thickness=ent_coef_thickness,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
    )

    buffer = RolloutBuffer()

    # Tracking
    ep_rewards = []
    ep_vals = []
    ep_lengths = []  # Track episode lengths
    ep_policy_std = []  # realised sampling std of the truncated normal
    ep_thickness_mean = []
    ep_thickness_within = []  # spread of layers within one design
    # Per-episode thresholds and target. Logging fires once per update, so
    # reading env.constraints there samples one episode in every stride.
    ep_constraints = []
    ep_targets = []
    sample_designs = []  # Track sample designs during warmup for debugging
    training_history = []  # periodic metrics rows for training_curves.png
    # Logging and checkpointing are checked once per update, so episode_count
    # rarely lands on a multiple of these intervals; fire on elapsed episodes.
    last_log_episode = 0
    last_plot_episode = 0
    last_hv_episode = 0
    last_hypervolume = None
    objectives = list(config.data.optimise_parameters)
    objective_targets = dict(getattr(config.data, "optimise_targets", {}) or {})

    # Create objective name -> index mapping for value head selection
    objective_to_idx = {obj: idx for idx, obj in enumerate(objectives)}

    def record_episode(vals, reward, stds, thicknesses):
        ep_rewards.append(reward)
        ep_vals.append(vals)
        ep_lengths.append(env.current_layer)
        ep_policy_std.append(float(np.mean(stds)))
        ep_thickness_mean.append(float(np.mean(thicknesses)))
        ep_thickness_within.append(float(np.std(thicknesses)))
        # Captured before reset() overwrites them for the next episode
        ep_constraints.append(dict(env.env.constraints))
        ep_targets.append(env.env.target_objective)
        # Only the last 100 are ever read
        if len(ep_constraints) > 200:
            del ep_constraints[:-100]
            del ep_targets[:-100]
        # Sample designs during warmup for debugging (keep last 100)
        if env.is_warmup:
            sample_designs.append(
                {
                    "episode": env.episode_count,
                    "target_obj": env.env.target_objective,
                    "state_array": env.env.current_state.get_array(),
                    "vals": vals.copy(),
                    "reward": reward,
                    "length": env.current_layer,
                }
            )
            if len(sample_designs) > 100:
                sample_designs.pop(0)

    # Annealing tracking
    lr_init = lr
    ent_coef_init = ent_coef
    current_ent_coef = ent_coef
    ent_coef_thickness_init = ent_coef_thickness
    current_ent_coef_thickness = ent_coef_thickness
    warmup_end_episode = 0  # Track when warmup ended for phase-based annealing reset
    was_warmup = True  # Track warmup state to detect transition

    # Resume from checkpoint if one exists
    ckpt = load_checkpoint(save_dir)
    if ckpt:
        policy.load_state_dict(ckpt["networks"]["policy"])
        agent.optimizer.load_state_dict(ckpt["optimizers"]["agent"])
        env.env.pareto_front_rewards = ckpt["pareto"]["rewards"]
        env.env.pareto_front_values = ckpt["pareto"]["values"]
        env.env.pareto_front_episodes = ckpt["pareto"]["episodes"]
        env.env.warmup_best_rewards = ckpt["pareto"]["warmup_best"]
        # Only-grows, so it cannot be rebuilt from the restored front alone
        spread = ckpt["pareto"].get("constraint_spread")
        if spread:
            env.env._constraint_spread.update(spread)
        env.env.n_evaluations = ckpt["pareto"].get("n_evaluations", 0)
        env.episode_count = ckpt["episode"]
        env.is_warmup = ckpt["meta"]["is_warmup"]
        env.env.is_warmup = ckpt["meta"]["is_warmup"]
        # Carry the log/plot clocks over so a resume keeps the same spacing
        last_log_episode = ckpt["episode"]
        last_plot_episode = ckpt["episode"]
        warmup_end_episode = ckpt["meta"]["warmup_end_episode"]
        was_warmup = ckpt["meta"]["is_warmup"]

        # Reload metrics history so training_history.csv stays continuous
        # across resumes (drop rows logged after the checkpoint episode)
        history_csv = Path(save_dir) / "training_history.csv"
        if history_csv.exists():
            try:
                import pandas as pd

                prior = pd.read_csv(history_csv)
                training_history = prior[prior["episode"] <= ckpt["episode"]].to_dict(
                    "records"
                )
            except Exception as e:
                if verbose:
                    print(f"  [history reload] skipped: {e}")

        if verbose:
            print(f"Resumed from checkpoint at episode {ckpt['episode']}")

    # Training loop
    obs, info = env.reset()
    mask = info["mask"]
    step_count = 0

    if verbose:
        print(f"Training for {total_episodes} episodes")

    # algorithm_runtime is the optimisation alone: everything the loop does for
    # reporting is timed into logging_io_seconds and subtracted.
    loop_start = time.perf_counter()
    logging_io_seconds = 0.0

    while env.episode_count < total_episodes:
        # Collect episodes for this update
        buffer.clear()
        episodes_collected = 0

        if lockstep_rollouts:
            obs, mask, n_steps = collect_lockstep(
                env,
                agent,
                buffer,
                episodes_per_update,
                objective_to_idx,
                bc_weight > 0,
                record_episode,
            )
            step_count += n_steps
        else:
            while episodes_collected < episodes_per_update:
                # Track episode data for Pareto BC loss (only if enabled)
                if bc_weight > 0:
                    episode_states = []
                    episode_materials = []
                    episode_thicknesses = []

                # Collect single episode. Names kept distinct from the BC tracking
                # above, which reuses the per-step actions for its own list.
                episode_done = False
                episode_stds = []
                episode_sampled_thicknesses = []
                while not episode_done:
                    # Recorded per step so the update can group advantages by
                    # target; the value head itself is shared across objectives.
                    target_obj_idx = objective_to_idx[env.env.target_objective]
                    material, thickness, log_prob, value = agent.act(obs, mask)
                    action = {
                        "material": material,
                        "thickness": np.array([thickness], dtype=np.float32),
                    }

                    # Track trajectory if BC loss enabled
                    if bc_weight > 0:
                        episode_states.append(env.env.current_state.copy())
                        episode_materials.append(torch.tensor(material))
                        episode_thicknesses.append(torch.tensor(thickness))

                    # Step environment (pass episode data on every step, used only when done=True)
                    step_kwargs = {}
                    if bc_weight > 0 and len(episode_states) > 0:
                        step_kwargs["episode_data"] = {
                            "states": episode_states,
                            "discrete_actions": episode_materials,
                            "continuous_actions": episode_thicknesses,
                        }

                    episode_stds.append(agent.last_thickness_std)
                    episode_sampled_thicknesses.append(thickness)

                    next_obs, reward, done, _, info = env.step(action, **step_kwargs)
                    next_mask = info["mask"]

                    buffer.add(
                        obs,
                        material,
                        thickness,
                        reward,
                        value,
                        log_prob,
                        done,
                        mask,
                        target_obj_idx,
                    )

                    if done:
                        episode_done = True
                        if "vals" in info:
                            record_episode(
                                info["vals"],
                                reward,
                                episode_stds,
                                episode_sampled_thicknesses,
                            )

                        episodes_collected += 1

                        obs, info = env.reset()
                        mask = info["mask"]
                    else:
                        obs = next_obs
                        mask = next_mask

                    step_count += 1

        # Finalize buffer
        _, _, _, last_value = agent.act(obs, mask)
        buffer.finalize(last_value, gamma, gae_lambda)

        # Detect warmup -> constrained transition
        if was_warmup and not env.is_warmup:
            warmup_end_episode = env.episode_count
            was_warmup = False
            if verbose:
                print(f"  Warmup complete at episode {warmup_end_episode}")
                print("  Resetting LR and entropy decay for constrained phase...")

        # Update LR and entropy with cosine annealing (separate for warmup/constrained phases)
        if env.is_warmup:
            if warmup_block_episodes and warmup_block_episodes != warmup_episodes:
                # Interleaved warmup: single decay across the whole warmup
                progress = min(
                    1.0, env.episode_count / (warmup_episodes * len(objectives))
                )
            else:
                # Block warmup: reset decay for EACH objective
                episode_in_current_objective = (
                    (env.episode_count - 1) % warmup_episodes
                ) + 1
                progress = min(1.0, episode_in_current_objective / warmup_episodes)
        else:
            # Constrained phase: decay over remaining episodes
            constrained_episodes = env.episode_count - warmup_end_episode
            if restart_decay_on_phase:
                # Restart decay every lr_decay_episodes (like cosine annealing with warm restarts)
                # Each constraint phase can get a fresh decay cycle
                episode_in_current_phase = (
                    (constrained_episodes - 1) % lr_decay_episodes
                ) + 1
                progress = min(1.0, episode_in_current_phase / lr_decay_episodes)
            else:
                # Decay once over entire constrained phase
                progress = min(1.0, constrained_episodes / lr_decay_episodes)

        # Cosine annealing: smooth decay with slower finish

        decay_mult = 0.5 * (1 + math.cos(math.pi * progress))
        current_lr = lr_final + (lr_init - lr_final) * decay_mult
        current_ent_coef = (
            ent_coef_final + (ent_coef_init - ent_coef_final) * decay_mult
        )
        current_ent_coef_thickness = (
            ent_coef_thickness_final
            + (ent_coef_thickness_init - ent_coef_thickness_final) * decay_mult
        )

        # Update agent LR and entropy
        for param_group in agent.optimizer.param_groups:
            param_group["lr"] = current_lr
        agent.ent_coef = current_ent_coef
        agent.ent_coef_thickness = current_ent_coef_thickness

        # Flush staged Pareto candidates (batched NDS) before policy update
        env.env.flush_pareto_candidates()

        # Update policy with BC loss from Pareto episodes (disabled during warmup)
        rollout_data = buffer.get()
        # Only use BC loss during constrained phase when exploring tradeoff region
        pareto_episodes = None
        if bc_weight > 0 and not env.is_warmup:
            if bc_selection == "target":
                # Imitate the archive's best designs for the CURRENT target
                # objective that satisfy the CURRENT constraint thresholds
                pareto_episodes = select_bc_episodes(
                    env.env.pareto_front_rewards,
                    env.env.pareto_front_episodes,
                    objectives,
                    env.env.target_objective,
                    env.env.constraints,
                )
            else:  # "all": original behaviour, whole front
                pareto_episodes = env.env.pareto_front_episodes
        ppo_logs = agent.update(
            rollout_data,
            n_epochs,
            batch_size,
            pareto_episodes=pareto_episodes,
            bc_weight=bc_weight,
            env_wrapper=env,
            bc_every=bc_every,
        )
        # Logging
        if env.episode_count - last_log_episode >= mlflow_log_freq:
            last_log_episode = env.episode_count
            io_start = time.perf_counter()
            if verbose:
                n_pareto = len(env.env.pareto_front_rewards)
                phase = "warmup" if env.is_warmup else "constrained"
                current_lr_display = agent.optimizer.param_groups[0]["lr"]
                air_bias = float(policy.material_head.bias[0].item())

                # Episode length stats
                ep_len_str = ""
                if ep_lengths:
                    recent_lengths = ep_lengths[-20:]
                    mean_len = np.mean(recent_lengths)
                    ep_len_str = f" | ep_len {mean_len:.1f}"

                print(
                    f"  [{phase}] episode {env.episode_count}/{total_episodes} | step {step_count} | "
                    f"pareto {n_pareto} | ent {current_ent_coef:.4f} | lr {current_lr_display:.2e} | "
                    f"air_bias {air_bias:.2f}{ep_len_str}"
                )

            metrics = {
                "step": step_count,
                "pareto.size": len(env.env.pareto_front_rewards),
                "schedule.lr": float(current_lr),
                "schedule.ent_coef": float(current_ent_coef),
                "schedule.ent_coef_thickness": float(current_ent_coef_thickness),
            }
            # Add pareto episodes count if BC loss enabled
            if bc_weight > 0 and hasattr(env.env, "pareto_front_episodes"):
                n_episodes_stored = sum(
                    1 for ep in env.env.pareto_front_episodes if ep is not None
                )
                metrics["pareto.episodes_stored"] = n_episodes_stored

            metrics.update({f"ppo.{k}": v for k, v in ppo_logs.items()})

            # Episode rewards
            if ep_rewards:
                window = ep_rewards[-100:]
                metrics["episode.reward_mean"] = float(np.mean(window))
                metrics["episode.reward_std"] = float(np.std(window))

            # Episode lengths
            if ep_lengths:
                length_window = ep_lengths[-100:]
                metrics["episode.length_mean"] = float(np.mean(length_window))
                metrics["episode.length_std"] = float(np.std(length_window))
                metrics["episode.length_min"] = float(np.min(length_window))
                metrics["episode.length_max"] = float(np.max(length_window))

            # Search width in thickness
            if ep_policy_std:
                std_window = ep_policy_std[-100:]
                metrics["policy.thickness_std"] = float(np.mean(std_window))
                metrics["policy.thickness_std_min"] = float(np.min(std_window))
            if ep_thickness_mean:
                metrics["episode.thickness_mean"] = float(
                    np.mean(ep_thickness_mean[-100:])
                )
                metrics["episode.thickness_spread"] = float(
                    np.mean(ep_thickness_within[-100:])
                )

            # Objective values
            if ep_vals:
                window = ep_vals[-100:]
                for obj in objectives:
                    vals = [v.get(obj, float("nan")) for v in window]
                    vals = [v for v in vals if not np.isnan(v)]
                    if vals:
                        metrics[f"vals.{obj}_mean"] = float(np.mean(vals))
                        # Best = closest to the objective's target
                        target = float(objective_targets.get(obj, 0.0))
                        metrics[f"vals.{obj}_best"] = float(
                            min(vals, key=lambda v: abs(v - target))
                        )
                        metrics[f"vals.{obj}_p10"] = float(np.percentile(vals, 10))
                        metrics[f"vals.{obj}_p90"] = float(np.percentile(vals, 90))

            # Hypervolume is the most expensive metric here and scales
            # quadratically with the archive, so it runs on its own schedule.
            if len(env.env.pareto_front_rewards) > 1:
                if env.episode_count - last_hv_episode >= hypervolume_freq:
                    last_hv_episode = env.episode_count
                    try:
                        last_hypervolume = env.env.compute_hypervolume(space="reward")
                    except Exception:
                        pass
                if last_hypervolume is not None:
                    metrics["pareto.hypervolume"] = last_hypervolume

            # Warmup best
            for obj, best in env.env.warmup_best_rewards.items():
                metrics[f"warmup_best.{obj}"] = best

            # Thresholds averaged over the window, across only the episodes
            # where each objective was constrained; *_frac is how often it was.
            if ep_constraints:
                window = ep_constraints[-100:]
                target_window = ep_targets[-100:]
                for obj in objectives:
                    held = [c[obj] for c in window if obj in c]
                    metrics[f"constraint.{obj}"] = float(np.mean(held)) if held else 0.0
                    metrics[f"constraint.{obj}_frac"] = len(held) / len(window)
                    metrics[f"target.{obj}_frac"] = target_window.count(obj) / len(
                        target_window
                    )

            # Monitor air material bias (check if it's shooting up)
            air_bias = float(policy.material_head.bias[0].item())
            metrics["policy.air_bias"] = air_bias

            # Also get air logit/probability from a sample observation
            with torch.no_grad():
                sample_logits = (
                    policy.material_head.weight
                    @ policy.material_head.weight.new_zeros(
                        policy.material_head.weight.shape[1]
                    )
                    + policy.material_head.bias
                )
                air_logit = float(sample_logits[0].item())
                metrics["policy.air_logit_init"] = air_logit

            metrics["episode"] = env.episode_count
            # Designs scored, not episodes run, so this matches the n_eval the
            # genetic trainer logs (see CoatingEnvironment.n_evaluations)
            metrics["evaluations"] = env.env.n_evaluations
            training_history.append(metrics)

            if mlflow.active_run():
                mlflow.log_metrics(
                    {k: v for k, v in metrics.items() if k != "episode"},
                    step=env.episode_count,
                )

            logging_io_seconds += time.perf_counter() - io_start

        # Periodic checkpointing
        if env.episode_count - last_plot_episode >= plot_freq:
            last_plot_episode = env.episode_count
            io_start = time.perf_counter()
            try:
                designs_df, values_df, rewards_df = env.env.export_pareto_dataframes()
                if not values_df.empty:
                    # Save CSVs
                    save_path = Path(save_dir)
                    save_path.mkdir(parents=True, exist_ok=True)
                    designs_df.to_csv(
                        save_path / "pareto_designs.csv",
                        index=False,
                    )
                    values_df.to_csv(
                        save_path / "pareto_values.csv",
                        index=False,
                    )
                    rewards_df.to_csv(
                        save_path / "pareto_rewards.csv",
                        index=False,
                    )

                    # Save model weights + training state
                    save_checkpoint(
                        save_dir,
                        env.episode_count,
                        {
                            "networks": {"policy": policy.state_dict()},
                            "optimizers": {"agent": agent.optimizer.state_dict()},
                            "pareto": {
                                "rewards": env.env.pareto_front_rewards,
                                "values": env.env.pareto_front_values,
                                "episodes": env.env.pareto_front_episodes,
                                "warmup_best": env.env.warmup_best_rewards,
                                "constraint_spread": env.env._constraint_spread,
                                "n_evaluations": env.env.n_evaluations,
                            },
                            "meta": {
                                "is_warmup": env.is_warmup,
                                "warmup_end_episode": warmup_end_episode,
                            },
                        },
                    )

                    if verbose:
                        print(
                            f"  Saved Pareto front checkpoint at episode {env.episode_count}"
                        )
            except Exception as e:
                if verbose:
                    print(f"  [checkpoint] skipped: {e}")

            try:
                from coatopt.utils.training_plots import save_training_curves

                save_training_curves(training_history, save_dir, objectives)
            except Exception as e:
                if verbose:
                    print(f"  [training curves] skipped: {e}")

            logging_io_seconds += time.perf_counter() - io_start

    algorithm_runtime = time.perf_counter() - loop_start - logging_io_seconds

    # Final Pareto export and plots
    try:
        from coatopt.utils.training_plots import save_training_curves

        save_training_curves(training_history, save_dir, objectives)
    except Exception as e:
        if verbose:
            print(f"  [training curves] skipped: {e}")

    designs_df, values_df, rewards_df = env.env.export_pareto_dataframes()
    if not designs_df.empty:
        try:
            from coatopt.utils.plot_design_diversity import (
                plot_cluster_designs,
                plot_design_diversity,
            )

            save_path = Path(save_dir)
            plot_design_diversity(designs_df, values_df, save_path)
            plot_cluster_designs(designs_df, values_df, save_path, materials=materials)
        except Exception as e:
            if verbose:
                print(f"  [diversity plot] skipped: {e}")

    # Return results
    return {
        "pareto_designs": designs_df,
        "pareto_values": values_df,
        "pareto_rewards": rewards_df,
        "model": None,
        "metadata": {
            "algorithm": "ppo_sequential",
            "total_episodes": total_episodes,
            "algorithm_runtime": round(algorithm_runtime, 2),
            "logging_runtime": round(logging_io_seconds, 2),
        },
    }
