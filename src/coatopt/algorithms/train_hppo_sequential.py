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
  lr                       = 3e-4
  lr_final                 = 3e-5           # Final LR (annealing target)
  lr_decay_episodes        = 10000          # Anneal over this many episodes per phase
  restart_decay_on_phase   = false          # Restart LR/entropy decay each constraint phase (like warm restarts)
  gamma                    = 0.99
  gae_lambda               = 0.95
  clip_range               = 0.2
  ent_coef                 = 0.01
  ent_coef_final           = 0.001          # Final entropy coefficient (annealing target)
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
from coatopt.utils.math_utils import TruncatedNormalDist
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
        # Warmup interleaving: cycle the target objective every
        # warmup_block_episodes instead of one long block per objective.
        # Default (None) = one block per objective, the original behaviour.
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
        tensor = state.get_observation_tensor(
            pre_type="lstm", max_thickness=self.env.max_thickness
        )
        base = tensor.numpy().flatten().astype(np.float32)
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
            # Alternate objectives during warmup, cycling every
            # warmup_block_episodes (defaults to one block per objective)
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

            # Constraint level (gradually tighten). "cycle" restarts the ramp
            # once it tops out; "ramp" climbs once and stays there. With
            # randomise_constraints the top level draws frac from U(0, 1),
            # which already covers every threshold the lower levels sample, so
            # restarting the ramp narrows the range rather than widening it.
            sweep = phase // len(self.objectives)
            if self.constraint_level_schedule == "ramp":
                level = min(sweep, self.steps_per_objective - 1)
            else:
                level = sweep % self.steps_per_objective

            # Set constraints on the other objectives. "reference" takes them
            # from one archived design so the subproblem is answerable by
            # construction; "box" draws each from the range the front spans.
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
        self.ptr = 0

    def add(self, obs, material, thickness, reward, value, log_prob, done, mask):
        self.observations.append(obs)
        self.materials.append(material)
        self.thicknesses.append(thickness)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        self.masks.append(mask)
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
        }


class LSTMEncoder(nn.Module):
    """Recurrent encoder of the layer stack.

    encode_step carries (h, c) so a rollout step feeds only the layer just
    placed instead of re-reading the whole stack.
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
        split = lambda z: z.view(b, t, self.heads, d // self.heads).transpose(1, 2)
        attended = torch.nn.functional.scaled_dot_product_attention(
            split(q), split(k), split(v), is_causal=True
        )
        x = x + self.proj(attended.transpose(1, 2).reshape(b, t, d))
        return x + self.ff(self.norm2(x))


class AttentionEncoder(nn.Module):
    """Causal self-attention encoder of the layer stack.

    Causal is not optional: the update reads position t from one pass over
    the episode's final observation, so if position t could see later layers
    it would use layers the policy had not placed yet and the PPO ratio would
    compare two different functions. No dropout, for the same reason - the
    rollout and the update must agree on the same state.

    On CPU this is much cheaper than the LSTM despite being O(T^2): 50 layers
    is one small matmul, where the LSTM is 50 dependent tiny ones. Rollout
    keeps no state and simply re-encodes the prefix placed so far.
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

# Thickness sampling width, as a fraction of (max_t - min_t) rather than an
# absolute value. An absolute clamp does not mean the same thing at every
# thickness range: at scale ~ the range the truncated normal is
# indistinguishable from uniform and d(log_prob)/d(log_std) falls to ~2e-4, so
# the head sits on a dead plateau and never learns a width. Measured on a
# 20-layer 3-objective run at [0.01, 0.40] the realised std stayed pinned at
# the uniform limit (0.1123) for all 150k episodes; the same run at
# [0.10, 0.40] escaped by chance around episode 13k and reached 0.024. These
# bounds put the whole usable interval in the informative region at any range.
LOG_STD_MIN, LOG_STD_MAX = -5.0, -1.2  # 0.7% .. 30% of the thickness range


class HybridActorCritic(nn.Module):
    """Actor-Critic with hybrid discrete+continuous actions.

    Discrete head: material selection (masked categorical)
    Continuous head: thickness (TruncatedNormal with bounds [min_t, max_t])
    Value head: state value V(s)

    pre_model_type selects how the layer stack is read before the MLP trunk:
    "linear" feeds the flattened observation straight in, "lstm" and
    "attention" encode the sequence first.
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
            assert (
                max_layers is not None and n_constraints is not None
            ), f"max_layers and n_constraints required for {pre_model_type}"

            # Observation structure: [layer_sequence (flattened), current_layer, constraints]
            n_features_per_layer = 1 + n_materials + 2  # thickness + one-hot + 2
            self.max_layers = max_layers
            self.n_features_per_layer = n_features_per_layer
            self.n_constraints = n_constraints

            self.encoder = PRE_MODELS[pre_model_type](
                n_features_per_layer, max_len=max_layers, **(pre_model_params or {})
            )

            # After the encoder: concat [encoded stack, objective weights, constraints]
            combined_dim = (
                self.encoder.out_dim + n_constraints + n_constraints
            )  # n_objectives = n_constraints
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

        # Value heads - one per objective for stable learning
        # Each head learns value function for when that objective is the target
        self.n_objectives = n_constraints  # Number of objectives
        self.value_heads = nn.ModuleList(
            [nn.Linear(prev_dim, 1) for _ in range(self.n_objectives)]
        )

    def _obs_layers(self, obs):
        """Layer-stack rows of an observation, without the objective tail."""
        tail = self.n_objectives + self.n_constraints
        return obs[:, :-tail].view(-1, self.max_layers, self.n_features_per_layer)

    def _seq_features(self, obs, episode_idx=None, step_idx=None):
        """Encoded layer stack at each step.

        Every encoder is causal, so output position t of one pass depends only
        on the first t layers. With episode_idx/step_idx, obs holds one final
        observation per episode and each step's feature is gathered from a
        single shared pass; otherwise obs is per-step and the output is read
        at each observation's own layer count.
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

        Same causality as _seq_features: position t depends only on the first
        t layers, so a recurrent encoder can carry its state between steps and
        read one new layer instead of a max_layers-long pass. A first step (no
        layers placed) restarts the state, and a missing state mid-episode
        rebuilds from the layers placed so far, so the caller cannot
        desynchronise it. Stateless encoders just re-encode that prefix.
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

    def _material_dist(self, features, mask):
        logits = self.material_head(features) + (1.0 - mask) * -1e8
        return torch.distributions.Categorical(logits=logits)

    def _thickness_dist(self, features, material):
        """Thickness distribution, conditioned on the chosen material.

        Shared so acting and evaluating cannot drift apart: PPO compares log
        probs from the two, which only means something if they build the same
        distribution.
        """
        material_onehot = torch.nn.functional.one_hot(
            material, num_classes=self.n_materials
        ).float()
        thickness_input = torch.cat([features, material_onehot], dim=-1)
        mean_raw = self.thickness_mean(thickness_input).squeeze(-1)
        width = self.max_t - self.min_t
        mean = self.min_t + width * torch.sigmoid(mean_raw)
        log_std = torch.clamp(
            self.thickness_logstd(thickness_input).squeeze(-1),
            LOG_STD_MIN,
            LOG_STD_MAX,
        )
        return TruncatedNormalDist(
            loc=mean,
            scale=width * torch.exp(log_std),
            a=torch.full_like(mean, self.min_t),
            b=torch.full_like(mean, self.max_t),
        )

    def forward(self, obs, mask, target_obj_idx, deterministic=False, lstm_state=None):
        """Forward pass returning actions, log_probs, and value.

        Args:
            obs: observation tensor
            mask: action mask
            target_obj_idx: index of target objective (selects which value head to use)
            deterministic: whether to sample or take argmax
            lstm_state: (h, c) carried from this episode's previous step, or
                None at its first step

        Returns:
            material: sampled material index
            thickness: sampled thickness value
            log_prob: total log probability (discrete + continuous)
            value: state value V(s) for target objective
            lstm_state: updated (h, c) to pass to the next step (None without LSTM)
        """
        seq_features = None
        if self.use_sequence:
            seq_features, lstm_state = self._seq_step(obs, lstm_state)
        features = self._trunk_features(obs, seq_features)

        dist_d = self._material_dist(features, mask)
        material = dist_d.logits.argmax(dim=-1) if deterministic else dist_d.sample()

        dist_c = self._thickness_dist(features, material)
        thickness = (
            dist_c.loc.clamp(self.min_t, self.max_t)
            if deterministic
            else dist_c.rsample()
        )

        log_prob = dist_d.log_prob(material) + dist_c.log_prob(thickness)
        value = self.value_heads[target_obj_idx](features).squeeze(-1)

        # Realised spread of the truncated normal, which is what sets the
        # search width. It is well below scale once the truncation bites.
        return material, thickness, log_prob, value, dist_c.variance.sqrt(), lstm_state

    def evaluate_actions(
        self,
        obs,
        mask,
        materials,
        thicknesses,
        target_obj_idx,
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
            target_obj_idx: index of target objective (selects which value head to use)
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

        dist_d = self._material_dist(features, mask)
        dist_c = self._thickness_dist(features, materials)

        log_prob = dist_d.log_prob(materials) + dist_c.log_prob(thicknesses)
        entropy = dist_d.entropy() + dist_c.entropy()
        value = self.value_heads[target_obj_idx](features).squeeze(-1)

        return log_prob, value, entropy


def select_bc_episodes(
    front_rewards,
    front_episodes,
    objectives,
    target_objective,
    constraints,
    top_k: int = 8,
):
    """Pick archive episodes matching the current constrained subproblem.

    Ranks Pareto-archive designs by (violation of the current constraint
    thresholds, then target-objective reward) and returns the episodes of the
    top_k: the archive's best feasible answers to the phase the policy is
    currently solving. Falls back to the least-violating designs when nothing
    on the front satisfies the thresholds yet. This keeps the BC conditioning
    consistent with the demonstrated behaviour (imitating the whole front
    under the current target teaches the policy to ignore the conditioning).
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
    policy, pareto_episodes, env_wrapper, target_obj_idx, bc_weight=0.1, batch_size=32
):
    """Compute behavior cloning loss from Pareto front episodes stored in environment.

    Args:
        policy: policy network
        pareto_episodes: list of Pareto front episodes
        env_wrapper: environment wrapper
        target_obj_idx: index of target objective (for value head selection)
        bc_weight: weight for BC loss
        batch_size: batch size for sampling
    """
    if not pareto_episodes or len(pareto_episodes) == 0:
        return 0.0

    # Filter out None episodes
    valid_episodes = [ep for ep in pareto_episodes if ep is not None]
    if not valid_episodes:
        return 0.0

    # Sample a subset of Pareto episodes first — iterating all N episodes to collect
    # transitions is O(N * ep_len) every update, which is expensive for large fronts.
    max_episodes = min(batch_size, len(valid_episodes))
    picks = torch.randperm(len(valid_episodes))[:max_episodes].tolist()
    sampled_episodes = [valid_episodes[i] for i in picks]

    # Index the pool rather than materialising every transition: the dicts for
    # thousands of steps were built and thrown away on every call.
    transitions = [
        (episode, i)
        for episode in sampled_episodes
        if episode.get("states")
        and episode.get("discrete_actions")
        and episode.get("continuous_actions")
        for i in range(len(episode["states"]))
    ]

    if not transitions:
        return 0.0

    # Sample batch
    n_samples = min(batch_size, len(transitions))

    picks = torch.randperm(len(transitions))[:n_samples].tolist()
    sampled = [transitions[i] for i in picks]

    # Prepare batch tensors
    obs_list = []
    mat_list = []
    thick_list = []
    mask_list = []

    for episode, step in sampled:
        state = episode["states"][step]
        # Get observation using wrapper's method (includes constraints)
        if hasattr(env_wrapper, "_get_obs"):
            obs = env_wrapper._get_obs(state)
        elif hasattr(state, "get_observation_tensor"):
            obs = state.get_observation_tensor(pre_type="lstm").numpy().flatten()
        else:
            obs = state
        obs_list.append(obs)

        # Actions
        mat = episode["discrete_actions"][step]
        thick = episode["continuous_actions"][step]
        mat_list.append(mat.item() if torch.is_tensor(mat) else mat)
        thick_list.append(thick.item() if torch.is_tensor(thick) else thick)
        mask_list.append(np.ones(policy.n_materials))  # Default mask

    # Convert to tensors
    obs_t = torch.FloatTensor(np.array(obs_list))
    mat_t = torch.LongTensor(mat_list)
    thick_t = torch.FloatTensor(thick_list)
    mask_t = torch.FloatTensor(np.array(mask_list))

    # Compute log probs from policy
    log_probs, _, _ = policy.evaluate_actions(
        obs_t, mask_t, mat_t, thick_t, target_obj_idx
    )
    bc_loss = -log_probs.mean()  # Negative log likelihood

    return bc_weight * bc_loss


class PPOAgent:
    """PPO agent with hybrid actions."""

    def __init__(
        self,
        policy: HybridActorCritic,
        lr: float = 3e-4,
        gamma: float = 0.99,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
    ):
        self.policy = policy
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.lstm_state = None  # carried across rollout steps of an episode
        self.last_thickness_std = float("nan")  # sampling width of the last act()

    def act(self, obs, mask, target_obj_idx, deterministic=False):
        """Sample action from policy.

        The LSTM state is carried between calls so each step feeds the LSTM
        only the layer just placed. _seq_step restarts it whenever the
        observation has no layers placed, so a reset needs no bookkeeping here.

        Args:
            obs: observation
            mask: action mask
            target_obj_idx: index of target objective
            deterministic: whether to sample or take argmax
        """
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            mask_t = torch.FloatTensor(mask).unsqueeze(0)
            material, thickness, log_prob, value, std, self.lstm_state = self.policy(
                obs_t, mask_t, target_obj_idx, deterministic, self.lstm_state
            )
            self.last_thickness_std = std.item()
            return (
                material.item(),
                thickness.item(),
                log_prob.item(),
                value.item(),
            )

    def update(
        self,
        rollout_data: dict,
        n_epochs: int,
        batch_size: int,
        target_obj_idx: int,
        pareto_episodes=None,
        bc_weight=0.1,
        env_wrapper=None,
    ):
        """PPO update using rollout data with optional BC loss from Pareto episodes.

        Args:
            rollout_data: dict with observations, actions, returns, etc.
            n_epochs: number of optimization epochs
            batch_size: minibatch size
            target_obj_idx: index of target objective (for value head selection)
            pareto_episodes: optional list of Pareto front episodes for BC loss
            bc_weight: weight for behavior cloning loss
            env_wrapper: environment wrapper (for BC loss)
        """
        obs = rollout_data["observations"]
        materials = rollout_data["materials"]
        thicknesses = rollout_data["thicknesses"]
        old_log_probs = rollout_data["log_probs"]
        returns = rollout_data["returns"]
        advantages = rollout_data["advantages"]
        masks = rollout_data["masks"]

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n_samples = len(obs)
        logs = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
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

                log_probs, values, entropy = self.policy.evaluate_actions(
                    obs[batch_idx],
                    masks[batch_idx],
                    materials[batch_idx],
                    thicknesses[batch_idx],
                    target_obj_idx,
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
                    - self.ent_coef * entropy.mean()
                )

                # Behaviour cloning rides the first minibatch of each epoch:
                # same number of archive samples as the old dedicated pass,
                # but its gradient joins the PPO gradient in one clipped step
                # instead of taking an optimiser step of its own.
                if use_bc and i_batch == 0:
                    bc_loss = compute_bc_loss_from_pareto(
                        self.policy,
                        pareto_episodes,
                        env_wrapper,
                        target_obj_idx,
                        bc_weight,
                        batch_size=32,
                    )
                    if torch.is_tensor(bc_loss):
                        logs["bc_loss"] += bc_loss.item()
                        loss = loss + bc_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                # Logging
                logs["policy_loss"] += policy_loss.item()
                logs["value_loss"] += value_loss.item()
                logs["entropy"] += entropy.mean().item()
                logs["clip_frac"] += (
                    ((ratio - 1.0).abs() > self.clip_range).float().mean().item()
                )
                n_updates += 1

        # Average over updates
        for k in logs:
            logs[k] /= max(n_updates, 1)

        return logs


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
    # "warmup" scales thresholds by this run's warmup best (run-dependent, so
    # repeat runs solve different problems); "absolute" scales by the
    # normalised reward's own 1.0, rising only if a run beats it.
    constraint_anchor = _get("constraint_anchor", "warmup", str).strip().lower()
    # Fraction of the front's own spread to widen each end of that range by
    constraint_extend_low = _get("constraint_extend_low", 0.2, float)
    constraint_extend_high = _get("constraint_extend_high", 0.2, float)
    # "box": each threshold drawn independently from its objective's range.
    # "reference": all thresholds taken from one archived design, so the
    # constrained subproblem always has at least that design as an answer.
    constraint_source = _get("constraint_source", "box", str).strip().lower()
    # How far past the reference point to ask, in units of that point's local
    # neighbour spacing on the front. 0.0 asks only to match it.
    constraint_ref_extend = _get("constraint_ref_extend", 1.0, float)

    # Parse hidden layers
    hidden_str = _get("hidden", "[256, 256]")
    hidden = eval(hidden_str)

    # These tensors are far too small to gain from intra-op threading, and on a
    # request_cpus=1 Condor slot torch would otherwise start one thread per host
    # core and thrash against the single allocated CPU. Attention pays for this
    # much more than the LSTM does: measured on a 6-core box, going from 1 to 8
    # threads cost attention 55% and the LSTM 7%. 0 leaves torch's default alone.
    torch_threads = _get("torch_threads", 1, int)
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
        # both are needed whatever the encoder: n_constraints sizes the
        # per-objective value heads, and only the encoders require max_layers
        max_layers=max_layers,
        n_constraints=n_constraints,
    )

    # Initialize air material (index 0) with strong negative bias
    # This prevents "masked action explosion" where air probability shoots up
    # when it's masked, then gets selected immediately when unmasked
    with torch.no_grad():
        initial_air_bias = -3.0
        policy.material_head.bias[0] = initial_air_bias

        # Start the thickness width at the wide end of its clamp. The default
        # linear init puts roughly half of states above LOG_STD_MAX, where the
        # clamp passes no gradient, so the head would have to random-walk down
        # before it could learn anything.
        policy.thickness_logstd.weight.mul_(0.01)
        policy.thickness_logstd.bias.fill_(LOG_STD_MAX)

    if verbose:
        print(
            f"Initialized air material bias: {initial_air_bias:.2f} (low probability)"
        )

    if verbose:
        if use_sequence:
            print(f"Layer-stack encoder: {pre_model_type} {pre_model_params}")
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
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
    )

    buffer = RolloutBuffer()

    # Tracking
    ep_rewards = []
    ep_vals = []
    ep_lengths = []  # Track episode lengths
    ep_policy_std = []          # realised sampling std of the truncated normal
    ep_thickness_mean = []
    ep_thickness_within = []    # spread of layers within one design
    # Per-episode constraint thresholds and target. Logging fires once per
    # update, so reading env.constraints there samples one episode out of
    # every episodes_per_update; when that stride is a whole number of
    # objective cycles the same phase is sampled every time and one
    # objective's column reads 0.0 for the entire run. Record per episode and
    # aggregate over the window instead.
    ep_constraints = []
    ep_targets = []
    sample_designs = []  # Track sample designs during warmup for debugging
    training_history = []  # periodic metrics rows for training_curves.png
    # Logging and checkpointing are checked once per update, so episode_count
    # jumps by episodes_per_update and rarely lands exactly on a multiple of
    # these intervals -- with episodes_per_update=4 and plot_freq=5000 it never
    # did, and a whole run wrote no checkpoints. Fire on elapsed episodes
    # instead, so any interval is honoured at the first update past it.
    last_log_episode = 0
    last_plot_episode = 0
    last_hv_episode = 0
    last_hypervolume = None
    objectives = list(config.data.optimise_parameters)
    objective_targets = dict(getattr(config.data, "optimise_targets", {}) or {})

    # Create objective name -> index mapping for value head selection
    objective_to_idx = {obj: idx for idx, obj in enumerate(objectives)}

    # Annealing tracking
    lr_init = lr
    ent_coef_init = ent_coef
    current_ent_coef = ent_coef
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
                training_history = prior[
                    prior["episode"] <= ckpt["episode"]
                ].to_dict("records")
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

    # algorithm_runtime is the optimisation alone. Everything the loop does
    # for reporting -- the metrics block, MLflow writes, checkpoints, CSVs and
    # curve plots -- is timed into logging_io_seconds and subtracted, and the
    # final plots fall outside the loop entirely. total_runtime minus
    # algorithm_runtime is therefore what reporting cost you, which is worth
    # watching: MLflow writes at mlflow_log_freq and hypervolume at
    # hypervolume_freq are both easy to make dominate a run by accident.
    loop_start = time.perf_counter()
    logging_io_seconds = 0.0

    while env.episode_count < total_episodes:
        # Collect episodes for this update
        buffer.clear()
        episodes_collected = 0

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
                # Get target objective index for value head selection
                target_obj_idx = objective_to_idx[env.env.target_objective]
                material, thickness, log_prob, value = agent.act(
                    obs, mask, target_obj_idx
                )
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
                    obs, material, thickness, reward, value, log_prob, done, mask
                )

                if done:
                    episode_done = True
                    if "vals" in info:
                        ep_rewards.append(reward)
                        ep_vals.append(info["vals"])
                        ep_lengths.append(env.current_layer)  # Track episode length
                        ep_policy_std.append(float(np.mean(episode_stds)))
                        ep_thickness_mean.append(
                            float(np.mean(episode_sampled_thicknesses))
                        )
                        ep_thickness_within.append(
                            float(np.std(episode_sampled_thicknesses))
                        )
                        # Captured before reset() overwrites them for the next
                        # episode, so these are the thresholds this episode ran under
                        ep_constraints.append(dict(env.env.constraints))
                        ep_targets.append(env.env.target_objective)
                        # Only the last 100 are ever read; cap so a 150k-episode
                        # run does not carry a dict per episode
                        if len(ep_constraints) > 200:
                            del ep_constraints[:-100]
                            del ep_targets[:-100]

                        # Sample designs during warmup for debugging (keep last 100)
                        if env.is_warmup:
                            # Get state array directly from environment
                            state_array = env.env.current_state.get_array()
                            design_info = {
                                "episode": env.episode_count,
                                "target_obj": env.env.target_objective,
                                "state_array": state_array,
                                "vals": info["vals"].copy(),
                                "reward": reward,
                                "length": env.current_layer,
                            }
                            sample_designs.append(design_info)
                            # Keep only last 100
                            if len(sample_designs) > 100:
                                sample_designs.pop(0)

                    episodes_collected += 1

                    obs, info = env.reset()
                    mask = info["mask"]
                else:
                    obs = next_obs
                    mask = next_mask

                step_count += 1

        # Finalize buffer
        target_obj_idx = objective_to_idx[env.env.target_objective]
        _, _, _, last_value = agent.act(obs, mask, target_obj_idx)
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

        # Update agent LR and entropy
        for param_group in agent.optimizer.param_groups:
            param_group["lr"] = current_lr
        agent.ent_coef = current_ent_coef

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
        target_obj_idx = objective_to_idx[env.env.target_objective]
        ppo_logs = agent.update(
            rollout_data,
            n_epochs,
            batch_size,
            target_obj_idx,
            pareto_episodes=pareto_episodes,
            bc_weight=bc_weight,
            env_wrapper=env,
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

            # Search width in thickness. Measured on a 20-layer stack, 0.02
            # costs almost nothing, 0.05 costs 5x transmission and 0.1 costs 95x.
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

            # Hypervolume. This is by far the most expensive metric here and
            # it scales roughly quadratically with the archive: measured at 3
            # objectives it costs 0.27 s at 2k points and 5.1 s at 8k, so at
            # mlflow_log_freq it would add hours to a run whose front grows
            # that far. Recompute on its own slower schedule and carry the
            # last value forward so the column stays continuous.
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

            # Constraint thresholds over the window, averaged only across the
            # episodes where each objective was actually constrained, so a
            # column no longer depends on which phase the log happened to land
            # on. constraint.<obj>_frac is how often it was constrained at all;
            # target.<obj>_frac how often it was the target. Both are 0.0
            # during warmup, when nothing is constrained.
            if ep_constraints:
                window = ep_constraints[-100:]
                target_window = ep_targets[-100:]
                for obj in objectives:
                    held = [c[obj] for c in window if obj in c]
                    metrics[f"constraint.{obj}"] = (
                        float(np.mean(held)) if held else 0.0
                    )
                    metrics[f"constraint.{obj}_frac"] = len(held) / len(window)
                    metrics[f"target.{obj}_frac"] = (
                        target_window.count(obj) / len(target_window)
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
            # One episode produces one design, so episodes are the evaluation
            # count. Named to match what the genetic trainer records, so the
            # two can be compared on designs evaluated rather than wall clock.
            # Designs scored, not episodes run: the genetic trainer logs
            # pymoo's n_eval here, and the two only agree when an episode costs
            # exactly one evaluation (see CoatingEnvironment.n_evaluations)
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
