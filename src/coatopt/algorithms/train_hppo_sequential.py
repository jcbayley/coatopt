#!/usr/bin/env python3
"""
Single-agent PPO with hybrid action space and sequential constraint scheduling.

Uses discrete material selection + continuous thickness (TruncatedNormal) with
constraint-based multi-objective training. Alternates between objectives during
warmup, then applies gradually tightening constraints.

Config section: [hppo_sequential]
  total_episodes           = 10000
  warmup_episodes          = 500            # Per objective
  epochs_per_step          = 200            # Episodes per phase
  steps_per_objective      = 10             # Constraint levels per objective
  episodes_per_update      = 10             # Episodes before PPO update
  n_epochs                 = 5              # SGD epochs per update
  batch_size               = 64
  constraint_penalty       = 3.0
  pareto_bonus             = 0.0            # Hypervolume improvement bonus
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
  use_lstm                 = false          # Use LSTM to process layer sequence
  lstm_hidden              = 128            # LSTM hidden size (if use_lstm=true)
  lstm_layers              = 1              # Number of LSTM layers (if use_lstm=true)
  hidden                   = [256, 256]     # MLP layers after LSTM/before policy heads
  seed                     = 42
  verbose                  = 1
  plot_freq                = 500
"""
import configparser
from pathlib import Path
from typing import Dict, List

import gymnasium as gym
import mlflow
import numpy as np
import torch
import torch.nn as nn

from coatopt.environments.environment import CoatingEnvironment
from coatopt.utils.configs import Config, load_config
from coatopt.utils.math_utils import TruncatedNormalDist
from coatopt.utils.utils import load_materials


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
        epochs_per_step: int = 200,
        steps_per_objective: int = 10,
        constraint_penalty: float = 3.0,
        mask_consecutive_materials: bool = True,
        min_layers_before_air: int = 4,
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
        self.total_warmup_episodes = warmup_episodes * len(
            self.objectives
        )  # Total warmup
        self.epochs_per_step = epochs_per_step
        self.steps_per_objective = steps_per_objective
        self.episode_count = 0
        self.is_warmup = True

        # Enable constrained training in base environment
        self.env.enable_constrained_training(
            warmup_episodes_per_objective=warmup_episodes,
            steps_per_objective=steps_per_objective,
            epochs_per_step=epochs_per_step,
            constraint_penalty=constraint_penalty,
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

        # Observation space (includes constraint thresholds)
        n_features = 1 + self.env.n_materials + 2
        n_constraints = len(self.env.optimise_parameters)
        obs_size = self.env.max_layers * n_features + 1 + n_constraints
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
        """Convert state to observation with constraint thresholds."""
        tensor = state.get_observation_tensor(pre_type="lstm")
        obs = tensor.numpy().flatten().astype(np.float32)
        # Append constraint thresholds
        for obj in self.env.optimise_parameters:
            obs = np.append(obs, self.env.constraints.get(obj, 0.0))
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
            # Alternate objectives during warmup (spend warmup_episodes_per_objective on each)
            obj_idx = (
                (self.episode_count - 1) // self.warmup_episodes_per_objective
            ) % len(self.objectives)
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
            phase = (constrained_episode - 1) // self.epochs_per_step

            # Alternate objectives
            obj_idx = phase % len(self.objectives)
            target_obj = self.objectives[obj_idx]

            # Constraint level (gradually tighten)
            level = (phase // len(self.objectives)) % self.steps_per_objective

            # Set constraints on other objectives
            constraints = {}
            for i, obj in enumerate(self.objectives):
                if i != obj_idx:
                    frac = (level + 1) / self.steps_per_objective
                    best = self.env.warmup_best_rewards.get(obj, 0.0)
                    constraints[obj] = frac * best

            self.env.target_objective = target_obj
            self.env.constraints = constraints

        obs = self._get_obs(state)
        info = {"mask": self.get_action_mask()}
        return obs, info

    def step(self, action):
        """Execute action (dict with material and thickness)."""
        material_idx = int(action["material"])
        thickness = float(action["thickness"][0])

        # CoatingEnvironment.step returns: state, rewards, terminated, finished, total_reward, full_action, vals
        state, rewards, terminated, finished, total_reward, full_action, vals = (
            self.env.step([material_idx, thickness])
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
        """Return all data as tensors."""
        return {
            "observations": torch.FloatTensor(np.array(self.observations)),
            "materials": torch.LongTensor(self.materials),
            "thicknesses": torch.FloatTensor(self.thicknesses),
            "log_probs": torch.FloatTensor(self.log_probs),
            "returns": torch.FloatTensor(self.returns),
            "advantages": torch.FloatTensor(self.advantages),
            "masks": torch.FloatTensor(np.array(self.masks)),
        }


class HybridActorCritic(nn.Module):
    """Actor-Critic with hybrid discrete+continuous actions.

    Discrete head: material selection (masked categorical)
    Continuous head: thickness (TruncatedNormal with bounds [min_t, max_t])
    Value head: state value V(s)

    Optional LSTM: processes layer sequence before MLP trunk.
    """

    def __init__(
        self,
        obs_dim: int,
        n_materials: int,
        min_thickness: float,
        max_thickness: float,
        hidden_dims: List[int] = [256, 256],
        use_lstm: bool = False,
        lstm_hidden: int = 128,
        lstm_layers: int = 1,
        max_layers: int = None,
        n_constraints: int = None,
    ):
        super().__init__()
        self.n_materials = n_materials
        self.min_t = min_thickness
        self.max_t = max_thickness
        self.use_lstm = use_lstm

        if use_lstm:
            assert (
                max_layers is not None and n_constraints is not None
            ), "max_layers and n_constraints required for LSTM"

            # Observation structure: [layer_sequence (flattened), current_layer, constraints]
            n_features_per_layer = 1 + n_materials + 2  # thickness + one-hot + 2
            self.max_layers = max_layers
            self.n_features_per_layer = n_features_per_layer
            self.n_constraints = n_constraints

            # LSTM processes layer sequence
            self.lstm = nn.LSTM(
                input_size=n_features_per_layer,
                hidden_size=lstm_hidden,
                num_layers=lstm_layers,
                batch_first=True,
            )

            # After LSTM: concat [lstm_output, constraints]
            combined_dim = lstm_hidden + n_constraints

            # Trunk MLP
            layers = []
            prev_dim = combined_dim
            for h in hidden_dims:
                layers.extend([nn.Linear(prev_dim, h), nn.ReLU()])
                prev_dim = h
            self.trunk = nn.Sequential(*layers)
        else:
            # Standard MLP trunk (no LSTM)
            layers = []
            prev_dim = obs_dim
            for h in hidden_dims:
                layers.extend([nn.Linear(prev_dim, h), nn.ReLU()])
                prev_dim = h
            self.trunk = nn.Sequential(*layers)

        # Discrete head (material)
        self.material_head = nn.Linear(prev_dim, n_materials)

        # Continuous head (thickness) - outputs mean and log_std
        self.thickness_mean = nn.Linear(prev_dim, 1)
        self.thickness_logstd = nn.Linear(prev_dim, 1)

        # Value head
        self.value_head = nn.Linear(prev_dim, 1)

    def forward(self, obs, mask, deterministic=False):
        """Forward pass returning actions, log_probs, and value.

        Returns:
            material: sampled material index
            thickness: sampled thickness value
            log_prob: total log probability (discrete + continuous)
            value: state value V(s)
        """
        if self.use_lstm:
            # Extract components from observation
            # obs = [layer_sequence (flattened), constraints]
            batch_size = obs.shape[0]

            # Extract constraints from the end
            layer_seq_flat = obs[:, : -self.n_constraints]
            constraints = obs[:, -self.n_constraints :]

            # Reshape layer sequence for LSTM: (batch, max_layers, features)
            layer_seq = layer_seq_flat.view(
                batch_size, self.max_layers, self.n_features_per_layer
            )

            # LSTM: use last layer's hidden state
            lstm_out, (h_n, c_n) = self.lstm(layer_seq)
            lstm_features = h_n[-1]  # Take last layer: (batch, lstm_hidden)

            # Concatenate LSTM features with constraints (no current_layer)
            combined = torch.cat([lstm_features, constraints], dim=1)
            features = self.trunk(combined)
        else:
            features = self.trunk(obs)

        # Discrete material selection (masked)
        logits = self.material_head(features)
        logits = logits + (1.0 - mask) * -1e8  # mask invalid actions
        dist_d = torch.distributions.Categorical(logits=logits)
        if deterministic:
            material = logits.argmax(dim=-1)
        else:
            material = dist_d.sample()
        log_prob_d = dist_d.log_prob(material)

        # Continuous thickness (TruncatedNormal)
        mean_raw = self.thickness_mean(features).squeeze(-1)
        # Use sigmoid to softly constrain mean to valid range
        mean = self.min_t + (self.max_t - self.min_t) * torch.sigmoid(mean_raw)

        log_std = self.thickness_logstd(features).squeeze(-1)
        log_std = torch.clamp(log_std, -4, 0)  # Clamp log_std
        std = torch.exp(log_std)

        # Create TruncatedNormal with bounds [min_t, max_t]
        dist_c = TruncatedNormalDist(
            loc=mean,
            scale=std,
            a=torch.full_like(mean, self.min_t),
            b=torch.full_like(mean, self.max_t),
        )
        if deterministic:
            thickness = mean.clamp(self.min_t, self.max_t)
        else:
            thickness = dist_c.rsample()
        log_prob_c = dist_c.log_prob(thickness)

        # Total log probability (joint = product of independent)
        log_prob = log_prob_d + log_prob_c

        # Value
        value = self.value_head(features).squeeze(-1)

        return material, thickness, log_prob, value

    def evaluate_actions(self, obs, mask, materials, thicknesses):
        """Evaluate log_probs and values for given actions."""
        if self.use_lstm:
            # Extract components from observation (same as forward)
            batch_size = obs.shape[0]

            # Extract constraints from the end
            layer_seq_flat = obs[:, : -self.n_constraints]
            constraints = obs[:, -self.n_constraints :]

            # Reshape and process with LSTM
            layer_seq = layer_seq_flat.view(
                batch_size, self.max_layers, self.n_features_per_layer
            )
            lstm_out, (h_n, c_n) = self.lstm(layer_seq)
            lstm_features = h_n[-1]  # Take last layer: (batch, lstm_hidden)

            # Concatenate and process (no current_layer)
            combined = torch.cat([lstm_features, constraints], dim=1)
            features = self.trunk(combined)
        else:
            features = self.trunk(obs)

        # Discrete
        logits = self.material_head(features)
        logits = logits + (1.0 - mask) * -1e8
        dist_d = torch.distributions.Categorical(logits=logits)
        log_prob_d = dist_d.log_prob(materials)
        entropy_d = dist_d.entropy()

        # Continuous
        mean_raw = self.thickness_mean(features).squeeze(-1)
        # Use sigmoid to softly constrain mean to valid range
        mean = self.min_t + (self.max_t - self.min_t) * torch.sigmoid(mean_raw)

        log_std = self.thickness_logstd(features).squeeze(-1)
        log_std = torch.clamp(log_std, -4, 0)
        std = torch.exp(log_std)

        # Create TruncatedNormal with bounds [min_t, max_t]
        dist_c = TruncatedNormalDist(
            loc=mean,
            scale=std,
            a=torch.full_like(mean, self.min_t),
            b=torch.full_like(mean, self.max_t),
        )
        log_prob_c = dist_c.log_prob(thicknesses)
        entropy_c = dist_c.entropy()

        log_prob = log_prob_d + log_prob_c
        # Use normal entropy (removed 3x weighting to allow convergence)
        entropy = entropy_d + entropy_c

        value = self.value_head(features).squeeze(-1)

        return log_prob, value, entropy


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

    def act(self, obs, mask, deterministic=False):
        """Sample action from policy."""
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            mask_t = torch.FloatTensor(mask).unsqueeze(0)
            material, thickness, log_prob, value = self.policy(
                obs_t, mask_t, deterministic
            )
            return (
                material.item(),
                thickness.item(),
                log_prob.item(),
                value.item(),
            )

    def update(self, rollout_data: dict, n_epochs: int, batch_size: int):
        """PPO update using rollout data."""
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
        logs = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "clip_frac": 0.0}
        n_updates = 0

        for epoch in range(n_epochs):
            indices = torch.randperm(n_samples)
            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                batch_idx = indices[start:end]

                log_probs, values, entropy = self.policy.evaluate_actions(
                    obs[batch_idx],
                    masks[batch_idx],
                    materials[batch_idx],
                    thicknesses[batch_idx],
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

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                # Check for NaN parameters
                for name, param in self.policy.named_parameters():
                    if torch.isnan(param).any():
                        print(f"ERROR: NaN detected in {name} after optimizer step!")
                        print(
                            f"Loss: {loss.item()}, Policy loss: {policy_loss.item()}, Value loss: {value_loss.item()}"
                        )
                        raise ValueError(f"NaN in parameters: {name}")

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
    materials_path = parser.get("general", "materials_path")
    materials = load_materials(materials_path)

    # Read hyperparameters
    total_episodes = _get("total_episodes", 10000, int)
    warmup_episodes = _get("warmup_episodes", 500, int)
    epochs_per_step = _get("epochs_per_step", 200, int)
    steps_per_objective = _get("steps_per_objective", 10, int)
    episodes_per_update = _get("episodes_per_update", 10, int)
    n_epochs = _get("n_epochs", 5, int)
    batch_size = _get("batch_size", 64, int)
    constraint_penalty = _get("constraint_penalty", 3.0, float)
    pareto_bonus = _get("pareto_bonus", 0.0, float)
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
    ent_decay_episodes = _get("ent_decay_episodes", total_episodes, int)
    vf_coef = _get("vf_coef", 0.5, float)
    max_grad_norm = _get("max_grad_norm", 0.5, float)
    min_layers_before_air = _get("min_layers_before_air", 4, int)
    mask_consecutive = _get(
        "mask_consecutive_materials", True, lambda x: x.lower() == "true"
    )
    use_lstm = _get("use_lstm", False, lambda x: x.lower() == "true")
    lstm_hidden = _get("lstm_hidden", 128, int)
    lstm_layers = _get("lstm_layers", 1, int)
    verbose = _get("verbose", 1, int)
    seed = _get("seed", 42, int)

    # Read logging frequencies
    mlflow_log_freq = parser.getint("general", "mlflow_log_freq", fallback=50)
    plot_freq = _get("plot_freq", 500, int)

    # Parse hidden layers
    hidden_str = _get("hidden", "[256, 256]")
    hidden = eval(hidden_str)

    # Set seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Log hyperparameters
    mlflow.log_params(
        {
            "total_episodes": total_episodes,
            "warmup_episodes": warmup_episodes,
            "epochs_per_step": epochs_per_step,
            "steps_per_objective": steps_per_objective,
            "episodes_per_update": episodes_per_update,
            "n_epochs": n_epochs,
            "batch_size": batch_size,
            "constraint_penalty": constraint_penalty,
            "pareto_bonus": pareto_bonus,
            "lr": lr,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "clip_range": clip_range,
            "ent_coef": ent_coef,
            "vf_coef": vf_coef,
            "hidden": str(hidden),
            "seed": seed,
        }
    )

    # Create environment with scheduling (like train_sb3_discrete)
    env = CoatOptHybridEnv(
        config=config,
        materials=materials,
        warmup_episodes=warmup_episodes,
        epochs_per_step=epochs_per_step,
        steps_per_objective=steps_per_objective,
        constraint_penalty=constraint_penalty,
        mask_consecutive_materials=mask_consecutive,
        min_layers_before_air=min_layers_before_air,
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

    # Get number of constraints for LSTM
    n_constraints = len(config.data.optimise_parameters)
    max_layers = config.data.n_layers

    policy = HybridActorCritic(
        obs_dim=obs_dim,
        n_materials=n_materials,
        min_thickness=min_thickness,
        max_thickness=max_thickness,
        hidden_dims=hidden,
        use_lstm=use_lstm,
        lstm_hidden=lstm_hidden,
        lstm_layers=lstm_layers,
        max_layers=max_layers if use_lstm else None,
        n_constraints=n_constraints if use_lstm else None,
    )

    if verbose:
        if use_lstm:
            print(f"Using LSTM: {lstm_layers} layer(s), hidden_size={lstm_hidden}")
        if restart_decay_on_phase:
            print(
                f"LR/entropy decay will restart every {lr_decay_episodes} episodes (warm restarts enabled)"
            )

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
    objectives = list(config.data.optimise_parameters)

    # Annealing tracking
    lr_init = lr
    ent_coef_init = ent_coef
    current_ent_coef = ent_coef
    total_warmup_episodes = warmup_episodes * len(objectives)
    warmup_end_episode = 0  # Track when warmup ended for phase-based annealing reset
    was_warmup = True  # Track warmup state to detect transition

    # Training loop
    obs, info = env.reset()
    mask = info["mask"]
    step_count = 0

    if verbose:
        print(f"Training for {total_episodes} episodes")

    while env.episode_count < total_episodes:
        # Collect episodes for this update
        buffer.clear()
        episodes_collected = 0

        while episodes_collected < episodes_per_update:
            material, thickness, log_prob, value = agent.act(obs, mask)
            action = {
                "material": material,
                "thickness": np.array([thickness], dtype=np.float32),
            }

            next_obs, reward, done, _, info = env.step(action)
            next_mask = info["mask"]

            buffer.add(obs, material, thickness, reward, value, log_prob, done, mask)

            if done:
                if "vals" in info:
                    ep_rewards.append(reward)
                    ep_vals.append(info["vals"])
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
                print(f"  Resetting LR and entropy decay for constrained phase...")

        # Update LR and entropy with cosine annealing (separate for warmup/constrained phases)
        if env.is_warmup:
            # Warmup phase: reset decay for EACH objective
            # Calculate progress within current objective's warmup phase
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
        import math

        decay_mult = 0.5 * (1 + math.cos(math.pi * progress))
        current_lr = lr_final + (lr_init - lr_final) * decay_mult
        current_ent_coef = (
            ent_coef_final + (ent_coef_init - ent_coef_final) * decay_mult
        )

        # Update agent LR and entropy
        for param_group in agent.optimizer.param_groups:
            param_group["lr"] = current_lr
        agent.ent_coef = current_ent_coef

        # Update policy
        rollout_data = buffer.get()
        ppo_logs = agent.update(rollout_data, n_epochs, batch_size)

        # Logging
        if env.episode_count % mlflow_log_freq == 0:
            if verbose:
                n_pareto = len(env.env.pareto_front_rewards)
                phase = "warmup" if env.is_warmup else "constrained"
                current_lr_display = agent.optimizer.param_groups[0]["lr"]
                print(
                    f"  [{phase}] episode {env.episode_count}/{total_episodes} | step {step_count} | "
                    f"pareto {n_pareto} | ent {current_ent_coef:.4f} | lr {current_lr_display:.2e}"
                )

            if mlflow.active_run():
                metrics = {
                    "step": step_count,
                    "pareto.size": len(env.env.pareto_front_rewards),
                }
                metrics.update({f"ppo.{k}": v for k, v in ppo_logs.items()})

                # Episode rewards
                if ep_rewards:
                    window = ep_rewards[-100:]
                    metrics["episode.reward_mean"] = float(np.mean(window))
                    metrics["episode.reward_std"] = float(np.std(window))

                # Objective values
                if ep_vals:
                    window = ep_vals[-100:]
                    for obj in objectives:
                        vals = [v.get(obj, float("nan")) for v in window]
                        vals = [v for v in vals if not np.isnan(v)]
                        if vals:
                            metrics[f"vals.{obj}_mean"] = float(np.mean(vals))
                            metrics[f"vals.{obj}_best"] = float(
                                np.min(vals) if obj == "absorption" else np.max(vals)
                            )

                # Hypervolume
                pareto = env.env.get_pareto_front(space="reward")
                if len(pareto) > 1:
                    try:
                        hv = env.env.compute_hypervolume(space="reward")
                        metrics["pareto.hypervolume"] = hv
                    except:
                        pass

                # Warmup best
                for obj, best in env.env.warmup_best_rewards.items():
                    metrics[f"warmup_best.{obj}"] = best

                mlflow.log_metrics(metrics, step=env.episode_count)

    # Return results
    designs_df, values_df, rewards_df = env.env.export_pareto_dataframes()
    return {
        "pareto_designs": designs_df,
        "pareto_values": values_df,
        "pareto_rewards": rewards_df,
        "model": None,
        "metadata": {
            "algorithm": "ppo_sequential",
            "total_episodes": total_episodes,
        },
    }
