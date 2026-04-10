#!/usr/bin/env python3
"""
Hybrid multi-agent + sequential PPO with dynamic constraint exploration.

Combines the benefits of:
- Multi-agent: Multiple agents explore in parallel with shared critic for knowledge transfer
- Sequential: Agents cycle through objectives with gradually tightening constraint bounds

Each agent:
- Cycles through target objectives (like sequential)
- Randomly samples constraint levels within current step bounds for better exploration
- Has its own actor network but shares critic with other agents

Config section: [hppo_hybrid]
  n_agents                 = 4              # Number of parallel agents
  total_episodes           = 10000
  warmup_episodes          = 500            # Per objective
  episodes_per_step          = 200            # Episodes per constraint phase
  steps_per_objective      = 10             # Constraint tightening steps (annealing schedule)
  episodes_per_update      = 10             # Complete episodes per agent before update
  n_epochs                 = 10             # SGD epochs per update
  batch_size               = 256
  constraint_penalty       = 3.0
  random_objective_order   = false          # If true, random; if false, cycle through objectives
  resample_constraints_freq = 1             # Resample constraints every N episodes per agent
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
  use_lstm                 = false          # Use shared LSTM for layer sequence processing
  lstm_hidden              = 128            # LSTM hidden size (if use_lstm=true)
  lstm_layers              = 1              # Number of LSTM layers (if use_lstm=true)
  hidden                   = [256, 256]     # MLP layers after LSTM/before policy heads
  seed                     = 42
  verbose                  = 1
  plot_freq                = 500
"""

import configparser
import math
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import gymnasium as gym
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from coatopt.environments.environment import CoatingEnvironment
from coatopt.utils.configs import Config, load_config
from coatopt.utils.math_utils import TruncatedNormalDist
from coatopt.utils.plotting import plot_pareto_front
from coatopt.utils.utils import load_materials

# ─────────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────────


class CoatOptHybridEnv(gym.Env):
    """
    Hybrid action wrapper: discrete material + continuous thickness.
    Supports dynamic constraint and objective switching.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: Config,
        materials: dict,
        constraint_penalty: float = 3.0,
        target_objective: str = "reflectivity",
        constraints: Dict[str, float] = None,
        min_layers_before_air: int = 4,
    ):
        super().__init__()
        self.base_env = CoatingEnvironment(config, materials)
        self.target_objective = target_objective
        self.constraints = constraints or {}
        self.min_layers_before_air = min_layers_before_air
        self.objectives = list(self.base_env.optimise_parameters)

        self.base_env.use_constrained_training = True
        self.base_env.is_warmup = True
        self.base_env.constraint_penalty = constraint_penalty
        self.base_env.target_objective = target_objective
        self.base_env.constraints = self.constraints

        self.n_materials = self.base_env.n_materials
        self.min_thickness = self.base_env.min_thickness
        self.max_thickness = self.base_env.max_thickness

        # Hybrid action space: dict with material (discrete) and thickness (continuous)
        self.action_space = gym.spaces.Dict(
            {
                "material": gym.spaces.Discrete(self.n_materials),
                "thickness": gym.spaces.Box(
                    low=np.array([self.min_thickness], dtype=np.float32),
                    high=np.array([self.max_thickness], dtype=np.float32),
                    dtype=np.float32,
                ),
            }
        )

        n_features = 1 + self.n_materials + 2
        # Observation: [layer_seq, objective_weights, constraints] (no current_layer for LSTM)
        n_objectives = len(self.objectives)
        obs_size = self.base_env.max_layers * n_features + n_objectives + n_objectives
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

        self.current_layer = 0
        self.prev_material = None

    def get_action_mask(self) -> np.ndarray:
        """Boolean mask over materials — shape (n_materials,)."""
        mask = np.ones(self.n_materials, dtype=bool)
        if self.prev_material is not None:
            mask[self.prev_material] = False
        if self.current_layer < self.min_layers_before_air:
            mask[self.base_env.air_material_index] = False
        if not mask.any():
            mask[:] = True
        return mask

    def _obs(self, state) -> np.ndarray:
        obs = (
            state.get_observation_tensor(pre_type="lstm")
            .numpy()
            .flatten()
            .astype(np.float32)
        )
        # Append objective weights (1.0 for target, 0.0 for others)
        for obj in self.objectives:
            weight = 1.0 if obj == self.target_objective else 0.0
            obs = np.append(obs, weight)
        # Append constraint thresholds
        for obj in self.objectives:
            obs = np.append(obs, self.constraints.get(obj, 0.0))
        # Note: NOT including current_layer - LSTM can track position from sequence
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        state = self.base_env.reset()
        self.current_layer = 0
        self.prev_material = None
        self.base_env.target_objective = self.target_objective
        self.base_env.constraints = self.constraints
        return self._obs(state), {"mask": self.get_action_mask()}

    def step(self, action: Dict):
        mat_idx = int(action["material"])
        thickness = float(action["thickness"][0])
        self.prev_material = mat_idx
        self.current_layer += 1

        coatopt_action = np.zeros(self.n_materials + 1, dtype=np.float32)
        coatopt_action[0] = thickness
        coatopt_action[1 + mat_idx] = 1.0

        state, rewards, _, finished, env_reward, _, vals = self.base_env.step(
            coatopt_action
        )
        obs = self._obs(state)
        reward = float(env_reward)
        info = {"mask": self.get_action_mask()}
        if finished:
            info["rewards"] = rewards
            info["vals"] = vals
            info["state_array"] = state.get_array()
        return obs, reward, finished, False, info

    def set_target(self, target: str, constraints: Dict[str, float]):
        self.target_objective = target
        self.constraints = constraints
        self.base_env.target_objective = target
        self.base_env.constraints = constraints


# ─────────────────────────────────────────────────────────────────────────────
# Rollout buffer (per agent)
# ─────────────────────────────────────────────────────────────────────────────


class RolloutBuffer:
    """Stores trajectories for on-policy PPO updates (episode-based collection)."""

    def __init__(
        self,
        max_layers: int,
        episodes_per_update: int,
        obs_dim: int,
        n_materials: int,
        gamma: float,
        gae_lambda: float,
    ):
        # Capacity: worst case = episodes_per_update * max_layers
        self.capacity = episodes_per_update * max_layers
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.pos = 0

        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.materials = np.zeros(self.capacity, dtype=np.int64)
        self.thicknesses = np.zeros(self.capacity, dtype=np.float32)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.values = np.zeros(self.capacity, dtype=np.float32)
        self.log_probs = np.zeros(self.capacity, dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)
        self.masks = np.zeros((self.capacity, n_materials), dtype=bool)

        # Computed during finalize()
        self.advantages = np.zeros(self.capacity, dtype=np.float32)
        self.returns = np.zeros(self.capacity, dtype=np.float32)

    def add(self, obs, material, thickness, reward, value, log_prob, done, mask):
        self.obs[self.pos] = obs
        self.materials[self.pos] = material
        self.thicknesses[self.pos] = thickness
        self.rewards[self.pos] = reward
        self.values[self.pos] = value
        self.log_probs[self.pos] = log_prob
        self.dones[self.pos] = done
        self.masks[self.pos] = mask
        self.pos += 1

    def finalize(self, last_value: float):
        """Compute advantages and returns using GAE."""
        # Use actual data collected (pos) not full capacity
        n_steps = self.pos
        advantages = np.zeros(n_steps, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                next_value = last_value
                next_non_terminal = 1.0 - self.dones[t]
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1.0 - self.dones[t]

            delta = (
                self.rewards[t]
                + self.gamma * next_value * next_non_terminal
                - self.values[t]
            )
            advantages[t] = last_gae = (
                delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            )

        self.advantages = advantages
        self.returns = advantages + self.values[:n_steps]

    def get(self):
        """Return all data as tensors (only actual collected data, not full capacity)."""
        n_steps = self.pos
        return {
            "obs": torch.FloatTensor(self.obs[:n_steps]),
            "materials": torch.LongTensor(self.materials[:n_steps]),
            "thicknesses": torch.FloatTensor(self.thicknesses[:n_steps]),
            "old_log_probs": torch.FloatTensor(self.log_probs[:n_steps]),
            "advantages": torch.FloatTensor(self.advantages),
            "returns": torch.FloatTensor(self.returns),
            "masks": torch.BoolTensor(self.masks[:n_steps]),
        }

    def clear(self):
        self.pos = 0


# ─────────────────────────────────────────────────────────────────────────────
# Policy network
# ─────────────────────────────────────────────────────────────────────────────


def _mlp(in_dim: int, out_dim: int, hidden: List[int]) -> nn.Sequential:
    layers, d = [], in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.Tanh()]
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class HybridActorCritic(nn.Module):
    """
    Hybrid policy: discrete material head + continuous thickness head + value head.

    π(material, thickness | s) = π_discrete(material | s) × π_continuous(thickness | s)

    Continuous action uses TruncatedNormalDist to ensure samples stay in [min_t, max_t].

    Optional shared LSTM: processes layer sequence before features (shared across agents).
    """

    def __init__(
        self,
        obs_dim: int,
        n_materials: int,
        hidden: List[int],
        min_t: float,
        max_t: float,
        shared_lstm: nn.Module = None,
        max_layers: int = None,
        n_constraints: int = None,
        n_objectives: int = None,
    ):
        super().__init__()
        self.n_materials = n_materials
        self.min_t = min_t
        self.max_t = max_t
        self.shared_lstm = shared_lstm
        self.use_lstm = shared_lstm is not None
        self.n_objectives = (
            n_objectives
            if n_objectives is not None
            else (n_constraints if n_constraints is not None else 2)
        )

        if self.use_lstm:
            assert max_layers is not None and n_constraints is not None
            # Observation structure: [layer_sequence (flattened), objective_weights, constraints]
            n_features_per_layer = 1 + n_materials + 2
            self.max_layers = max_layers
            self.n_features_per_layer = n_features_per_layer
            self.n_constraints = n_constraints

            # LSTM output is shared, so input to features is lstm_hidden + objective_weights + constraints
            lstm_hidden = shared_lstm.hidden_size
            input_dim = lstm_hidden + self.n_objectives + n_constraints
            self.features = _mlp(input_dim, hidden[-1], hidden[:-1])
        else:
            # Shared feature extractor (standard MLP)
            self.features = _mlp(obs_dim, hidden[-1], hidden[:-1])

        feat_dim = hidden[-1]

        # Policy heads
        self.material_head = nn.Linear(feat_dim, n_materials)  # logits
        self.thickness_mean = nn.Linear(feat_dim, 1)
        self.thickness_logstd = nn.Linear(feat_dim, 1)  # learnable per-state std

        # Value heads - one per objective for stable learning
        # Each head learns value function for when that objective is the target
        self.value_heads = nn.ModuleList(
            [nn.Linear(feat_dim, 1) for _ in range(self.n_objectives)]
        )

    def forward(
        self,
        obs: torch.Tensor,
        mask: torch.Tensor = None,
        target_obj_idx: int = 0,
        deterministic: bool = False,
    ) -> Tuple:
        """
        Sample action and compute value.

        Args:
            obs: observation tensor
            mask: action mask
            target_obj_idx: index of target objective (selects which value head to use)
            deterministic: whether to sample or take argmax

        Returns: (material, thickness, log_prob, value)
        """
        if self.use_lstm:
            # Extract components from observation (SAME AS SEQUENTIAL)
            # obs = [layer_sequence (flattened), objective_weights, constraints]
            batch_size = obs.shape[0]

            # Extract objective weights and constraints from the end
            layer_seq_flat = obs[:, : -(self.n_objectives + self.n_constraints)]
            objective_weights = obs[
                :, -(self.n_objectives + self.n_constraints) : -self.n_constraints
            ]
            constraints = obs[:, -self.n_constraints :]

            # Reshape layer sequence for LSTM: (batch, max_layers, features)
            layer_seq = layer_seq_flat.view(
                batch_size, self.max_layers, self.n_features_per_layer
            )

            # LSTM: use last layer's hidden state
            lstm_out, (h_n, c_n) = self.shared_lstm(layer_seq)
            lstm_features = h_n[-1]  # Take last layer: (batch, lstm_hidden)

            # Concatenate LSTM features with objective weights and constraints
            combined = torch.cat([lstm_features, objective_weights, constraints], dim=1)
            feat = self.features(combined)
        else:
            feat = self.features(obs)

        # Discrete: material selection with masking
        logits = self.material_head(feat)
        if mask is not None:
            logits = logits.masked_fill(~mask, float("-inf"))
        probs = F.softmax(logits, dim=-1)

        if deterministic:
            material = probs.argmax(dim=-1)
        else:
            material = torch.distributions.Categorical(probs).sample()
        log_prob_d = torch.log(probs + 1e-8).gather(1, material.unsqueeze(1)).squeeze(1)

        # Continuous: thickness using TruncatedNormal
        mean_raw = self.thickness_mean(feat).squeeze(-1)
        # Use sigmoid to softly constrain mean to valid range
        mean = self.min_t + (self.max_t - self.min_t) * torch.sigmoid(mean_raw)

        log_std = self.thickness_logstd(feat).squeeze(-1).clamp(-4, 0)  # Constrain std
        std = torch.exp(log_std)

        # TruncatedNormal handles bounds and log_prob automatically
        dist = TruncatedNormalDist(
            loc=mean,
            scale=std,
            a=torch.full_like(mean, self.min_t),
            b=torch.full_like(mean, self.max_t),
        )

        if deterministic:
            thickness = mean.clamp(self.min_t, self.max_t)
        else:
            thickness = dist.rsample()
            # Clamp to ensure numerical precision doesn't cause validation errors
            thickness = thickness.clamp(self.min_t, self.max_t)

        log_prob_c = dist.log_prob(thickness)

        # Joint log prob
        log_prob = log_prob_d + log_prob_c

        # Value - select head based on target objective
        value = self.value_heads[target_obj_idx](feat).squeeze(-1)

        return material, thickness, log_prob, value

    def evaluate(
        self,
        obs: torch.Tensor,
        materials: torch.Tensor,
        thicknesses: torch.Tensor,
        mask: torch.Tensor,
        target_obj_idx: int = 0,
    ) -> Tuple:
        """
        Evaluate log_prob and value for given actions.

        Args:
            obs: observation tensor
            materials: material actions
            thicknesses: thickness actions
            mask: action mask
            target_obj_idx: index of target objective (selects which value head to use)

        Returns: (log_prob, value, entropy)
        """
        if self.use_lstm:
            # Extract components from observation (SAME AS SEQUENTIAL)
            # obs = [layer_sequence (flattened), objective_weights, constraints]
            batch_size = obs.shape[0]

            # Extract objective weights and constraints from the end
            layer_seq_flat = obs[:, : -(self.n_objectives + self.n_constraints)]
            objective_weights = obs[
                :, -(self.n_objectives + self.n_constraints) : -self.n_constraints
            ]
            constraints = obs[:, -self.n_constraints :]

            # Reshape layer sequence for LSTM: (batch, max_layers, features)
            layer_seq = layer_seq_flat.view(
                batch_size, self.max_layers, self.n_features_per_layer
            )

            # LSTM: use last layer's hidden state
            lstm_out, (h_n, c_n) = self.shared_lstm(layer_seq)
            lstm_features = h_n[-1]  # Take last layer: (batch, lstm_hidden)

            # Concatenate LSTM features with objective weights and constraints
            combined = torch.cat([lstm_features, objective_weights, constraints], dim=1)
            feat = self.features(combined)
        else:
            feat = self.features(obs)

        # Discrete
        logits = self.material_head(feat).masked_fill(~mask, float("-inf"))
        probs = F.softmax(logits, dim=-1)
        log_probs_d = torch.log(probs + 1e-8)
        log_prob_d = log_probs_d.gather(1, materials.unsqueeze(1)).squeeze(1)
        entropy_d = -(probs * log_probs_d).sum(-1)

        # Continuous: use TruncatedNormal to evaluate stored actions
        mean_raw = self.thickness_mean(feat).squeeze(-1)
        # Use sigmoid to softly constrain mean to valid range
        mean = self.min_t + (self.max_t - self.min_t) * torch.sigmoid(mean_raw)

        log_std = self.thickness_logstd(feat).squeeze(-1).clamp(-4, 0)
        std = torch.exp(log_std)

        dist = TruncatedNormalDist(
            loc=mean,
            scale=std,
            a=torch.full_like(mean, self.min_t),
            b=torch.full_like(mean, self.max_t),
        )
        # Clamp to ensure numerical precision doesn't cause validation errors
        thicknesses_clamped = thicknesses.clamp(self.min_t, self.max_t)
        log_prob_c = dist.log_prob(thicknesses_clamped)
        entropy_c = dist.entropy()

        # Joint
        log_prob = log_prob_d + log_prob_c
        entropy = entropy_d + entropy_c

        # Value - select head based on target objective
        value = self.value_heads[target_obj_idx](feat).squeeze(-1)

        return log_prob, value, entropy


# ─────────────────────────────────────────────────────────────────────────────
# PPO agent
# ─────────────────────────────────────────────────────────────────────────────


class PPOAgent:
    """Single PPO agent with hybrid action space and optional shared critic."""

    def __init__(
        self,
        obs_dim: int,
        n_materials: int,
        hidden: List[int],
        min_t: float,
        max_t: float,
        lr: float,
        gamma: float,
        gae_lambda: float,
        clip_range: float,
        ent_coef: float,
        vf_coef: float,
        max_grad_norm: float,
        device: str,
        shared_value_net: nn.Module = None,  # Optional shared critic
        shared_lstm: nn.Module = None,  # Optional shared LSTM
        max_layers: int = None,
        n_constraints: int = None,
        n_objectives: int = None,
    ):
        self.device = device
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.shared_value_net = shared_value_net

        self.policy = HybridActorCritic(
            obs_dim,
            n_materials,
            hidden,
            min_t,
            max_t,
            shared_lstm=shared_lstm,
            max_layers=max_layers,
            n_constraints=n_constraints,
            n_objectives=n_objectives,
        ).to(device)
        # If using shared critic, only optimize policy parameters
        if shared_value_net is None:
            self.optimizer = Adam(self.policy.parameters(), lr=lr)
        else:
            # Only optimize policy heads, not value heads
            policy_params = [
                p for n, p in self.policy.named_parameters() if "value_heads" not in n
            ]
            self.optimizer = Adam(policy_params, lr=lr)

    @torch.no_grad()
    def act(
        self,
        obs: np.ndarray,
        mask: np.ndarray,
        target_obj_idx: int,
        deterministic: bool = False,
    ):
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        mask_t = torch.BoolTensor(mask).unsqueeze(0).to(self.device)
        material, thickness, log_prob, value = self.policy(
            obs_t, mask_t, target_obj_idx, deterministic
        )

        # Use shared critic if available
        if self.shared_value_net is not None:
            # Process observation to get features (handles LSTM if enabled)
            if self.policy.use_lstm:
                batch_size = obs_t.shape[0]
                layer_seq_flat = obs_t[:, : -self.policy.n_constraints]
                constraints = obs_t[:, -self.policy.n_constraints :]
                layer_seq = layer_seq_flat.view(
                    batch_size, self.policy.max_layers, self.policy.n_features_per_layer
                )
                lstm_out, (h_n, c_n) = self.policy.shared_lstm(layer_seq)
                lstm_features = h_n[-1]
                combined = torch.cat([lstm_features, constraints], dim=1)
                feat = self.policy.features(combined)
            else:
                feat = self.policy.features(obs_t)
            value = self.shared_value_net(feat).squeeze(-1)
            return material.item(), thickness.item(), log_prob.item(), value.item()

        return material.item(), thickness.item(), log_prob.item(), value.item()

    def update(
        self,
        rollout_data: dict,
        n_epochs: int,
        batch_size: int,
        target_obj_idx: int,
        update_value: bool = True,
    ) -> Dict[str, float]:
        """PPO update using collected rollout.

        Args:
            rollout_data: dict with observations, actions, returns, etc.
            n_epochs: number of optimization epochs
            batch_size: minibatch size
            target_obj_idx: index of target objective (for value head selection)
            update_value: whether to update value network
        """
        # Move to device
        for k in rollout_data:
            rollout_data[k] = rollout_data[k].to(self.device)

        obs = rollout_data["obs"]
        materials = rollout_data["materials"]
        thicknesses = rollout_data["thicknesses"]
        old_log_probs = rollout_data["old_log_probs"]
        advantages = rollout_data["advantages"]
        returns = rollout_data["returns"]
        masks = rollout_data["masks"]

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n_samples = obs.shape[0]
        logs = {"policy_loss": 0, "value_loss": 0, "entropy": 0, "clip_frac": 0}
        n_updates = 0

        for epoch in range(n_epochs):
            indices = torch.randperm(n_samples)
            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                batch_idx = indices[start:end]

                log_probs, values, entropy = self.policy.evaluate(
                    obs[batch_idx],
                    materials[batch_idx],
                    thicknesses[batch_idx],
                    masks[batch_idx],
                    target_obj_idx,
                )

                # Use shared critic if available
                if self.shared_value_net is not None:
                    # Process observation to get features (handles LSTM if enabled)
                    if self.policy.use_lstm:
                        batch_size_inner = obs[batch_idx].shape[0]
                        layer_seq_flat = obs[batch_idx][:, : -self.policy.n_constraints]
                        constraints = obs[batch_idx][:, -self.policy.n_constraints :]
                        layer_seq = layer_seq_flat.view(
                            batch_size_inner,
                            self.policy.max_layers,
                            self.policy.n_features_per_layer,
                        )
                        lstm_out, (h_n, c_n) = self.policy.shared_lstm(layer_seq)
                        lstm_features = h_n[-1]
                        combined = torch.cat([lstm_features, constraints], dim=1)
                        feat = self.policy.features(combined)
                    else:
                        feat = self.policy.features(obs[batch_idx])
                    values = self.shared_value_net(feat).squeeze(-1)

                # Policy loss (PPO clip objective)
                ratio = torch.exp(log_probs - old_log_probs[batch_idx])
                adv = advantages[batch_idx]
                surr1 = ratio * adv
                surr2 = (
                    torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * adv
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (only if not using shared critic or if updating shared critic)
                value_loss = F.mse_loss(values, returns[batch_idx])

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss (exclude value loss if using shared critic and not updating it)
                if self.shared_value_net is None or update_value:
                    loss = (
                        policy_loss
                        + self.vf_coef * value_loss
                        + self.ent_coef * entropy_loss
                    )
                else:
                    loss = policy_loss + self.ent_coef * entropy_loss

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Logging
                logs["policy_loss"] += policy_loss.item()
                logs["value_loss"] += value_loss.item()
                logs["entropy"] += entropy.mean().item()
                logs["clip_frac"] += (
                    ((ratio - 1.0).abs() > self.clip_range).float().mean().item()
                )
                n_updates += 1

        return {k: v / n_updates for k, v in logs.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Hybrid multi-agent trainer
# ─────────────────────────────────────────────────────────────────────────────


class HybridMultiAgentPPO:
    """
    Hybrid multi-agent + sequential PPO.

    Multiple agents with shared critic, each cycling through objectives
    with randomly sampled constraints within annealing bounds.
    """

    def __init__(
        self,
        config: Config,
        materials: dict,
        n_agents: int = 4,
        episodes_per_update: int = 2048,
        n_epochs: int = 10,
        batch_size: int = 256,
        warmup_episodes_per_obj: int = 500,
        episodes_per_step: int = 200,
        steps_per_objective: int = 10,
        constraint_penalty: float = 3.0,
        random_objective_order: bool = False,
        resample_constraints_freq: int = 1,
        hidden: List[int] = None,
        lr: float = 3e-4,
        lr_final: float = None,
        lr_decay_episodes: int = 10000,
        restart_decay_on_phase: bool = False,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        ent_coef_final: float = None,
        ent_decay_episodes: int = 10000,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        min_layers_before_air: int = 4,
        use_lstm: bool = False,
        lstm_hidden: int = 128,
        lstm_layers: int = 1,
        device: str = "cpu",
        verbose: int = 1,
        log_freq: int = 50,
        plot_freq: int = 500,
        save_dir: str = None,
    ):
        if hidden is None:
            hidden = [256, 256]

        self.n_agents = n_agents
        self.episodes_per_update = episodes_per_update
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = device
        self.verbose = verbose
        self.objectives = list(config.data.optimise_parameters)
        self.n_objectives = len(self.objectives)

        # Create objective name -> index mapping for value head selection
        self.objective_to_idx = {obj: idx for idx, obj in enumerate(self.objectives)}

        self.warmup_episodes_per_obj = warmup_episodes_per_obj
        # Total warmup episodes accounts for multiple agents cycling through objectives
        self.total_warmup_episodes = (
            warmup_episodes_per_obj * self.n_objectives * n_agents
        )
        self.episodes_per_step = episodes_per_step
        self.steps_per_objective = steps_per_objective
        self.random_objective_order = random_objective_order
        self.resample_constraints_freq = resample_constraints_freq

        # Annealing settings
        self.lr_init = lr
        self.lr_final = lr_final if lr_final is not None else lr
        self.lr_decay_episodes = lr_decay_episodes
        self.restart_decay_on_phase = restart_decay_on_phase
        self.ent_coef_init = ent_coef
        self.ent_coef_final = ent_coef_final if ent_coef_final is not None else ent_coef
        self.ent_decay_episodes = ent_decay_episodes
        self.current_ent_coef = ent_coef

        self.warmup_best: Dict[str, float] = {o: 0.0 for o in self.objectives}
        max_layers = config.data.n_layers
        self.max_layers = max_layers

        # Create environments (one per agent)
        self.envs = [
            CoatOptHybridEnv(
                config,
                materials,
                constraint_penalty,
                target_objective=self.objectives[0],  # Will be updated
                constraints={},
                min_layers_before_air=min_layers_before_air,
            )
            for i in range(n_agents)
        ]
        for env in self.envs:
            env.base_env.is_warmup = True

        # Share Pareto front across all agents
        primary = self.envs[0].base_env
        for env in self.envs[1:]:
            env.base_env.pareto_front_rewards = primary.pareto_front_rewards
            env.base_env.pareto_front_values = primary.pareto_front_values

        obs_dim = self.envs[0].observation_space.shape[0]
        n_materials = self.envs[0].n_materials
        min_t = self.envs[0].min_thickness
        max_t = self.envs[0].max_thickness

        # Create shared critic (key feature of hybrid approach)
        feat_dim = hidden[-1]
        self.shared_value_net = nn.Linear(feat_dim, 1).to(device)
        self.value_optimizer = Adam(self.shared_value_net.parameters(), lr=lr)
        if verbose:
            print(f"Using shared critic across {n_agents} agents")

        # Create shared LSTM if enabled
        self.use_lstm = use_lstm
        if use_lstm:
            n_features_per_layer = 1 + n_materials + 2
            n_constraints = len(self.objectives)
            self.shared_lstm = nn.LSTM(
                input_size=n_features_per_layer,
                hidden_size=lstm_hidden,
                num_layers=lstm_layers,
                batch_first=True,
            ).to(device)
            self.lstm_optimizer = Adam(self.shared_lstm.parameters(), lr=lr)
            if verbose:
                print(
                    f"Using shared LSTM: {lstm_layers} layer(s), hidden_size={lstm_hidden}, across {n_agents} agents"
                )
        else:
            self.shared_lstm = None
            n_constraints = None

        # Create PPO agents and buffers
        self.agents = [
            PPOAgent(
                obs_dim,
                n_materials,
                hidden,
                min_t,
                max_t,
                lr,
                gamma,
                gae_lambda,
                clip_range,
                ent_coef,
                vf_coef,
                max_grad_norm,
                device,
                shared_value_net=self.shared_value_net,
                shared_lstm=self.shared_lstm,
                max_layers=self.max_layers if use_lstm else None,
                n_constraints=n_constraints,
                n_objectives=self.n_objectives,
            )
            for _ in range(n_agents)
        ]
        self.buffers = [
            RolloutBuffer(
                self.max_layers,
                episodes_per_update,
                obs_dim,
                n_materials,
                gamma,
                gae_lambda,
            )
            for _ in range(n_agents)
        ]

        self._obs = [None] * n_agents
        self._mask = [None] * n_agents
        self.step_count = 0
        self.episode_count = 0
        self.warmup_done = False
        self._warmup_end_episode = (
            0  # Track when warmup ended for phase-based annealing reset
        )

        # Per-agent state tracking
        self._agent_episode_count = [0] * n_agents  # Episodes per agent
        self._agent_target_objectives = [
            self.objectives[i % self.n_objectives] for i in range(n_agents)
        ]
        self._agent_constraints = [{} for _ in range(n_agents)]

        # Monitoring
        self.log_freq = log_freq
        self.plot_freq = plot_freq
        self.save_dir = Path(save_dir) if save_dir else None
        self._ep_rewards = []
        self._ep_vals = []
        self._ppo_logs = {}

    def _reset(self, i: int):
        obs, info = self.envs[i].reset()
        self._obs[i] = obs
        self._mask[i] = info["mask"]

    def _update_agent_target_and_constraints(self, agent_idx: int):
        """
        Update target objective and constraints for a single agent.

        During warmup: cycle through objectives, no constraints
        After warmup: cycle through objectives, randomly sample constraints within current step bounds
        """
        agent_ep = self._agent_episode_count[agent_idx]

        if not self.warmup_done:
            # Warmup: cycle through objectives
            # Offset by agent_idx so each agent starts with different objective
            obj_idx = (
                (agent_ep // self.warmup_episodes_per_obj) + agent_idx
            ) % self.n_objectives
            target = self.objectives[obj_idx]
            constraints = {}
            self.envs[agent_idx].base_env.is_warmup = True
        else:
            # Constrained phase
            self.envs[agent_idx].base_env.is_warmup = False

            # Determine current constraint step (global across all agents)
            constrained_episodes = self.episode_count - self.total_warmup_episodes
            current_phase = constrained_episodes // self.episodes_per_step

            # Which objective to target (cycle or random)
            if self.random_objective_order:
                obj_idx = random.randint(0, self.n_objectives - 1)
            else:
                obj_idx = current_phase % self.n_objectives
            target = self.objectives[obj_idx]

            # Current constraint step (increases over time, resets after steps_per_objective)
            step = (current_phase // self.n_objectives) % self.steps_per_objective

            # Maximum constraint level for this step
            max_constraint_frac = (step + 1) / self.steps_per_objective

            # Randomly sample constraint level for each non-target objective
            # between 0 and max_constraint_frac * warmup_best
            constraints = {}
            for i, obj in enumerate(self.objectives):
                if i != obj_idx:
                    # Random constraint between 0 and current max
                    random_frac = random.uniform(0, max_constraint_frac)
                    constraints[obj] = random_frac * self.warmup_best[obj]

        self._agent_target_objectives[agent_idx] = target
        self._agent_constraints[agent_idx] = constraints
        self.envs[agent_idx].set_target(target, constraints)

    def _log_progress(self, episode: int, step: int):
        """Log to MLflow and print progress."""
        if episode % self.log_freq != 0 and episode != 0:
            return

        # Print progress
        if self.verbose:
            n_pareto = len(self.envs[0].base_env.pareto_front_rewards)
            phase = "warmup" if not self.warmup_done else "constrained"
            lr = self.agents[0].optimizer.param_groups[0]["lr"]

            # Show current constraint step
            if not self.warmup_done:
                print(
                    f"  [{phase}] ep {episode:>6d} (step {step:>7d}) | "
                    f"pareto {n_pareto:>3d} | ent {self.current_ent_coef:.4f} | lr {lr:.2e}"
                )
            else:
                constrained_episodes = self.episode_count - self.total_warmup_episodes
                current_phase = constrained_episodes // self.episodes_per_step
                step_in_cycle = (
                    current_phase // self.n_objectives
                ) % self.steps_per_objective
                print(
                    f"  [{phase}] ep {episode:>6d} (step {step:>7d}) | "
                    f"pareto {n_pareto:>3d} | constraint_step {step_in_cycle}/{self.steps_per_objective} | "
                    f"ent {self.current_ent_coef:.4f} | lr {lr:.2e}"
                )

        # Log to MLflow
        if not mlflow.active_run():
            return

        metrics = {"episode": episode, "step": step}
        metrics.update({f"ppo.{k}": v for k, v in self._ppo_logs.items()})

        if self._ep_rewards:
            window = self._ep_rewards[-100:]
            metrics["episode.reward_mean"] = float(np.mean(window))
            metrics["episode.reward_std"] = float(np.std(window))

        if self._ep_vals:
            window = self._ep_vals[-100:]
            for obj in self.objectives:
                vals = [v.get(obj, float("nan")) for v in window]
                vals = [v for v in vals if not np.isnan(v)]
                if vals:
                    metrics[f"vals.{obj}_mean"] = float(np.mean(vals))
                    metrics[f"vals.{obj}_best"] = float(
                        np.min(vals) if obj == "absorption" else np.max(vals)
                    )

        pareto = self.envs[0].base_env.get_pareto_front(space="reward")
        metrics["pareto.size"] = len(pareto)
        if len(pareto) > 1:
            try:
                hv = self.envs[0].base_env.compute_hypervolume(space="reward")
                metrics["pareto.hypervolume"] = hv
            except:
                pass

        for obj, best in self.warmup_best.items():
            metrics[f"warmup_best.{obj}"] = best

        mlflow.log_metrics(metrics, step=step)

        # Periodic checkpointing
        if self.save_dir and episode % self.plot_freq == 0 and episode > 0:
            try:
                designs_df, values_df, rewards_df = self.envs[
                    0
                ].base_env.export_pareto_dataframes()
                if not values_df.empty:
                    save_path = Path(self.save_dir)
                    save_path.mkdir(parents=True, exist_ok=True)

                    # Save combined Pareto front CSV (like end-of-training)
                    combined_pareto = pd.concat(
                        [designs_df, values_df, rewards_df.add_suffix("_reward")],
                        axis=1,
                    )
                    pareto_path = save_path / f"pareto_front_ep{episode}.csv"
                    combined_pareto.to_csv(pareto_path, index=False)

                    # Save plot
                    plot_pareto_front(
                        values_df,
                        self.objectives,
                        self.save_dir,
                        "vals",
                        f"ppo_hybrid_ep{episode}",
                    )

                    if self.verbose:
                        print(f"  Saved Pareto front checkpoint at episode {episode}")
            except Exception as e:
                if self.verbose:
                    print(f"  [checkpoint] skipped: {e}")

    def train(self, total_episodes: int) -> dict:
        """Train the hybrid multi-agent system."""
        # Initialize all agents
        for i in range(self.n_agents):
            self._update_agent_target_and_constraints(i)
            self._reset(i)

        if self.verbose:
            print(f"Warmup for {self.total_warmup_episodes} episodes...")
            print(f"Training with {self.n_agents} agents, shared critic\n")

        while self.episode_count < total_episodes:
            # Check if warmup is complete
            if (
                not self.warmup_done
                and self.episode_count >= self.total_warmup_episodes
            ):
                self.warmup_done = True
                self._warmup_end_episode = (
                    self.episode_count
                )  # Record when warmup ended
                print(f"\nWarmup complete at episode {self.episode_count}")
                print(f"Best warmup rewards: {self.warmup_best}")
                print("Resetting LR and entropy decay for constrained phase...")
                print(
                    "Starting constrained phase with random constraint exploration...\n"
                )

            # Collect episodes_per_update complete episodes for each agent
            for agent_idx in range(self.n_agents):
                self.buffers[agent_idx].clear()
                episodes_collected = 0

                while episodes_collected < self.episodes_per_update:
                    obs = self._obs[agent_idx]
                    mask = self._mask[agent_idx]

                    # Sample action
                    target_obj_idx = self.objective_to_idx[
                        self._agent_target_objectives[agent_idx]
                    ]
                    material, thickness, log_prob, value = self.agents[agent_idx].act(
                        obs, mask, target_obj_idx
                    )
                    action = {
                        "material": material,
                        "thickness": np.array([thickness], dtype=np.float32),
                    }

                    # Step environment
                    next_obs, reward, done, _, info = self.envs[agent_idx].step(action)
                    next_mask = info["mask"]

                    # Store transition
                    self.buffers[agent_idx].add(
                        obs, material, thickness, reward, value, log_prob, done, mask
                    )

                    self.step_count += 1

                    # Episode complete
                    if done:
                        # Update warmup bounds
                        if not self.warmup_done and "vals" in info:
                            norm = self.envs[
                                agent_idx
                            ].base_env.compute_objective_rewards(
                                info["vals"], normalised=True
                            )
                            for obj in self.objectives:
                                self.warmup_best[obj] = max(
                                    self.warmup_best[obj], norm.get(obj, 0.0)
                                )

                        # Track episode stats
                        if "vals" in info:
                            self._ep_rewards.append(reward)
                            self._ep_vals.append(info["vals"])
                            self.episode_count += 1
                            self._agent_episode_count[agent_idx] += 1
                            episodes_collected += 1

                        # Update target and constraints for this agent
                        if (
                            self._agent_episode_count[agent_idx]
                            % self.resample_constraints_freq
                            == 0
                        ):
                            self._update_agent_target_and_constraints(agent_idx)

                        self._reset(agent_idx)
                        obs = self._obs[agent_idx]
                        mask = self._mask[agent_idx]
                    else:
                        self._obs[agent_idx] = next_obs
                        self._mask[agent_idx] = next_mask
                        obs = next_obs
                        mask = next_mask

                # Finalize buffer with bootstrap value
                target_obj_idx = self.objective_to_idx[
                    self._agent_target_objectives[agent_idx]
                ]
                _, _, _, last_value = self.agents[agent_idx].act(
                    self._obs[agent_idx], self._mask[agent_idx], target_obj_idx
                )
                self.buffers[agent_idx].finalize(last_value)

            # Update LR and entropy with cosine annealing (separate for warmup/constrained phases)
            if not self.warmup_done:
                # Warmup phase: decay from init to final over warmup episodes
                progress = min(1.0, self.episode_count / self.total_warmup_episodes)
            else:
                # Constrained phase: decay over remaining episodes
                constrained_episodes = self.episode_count - self._warmup_end_episode
                if self.restart_decay_on_phase:
                    # Restart decay every lr_decay_episodes (like cosine annealing with warm restarts)
                    episode_in_current_phase = (
                        (constrained_episodes - 1) % self.lr_decay_episodes
                    ) + 1
                    progress = min(
                        1.0, episode_in_current_phase / self.lr_decay_episodes
                    )
                else:
                    # Decay once over entire constrained phase
                    progress = min(1.0, constrained_episodes / self.lr_decay_episodes)

            # Cosine annealing: smooth decay with slower finish
            decay_mult = 0.5 * (1 + math.cos(math.pi * progress))
            current_lr = self.lr_final + (self.lr_init - self.lr_final) * decay_mult
            self.current_ent_coef = (
                self.ent_coef_final
                + (self.ent_coef_init - self.ent_coef_final) * decay_mult
            )

            # Update all agents LR and entropy
            for agent in self.agents:
                for param_group in agent.optimizer.param_groups:
                    param_group["lr"] = current_lr
                agent.ent_coef = self.current_ent_coef

            # Update shared critic and LSTM LR
            for param_group in self.value_optimizer.param_groups:
                param_group["lr"] = current_lr
            if self.use_lstm:
                for param_group in self.lstm_optimizer.param_groups:
                    param_group["lr"] = current_lr

            # Update all agents: policies first, then shared critic

            # Update policies only (not value)
            for agent_idx in range(self.n_agents):
                rollout_data = self.buffers[agent_idx].get()
                target_obj_idx = self.objective_to_idx[
                    self._agent_target_objectives[agent_idx]
                ]
                self._ppo_logs = self.agents[agent_idx].update(
                    rollout_data,
                    self.n_epochs,
                    self.batch_size,
                    target_obj_idx,
                    update_value=False,
                )

            # Update shared critic with all agents' data
            all_obs, all_returns = [], []
            for agent_idx in range(self.n_agents):
                rollout_data = self.buffers[agent_idx].get()
                all_obs.append(rollout_data["obs"])
                all_returns.append(rollout_data["returns"])

            # Combine data from all agents
            combined_obs = torch.cat(all_obs, dim=0).to(self.device)
            combined_returns = torch.cat(all_returns, dim=0).to(self.device)

            # Update shared critic
            n_samples = combined_obs.shape[0]
            for epoch in range(self.n_epochs):
                indices = torch.randperm(n_samples)
                for start in range(0, n_samples, self.batch_size):
                    end = start + self.batch_size
                    batch_idx = indices[start:end]

                    # Process observation to get features (handles LSTM if enabled)
                    if self.use_lstm:
                        batch_size_inner = combined_obs[batch_idx].shape[0]
                        n_constraints = len(self.objectives)
                        layer_seq_flat = combined_obs[batch_idx][:, :-n_constraints]
                        constraints = combined_obs[batch_idx][:, -n_constraints:]
                        layer_seq = layer_seq_flat.view(
                            batch_size_inner,
                            self.max_layers,
                            1 + self.envs[0].n_materials + 2,
                        )
                        lstm_out, (h_n, c_n) = self.shared_lstm(layer_seq)
                        lstm_features = h_n[-1]
                        combined = torch.cat([lstm_features, constraints], dim=1)
                        feat = self.agents[0].policy.features(combined)
                    else:
                        feat = self.agents[0].policy.features(combined_obs[batch_idx])
                    values = self.shared_value_net(feat).squeeze(-1)
                    value_loss = F.mse_loss(values, combined_returns[batch_idx])

                    self.value_optimizer.zero_grad()
                    if self.use_lstm:
                        self.lstm_optimizer.zero_grad()

                    value_loss.backward()

                    nn.utils.clip_grad_norm_(
                        self.shared_value_net.parameters(), self.agents[0].max_grad_norm
                    )
                    self.value_optimizer.step()

                    # Update shared LSTM if used
                    if self.use_lstm:
                        nn.utils.clip_grad_norm_(
                            self.shared_lstm.parameters(), self.agents[0].max_grad_norm
                        )
                        self.lstm_optimizer.step()

            # Logging and progress
            self._log_progress(self.episode_count, self.step_count)

        designs_df, values_df, rewards_df = self.envs[
            0
        ].base_env.export_pareto_dataframes()
        return {
            "pareto_designs": designs_df,
            "pareto_values": values_df,
            "pareto_rewards": rewards_df,
            "model": None,
            "metadata": {
                "n_agents": self.n_agents,
                "total_episodes": total_episodes,
                "total_steps": self.step_count,
            },
        }


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def train(config_path: str, save_dir: str = None) -> dict:
    parser = configparser.ConfigParser()
    parser.read(config_path)
    section = "hppo_hybrid"

    def _get(key, fallback, cast=str):
        return cast(parser.get(section, key, fallback=str(fallback)))

    materials_path = parser.get("general", "materials_path")
    materials = load_materials(str(materials_path))
    config = load_config(config_path)

    seed = _get("seed", 42, int)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    n_agents = _get("n_agents", 4, int)
    total_episodes = _get("total_episodes", 10000, int)
    warmup_episodes = _get("warmup_episodes", 500, int)
    episodes_per_step = _get("episodes_per_step", 200, int)
    steps_per_objective = _get("steps_per_objective", 10, int)
    episodes_per_update = _get("episodes_per_update", 2048, int)
    n_epochs = _get("n_epochs", 10, int)
    batch_size = _get("batch_size", 256, int)
    constraint_penalty = _get("constraint_penalty", 3.0, float)
    random_objective_order = _get(
        "random_objective_order", False, lambda x: x.lower() == "true"
    )
    resample_constraints_freq = _get("resample_constraints_freq", 1, int)
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
    use_lstm = _get("use_lstm", False, lambda x: x.lower() == "true")
    lstm_hidden = _get("lstm_hidden", 128, int)
    lstm_layers = _get("lstm_layers", 1, int)
    verbose = _get("verbose", 1, int)
    hidden_str = parser.get(section, "hidden", fallback="[256, 256]")
    hidden = eval(hidden_str)

    # Read mlflow_log_freq from [general] section
    mlflow_log_freq = parser.getint("general", "mlflow_log_freq", fallback=50)
    plot_freq = _get("plot_freq", 500, int)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if save_dir is None:
        base = parser.get("general", "save_dir", fallback="./runs")
        run_name = parser.get("general", "run_name", fallback="")
        date_str = datetime.now().strftime("%Y%m%d")
        suffix = f"-{run_name}" if run_name else ""
        save_dir = Path(base) / f"{date_str}-ppo_hybrid{suffix}"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Hybrid Multi-Agent + Sequential PPO")
    print(f"  Agents      : {n_agents}")
    print(f"  Episodes    : {total_episodes:,}")
    print(f"  Device      : {device}")
    print(f"  Objectives  : {list(config.data.optimise_parameters)}")
    print(f"  Constraint steps: {steps_per_objective}")
    print(f"  Random obj order: {random_objective_order}")
    print(f"{'='*60}\n")

    # Log parameters to MLflow
    if mlflow.active_run():
        mlflow.log_params(
            {
                "algorithm": "hppo_hybrid",
                "n_agents": n_agents,
                "total_episodes": total_episodes,
                "warmup_episodes": warmup_episodes,
                "episodes_per_step": episodes_per_step,
                "steps_per_objective": steps_per_objective,
                "episodes_per_update": episodes_per_update,
                "n_epochs": n_epochs,
                "batch_size": batch_size,
                "constraint_penalty": constraint_penalty,
                "random_objective_order": random_objective_order,
                "resample_constraints_freq": resample_constraints_freq,
                "lr": lr,
                "gamma": gamma,
                "hidden": str(hidden),
                "seed": seed,
            }
        )

    trainer = HybridMultiAgentPPO(
        config=config,
        materials=materials,
        n_agents=n_agents,
        episodes_per_update=episodes_per_update,
        n_epochs=n_epochs,
        batch_size=batch_size,
        warmup_episodes_per_obj=warmup_episodes,
        episodes_per_step=episodes_per_step,
        steps_per_objective=steps_per_objective,
        constraint_penalty=constraint_penalty,
        random_objective_order=random_objective_order,
        resample_constraints_freq=resample_constraints_freq,
        hidden=hidden,
        lr=lr,
        lr_final=lr_final,
        lr_decay_episodes=lr_decay_episodes,
        restart_decay_on_phase=restart_decay_on_phase,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        ent_coef_final=ent_coef_final,
        ent_decay_episodes=ent_decay_episodes,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        min_layers_before_air=min_layers_before_air,
        use_lstm=use_lstm,
        lstm_hidden=lstm_hidden,
        lstm_layers=lstm_layers,
        device=device,
        verbose=verbose,
        log_freq=mlflow_log_freq,
        plot_freq=plot_freq,
        save_dir=str(save_dir),
    )

    start = time.time()
    results = trainer.train(total_episodes)
    results["metadata"]["training_time_s"] = time.time() - start

    print(f"\nTraining done in {results['metadata']['training_time_s']/60:.1f} min")
    print(
        f"Total: {results['metadata']['total_episodes']} episodes, "
        f"{results['metadata']['total_steps']:,} steps"
    )
    print(f"Pareto front: {len(results['pareto_rewards'])} solutions")

    return results


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Hybrid Multi-Agent + Sequential PPO for CoatOpt"
    )
    ap.add_argument("--config", required=True, help="Path to config INI")
    ap.add_argument("--save-dir", default=None, help="Override save directory")
    args = ap.parse_args()
    train(config_path=args.config, save_dir=args.save_dir)
