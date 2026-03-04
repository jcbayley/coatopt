#!/usr/bin/env python3
"""
Multi-agent PPO with hybrid action space (discrete material + continuous thickness).

Config section: [ppo_multiagent]
  n_agents              = 6
  total_episodes        = 10000       # total episodes to train (across all agents)
  warmup_episodes       = 500         # per objective
  n_steps               = 2048        # steps per agent before update
  n_epochs              = 10          # optimization epochs per update
  batch_size            = 256         # minibatch size for SGD
  constraint_penalty    = 3.0
  lr                    = 3e-4
  gamma                 = 0.99
  gae_lambda            = 0.95
  clip_range            = 0.2
  ent_coef              = 0.01
  vf_coef               = 0.5
  max_grad_norm         = 0.5
  hidden                = [256, 256]
  min_layers_before_air = 4
  seed                  = 42
  verbose               = 1
"""

import configparser
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import gymnasium as gym
import mlflow
import numpy as np
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
    Action space: Dict(material=Discrete(n_materials), thickness=Box([min, max]))
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
        obs_size = self.base_env.max_layers * n_features + 1 + len(self.objectives)
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
        for obj in self.objectives:
            obs = np.append(obs, self.constraints.get(obj, 0.0))
        obs = np.append(obs, self.current_layer / self.base_env.max_layers)
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
        reward = float(env_reward)  # Respects use_intermediate_reward
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
    """Stores trajectories for on-policy PPO updates."""

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        n_materials: int,
        gamma: float,
        gae_lambda: float,
    ):
        self.capacity = capacity
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.pos = 0

        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.materials = np.zeros(capacity, dtype=np.int64)
        self.thicknesses = np.zeros(capacity, dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)
        self.log_probs = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.masks = np.zeros((capacity, n_materials), dtype=bool)

        # Computed during finalize()
        self.advantages = np.zeros(capacity, dtype=np.float32)
        self.returns = np.zeros(capacity, dtype=np.float32)

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
        assert self.pos == self.capacity, "Buffer not full"

        advantages = np.zeros_like(self.rewards)
        last_gae = 0.0

        for t in reversed(range(self.capacity)):
            if t == self.capacity - 1:
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
        self.returns = advantages + self.values

    def get(self):
        """Return all data as tensors."""
        assert self.pos == self.capacity
        return {
            "obs": torch.FloatTensor(self.obs),
            "materials": torch.LongTensor(self.materials),
            "thicknesses": torch.FloatTensor(self.thicknesses),
            "old_log_probs": torch.FloatTensor(self.log_probs),
            "advantages": torch.FloatTensor(self.advantages),
            "returns": torch.FloatTensor(self.returns),
            "masks": torch.BoolTensor(self.masks),
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
    """

    def __init__(
        self,
        obs_dim: int,
        n_materials: int,
        hidden: List[int],
        min_t: float,
        max_t: float,
    ):
        super().__init__()
        self.n_materials = n_materials
        self.min_t = min_t
        self.max_t = max_t

        # Shared feature extractor
        self.features = _mlp(obs_dim, hidden[-1], hidden[:-1])
        feat_dim = hidden[-1]

        # Policy heads
        self.material_head = nn.Linear(feat_dim, n_materials)  # logits
        self.thickness_mean = nn.Linear(feat_dim, 1)
        self.thickness_logstd = nn.Linear(feat_dim, 1)  # learnable per-state std

        # Value head
        self.value_head = nn.Linear(feat_dim, 1)

    def forward(
        self, obs: torch.Tensor, mask: torch.Tensor = None, deterministic: bool = False
    ) -> Tuple:
        """
        Sample action and compute value.
        Returns: (material, thickness, log_prob, value)
        """
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

        # Value
        value = self.value_head(feat).squeeze(-1)

        return material, thickness, log_prob, value

    def evaluate(
        self,
        obs: torch.Tensor,
        materials: torch.Tensor,
        thicknesses: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple:
        """
        Evaluate log_prob and value for given actions.
        Returns: (log_prob, value, entropy)
        """
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

        # Value
        value = self.value_head(feat).squeeze(-1)

        return log_prob, value, entropy


# ─────────────────────────────────────────────────────────────────────────────
# PPO agent
# ─────────────────────────────────────────────────────────────────────────────


class PPOAgent:
    """Single PPO agent with hybrid action space."""

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
    ):
        self.device = device
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

        self.policy = HybridActorCritic(obs_dim, n_materials, hidden, min_t, max_t).to(
            device
        )
        self.optimizer = Adam(self.policy.parameters(), lr=lr)

    @torch.no_grad()
    def act(self, obs: np.ndarray, mask: np.ndarray, deterministic: bool = False):
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        mask_t = torch.BoolTensor(mask).unsqueeze(0).to(self.device)
        material, thickness, log_prob, value = self.policy(obs_t, mask_t, deterministic)
        return material.item(), thickness.item(), log_prob.item(), value.item()

    def update(
        self, rollout_data: dict, n_epochs: int, batch_size: int
    ) -> Dict[str, float]:
        """PPO update using collected rollout."""
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
                )

                # Policy loss (PPO clip objective)
                ratio = torch.exp(log_probs - old_log_probs[batch_idx])
                adv = advantages[batch_idx]
                surr1 = ratio * adv
                surr2 = (
                    torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * adv
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(values, returns[batch_idx])

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    policy_loss
                    + self.vf_coef * value_loss
                    + self.ent_coef * entropy_loss
                )

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
# Multi-agent trainer
# ─────────────────────────────────────────────────────────────────────────────


class MultiAgentPPO:
    """N PPO agents with constraint-based targeting and shared Pareto front."""

    def __init__(
        self,
        config: Config,
        materials: dict,
        n_agents: int = 6,
        n_steps: int = 2048,
        n_epochs: int = 10,
        batch_size: int = 256,
        warmup_episodes_per_obj: int = 500,
        constraint_penalty: float = 3.0,
        hidden: List[int] = None,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        min_layers_before_air: int = 4,
        device: str = "cpu",
        verbose: int = 1,
        log_freq: int = 500,
        plot_freq: int = 5000,
        save_dir: str = None,
    ):
        if hidden is None:
            hidden = [256, 256]

        self.n_agents = n_agents
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = device
        self.verbose = verbose
        self.objectives = list(config.data.optimise_parameters)
        assert len(self.objectives) == 2

        n_obj = len(self.objectives)
        self.warmup_best: Dict[str, float] = {o: 0.0 for o in self.objectives}
        max_layers = config.data.n_layers
        self.warmup_steps = warmup_episodes_per_obj * n_obj * max_layers

        # Create environments (one per agent)
        self.envs = [
            CoatOptHybridEnv(
                config,
                materials,
                constraint_penalty,
                target_objective=self.objectives[i % n_obj],
                constraints={},
                min_layers_before_air=min_layers_before_air,
            )
            for i in range(n_agents)
        ]
        for env in self.envs:
            env.base_env.is_warmup = True

        # Share Pareto front
        primary = self.envs[0].base_env
        for env in self.envs[1:]:
            env.base_env.pareto_front_rewards = primary.pareto_front_rewards
            env.base_env.pareto_front_values = primary.pareto_front_values

        obs_dim = self.envs[0].observation_space.shape[0]
        n_materials = self.envs[0].n_materials
        min_t = self.envs[0].min_thickness
        max_t = self.envs[0].max_thickness

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
            )
            for _ in range(n_agents)
        ]
        self.buffers = [
            RolloutBuffer(n_steps, obs_dim, n_materials, gamma, gae_lambda)
            for _ in range(n_agents)
        ]

        self._obs = [None] * n_agents
        self._mask = [None] * n_agents
        self.step_count = 0
        self.episode_count = 0  # Total episodes completed (all agents)
        self.warmup_done = False
        self._base_constraints = [{} for _ in range(n_agents)]
        self._agent_targets = [self.objectives[i % n_obj] for i in range(n_agents)]

        # Monitoring (episode-based)
        self.log_freq = log_freq  # From [general] mlflow_log_freq
        self.plot_freq = plot_freq
        self.save_dir = Path(save_dir) if save_dir else None
        self._ep_rewards = []
        self._ep_vals = []
        self._ppo_logs = {}

    def _reset(self, i: int):
        obs, info = self.envs[i].reset()
        self._obs[i] = obs
        self._mask[i] = info["mask"]

    def _assign_constraints(self):
        """Assign constraint levels after warmup."""
        obj0, obj1 = self.objectives
        n, half = self.n_agents, self.n_agents // 2

        print("\nPost-warmup constraint assignments:")
        for i in range(n):
            if i < half:
                frac = i / max(half - 1, 1)
                c = {obj1: frac * self.warmup_best[obj1]} if frac > 0 else {}
                target = obj0
            else:
                j = i - half
                frac = j / max(n - half - 1, 1)
                c = {obj0: frac * self.warmup_best[obj0]} if frac > 0 else {}
                target = obj1

            self.envs[i].set_target(target, c)
            self.envs[i].base_env.is_warmup = False
            self._base_constraints[i] = c
            self._agent_targets[i] = target
            c_str = ", ".join(f"{k}≥{v:.3f}" for k, v in c.items()) or "free"
            print(f"  Agent {i:2d}: target={target:13s} constraint=[{c_str}]")

        for i in range(self.n_agents):
            self._reset(i)

    def _log_progress(self, episode: int, step: int):
        """Log to MLflow and print (works in warmup and constrained phases)."""
        # Log every log_freq episodes
        if episode % self.log_freq != 0 and episode != 0:
            return

        # Print progress
        if self.verbose:
            ent = self._ppo_logs.get("entropy", 0.0)
            n_pareto = len(self.envs[0].base_env.pareto_front_rewards)
            phase = "warmup" if not self.warmup_done else "constrained"
            print(
                f"  [{phase}] episode {episode:>6d} (step {step:>7d}) | "
                f"pareto {n_pareto:>3d} | entropy {ent:.3f}"
            )

        # Log to MLflow if active
        if not mlflow.active_run():
            if self.verbose and episode == self.log_freq:
                print("  Warning: MLflow run not active")
            return

        metrics = {"episode": episode, "step": step}
        metrics.update({f"ppo.{k}": v for k, v in self._ppo_logs.items()})

        if self._ep_rewards:
            window = self._ep_rewards[-100:]
            metrics["episode.reward_mean"] = float(np.mean(window))
            metrics["episode.reward_std"] = float(np.std(window))
            metrics["episode.reward_min"] = float(np.min(window))
            metrics["episode.reward_max"] = float(np.max(window))
            neg_frac = sum(1 for r in window if r < 0) / len(window)
            metrics["episode.negative_fraction"] = neg_frac

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

        # Plot
        if self.save_dir and episode % self.plot_freq == 0:
            try:
                _, values_df, _ = self.envs[0].base_env.export_pareto_dataframes()
                if not values_df.empty:
                    plot_pareto_front(
                        values_df,
                        self.objectives,
                        self.save_dir,
                        "vals",
                        f"ppo_multiagent_ep{episode}",
                    )
            except Exception as e:
                if self.verbose:
                    print(f"  [plot] skipped: {e}")

    def train(self, total_episodes: int) -> dict:
        for i in range(self.n_agents):
            self._reset(i)

        warmup_eps_total = (
            len(self.objectives) * self.warmup_steps // self.envs[0].base_env.max_layers
        )
        if self.verbose:
            print(
                f"Warmup for {warmup_eps_total} episodes (~{self.warmup_steps} steps)..."
            )

        while self.episode_count < total_episodes:
            # Warmup transition
            if not self.warmup_done and self.step_count >= self.warmup_steps:
                self.warmup_done = True
                print(f"\nWarmup complete at step {self.step_count}")
                print(f"Best: {self.warmup_best}")
                self._assign_constraints()

            # Collect n_steps for each agent
            for agent_idx in range(self.n_agents):
                self.buffers[agent_idx].clear()

                for _ in range(self.n_steps):
                    obs = self._obs[agent_idx]
                    mask = self._mask[agent_idx]

                    # Sample action
                    material, thickness, log_prob, value = self.agents[agent_idx].act(
                        obs, mask
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

                    # Update warmup bounds
                    if not self.warmup_done and done and "vals" in info:
                        norm = self.envs[agent_idx].base_env.compute_objective_rewards(
                            info["vals"], normalised=True
                        )
                        for obj in self.objectives:
                            self.warmup_best[obj] = max(
                                self.warmup_best[obj], norm.get(obj, 0.0)
                            )

                    # Track episode stats
                    if done and "vals" in info:
                        self._ep_rewards.append(reward)
                        self._ep_vals.append(info["vals"])
                        self.episode_count += 1  # Increment total episode counter
                        self._reset(agent_idx)
                        obs = self._obs[agent_idx]
                        mask = self._mask[agent_idx]
                    else:
                        self._obs[agent_idx] = next_obs
                        self._mask[agent_idx] = next_mask
                        obs = next_obs
                        mask = next_mask

                    self.step_count += 1

                # Finalize buffer with bootstrap value
                _, _, _, last_value = self.agents[agent_idx].act(
                    self._obs[agent_idx], self._mask[agent_idx]
                )
                self.buffers[agent_idx].finalize(last_value)

            # Update all agents
            for agent_idx in range(self.n_agents):
                rollout_data = self.buffers[agent_idx].get()
                self._ppo_logs = self.agents[agent_idx].update(
                    rollout_data, self.n_epochs, self.batch_size
                )

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
    section = "ppo_multiagent"

    def _get(key, fallback, cast=str):
        return cast(parser.get(section, key, fallback=str(fallback)))

    materials_path = parser.get("general", "materials_path")
    materials = load_materials(str(materials_path))
    config = load_config(config_path)

    seed = _get("seed", 42, int)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    n_agents = _get("n_agents", 6, int)
    total_episodes = _get("total_episodes", 10000, int)
    warmup_episodes = _get("warmup_episodes", 500, int)
    n_steps = _get("n_steps", 2048, int)
    n_epochs = _get("n_epochs", 10, int)
    batch_size = _get("batch_size", 256, int)
    constraint_penalty = _get("constraint_penalty", 3.0, float)
    lr = _get("lr", 3e-4, float)
    gamma = _get("gamma", 0.99, float)
    gae_lambda = _get("gae_lambda", 0.95, float)
    clip_range = _get("clip_range", 0.2, float)
    ent_coef = _get("ent_coef", 0.01, float)
    vf_coef = _get("vf_coef", 0.5, float)
    max_grad_norm = _get("max_grad_norm", 0.5, float)
    min_layers_before_air = _get("min_layers_before_air", 4, int)
    verbose = _get("verbose", 1, int)
    hidden_str = parser.get(section, "hidden", fallback="[256, 256]")
    hidden = eval(hidden_str)

    # Read mlflow_log_freq from [general] section
    mlflow_log_freq = parser.getint("general", "mlflow_log_freq", fallback=50)
    plot_freq = _get("plot_freq", 500, int)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config.data.n_layers = parser.getint(
        "data", "n_layers", fallback=config.data.n_layers
    )

    if save_dir is None:
        base = parser.get("general", "save_dir", fallback="./runs")
        run_name = parser.get("general", "run_name", fallback="")
        date_str = datetime.now().strftime("%Y%m%d")
        suffix = f"-{run_name}" if run_name else ""
        save_dir = Path(base) / f"{date_str}-ppo_multiagent{suffix}"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Multi-agent PPO (hybrid discrete+continuous)")
    print(f"  Agents      : {n_agents}")
    print(f"  Episodes    : {total_episodes:,}")
    print(f"  Device      : {device}")
    print(f"  Objectives  : {list(config.data.optimise_parameters)}")
    print(f"{'='*60}\n")

    trainer = MultiAgentPPO(
        config=config,
        materials=materials,
        n_agents=n_agents,
        n_steps=n_steps,
        n_epochs=n_epochs,
        batch_size=batch_size,
        warmup_episodes_per_obj=warmup_episodes,
        constraint_penalty=constraint_penalty,
        hidden=hidden,
        lr=lr,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        min_layers_before_air=min_layers_before_air,
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

    ap = argparse.ArgumentParser(description="Multi-agent PPO for CoatOpt")
    ap.add_argument("--config", required=True, help="Path to config INI")
    ap.add_argument("--save-dir", default=None, help="Override save directory")
    args = ap.parse_args()
    train(config_path=args.config, save_dir=args.save_dir)
