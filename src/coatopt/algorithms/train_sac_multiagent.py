#!/usr/bin/env python3
"""
Multi-agent Discrete SAC with Q-value action masking for CoatOpt.

N SAC agents each hold a fixed constraint level, together spanning the Pareto
front (like MOEA/D but with constraint thresholds instead of weight vectors).
All agents share one replay buffer — SAC is off-policy so this is valid and
gives free cross-agent data sharing.

Action masking: invalid actions get -inf logits in the actor so the sampled
distribution is always valid. No env-level masking needed.

Observation includes constraint thresholds so the shared Q-functions can
distinguish which regime each transition came from.

Config section: [sac_multiagent]
  n_agents              = 6
  total_timesteps       = 300000
  warmup_episodes       = 100        # per objective
  n_thickness_bins      = 20
  buffer_size           = 100000
  batch_size            = 256
  train_every           = 1          # gradient steps per env step
  constraint_penalty    = 10.0
  lr                    = 3e-4
  gamma                 = 0.99
  tau                   = 0.005
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
from coatopt.utils.plotting import plot_pareto_front
from coatopt.utils.utils import load_materials

# ─────────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────────


class CoatOptFlatEnv(gym.Env):
    """
    Flat Discrete(n_materials * n_thickness_bins) wrapper with a fixed constraint.
    No objective cycling — each agent gets one target and one constraint level.
    Action mask is returned in info; masking is enforced in the actor, not here.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: Config,
        materials: dict,
        n_thickness_bins: int = 20,
        constraint_penalty: float = 10.0,
        target_objective: str = "reflectivity",
        constraints: Dict[str, float] = None,
        min_layers_before_air: int = 4,
    ):
        super().__init__()
        self.base_env = CoatingEnvironment(config, materials)
        self.n_thickness_bins = n_thickness_bins
        self.target_objective = target_objective
        self.constraints = constraints or {}
        self.min_layers_before_air = min_layers_before_air
        self.objectives = list(self.base_env.optimise_parameters)

        # Enable constrained-mode reward (penalty applied by CoatingEnvironment.step)
        self.base_env.use_constrained_training = True
        self.base_env.is_warmup = True  # overridden by trainer
        self.base_env.constraint_penalty = constraint_penalty
        self.base_env.target_objective = target_objective
        self.base_env.constraints = self.constraints

        self.thickness_bins = np.linspace(
            self.base_env.min_thickness, self.base_env.max_thickness, n_thickness_bins
        )
        self.n_materials = self.base_env.n_materials
        self.n_actions = self.n_materials * n_thickness_bins
        self.action_space = gym.spaces.Discrete(self.n_actions)

        n_features = 1 + self.n_materials + 2  # thickness + one-hot mat + n + k
        obs_size = self.base_env.max_layers * n_features + 1 + len(self.objectives)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

        self.current_layer = 0
        self.prev_material = None

    def _decode(self, action: int) -> Tuple[int, int]:
        return action // self.n_thickness_bins, action % self.n_thickness_bins

    def get_action_mask(self) -> np.ndarray:
        mask = np.ones(self.n_actions, dtype=bool)
        for m in range(self.n_materials):
            s, e = m * self.n_thickness_bins, (m + 1) * self.n_thickness_bins
            if self.prev_material is not None and m == self.prev_material:
                mask[s:e] = False
            if (
                m == self.base_env.air_material_index
                and self.current_layer < self.min_layers_before_air
            ):
                mask[s:e] = False
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
        mask = self.get_action_mask()
        return self._obs(state), {"mask": mask}

    def step(self, action: int):
        mat_idx, t_bin = self._decode(action)
        thickness = self.thickness_bins[t_bin]
        self.prev_material = mat_idx
        self.current_layer += 1

        coatopt_action = np.zeros(self.n_materials + 1, dtype=np.float32)
        coatopt_action[0] = thickness
        coatopt_action[1 + mat_idx] = 1.0

        state, rewards, _, finished, env_reward, _, vals = self.base_env.step(
            coatopt_action
        )
        obs = self._obs(state)
        mask = self.get_action_mask()
        # Use env_reward as-is (respects use_intermediate_reward config)
        reward = float(env_reward)
        info = {"mask": mask}
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
# Replay buffer  (shared across all agents)
# ─────────────────────────────────────────────────────────────────────────────


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, n_actions: int, device):
        self.capacity = capacity
        self.device = device
        self.pos = self.size = 0
        self.obs = np.zeros((capacity, obs_dim), np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), np.float32)
        self.actions = np.zeros(capacity, np.int64)
        self.rewards = np.zeros(capacity, np.float32)
        self.dones = np.zeros(capacity, np.float32)
        self.masks = np.ones((capacity, n_actions), bool)
        self.next_masks = np.ones((capacity, n_actions), bool)

    def push(self, obs, action, reward, next_obs, done, mask, next_mask):
        i = self.pos
        self.obs[i] = obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_obs[i] = next_obs
        self.dones[i] = done
        self.masks[i] = mask
        self.next_masks[i] = next_mask
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        d = self.device
        return (
            torch.FloatTensor(self.obs[idx]).to(d),
            torch.LongTensor(self.actions[idx]).to(d),
            torch.FloatTensor(self.rewards[idx]).to(d),
            torch.FloatTensor(self.next_obs[idx]).to(d),
            torch.FloatTensor(self.dones[idx]).to(d),
            torch.BoolTensor(self.masks[idx]).to(d),
            torch.BoolTensor(self.next_masks[idx]).to(d),
        )

    def __len__(self):
        return self.size


# ─────────────────────────────────────────────────────────────────────────────
# Networks
# ─────────────────────────────────────────────────────────────────────────────


def _mlp(in_dim: int, out_dim: int, hidden: List[int]) -> nn.Sequential:
    layers, d = [], in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: List[int]):
        super().__init__()
        self.net = _mlp(obs_dim, n_actions, hidden)

    def probs_and_log_probs(self, obs: torch.Tensor, mask: torch.Tensor):
        logits = self.net(obs).masked_fill(~mask, float("-inf"))
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        return probs, log_probs

    def sample(self, obs: torch.Tensor, mask: torch.Tensor):
        probs, log_probs = self.probs_and_log_probs(obs, mask)
        action = torch.distributions.Categorical(probs).sample()
        log_prob = log_probs.gather(1, action.unsqueeze(1)).squeeze(1)
        return action, log_prob


class Critic(nn.Module):
    """Outputs Q-value for every action given obs (no action input required)."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: List[int]):
        super().__init__()
        self.net = _mlp(obs_dim, n_actions, hidden)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


# ─────────────────────────────────────────────────────────────────────────────
# SAC agent
# ─────────────────────────────────────────────────────────────────────────────


class SACAgent:
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden: List[int],
        lr: float,
        gamma: float,
        tau: float,
        device: str,
    ):
        self.gamma = gamma
        self.tau = tau
        self.device = device

        self.actor = Actor(obs_dim, n_actions, hidden).to(device)
        self.c1 = Critic(obs_dim, n_actions, hidden).to(device)
        self.c2 = Critic(obs_dim, n_actions, hidden).to(device)
        self.tc1 = Critic(obs_dim, n_actions, hidden).to(device)
        self.tc2 = Critic(obs_dim, n_actions, hidden).to(device)
        self.tc1.load_state_dict(self.c1.state_dict())
        self.tc2.load_state_dict(self.c2.state_dict())

        self.a_opt = Adam(self.actor.parameters(), lr=lr)
        self.c1_opt = Adam(self.c1.parameters(), lr=lr)
        self.c2_opt = Adam(self.c2.parameters(), lr=lr)

        # Auto-tune temperature: target ~98% of max entropy
        self.target_entropy = -np.log(1.0 / n_actions) * 0.98
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha = self.log_alpha.exp().item()
        self.alpha_opt = Adam([self.log_alpha], lr=lr)

    @torch.no_grad()
    def act(self, obs: np.ndarray, mask: np.ndarray) -> int:
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        mask_t = torch.BoolTensor(mask).unsqueeze(0).to(self.device)
        probs, _ = self.actor.probs_and_log_probs(obs_t, mask_t)
        return torch.distributions.Categorical(probs).sample().item()

    def update(self, batch) -> Dict[str, float]:
        obs, acts, rews, next_obs, dones, masks, next_masks = batch

        with torch.no_grad():
            next_probs, next_log_probs = self.actor.probs_and_log_probs(
                next_obs, next_masks
            )
            next_q = torch.min(self.tc1(next_obs), self.tc2(next_obs))
            # V(s') = E_pi[Q(s',a') - alpha * log pi(a'|s')]
            next_v = (next_probs * (next_q - self.alpha * next_log_probs)).sum(-1)
            target = rews + (1.0 - dones) * self.gamma * next_v

        # Critic update
        q1 = self.c1(obs).gather(1, acts.unsqueeze(1)).squeeze(1)
        q2 = self.c2(obs).gather(1, acts.unsqueeze(1)).squeeze(1)
        c1_loss = F.mse_loss(q1, target)
        c2_loss = F.mse_loss(q2, target)
        for opt, loss in [(self.c1_opt, c1_loss), (self.c2_opt, c2_loss)]:
            opt.zero_grad()
            loss.backward()
            opt.step()

        # Actor update
        probs, log_probs = self.actor.probs_and_log_probs(obs, masks)
        with torch.no_grad():
            q_min = torch.min(self.c1(obs), self.c2(obs))
        actor_loss = (probs * (self.alpha * log_probs - q_min)).sum(-1).mean()
        self.a_opt.zero_grad()
        actor_loss.backward()
        self.a_opt.step()

        # Temperature update - only compute entropy over valid (unmasked) actions
        valid_mask = masks.float()
        entropy = -(probs.detach() * log_probs.detach() * valid_mask).sum(-1).mean()
        alpha_loss = -(self.log_alpha * (self.target_entropy - entropy)).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()
        self.alpha = self.log_alpha.exp().item()

        # Soft target update
        for p, tp in [
            *zip(self.c1.parameters(), self.tc1.parameters()),
            *zip(self.c2.parameters(), self.tc2.parameters()),
        ]:
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

        return {
            "c1_loss": c1_loss.item(),
            "c2_loss": c2_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha": self.alpha,
            "entropy": entropy.item(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Multi-agent trainer
# ─────────────────────────────────────────────────────────────────────────────


class MultiAgentSAC:
    """
    N SAC agents sharing one replay buffer, each targeting a fixed Pareto point.

    Warmup: all agents run freely (no constraints) to discover per-objective
    reward bounds.  Then agents are assigned constraint levels that evenly span
    the Pareto front from one extreme to the other.
    """

    def __init__(
        self,
        config: Config,
        materials: dict,
        n_agents: int = 6,
        n_thickness_bins: int = 20,
        buffer_size: int = 100_000,
        batch_size: int = 256,
        warmup_episodes_per_obj: int = 100,
        train_every: int = 1,
        constraint_penalty: float = 10.0,
        hidden: List[int] = None,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        min_layers_before_air: int = 4,
        device: str = "cpu",
        verbose: int = 1,
        # Exploration
        explore_eps: float = 0.2,  # random-action prob during warmup
        constraint_noise: float = 0.05,  # Gaussian std on constraint thresholds (as fraction of value)
        clear_buffer_on_warmup_end: bool = True,
        # Monitoring
        log_freq: int = 500,  # MLflow log every N env steps
        plot_freq: int = 5000,  # Save Pareto plot every N env steps
        save_dir: str = None,
    ):
        if hidden is None:
            hidden = [256, 256]

        self.n_agents = n_agents
        self.batch_size = batch_size
        self.device = device
        self.train_every = train_every
        self.verbose = verbose
        self.objectives = list(config.data.optimise_parameters)
        assert len(self.objectives) == 2, "Currently supports exactly 2 objectives"

        n_obj = len(self.objectives)
        self.warmup_best: Dict[str, float] = {o: 0.0 for o in self.objectives}
        # Warmup ends after each objective has had warmup_episodes_per_obj episodes
        # per agent slot assigned to it; estimate in env steps
        max_layers = config.data.n_layers
        self.warmup_steps = warmup_episodes_per_obj * n_obj * max_layers

        # All envs start with warmup mode, alternating objectives for breadth
        self.envs = [
            CoatOptFlatEnv(
                config,
                materials,
                n_thickness_bins,
                constraint_penalty,
                target_objective=self.objectives[i % n_obj],
                constraints={},
                min_layers_before_air=min_layers_before_air,
            )
            for i in range(n_agents)
        ]
        for env in self.envs:
            env.base_env.is_warmup = True

        # Share a single Pareto archive across all envs.
        # All CoatingEnvironment.update_pareto_front() calls will mutate the
        # same list objects, so there is always exactly one globally
        # non-dominated front regardless of which agent found each solution.
        primary = self.envs[0].base_env
        for env in self.envs[1:]:
            env.base_env.pareto_front_rewards = primary.pareto_front_rewards
            env.base_env.pareto_front_values = primary.pareto_front_values

        obs_dim = self.envs[0].observation_space.shape[0]
        n_actions = self.envs[0].n_actions

        # One shared replay buffer
        self.buffer = ReplayBuffer(buffer_size, obs_dim, n_actions, device)

        # One SAC agent per env (independent networks, shared buffer)
        self.agents = [
            SACAgent(obs_dim, n_actions, hidden, lr, gamma, tau, device)
            for _ in range(n_agents)
        ]

        self._obs: List = [None] * n_agents
        self._mask: List = [None] * n_agents
        self.step_count = 0
        self.warmup_done = False

        # Exploration
        self.explore_eps = explore_eps
        self.constraint_noise = constraint_noise
        self.clear_buffer_on_warmup_end = clear_buffer_on_warmup_end
        # Base constraints per agent (set by _assign_constraints, used for jitter)
        self._base_constraints: List[Dict[str, float]] = [{} for _ in range(n_agents)]
        self._agent_targets: List[str] = [
            self.objectives[i % len(self.objectives)] for i in range(n_agents)
        ]

        # Monitoring
        self.log_freq = log_freq
        self.plot_freq = plot_freq
        self.save_dir = Path(save_dir) if save_dir else None
        # Rolling episode stats (across all agents)
        self._ep_rewards: List[float] = []
        self._ep_vals: List[Dict] = []
        self._sac_logs: Dict[str, float] = {}

    def _reset(self, i: int):
        obs, info = self.envs[i].reset()
        self._obs[i] = obs
        self._mask[i] = info["mask"]

    def _assign_constraints(self):
        """After warmup: spread agents across Pareto front via constraint fractions.

        First half targets obj0 with increasing constraint on obj1.
        Second half targets obj1 with increasing constraint on obj0.
        Together they cover both arms of the trade-off curve.
        """
        obj0, obj1 = self.objectives[0], self.objectives[1]
        n = self.n_agents
        half = n // 2

        print("\nPost-warmup constraint assignments:")
        for i in range(n):
            if i < half:
                # target obj0, increasing constraint on obj1
                frac = i / max(half - 1, 1)
                c = {obj1: frac * self.warmup_best[obj1]} if frac > 0 else {}
                target = obj0
            else:
                # target obj1, increasing constraint on obj0
                j = i - half
                frac = j / max(n - half - 1, 1)
                c = {obj0: frac * self.warmup_best[obj0]} if frac > 0 else {}
                target = obj1

            self.envs[i].set_target(target, c)
            self.envs[i].base_env.is_warmup = False
            self._base_constraints[i] = c
            self._agent_targets[i] = target
            c_str = ", ".join(f"{k}≥{v:.3f}" for k, v in c.items()) or "free"
            print(f"  Agent {i:2d}: target={target:13s}  constraint=[{c_str}]")

        if self.clear_buffer_on_warmup_end:
            self.buffer.pos = 0
            self.buffer.size = 0
            print("  Replay buffer cleared (warmup transitions discarded).")

        for i in range(self.n_agents):
            self._reset(i)

    def _act(self, i: int, obs: np.ndarray, mask: np.ndarray) -> int:
        """Epsilon-greedy during warmup; SAC policy otherwise."""
        if not self.warmup_done and np.random.random() < self.explore_eps:
            valid = np.where(mask)[0]
            return int(np.random.choice(valid))
        return self.agents[i].act(obs, mask)

    def _apply_constraint_noise(self, i: int):
        """Jitter this agent's constraint thresholds slightly each episode."""
        if not self.warmup_done and self.constraint_noise > 0:
            return
        base = self._base_constraints[i]
        if not base or self.constraint_noise <= 0:
            return
        noisy = {
            k: float(
                np.clip(v * (1.0 + np.random.randn() * self.constraint_noise), 0.0, 1.0)
            )
            for k, v in base.items()
        }
        self.envs[i].set_target(self._agent_targets[i], noisy)

    def _step_env(self, i: int):
        obs, mask = self._obs[i], self._mask[i]
        action = self._act(i, obs, mask)
        next_obs, reward, done, _, info = self.envs[i].step(action)
        next_mask = info["mask"]

        self.buffer.push(obs, action, reward, next_obs, float(done), mask, next_mask)

        if done:
            if not self.warmup_done and "vals" in info:
                norm = self.envs[i].base_env.compute_objective_rewards(
                    info["vals"], normalised=True
                )
                for obj in self.objectives:
                    self.warmup_best[obj] = max(
                        self.warmup_best[obj], norm.get(obj, 0.0)
                    )
            # Accumulate episode stats for logging
            if "vals" in info:
                self._ep_rewards.append(reward)
                self._ep_vals.append(info["vals"])
            self._apply_constraint_noise(i)
            self._reset(i)
        else:
            self._obs[i] = next_obs
            self._mask[i] = next_mask

        self.step_count += 1

    def _log_progress(self, step: int, total_timesteps: int):
        """Log metrics to MLflow and save Pareto plot periodically."""
        primary_env = self.envs[0].base_env

        # ── MLflow metrics ──────────────────────────────────────────────────
        if mlflow.active_run() and step % self.log_freq == 0:
            metrics: Dict[str, float] = {"step": step}

            # SAC training stats (from last batch update)
            metrics.update({f"sac/{k}": v for k, v in self._sac_logs.items()})

            # Episode reward stats over recent window
            if self._ep_rewards:
                window = self._ep_rewards[-100:]
                metrics["episode/reward_mean"] = float(np.mean(window))
                metrics["episode/reward_std"] = float(np.std(window))
                metrics["episode/reward_min"] = float(np.min(window))
                metrics["episode/reward_max"] = float(np.max(window))

                # % of recent episodes with negative reward (constraint violations)
                negative_frac = sum(1 for r in window if r < 0) / len(window)
                metrics["episode/negative_fraction"] = negative_frac

            # Per-objective value stats (physical units)
            if self._ep_vals:
                window_vals = self._ep_vals[-100:]
                for obj in self.objectives:
                    vals = [v.get(obj, float("nan")) for v in window_vals]
                    vals = [v for v in vals if not np.isnan(v)]
                    if vals:
                        metrics[f"vals/{obj}_mean"] = float(np.mean(vals))
                        metrics[f"vals/{obj}_best"] = float(
                            np.min(vals) if obj == "absorption" else np.max(vals)
                        )

            # Pareto front size + hypervolume
            pareto = primary_env.get_pareto_front(space="reward")
            metrics["pareto/size"] = len(pareto)
            if len(pareto) > 1:
                try:
                    hv = primary_env.compute_hypervolume(space="reward")
                    metrics["pareto/hypervolume"] = hv
                except Exception:
                    pass

            # Warmup best bounds (useful to see they're sensible)
            for obj, best in self.warmup_best.items():
                metrics[f"warmup_best/{obj}"] = best

            mlflow.log_metrics(metrics, step=step)

        # ── Pareto plot ──────────────────────────────────────────────────────
        if self.save_dir and step % self.plot_freq == 0:
            try:
                _, values_df, _ = primary_env.export_pareto_dataframes()
                if not values_df.empty:
                    plot_pareto_front(
                        df=values_df,
                        objectives=self.objectives,
                        save_dir=self.save_dir,
                        plot_type="vals",
                        algorithm_name=f"sac_multiagent_step{step}",
                    )
            except Exception as e:
                if self.verbose:
                    print(f"  [plot] skipped: {e}")

    def train(self, total_timesteps: int) -> dict:
        for i in range(self.n_agents):
            self._reset(i)

        if self.verbose:
            print(f"Warmup for ~{self.warmup_steps} steps ({self.n_agents} agents)...")

        log_interval = max(1, total_timesteps // 20)

        while self.step_count < total_timesteps:
            # Warmup → constrained transition
            if not self.warmup_done and self.step_count >= self.warmup_steps:
                self.warmup_done = True
                print(f"\nWarmup complete at step {self.step_count}.")
                print(
                    f"Best per objective: { {k: f'{v:.4f}' for k, v in self.warmup_best.items()} }"
                )
                self._assign_constraints()

            # Collect one step from each agent
            for i in range(self.n_agents):
                self._step_env(i)

            # Train all agents from the shared buffer.
            # Each agent draws its own independent sample so they benefit from
            # each other's collected data without all repeating the same update.
            if len(self.buffer) >= self.batch_size:
                for _ in range(self.train_every):
                    for agent in self.agents:
                        self._sac_logs = agent.update(
                            self.buffer.sample(self.batch_size)
                        )

            # Logging + plotting
            self._log_progress(self.step_count, total_timesteps)

            if self.verbose and self.step_count % log_interval == 0:
                ent = self._sac_logs.get("entropy", float("nan"))
                alph = self._sac_logs.get("alpha", float("nan"))
                n_pareto = len(self.envs[0].base_env.pareto_front_rewards)
                print(
                    f"  step {self.step_count:>7d}/{total_timesteps}"
                    f" | buffer {len(self.buffer):>6d}"
                    f" | pareto {n_pareto:>3d}"
                    f" | entropy {ent:.3f}"
                    f" | alpha {alph:.4f}"
                )

        # All envs share the same Pareto lists — export from primary env only.
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
                "total_timesteps": total_timesteps,
                "warmup_best": self.warmup_best,
            },
        }


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def train(config_path: str, save_dir: str = None) -> dict:
    """Train multi-agent discrete SAC on CoatOpt.

    Args:
        config_path: Path to config INI file (needs [sac_multiagent] section)
        save_dir:    Override for the output directory

    Returns:
        Results dict compatible with save_training_results()
    """
    parser = configparser.ConfigParser()
    parser.read(config_path)
    section = "sac_multiagent"

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
    total_timesteps = _get("total_timesteps", 300_000, int)
    warmup_episodes = _get("warmup_episodes", 100, int)
    n_thickness_bins = _get("n_thickness_bins", 20, int)
    buffer_size = _get("buffer_size", 100_000, int)
    batch_size = _get("batch_size", 256, int)
    train_every = _get("train_every", 1, int)
    constraint_penalty = _get("constraint_penalty", 10.0, float)
    lr = _get("lr", 3e-4, float)
    gamma = _get("gamma", 0.99, float)
    tau = _get("tau", 0.005, float)
    min_layers_before_air = _get("min_layers_before_air", 4, int)
    verbose = _get("verbose", 1, int)
    hidden_str = parser.get(section, "hidden", fallback="[256, 256]")
    hidden = eval(hidden_str)
    explore_eps = _get("explore_eps", 0.2, float)
    constraint_noise = _get("constraint_noise", 0.05, float)
    clear_buffer = parser.getboolean(
        section, "clear_buffer_on_warmup_end", fallback=True
    )
    log_freq = _get("log_freq", 500, int)
    plot_freq = _get("plot_freq", 5000, int)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config.data.n_layers = parser.getint(
        "data", "n_layers", fallback=config.data.n_layers
    )

    if save_dir is None:
        base = parser.get("general", "save_dir", fallback="./runs")
        run_name = parser.get("general", "run_name", fallback="")
        date_str = datetime.now().strftime("%Y%m%d")
        suffix = f"-{run_name}" if run_name else ""
        save_dir = Path(base) / f"{date_str}-sac_multiagent{suffix}"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("  Multi-agent Discrete SAC")
    print(f"  Agents          : {n_agents}")
    print(f"  Timesteps       : {total_timesteps:,}")
    print(f"  Buffer size     : {buffer_size:,}")
    print(f"  Device          : {device}")
    print(f"  Objectives      : {list(config.data.optimise_parameters)}")
    print(f"{'='*60}\n")

    trainer = MultiAgentSAC(
        config=config,
        materials=materials,
        n_agents=n_agents,
        n_thickness_bins=n_thickness_bins,
        buffer_size=buffer_size,
        batch_size=batch_size,
        warmup_episodes_per_obj=warmup_episodes,
        train_every=train_every,
        constraint_penalty=constraint_penalty,
        hidden=hidden,
        lr=lr,
        gamma=gamma,
        tau=tau,
        min_layers_before_air=min_layers_before_air,
        device=device,
        verbose=verbose,
        explore_eps=explore_eps,
        constraint_noise=constraint_noise,
        clear_buffer_on_warmup_end=clear_buffer,
        log_freq=log_freq,
        plot_freq=plot_freq,
        save_dir=str(save_dir),
    )

    start = time.time()
    results = trainer.train(total_timesteps)
    results["metadata"]["training_time_s"] = time.time() - start

    print(f"\nTraining done in {results['metadata']['training_time_s']/60:.1f} min")
    n_pareto = len(results["pareto_rewards"])
    print(f"Combined Pareto front: {n_pareto} solutions")

    return results


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Multi-agent Discrete SAC for CoatOpt")
    ap.add_argument("--config", required=True, help="Path to config INI")
    ap.add_argument("--save-dir", default=None, help="Override save directory")
    args = ap.parse_args()

    train(config_path=args.config, save_dir=args.save_dir)
