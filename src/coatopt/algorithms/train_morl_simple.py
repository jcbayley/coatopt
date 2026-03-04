#!/usr/bin/env python3
"""
MORLD (MOSAC population) for multi-objective optimization.

Uses MOSAC agents with weight adaptation (PSA) to explore the Pareto front.
Trains a population of agents with different weight vectors that adapt toward
underexplored regions.

Config section: [morl] or [morld]
  total_timesteps          = 500000
  seed                     = 42
  verbose                  = 1
  plot_freq                = 10000
  eval_freq                = 10000
  net_arch                 = [256, 256]
  pop_size                 = 8              # Population size
  scalarization_method     = ws             # Weighted sum (ws only, tch broken)
  weight_adaptation_method = PSA            # PSA or none
  weight_init_method       = uniform
  neighborhood_size        = 2
  shared_buffer            = true
  exchange_every           = 50000          # Weight exchange frequency
  gamma                    = 0.99
  learning_rate            = 3e-4
"""
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium.spaces import Box

from coatopt.environments.environment import CoatingEnvironment
from coatopt.utils.configs import Config, load_config
from coatopt.utils.plotting import plot_pareto_front
from coatopt.utils.utils import load_materials


# ============================================================================
# MO-GYMNASIUM WRAPPER
# ============================================================================
class CoatOptEnvSpec:
    """Minimal spec object for MORL-baselines compatibility."""

    def __init__(self, env_id: str = "CoatOpt-v0"):
        self.id = env_id
        self.name = env_id


class CoatOptMOGymWrapper(gym.Env):
    """MO-Gymnasium compatible wrapper for CoatingEnvironment."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: Config,
        materials: dict,
        consecutive_material_penalty: float = 0.2,
    ):
        super().__init__()
        self.env = CoatingEnvironment(config, materials)

        # Spec for MORL-baselines compatibility
        self.spec = CoatOptEnvSpec("CoatOpt-v0")

        # Consecutive material penalty
        self.consecutive_material_penalty = consecutive_material_penalty
        self.previous_material_idx = None

        # Multi-objective settings
        self.objectives = config.data.optimise_parameters
        self.reward_dim = len(self.objectives)

        # Observation space
        n_features = 1 + self.env.n_materials + 2
        obs_size = self.env.max_layers * n_features
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

        # Action space
        self.action_space = Box(
            low=np.array([0.0, self.env.min_thickness], dtype=np.float32),
            high=np.array(
                [float(self.env.n_materials - 1), self.env.max_thickness],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

        # MO-Gymnasium required: reward_space
        # compute_objective_rewards returns exp((raw - min) / (max - min)).
        # For raw in [min, max] this maps to [1, e].  Values outside the
        # configured objective_bounds can fall below 1, so use 0 as the safe
        # lower bound.  The upper bound is e (np.e ≈ 2.718).
        self.reward_space = Box(
            low=np.zeros(self.reward_dim, dtype=np.float32),
            high=np.full(self.reward_dim, np.e, dtype=np.float32),
            dtype=np.float32,
        )

    def _get_obs(self, state) -> np.ndarray:
        """Convert CoatingState to fixed-size numpy array."""
        tensor = state.get_observation_tensor(pre_type="lstm")
        return tensor.numpy().flatten().astype(np.float32)

    def reset(self, seed=None, options=None):
        """Reset environment."""
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        state = self.env.reset()

        # Reset material tracking
        self.previous_material_idx = None

        return self._get_obs(state), {}

    def step(self, action):
        """Take action and return vector reward."""
        # Decode action
        material_idx = int(np.clip(np.round(action[0]), 0, self.env.n_materials - 1))
        thickness = float(action[1])

        # Check for consecutive same material penalty
        consecutive_penalty = 0.0
        if (
            self.previous_material_idx is not None
            and material_idx == self.previous_material_idx
        ):
            consecutive_penalty = self.consecutive_material_penalty

        # Update previous material tracking
        self.previous_material_idx = material_idx

        # Build CoatOpt action
        coatopt_action = np.zeros(self.env.n_materials + 1, dtype=np.float32)
        coatopt_action[0] = thickness
        coatopt_action[1 + material_idx] = 1.0

        # Step environment
        state, rewards, terminated, finished, total_reward, _, vals = self.env.step(
            coatopt_action
        )

        obs = self._get_obs(state)
        done = finished
        truncated = False

        # Build info
        info = {
            "rewards": rewards,
            "vals": vals,
            "finished": finished,
            "consecutive_penalty": consecutive_penalty,
        }

        # Vector reward (MO-Gymnasium API)
        if done:
            # Final episode reward based on actual objective values
            # Get normalised rewards for all objectives
            normalised_rewards = self.env.compute_objective_rewards(
                vals, normalised=True
            )
            vec_reward = np.array(
                [normalised_rewards.get(obj, 0.0) for obj in self.objectives],
                dtype=np.float32,
            )
            # Apply consecutive penalty to all objectives
            vec_reward = vec_reward - consecutive_penalty
            info["state_array"] = state.get_array()
        else:
            # Intermediate reward: apply penalty if consecutive material used
            vec_reward = np.full(
                self.reward_dim, -consecutive_penalty, dtype=np.float32
            )

        return obs, vec_reward, done, truncated, info


def setup_morl_training(config_path: str, algorithm: str = "morld"):
    """Shared setup for all MORL algorithms.

    Args:
        config_path: Path to config INI file
        algorithm: Algorithm name (currently only "morld")

    Returns:
        Dictionary with config, env, eval_env, materials, save_dir, and algorithm params
    """
    import configparser

    config = load_config(config_path)
    parser = configparser.ConfigParser(inline_comment_prefixes=(";", "#"))
    parser.read(config_path)

    # Common parameters (try algorithm-specific section, fall back to 'morl')
    section = algorithm if parser.has_section(algorithm) else "morl"

    total_timesteps = parser.getint(section, "total_timesteps")
    seed = parser.getint(section, "seed", fallback=42)
    verbose = parser.getint(section, "verbose", fallback=1)
    plot_freq = parser.getint(section, "plot_freq", fallback=10000)
    eval_freq = parser.getint(section, "eval_freq", fallback=10000)
    net_arch = eval(parser.get(section, "net_arch", fallback="[256, 256]"))

    # Directories
    save_dir = Path(parser.get("general", "save_dir"))
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "plots").mkdir(exist_ok=True)

    # Materials
    materials_path = parser.get("general", "materials_path")
    if materials_path is None:
        materials_path = Path(__file__).parent / "config" / "materials.json"
    materials = load_materials(str(materials_path))

    # Create environments
    env = CoatOptMOGymWrapper(config, materials)
    eval_env = CoatOptMOGymWrapper(config, materials)

    print(
        f"\nEnvironment: obs={env.observation_space.shape}, "
        f"action={env.action_space}, reward_dim={env.reward_dim}"
    )

    return {
        "config": config,
        "parser": parser,
        "section": section,
        "env": env,
        "eval_env": eval_env,
        "materials": materials,
        "save_dir": save_dir,
        "config_path": config_path,
        "total_timesteps": total_timesteps,
        "seed": seed,
        "verbose": verbose,
        "plot_freq": plot_freq,
        "eval_freq": eval_freq,
        "net_arch": net_arch,
    }


def create_morl_agent(algorithm: str, setup_dict: dict):
    """Factory to create MORL algorithm agent.

    Args:
        algorithm: "morld"
        setup_dict: Dictionary from setup_morl_training()

    Returns:
        Initialized agent
    """
    env = setup_dict["env"]
    parser = setup_dict["parser"]
    section = setup_dict["section"]
    seed = setup_dict["seed"]
    net_arch = setup_dict["net_arch"]

    if algorithm == "morld":
        from morl_baselines.multi_policy.morld.morld import MORLD

        pop_size = parser.getint(section, "pop_size", fallback=8)
        # NOTE: "tch" (Tchebycheff) is broken in morl-baselines for MOSAC — it
        # receives batched PyTorch tensors but expects numpy scalars in the
        # reference-point loop, causing "Boolean value of Tensor is ambiguous".
        # Use "ws" (weighted sum) only. Non-convex Pareto coverage is instead
        # achieved via weight_adaptation_method="PSA" + larger pop_size.
        scalarization = parser.get(section, "scalarization_method", fallback="ws")
        # "PSA" shifts weights toward underexplored Pareto regions during training.
        weight_adaptation = parser.get(
            section, "weight_adaptation_method", fallback="PSA"
        )
        weight_init = parser.get(section, "weight_init_method", fallback="uniform")
        neighborhood_size = parser.getint(section, "neighborhood_size", fallback=2)
        shared_buffer = parser.getboolean(section, "shared_buffer", fallback=True)
        exchange_every = parser.getint(
            section,
            "exchange_every",
            fallback=setup_dict["total_timesteps"] // 10,
        )
        gamma = parser.getfloat(section, "gamma", fallback=0.99)
        learning_rate = parser.getfloat(section, "learning_rate", fallback=3e-4)

        policy_args = {"net_arch": net_arch}
        if learning_rate != 3e-4:
            policy_args["learning_rate"] = learning_rate

        return MORLD(
            env=env,
            scalarization_method=scalarization,
            evaluation_mode="ser",
            policy_name="MOSAC",
            gamma=gamma,
            pop_size=pop_size,
            seed=seed,
            exchange_every=exchange_every,
            neighborhood_size=neighborhood_size,
            shared_buffer=shared_buffer,
            weight_init_method=weight_init,
            weight_adaptation_method=(
                weight_adaptation if weight_adaptation != "none" else None
            ),
            log=False,
            device="auto",
            policy_args=policy_args,
        )

    else:
        raise ValueError(
            f"Unknown algorithm: '{algorithm}'. Supported: morld\n"
            "Note: pgmorl and moppo are excluded — they require vectorised envs "
            "and bypass the CoatingEnvironment, so designs/values cannot be tracked."
        )


def run_morl_training_loop(agent, setup_dict: dict, algorithm: str):
    """Shared training loop for all MORL algorithms.

    Args:
        agent: Initialized MORL agent
        setup_dict: Dictionary from setup_morl_training()
        algorithm: Algorithm name for logging
    """
    import pandas as pd

    env = setup_dict["env"]
    eval_env = setup_dict["eval_env"]
    save_dir = setup_dict["save_dir"]
    total_timesteps = setup_dict["total_timesteps"]
    objectives = list(env.objectives)
    reward_dim = env.reward_dim
    ref_point = np.zeros(reward_dim, dtype=np.float32)

    print(f"\n{'='*60}")
    print(f"  Algorithm  : {algorithm.upper()}")
    print(f"  Objectives : {objectives}")
    print(f"  Reward dim : {reward_dim}")
    print(f"  Timesteps  : {total_timesteps:,}")
    print(f"  Save dir   : {save_dir}")
    print(f"{'='*60}")
    print("Starting training — morl-baselines handles the inner loop.\n")

    start_time = time.time()

    agent.train(
        total_timesteps=total_timesteps,
        eval_env=eval_env,
        ref_point=ref_point,
        known_pareto_front=None,
    )

    end_time = time.time()
    duration_min = (end_time - start_time) / 60

    designs_df, values_df, rewards_df = env.env.export_pareto_dataframes()

    if not rewards_df.empty:
        print(f"\nPareto front: {len(rewards_df)} solutions")
        for obj in objectives:
            col = values_df[obj]
            print(
                f"  {obj}: min={col.min():.4g}  max={col.max():.4g}  mean={col.mean():.4g}"
            )
        try:
            plot_path = plot_pareto_front(
                df=values_df,
                objectives=objectives,
                save_dir=save_dir,
                plot_type="vals",
                algorithm_name=algorithm,
            )
            print(f"Saved plot → {plot_path}")
        except Exception as e:
            print(f"Warning: Failed to plot: {e}")
    else:
        print("\nWarning: No Pareto solutions found.")

    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  Duration       : {duration_min:.1f} min")
    print(f"  Pareto front   : {len(rewards_df)} solutions")
    print(f"{'='*60}")

    # Saving is handled by run.py via save_training_results — do not duplicate here.
    return {
        "pareto_designs": designs_df,
        "pareto_values": values_df,
        "pareto_rewards": rewards_df,
        "model": None,
        "metadata": {
            "total_timesteps": total_timesteps,
            "seed": setup_dict["seed"],
        },
    }


def train(config_path: str, algorithm: str = "morld", save_dir: str = None):
    """Unified training function for MORL algorithms that work with a single env.

    Args:
        config_path: Path to config INI file
        algorithm: Algorithm to use — currently only "morld" (MOSAC population).
            pgmorl and moppo are not supported: they require vectorised envs and
            bypass the CoatingEnvironment, so designs/values cannot be tracked.
        save_dir: Optional override for save directory

    Returns:
        Results dict with pareto_designs, pareto_values, pareto_rewards, model, metadata
    """
    setup = setup_morl_training(config_path, algorithm)
    if save_dir:
        setup["save_dir"] = Path(save_dir)
        setup["save_dir"].mkdir(parents=True, exist_ok=True)

    agent = create_morl_agent(algorithm, setup)
    return run_morl_training_loop(agent, setup, algorithm)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train MORL on CoatOpt")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config INI file",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="morld",
        choices=["morld"],
        help="MORL algorithm to use",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Override save directory from config",
    )

    args = parser.parse_args()

    # Use unified training interface
    agent, pareto_df = train(
        config_path=args.config,
        algorithm=args.algorithm,
        save_dir=args.save_dir,
    )
