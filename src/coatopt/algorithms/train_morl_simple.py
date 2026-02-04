#!/usr/bin/env python3
import json
import time
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gymnasium.spaces import Box

from coatopt.environments.environment import CoatingEnvironment
from coatopt.utils.configs import Config, DataConfig, TrainingConfig, load_config
from coatopt.utils.plotting import plot_pareto_front
from coatopt.utils.utils import load_materials, save_run_metadata


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
        # Rewards are normalised to roughly [0, 1] for each objective
        self.reward_space = Box(
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
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
        algorithm: Algorithm name ("morld", "pgmorl", "moppo")

    Returns:
        Dictionary with config, env, eval_env, materials, save_dir, and algorithm params
    """
    import configparser

    config = load_config(config_path)
    parser = configparser.ConfigParser()
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
        algorithm: "morld", "pgmorl", or "moppo"
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

        pop_size = parser.getint(section, "pop_size", fallback=6)

        return MORLD(
            env=env,
            scalarization_method="ws",
            evaluation_mode="ser",
            policy_name="MOSAC",
            gamma=0.99,
            pop_size=pop_size,
            seed=seed,
            exchange_every=setup_dict["total_timesteps"] // 10,
            neighborhood_size=1,
            shared_buffer=True,
            weight_init_method="uniform",
            weight_adaptation_method=None,
            log=False,
            device="auto",
            policy_args={"net_arch": net_arch},
        )

    elif algorithm == "pgmorl":
        from morl_baselines.multi_policy.pgmorl.pgmorl import PGMORL

        pop_size = parser.getint(section, "pop_size", fallback=6)
        warmup_iterations = parser.getint(section, "warmup_iterations", fallback=80)
        evolutionary_iterations = parser.getint(
            section, "evolutionary_iterations", fallback=20
        )
        num_weight_candidates = parser.getint(
            section, "num_weight_candidates", fallback=7
        )

        return PGMORL(
            env_id=env.spec.id,
            origin=np.array([0.0, 0.0]),
            num_envs=1,
            pop_size=pop_size,
            warmup_iterations=warmup_iterations,
            evolutionary_iterations=evolutionary_iterations,
            num_weight_candidates=num_weight_candidates,
            gamma=0.99,
            seed=seed,
            log=False,
            project_name="coatopt-pgmorl",
            experiment_name=setup_dict["save_dir"].name,
            policy_args={"net_arch": net_arch},
        )

    elif algorithm == "moppo":
        from morl_baselines.single_policy.ser.mo_ppo import MOPPO

        return MOPPO(
            id=0,
            env=env,
            gamma=0.99,
            seed=seed,
            log=False,
            policy_args={"net_arch": net_arch},
        )

    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def run_morl_training_loop(agent, setup_dict: dict, algorithm: str):
    """Shared training loop for all MORL algorithms.

    Args:
        agent: Initialized MORL agent
        setup_dict: Dictionary from setup_morl_training()
        algorithm: Algorithm name for logging
    """
    import pandas as pd

    eval_env = setup_dict["eval_env"]
    save_dir = setup_dict["save_dir"]
    total_timesteps = setup_dict["total_timesteps"]

    ref_point = np.array([0.0, 0.0])
    start_time = time.time()

    print(f"\n{'='*60}")
    print(f"Training {algorithm.upper()}")
    print(f"{'='*60}")

    # Single training call - morl-baselines handles pareto tracking internally
    agent.train(
        total_timesteps=total_timesteps,
        eval_env=eval_env,
        ref_point=ref_point,
        known_pareto_front=None,
    )

    end_time = time.time()
    duration_min = (end_time - start_time) / 60

    # Extract Pareto front from agent's archive
    pareto_solutions = []
    if hasattr(agent, "archive") and hasattr(agent.archive, "individuals"):
        # Multi-policy algorithms (MORLD, PGMORL)
        for individual in agent.archive.individuals:
            pareto_solutions.append(
                {
                    "reflectivity": float(individual.reward_vector[0]),
                    "absorption": float(individual.reward_vector[1]),
                }
            )
    elif hasattr(agent, "np_archive"):
        # Alternative archive structure
        for sol in agent.np_archive:
            pareto_solutions.append(
                {
                    "reflectivity": float(sol[0]),
                    "absorption": float(sol[1]),
                }
            )

    # Save Pareto front to CSV
    if pareto_solutions:
        pareto_df = pd.DataFrame(pareto_solutions)
        pareto_csv = save_dir / "pareto_front.csv"
        pareto_df.to_csv(pareto_csv, index=False)
        print(f"\nSaved Pareto front ({len(pareto_df)} solutions) to {pareto_csv}")

        # Plot Pareto front using shared utility
        try:
            objectives = list(pareto_df.columns)  # ['reflectivity', 'absorption']
            plot_path = plot_pareto_front(
                df=pareto_df,
                objectives=objectives,
                save_dir=save_dir,
                plot_type="vals",
                algorithm_name=algorithm,
            )
            print(f"Saved plot to {plot_path}")
        except Exception as e:
            print(f"Warning: Failed to plot: {e}")
    else:
        pareto_df = pd.DataFrame()
        print("\nWarning: No Pareto solutions found in agent archive")

    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Duration: {duration_min:.1f} minutes")
    print(f"Pareto front size: {len(pareto_df)}")
    print(f"{'='*60}")

    # Save metadata
    save_run_metadata(
        save_dir=save_dir,
        algorithm_name=algorithm.upper(),
        start_time=start_time,
        end_time=end_time,
        pareto_front_size=len(pareto_df),
        total_episodes=None,
        config_path=setup_dict["config_path"],
        additional_info={
            "total_timesteps": total_timesteps,
            "seed": setup_dict["seed"],
        },
    )

    return agent, pareto_df


def train(config_path: str, algorithm: str = "morld", save_dir: str = None):
    """Unified training function for all MORL algorithms.

    Args:
        config_path: Path to config INI file
        algorithm: Algorithm to use ("morld", "pgmorl", "moppo")
        save_dir: Optional override for save directory

    Returns:
        Trained agent and Pareto front tracker

    Example:
        # Use PGMORL
        agent, tracker = train("config.ini", algorithm="pgmorl")

        # Use MORLD (default)
        agent, tracker = train("config.ini")
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
        choices=["morld", "pgmorl", "moppo"],
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
