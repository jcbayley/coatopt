#!/usr/bin/env python3
import json
from pathlib import Path

import gymnasium as gym
from gymnasium.spaces import Box
import matplotlib.pyplot as plt
import numpy as np

from coatopt.environments.environment import CoatingEnvironment
from coatopt.utils.configs import Config, DataConfig, TrainingConfig
from coatopt.utils.utils import load_materials, save_run_metadata
import time


# ============================================================================
# MO-GYMNASIUM WRAPPER
# ============================================================================
class CoatOptEnvSpec:
    """Minimal spec object for MORL-baselines compatibility."""

    def __init__(self, env_id: str = "CoatOpt-v0"):
        self.id = env_id
        self.name = env_id


class CoatOptMOGymWrapper(gym.Env):
    """MO-Gymnasium compatible wrapper for CoatingEnvironment.
    """

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
        if self.previous_material_idx is not None and material_idx == self.previous_material_idx:
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
            normalised_rewards = self.env.compute_objective_rewards(vals, normalised=True)
            vec_reward = np.array([normalised_rewards.get(obj, 0.0) for obj in self.objectives], dtype=np.float32)
            # Apply consecutive penalty to all objectives
            vec_reward = vec_reward - consecutive_penalty
            info["state_array"] = state.get_array()
        else:
            # Intermediate reward: apply penalty if consecutive material used
            vec_reward = np.full(self.reward_dim, -consecutive_penalty, dtype=np.float32)

        return obs, vec_reward, done, truncated, info


class ParetoFrontTracker:
    """Track and visualize Pareto front during training."""

    # Material colors for stack plots
    MATERIAL_COLORS = {
        0: "lightgray",  # air
        1: "steelblue",  # SiO2
        2: "coral",  # Ta2O5
        3: "mediumseagreen",
        4: "gold",
        5: "mediumpurple",
    }

    def __init__(
        self,
        save_dir: str = ".",
        materials: dict = None,
        objectives: list = None,
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.materials = materials or {}
        self.objectives = objectives or ["reflectivity", "absorption"]

        # Pareto front storage: list of dicts with rewards, vals, state_array
        self.pareto_front = []
        self.all_solutions = []

        # Training history
        self.episode_rewards = []  # List of reward vectors
        self.episode_count = 0

    def update(self, vec_reward: np.ndarray, vals: dict, state_array=None):
        """Add a new solution and update Pareto front."""
        self.episode_count += 1

        solution = {
            "reward": vec_reward.copy(),
            "vals": vals.copy() if vals else {},
            "state_array": state_array.copy() if state_array is not None else None,
            "episode": self.episode_count,
        }

        self.all_solutions.append(solution)
        self.episode_rewards.append(vec_reward.copy())

        # Update Pareto front
        self._update_pareto_front(solution)

    def _update_pareto_front(self, new_solution: dict):
        """Update Pareto front with new solution using dominance check."""
        new_reward = new_solution["reward"]

        # Check if new solution is dominated by any existing front member
        is_dominated = False
        to_remove = []

        for i, sol in enumerate(self.pareto_front):
            existing_reward = sol["reward"]

            # Check dominance
            if self._dominates(existing_reward, new_reward):
                is_dominated = True
                break
            elif self._dominates(new_reward, existing_reward):
                to_remove.append(i)

        if not is_dominated:
            # Remove dominated solutions
            for i in reversed(to_remove):
                self.pareto_front.pop(i)
            # Add new solution
            self.pareto_front.append(new_solution)

    def _dominates(self, a: np.ndarray, b: np.ndarray) -> bool:
        """Check if solution a dominates solution b (all >= and at least one >)."""
        return np.all(a >= b) and np.any(a > b)

    def plot_pareto_front(self, title_suffix: str = ""):
        """Plot current Pareto front."""
        if not self.pareto_front:
            return

        fig, axs = plt.subplots(1, 2, figsize=(14, 6))

        # Plot 1: Pareto front in objective space
        ax = axs[0]

        # All solutions (faded)
        if self.all_solutions:
            all_rewards = np.array([s["reward"] for s in self.all_solutions])
            ax.scatter(
                all_rewards[:, 0],
                all_rewards[:, 1],
                alpha=0.1,
                c="gray",
                s=10,
                label="All solutions",
            )

        # Pareto front (highlighted)
        front_rewards = np.array([s["reward"] for s in self.pareto_front])
        ax.scatter(
            front_rewards[:, 0],
            front_rewards[:, 1],
            c="red",
            s=50,
            marker="*",
            label=f"Pareto front ({len(self.pareto_front)})",
            zorder=5,
        )

        # Connect Pareto front points
        if len(front_rewards) > 1:
            sorted_idx = np.argsort(front_rewards[:, 0])
            ax.plot(
                front_rewards[sorted_idx, 0],
                front_rewards[sorted_idx, 1],
                "r--",
                alpha=0.5,
            )

        ax.set_xlabel(f"{self.objectives[0]} (normalised)")
        ax.set_ylabel(f"{self.objectives[1]} (normalised)")
        ax.set_title(f"Pareto Front{title_suffix}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Raw objective values for Pareto front
        ax = axs[1]

        if self.pareto_front:
            reflectivities = [
                s["vals"].get("reflectivity", 0) for s in self.pareto_front
            ]
            absorptions = [s["vals"].get("absorption", 1e-3) for s in self.pareto_front]

            scatter = ax.scatter(
                reflectivities,
                absorptions,
                c=range(len(self.pareto_front)),
                cmap="viridis",
                s=100,
                marker="*",
            )
            plt.colorbar(scatter, ax=ax, label="Solution index")

        ax.set_xlabel("Reflectivity")
        ax.set_ylabel("Absorption")
        ax.set_yscale("log")
        ax.set_title("Pareto Front (Raw Values)")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.save_dir / "pareto_front.png", dpi=150)
        plt.close(fig)

    def plot_training_progress(self):
        """Plot training progress metrics."""
        if len(self.episode_rewards) < 2:
            return

        rewards = np.array(self.episode_rewards)
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))

        # Objective 1 over time
        ax = axs[0, 0]
        ax.plot(rewards[:, 0], alpha=0.6)
        ax.set_title(f"{self.objectives[0]} (normalised)")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        if len(rewards) > 50:
            window = min(50, len(rewards) // 5)
            rolling = np.convolve(rewards[:, 0], np.ones(window) / window, mode="valid")
            ax.plot(range(window - 1, len(rewards)), rolling, "r-", linewidth=2)

        # Objective 2 over time
        ax = axs[0, 1]
        ax.plot(rewards[:, 1], alpha=0.6)
        ax.set_title(f"{self.objectives[1]} (normalised)")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        if len(rewards) > 50:
            window = min(50, len(rewards) // 5)
            rolling = np.convolve(rewards[:, 1], np.ones(window) / window, mode="valid")
            ax.plot(range(window - 1, len(rewards)), rolling, "r-", linewidth=2)

        # Pareto front size over time
        ax = axs[1, 0]
        # Recalculate front size history (approximate)
        ax.axhline(len(self.pareto_front), color="r", linestyle="--", label="Current")
        ax.set_title("Pareto Front Size")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Size")
        ax.legend()

        # Hypervolume proxy: sum of rewards
        ax = axs[1, 1]
        sum_rewards = rewards.sum(axis=1)
        ax.plot(sum_rewards, alpha=0.6)
        ax.set_title("Sum of Rewards (Hypervolume Proxy)")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Sum")
        if len(sum_rewards) > 50:
            window = min(50, len(sum_rewards) // 5)
            rolling = np.convolve(sum_rewards, np.ones(window) / window, mode="valid")
            ax.plot(range(window - 1, len(sum_rewards)), rolling, "r-", linewidth=2)

        plt.tight_layout()
        plt.savefig(self.save_dir / "training_progress.png", dpi=150)
        plt.close(fig)

    def plot_best_designs(self, n_designs: int = 5):
        """Plot coating stack structure for Pareto front designs."""
        designs_with_state = [
            d for d in self.pareto_front if d["state_array"] is not None
        ]
        if not designs_with_state:
            return

        # Sort by reflectivity (first objective)
        designs_with_state.sort(key=lambda x: x["reward"][0], reverse=True)
        designs_to_plot = designs_with_state[:n_designs]

        n_plots = len(designs_to_plot)
        fig, axs = plt.subplots(1, n_plots, figsize=(4 * n_plots, 6), squeeze=False)

        for i, design in enumerate(designs_to_plot):
            ax = axs[0, i]
            state_array = design["state_array"]
            vals = design["vals"]
            self._plot_single_stack(ax, state_array, vals, rank=i + 1)

        plt.tight_layout()
        plt.savefig(self.save_dir / "best_designs.png", dpi=150)
        plt.close(fig)

    def _plot_single_stack(self, ax, state_array, vals, rank: int):
        """Plot a single coating stack as vertical bars."""
        # state_array shape: (n_layers, n_materials + 1)
        # Column 0: thickness, Columns 1+: one-hot material

        # Filter to active layers (thickness > 0)
        thicknesses = state_array[:, 0]
        active_mask = thicknesses > 1e-12
        active_thicknesses = thicknesses[active_mask] * 1e9  # Convert to nm

        if len(active_thicknesses) == 0:
            ax.text(0.5, 0.5, "Empty design", ha="center", va="center")
            ax.set_title(f"Rank {rank}")
            return

        # Get material indices
        material_onehot = state_array[active_mask, 1:]
        material_indices = np.argmax(material_onehot, axis=1)

        # Plot stacked bars from bottom to top
        y_pos = 0
        for layer_idx, (thickness, mat_idx) in enumerate(
            zip(active_thicknesses, material_indices)
        ):
            color = self.MATERIAL_COLORS.get(mat_idx, "gray")
            mat_name = self.materials.get(mat_idx, {}).get("name", f"M{mat_idx}")

            ax.bar(
                0,
                thickness,
                bottom=y_pos,
                width=0.6,
                color=color,
                edgecolor="black",
                linewidth=0.5,
                label=(
                    mat_name
                    if layer_idx == 0 or mat_idx not in material_indices[:layer_idx]
                    else ""
                ),
            )
            y_pos += thickness

        # Labels and title
        R = vals.get("reflectivity", 0)
        A = vals.get("absorption", 0)
        ax.set_title(f"Rank {rank}\nR={R:.4f}, A={A:.2e}")
        ax.set_ylabel("Thickness (nm)")
        ax.set_xticks([])
        ax.set_xlim(-0.5, 0.5)

        # Legend with unique materials
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label:
            ax.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=8)

    def save_pareto_front(self, filename: str = "pareto_front.json"):
        """Save Pareto front solutions to JSON."""
        data = []
        for sol in self.pareto_front:
            entry = {
                "reward": sol["reward"].tolist(),
                "vals": {k: float(v) for k, v in sol["vals"].items() if v is not None},
                "episode": sol["episode"],
            }
            data.append(entry)

        with open(self.save_dir / filename, "w") as f:
            json.dump(data, f, indent=2)

    def save_pareto_front_to_csv(self, filename: str = "pareto_front.csv"):
        """Save BOTH Pareto fronts (value and reward space) to CSV files.

        Args:
            filename: Base name of CSV files (will create two files:
                     {base}_values.csv and {base}_rewards.csv)
        """
        import pandas as pd
        from pathlib import Path

        # Build base filename
        filepath = Path(filename)
        if not filepath.is_absolute():
            filepath = self.save_dir / filename

        # Get base name without extension
        base_name = filepath.stem
        base_dir = filepath.parent

        saved_files = []

        # === Save VALUE space Pareto front ===
        if self.pareto_front:
            values_filepath = base_dir / f"{base_name}_values.csv"
            data = []
            for design in self.pareto_front:
                row = {
                    "episode": design.get("episode", -1),
                }

                # Add objective values
                vals = design.get("vals", {})
                for key, value in vals.items():
                    row[key] = value

                # Add state array information if available
                state_array = design.get("state_array")
                if state_array is not None:
                    thicknesses = state_array[:, 0]
                    active_mask = thicknesses > 1e-12
                    active_thicknesses = thicknesses[active_mask]

                    if state_array.shape[1] > 1:
                        material_onehot = state_array[active_mask, 1:]
                        material_indices = np.argmax(material_onehot, axis=1)
                        row["thicknesses"] = ",".join(map(str, active_thicknesses))
                        row["materials"] = ",".join(map(str, material_indices))
                        row["n_layers"] = len(active_thicknesses)

                data.append(row)

            df = pd.DataFrame(data)
            # Sort by absorption ascending (lower is better)
            if "absorption" in df.columns:
                df = df.sort_values(by="absorption", ascending=True)
            df.to_csv(values_filepath, index=False)
            saved_files.append(str(values_filepath))
            print(f"Saved VALUE space Pareto front to {values_filepath} ({len(df)} points)")

        # === Save REWARD space Pareto front ===
        if self.pareto_front:
            rewards_filepath = base_dir / f"{base_name}_rewards.csv"
            data = []
            for design in self.pareto_front:
                row = {
                    "episode": design.get("episode", -1),
                    "reflectivity_reward": design["reward"][0],
                    "absorption_reward": design["reward"][1],
                }

                # Add original objective values for reference
                vals = design.get("vals", {})
                row["reflectivity"] = vals.get("reflectivity", 0)
                row["absorption"] = vals.get("absorption", 0)

                # Add state array information if available
                state_array = design.get("state_array")
                if state_array is not None:
                    thicknesses = state_array[:, 0]
                    active_mask = thicknesses > 1e-12
                    active_thicknesses = thicknesses[active_mask]

                    if state_array.shape[1] > 1:
                        material_onehot = state_array[active_mask, 1:]
                        material_indices = np.argmax(material_onehot, axis=1)
                        row["n_layers"] = len(active_thicknesses)

                data.append(row)

            df = pd.DataFrame(data)
            # Sort by absorption reward descending (higher is better)
            df = df.sort_values(by="absorption_reward", ascending=False)
            df.to_csv(rewards_filepath, index=False)
            saved_files.append(str(rewards_filepath))
            print(f"Saved REWARD space Pareto front to {rewards_filepath} ({len(df)} points)")

        return saved_files




# ============================================================================
# TRAINING METRICS TRACKER
# ============================================================================
class TrainingMetricsTracker:
    """Track and plot training metrics without wandb."""

    def __init__(self, save_dir: Path, objectives: list):
        self.save_dir = save_dir
        self.objectives = objectives

        # Metrics storage
        self.timesteps = []
        self.hypervolumes = []
        self.pareto_front_sizes = []
        self.policy_losses = {i: [] for i in range(10)}  # Up to 10 policies
        self.critic_losses = {i: [] for i in range(10)}
        self.mean_rewards = {obj: [] for obj in objectives}

    def log_metrics(
        self,
        timestep: int,
        hypervolume: float = None,
        pareto_size: int = None,
        policy_losses: dict = None,
        critic_losses: dict = None,
        mean_rewards: dict = None,
    ):
        """Log training metrics at a timestep."""
        self.timesteps.append(timestep)

        if hypervolume is not None:
            self.hypervolumes.append(hypervolume)
        if pareto_size is not None:
            self.pareto_front_sizes.append(pareto_size)

        if policy_losses:
            for idx, loss in policy_losses.items():
                if idx in self.policy_losses:
                    self.policy_losses[idx].append(loss)

        if critic_losses:
            for idx, loss in critic_losses.items():
                if idx in self.critic_losses:
                    self.critic_losses[idx].append(loss)

        if mean_rewards:
            for obj, val in mean_rewards.items():
                if obj in self.mean_rewards:
                    self.mean_rewards[obj].append(val)

    def plot_training_metrics(self, title_suffix: str = ""):
        """Plot all training metrics."""
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))

        # Hypervolume over time
        ax = axs[0, 0]
        if self.hypervolumes:
            ax.plot(self.timesteps[: len(self.hypervolumes)], self.hypervolumes, "b-")
            ax.set_xlabel("Timesteps")
            ax.set_ylabel("Hypervolume")
            ax.set_title(f"Hypervolume{title_suffix}")
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, "No hypervolume data", ha="center", va="center")
            ax.set_title("Hypervolume")

        # Pareto front size over time
        ax = axs[0, 1]
        if self.pareto_front_sizes:
            ax.plot(
                self.timesteps[: len(self.pareto_front_sizes)],
                self.pareto_front_sizes,
                "g-",
            )
            ax.set_xlabel("Timesteps")
            ax.set_ylabel("Pareto Front Size")
            ax.set_title(f"Pareto Front Size{title_suffix}")
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, "No Pareto front data", ha="center", va="center")
            ax.set_title("Pareto Front Size")

        # Policy losses
        ax = axs[1, 0]
        has_policy_loss = False
        for idx, losses in self.policy_losses.items():
            if losses:
                ax.plot(losses, label=f"Policy {idx}", alpha=0.7)
                has_policy_loss = True
        if has_policy_loss:
            ax.set_xlabel("Update Step")
            ax.set_ylabel("Policy Loss")
            ax.set_title("Policy Losses")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, "No policy loss data", ha="center", va="center")
            ax.set_title("Policy Losses")

        # Mean rewards per objective
        ax = axs[1, 1]
        has_rewards = False
        for obj, rewards in self.mean_rewards.items():
            if rewards:
                ax.plot(rewards, label=obj, alpha=0.7)
                has_rewards = True
        if has_rewards:
            ax.set_xlabel("Evaluation Step")
            ax.set_ylabel("Mean Reward")
            ax.set_title("Mean Rewards by Objective")
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, "No reward data", ha="center", va="center")
            ax.set_title("Mean Rewards")

        plt.tight_layout()
        plt.savefig(self.save_dir / "training_metrics.png", dpi=150)
        plt.close(fig)


# ============================================================================
# TRAINING WITH MORL/D
# ============================================================================
def train_morld(config_path: str, save_dir: str = None):
    """Train using MORL/D algorithm with periodic plotting.

    Args:
        config_path: Path to config INI file
        save_dir: Directory to save results (overrides config if provided)

    Returns:
        MORL/D agent and Pareto front tracker
    """
    import configparser

    parser = configparser.ConfigParser()
    parser.read(config_path)

    # [General] section
    if save_dir is None:
        save_dir = parser.get('General', 'save_dir')
    materials_path = parser.get('General', 'materials_path')

    # [morl] section
    total_timesteps = parser.getint('morl', 'total_timesteps')
    pop_size = parser.getint('morl', 'pop_size')
    seed = parser.getint('morl', 'seed')
    verbose = parser.getint('morl', 'verbose')
    plot_freq = parser.getint('morl', 'plot_freq')
    eval_freq = parser.getint('morl', 'eval_freq')

    # Network architecture
    net_arch_str = parser.get('morl', 'net_arch', fallback='[256, 256]')
    net_arch = eval(net_arch_str)

    # [Data] section
    n_layers = parser.getint('Data', 'n_layers')
    try:
        from morl_baselines.multi_policy.morld.morld import MORLD
    except ImportError:
        raise ImportError(
            "morl-baselines not installed. Install with: pip install morl-baselines"
        )

    # Set up directories
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = save_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Load materials
    if materials_path is None:
        materials_path = Path(__file__).parent / "config" / "materials.json"

    materials = load_materials(str(materials_path))
    print(f"Loaded {len(materials)} materials from {materials_path}")

    # Create minimal config
    config = Config(
        data=DataConfig(
            n_layers=n_layers,
            min_thickness=10e-9,
            max_thickness=500e-9,
            optimise_parameters=["reflectivity", "absorption"],
            optimise_targets={"reflectivity": 0.99999, "absorption": 0.0},
            use_optical_thickness=False,
            ignore_air_option=False,
            ignore_substrate_option=False,
            use_intermediate_reward=False,
            combine="sum",
            use_reward_normalisation=True,
            reward_normalisation_apply_clipping=True,
            objective_bounds={
                "reflectivity": [0.0, 0.99999],
                "absorption": [1e-3, 0.0],
            },
            apply_air_penalty=True,
            air_penalty_weight=0.5,
            apply_preference_constraints=False,
        ),
        training=TrainingConfig(cycle_weights="random"),
    )

    # Create MO-Gymnasium compatible environment
    env = CoatOptMOGymWrapper(config, materials)
    eval_env = CoatOptMOGymWrapper(config, materials)

    print(f"Environment created:")
    print(f"  Observation space: {env.observation_space.shape}")
    print(f"  Action space: {env.action_space}")
    print(f"  Reward space: {env.reward_space}")
    print(f"  Reward dim: {env.reward_dim}")
    print(f"  Max layers: {env.env.max_layers}")
    print(f"  N materials: {env.env.n_materials}")

    # Reference point for hypervolume (worst case)
    ref_point = np.array([0.0, 0.0])

    print(f"\nMORL Network Architecture:")
    print(f"  Policy network: {net_arch}")

    # Create MORL/D agent (wandb disabled with log=False)
    agent = MORLD(
        env=env,
        scalarization_method="ws",  # Weighted sum
        evaluation_mode="ser",  # Expected reward under single policy
        policy_name="MOSAC",  # Multi-objective SAC as base policy
        gamma=0.99,
        pop_size=pop_size,
        seed=seed,
        exchange_every=int(total_timesteps / 10),  # Share every 10% of training
        neighborhood_size=1,
        shared_buffer=True,
        weight_init_method="uniform",
        weight_adaptation_method=None,
        log=False,  # Disable wandb logging
        device="auto",
        policy_args={"net_arch": net_arch},  # Network architecture for actor and critic
    )

    # Initialize trackers
    tracker = ParetoFrontTracker(
        save_dir=str(save_dir),
        materials=materials,
        objectives=["reflectivity", "absorption"],
    )
    metrics = TrainingMetricsTracker(
        save_dir=save_dir,
        objectives=["reflectivity", "absorption"],
    )

    # Train in intervals for periodic evaluation and plotting
    n_intervals = max(1, total_timesteps // eval_freq)
    timesteps_per_interval = total_timesteps // n_intervals

    # Start timing
    start_time = time.time()

    for interval in range(n_intervals):
        current_timesteps = (interval + 1) * timesteps_per_interval

        # Train for this interval
        print(f"\n--- Training interval {interval + 1}/{n_intervals} ---")
        agent.train(
            total_timesteps=current_timesteps,
            eval_env=eval_env,
            ref_point=ref_point,
            known_pareto_front=None,
            num_eval_episodes_for_front=5,
            num_eval_weights_for_eval=20,
            reset_num_timesteps=False,
        )

        # Evaluate current policies and update Pareto front
        print(f"Evaluating at timestep {current_timesteps}...")
        eval_results = evaluate_pareto_front(
            agent, eval_env, tracker, n_eval_episodes=5
        )

        # Calculate hypervolume
        if tracker.pareto_front:
            hv = calculate_hypervolume(tracker.pareto_front, ref_point)
        else:
            hv = 0.0

        # Log metrics
        metrics.log_metrics(
            timestep=current_timesteps,
            hypervolume=hv,
            pareto_size=len(tracker.pareto_front),
            mean_rewards=eval_results.get("mean_rewards", {}),
        )

        # Print progress
        if verbose:
            print(
                f"  Timestep {current_timesteps}: "
                f"Pareto size={len(tracker.pareto_front)}, "
                f"HV={hv:.4f}"
            )
            if tracker.pareto_front:
                best_R = max(s["vals"].get("reflectivity", 0) for s in tracker.pareto_front)
                best_A = min(s["vals"].get("absorption", 1) for s in tracker.pareto_front)
                print(f"  Best R={best_R:.6f}, Best A={best_A:.2e}")

        # Periodic plotting
        if current_timesteps % plot_freq == 0 or interval == n_intervals - 1:
            print(f"Saving plots at timestep {current_timesteps}...")

            # Plot Pareto front
            tracker.plot_pareto_front(f" (t={current_timesteps})")

            # Plot training metrics
            metrics.plot_training_metrics(f" (t={current_timesteps})")

            # Plot best coating designs
            if tracker.pareto_front:
                tracker.plot_best_designs(n_designs=min(5, len(tracker.pareto_front)))

            # Save timestamped copies
            import shutil

            for src_name in ["pareto_front.png", "best_designs.png", "training_metrics.png"]:
                src = save_dir / src_name
                if src.exists():
                    dst = plots_dir / f"{src.stem}_t{current_timesteps}{src.suffix}"
                    shutil.copy(src, dst)

    print(f"\nTraining complete!")
    end_time = time.time()

    # Final evaluation with more episodes
    print("\nFinal evaluation...")
    evaluate_pareto_front(agent, eval_env, tracker, n_eval_episodes=20)

    # Final plots
    tracker.plot_pareto_front(" (final)")
    tracker.plot_training_progress()
    tracker.plot_best_designs()
    metrics.plot_training_metrics(" (final)")
    tracker.save_pareto_front()

    # Save Pareto front to CSV (like SB3 runs)
    print("\nSaving Pareto front to CSV...")
    tracker.save_pareto_front_to_csv("pareto_front.csv")

    print(f"\nResults saved to {save_dir}")
    print(f"Pareto front size: {len(tracker.pareto_front)}")

    # Print final Pareto front summary
    if tracker.pareto_front:
        print("\nPareto Front Summary:")
        print("-" * 60)
        for i, sol in enumerate(
            sorted(tracker.pareto_front, key=lambda x: -x["vals"].get("reflectivity", 0))
        ):
            R = sol["vals"].get("reflectivity", 0)
            A = sol["vals"].get("absorption", 0)
            print(f"  Solution {i + 1}: R={R:.6f}, A={A:.2e}")

    # Save run metadata
    save_run_metadata(
        save_dir=save_dir,
        algorithm_name="MORLD",
        start_time=start_time,
        end_time=end_time,
        pareto_front_size=len(tracker.pareto_front),
        total_episodes=None,
        config_path=config_path,
        additional_info={
            "total_timesteps": total_timesteps,
            "pop_size": pop_size,
            "seed": seed,
        }
    )

    return agent, tracker


def calculate_hypervolume(pareto_front: list, ref_point: np.ndarray) -> float:
    """Calculate hypervolume indicator for the Pareto front."""
    if not pareto_front:
        return 0.0

    try:
        from pymoo.indicators.hv import HV
        points = np.array([s["reward"] for s in pareto_front])
        # Negate because pymoo expects minimization
        hv_indicator = HV(ref_point=-ref_point)
        return float(hv_indicator(-points))
    except ImportError:
        # Simple 2D hypervolume calculation
        points = np.array([s["reward"] for s in pareto_front])
        points = points[np.argsort(points[:, 0])]  # Sort by first objective

        hv = 0.0
        prev_x = ref_point[0]
        for point in points:
            if point[1] > ref_point[1]:
                hv += (point[0] - prev_x) * (point[1] - ref_point[1])
                prev_x = point[0]
        return hv


def evaluate_pareto_front(
    agent, env: CoatOptMOGymWrapper, tracker: ParetoFrontTracker, n_eval_episodes: int = 10
) -> dict:
    """Evaluate agent's policies to extract Pareto front.

    MORLD stores policies in agent.population, each with its own weight vector.
    We evaluate each policy multiple times and track results.

    Returns dict with evaluation statistics.
    """
    all_rewards = []
    objective_rewards = {obj: [] for obj in env.objectives}

    # Get policies from MORLD population
    policies = getattr(agent, "population", [])

    if not policies:
        # Fallback: try archive if population is empty
        policies = getattr(agent, "archive", {}).get("policies", [])

    if not policies:
        print("Warning: No policies found in agent for evaluation")
        return {"mean_rewards": {obj: 0.0 for obj in env.objectives}, "n_episodes": 0}

    episodes_per_policy = max(1, n_eval_episodes // len(policies))

    for policy in policies:
        # Get the wrapped policy that has the eval method
        wrapped_policy = getattr(policy, "policy", policy)

        for ep in range(episodes_per_policy):
            obs, _ = env.reset()
            done = False
            episode_reward = np.zeros(env.reward_dim, dtype=np.float32)

            while not done:
                # Get action from the policy
                # Try different interfaces that MORL-baselines policies might have
                if hasattr(wrapped_policy, "eval"):
                    action = wrapped_policy.eval(obs, policy.weights if hasattr(policy, "weights") else None)
                elif hasattr(wrapped_policy, "act"):
                    action = wrapped_policy.act(obs)
                elif hasattr(wrapped_policy, "get_action"):
                    action = wrapped_policy.get_action(obs)
                elif hasattr(wrapped_policy, "select_action"):
                    action = wrapped_policy.select_action(obs)
                else:
                    # Last resort: random action
                    action = env.action_space.sample()

                # Handle tensor outputs
                if hasattr(action, "cpu"):
                    action = action.cpu().numpy()
                if hasattr(action, "detach"):
                    action = action.detach().numpy()

                obs, reward, done, truncated, info = env.step(action)
                episode_reward += reward
                done = done or truncated

            all_rewards.append(episode_reward)

            # Add to tracker
            if info.get("vals"):
                tracker.update(
                    episode_reward,
                    info["vals"],
                    info.get("state_array"),
                )

                # Track per-objective rewards
                for i, obj in enumerate(env.objectives):
                    objective_rewards[obj].append(episode_reward[i])

    # Calculate statistics
    results = {
        "mean_rewards": {
            obj: np.mean(rewards) if rewards else 0.0
            for obj, rewards in objective_rewards.items()
        },
        "n_episodes": len(all_rewards),
    }

    return results


# ============================================================================
# MANUAL TRAINING LOOP (Alternative without MORL/D dependency)
# ============================================================================
def train_manual(
    total_timesteps: int = 100_000,
    n_layers: int = 20,
    materials_path: str = None,
    save_dir: str = "./morl_output",
    plot_freq: int = 5000,
    verbose: int = 1,
):
    """Manual training loop using linear scalarization with varying weights.

    This is a simpler alternative that doesn't require morl-baselines,
    using random weight sampling to explore the Pareto front.

    Args:
        total_timesteps: Total training timesteps
        n_layers: Number of layers in coating stack
        materials_path: Path to materials JSON file
        save_dir: Directory for output files
        plot_freq: How often to plot progress
        verbose: Verbosity level

    Returns:
        Pareto front tracker with discovered solutions
    """
    # Set up directories
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load materials
    if materials_path is None:
        materials_path = Path(__file__).parent / "config" / "materials.json"

    materials = load_materials(str(materials_path))
    print(f"Loaded {len(materials)} materials from {materials_path}")

    # Create config
    config = Config(
        data=DataConfig(
            n_layers=n_layers,
            min_thickness=10e-9,
            max_thickness=500e-9,
            optimise_parameters=["reflectivity", "absorption"],
            optimise_targets={"reflectivity": 0.99999, "absorption": 0.0},
            use_optical_thickness=False,
            ignore_air_option=False,
            ignore_substrate_option=False,
            use_intermediate_reward=False,
            combine="sum",
            use_reward_normalisation=True,
            reward_normalisation_apply_clipping=True,
            objective_bounds={
                "reflectivity": [0.0, 0.99999],
                "absorption": [1e-3, 0.0],
            },
            apply_air_penalty=True,
            air_penalty_weight=0.5,
            apply_preference_constraints=False,
        ),
        training=TrainingConfig(cycle_weights="random"),
    )

    # Create environment
    env = CoatOptMOGymWrapper(config, materials)
    print(f"Environment created:")
    print(f"  Observation space: {env.observation_space.shape}")
    print(f"  Action space: {env.action_space}")
    print(f"  Reward space: {env.reward_space}")
    print(f"  Reward dim: {env.reward_dim}")

    # Initialize tracker
    tracker = ParetoFrontTracker(
        save_dir=str(save_dir),
        materials=materials,
        objectives=["reflectivity", "absorption"],
    )

    # Simple random policy exploration
    print(f"\nStarting exploration for {total_timesteps} timesteps...")

    timestep = 0
    episode = 0

    while timestep < total_timesteps:
        obs, _ = env.reset()
        done = False
        episode_reward = np.zeros(env.reward_dim, dtype=np.float32)
        episode_length = 0

        while not done and timestep < total_timesteps:
            # Random action (baseline exploration)
            action = env.action_space.sample()

            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            timestep += 1
            done = done or truncated

        # Update tracker with episode result
        if info.get("vals"):
            tracker.update(
                episode_reward,
                info["vals"],
                info.get("state_array"),
            )

        episode += 1

        # Logging
        if verbose and episode % 100 == 0:
            print(
                f"Episode {episode}, Timestep {timestep}, "
                f"Pareto front size: {len(tracker.pareto_front)}"
            )

        # Periodic plotting
        if timestep % plot_freq == 0 or timestep >= total_timesteps:
            tracker.plot_pareto_front(f" (t={timestep})")
            tracker.plot_training_progress()
            if tracker.pareto_front:
                tracker.plot_best_designs()

    # Final save
    tracker.save_pareto_front()

    # Save Pareto front to CSV (like SB3 runs)
    print("\nSaving Pareto front to CSV...")
    tracker.save_pareto_front_to_csv("pareto_front.csv")

    print(f"\nExploration complete!")
    print(f"Total episodes: {episode}")
    print(f"Pareto front size: {len(tracker.pareto_front)}")

    return tracker


# ============================================================================
# TRAINING WITH NL-MO-PPO (Non-Linear Multi-Objective PPO)
# ============================================================================
def train_nlmoppo(
    total_timesteps: int = 100_000,
    n_layers: int = 20,
    materials_path: str = None,
    save_dir: str = "./morl_output",
    n_envs: int = 4,
    n_preferences: int = 10,
    seed: int = 42,
    plot_freq: int = 10_000,
    verbose: int = 1,
):
    """Train using NL-MO-PPO algorithm with multiple preference vectors.

    NL-MO-PPO is a single-policy algorithm that can use non-linear utility
    functions. We train it multiple times with different preference vectors
    to build a Pareto front.

    Args:
        total_timesteps: Timesteps PER preference vector
        n_layers: Number of layers in coating stack
        materials_path: Path to materials JSON file
        save_dir: Directory for output files
        n_envs: Number of parallel environments
        n_preferences: Number of different preference vectors to try
        seed: Random seed
        plot_freq: How often to save plots
        verbose: Verbosity level

    Returns:
        Pareto front tracker with discovered solutions
    """
    try:
        from morl_baselines.single_policy.ser.mo_ppo import MOPPO
    except ImportError:
        try:
            from morl_baselines.single_policy.ser.nl_mo_ppo import NLMOPPO
        except ImportError:
            raise ImportError(
                "morl-baselines not installed or MOPPO/NLMOPPO not found. "
                "Install with: pip install morl-baselines"
            )

    try:
        import mo_gymnasium as mo_gym
        from mo_gymnasium.utils import MORecordEpisodeStatistics
    except ImportError:
        raise ImportError("mo-gymnasium not installed. Install with: pip install mo-gymnasium")

    import torch

    # Set up directories
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = save_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Load materials
    if materials_path is None:
        materials_path = Path(__file__).parent / "config" / "materials.json"

    materials = load_materials(str(materials_path))
    print(f"Loaded {len(materials)} materials from {materials_path}")

    # Create config
    config = Config(
        data=DataConfig(
            n_layers=n_layers,
            min_thickness=10e-9,
            max_thickness=500e-9,
            optimise_parameters=["reflectivity", "absorption"],
            optimise_targets={"reflectivity": 0.99999, "absorption": 0.0},
            use_optical_thickness=False,
            ignore_air_option=False,
            ignore_substrate_option=False,
            use_intermediate_reward=False,
            combine="sum",
            use_reward_normalisation=True,
            reward_normalisation_apply_clipping=True,
            objective_bounds={
                "reflectivity": [0.0, 0.99999],
                "absorption": [1e-3, 0.0],
            },
            apply_air_penalty=True,
            air_penalty_weight=0.5,
            apply_preference_constraints=False,
        ),
        training=TrainingConfig(cycle_weights="random"),
    )

    # Create vectorized environment for PPO
    def make_env():
        env = CoatOptMOGymWrapper(config, materials)
        env = MORecordEpisodeStatistics(env)
        return env

    envs = mo_gym.MOSyncVectorEnv([make_env for _ in range(n_envs)])
    eval_env = make_env()

    print(f"Environment created:")
    print(f"  Observation space: {envs.single_observation_space.shape}")
    print(f"  Action space: {envs.single_action_space}")
    print(f"  Reward dim: {eval_env.reward_dim}")
    print(f"  N parallel envs: {n_envs}")

    # Initialize tracker
    tracker = ParetoFrontTracker(
        save_dir=str(save_dir),
        materials=materials,
        objectives=["reflectivity", "absorption"],
    )

    # Generate preference vectors to explore Pareto front
    preferences = []
    for i in range(n_preferences):
        # Uniform spacing along the preference simplex
        w1 = i / (n_preferences - 1) if n_preferences > 1 else 0.5
        w2 = 1.0 - w1
        preferences.append(torch.tensor([w1, w2], dtype=torch.float32))

    print(f"\nTraining NL-MO-PPO with {n_preferences} preference vectors...")
    print(f"  Timesteps per preference: {total_timesteps}")
    print(f"  Total timesteps: {total_timesteps * n_preferences}")

    # Train with each preference vector
    for pref_idx, pref in enumerate(preferences):
        print(f"\n--- Preference {pref_idx + 1}/{n_preferences}: {pref.numpy()} ---")

        # Utility function: weighted sum with current preference
        def utility_func(rewards: torch.Tensor) -> torch.Tensor:
            # rewards shape: (batch, reward_dim) or (reward_dim,)
            if rewards.dim() == 1:
                return (rewards * pref).sum()
            return (rewards * pref).sum(dim=-1)

        # Create fresh agent for each preference
        try:
            # Try MOPPO first (more common)
            from morl_baselines.single_policy.ser.mo_ppo import MOPPO
            agent = MOPPO(
                id=pref_idx,
                envs=envs,
                log=False,  # Disable wandb
                total_timesteps=total_timesteps,
                learning_rate=3e-4,
                num_steps=128,
                gamma=0.99,
                gae_lambda=0.95,
                num_minibatches=4,
                update_epochs=4,
                clip_coef=0.2,
                ent_coef=0.01,
                vf_coef=0.5,
                seed=seed + pref_idx,
                device="auto",
            )
        except Exception:
            # Fall back to NLMOPPO
            from morl_baselines.single_policy.ser.nl_mo_ppo import NLMOPPO
            agent = NLMOPPO(
                id=pref_idx,
                envs=envs,
                log=False,
                total_timesteps=total_timesteps,
                learning_rate=3e-4,
                num_steps=128,
                gamma=0.99,
                gae_lambda=0.95,
                num_minibatches=4,
                update_epochs=4,
                clip_coef=0.2,
                ent_coef=0.01,
                vf_coef=0.5,
                seed=seed + pref_idx,
                device="auto",
            )

        # Train
        try:
            result = agent.train(
                eval_env=eval_env,
                w=pref,  # Some versions use 'w' instead of 'pref'
            )
        except TypeError:
            try:
                result = agent.train(
                    eval_env=eval_env,
                    pref=pref,
                )
            except TypeError:
                # Simplest form
                result = agent.train(eval_env=eval_env)

        # Evaluate this policy and add to Pareto front
        print(f"  Evaluating policy...")
        evaluate_single_policy(agent, eval_env, tracker, n_episodes=5)

        # Periodic plotting
        if (pref_idx + 1) % max(1, n_preferences // 5) == 0:
            tracker.plot_pareto_front(f" (pref {pref_idx + 1}/{n_preferences})")
            tracker.plot_training_progress()
            if tracker.pareto_front:
                tracker.plot_best_designs()

        if verbose:
            print(f"  Pareto front size: {len(tracker.pareto_front)}")

    # Final evaluation and plots
    print(f"\nTraining complete!")
    tracker.plot_pareto_front(" (final)")
    tracker.plot_training_progress()
    tracker.plot_best_designs()
    tracker.save_pareto_front()

    # Save Pareto front to CSV (like SB3 runs)
    print("\nSaving Pareto front to CSV...")
    tracker.save_pareto_front_to_csv("pareto_front.csv")

    print(f"\nResults saved to {save_dir}")
    print(f"Pareto front size: {len(tracker.pareto_front)}")

    # Print Pareto front summary
    if tracker.pareto_front:
        print("\nPareto Front Summary:")
        print("-" * 60)
        for i, sol in enumerate(
            sorted(tracker.pareto_front, key=lambda x: -x["vals"].get("reflectivity", 0))
        ):
            R = sol["vals"].get("reflectivity", 0)
            A = sol["vals"].get("absorption", 0)
            print(f"  Solution {i + 1}: R={R:.6f}, A={A:.2e}")

    envs.close()
    return tracker


def evaluate_single_policy(agent, env, tracker, n_episodes: int = 5):
    """Evaluate a single trained policy and add results to tracker."""
    import torch

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = np.zeros(env.reward_dim, dtype=np.float32)

        while not done:
            # Get action from agent
            if hasattr(agent, "eval"):
                action = agent.eval(obs)
            elif hasattr(agent, "act"):
                action = agent.act(obs)
            elif hasattr(agent, "get_action"):
                action, _, _, _ = agent.get_action(
                    torch.tensor(obs).unsqueeze(0).float()
                )
                action = action.squeeze().cpu().numpy()
            else:
                # Try accessing the actor network directly
                with torch.no_grad():
                    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                    if hasattr(agent, "actor"):
                        action = agent.actor(obs_tensor)
                    elif hasattr(agent, "policy"):
                        action = agent.policy(obs_tensor)
                    else:
                        action = env.action_space.sample()

            # Handle tensor outputs
            if hasattr(action, "cpu"):
                action = action.cpu().numpy()
            if hasattr(action, "detach"):
                action = action.detach().numpy()
            if hasattr(action, "squeeze"):
                action = np.squeeze(action)

            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            done = done or truncated

        # Add to tracker
        if info.get("vals"):
            tracker.update(
                episode_reward,
                info["vals"],
                info.get("state_array"),
            )


# ============================================================================
# CLI
# ============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train MORL on CoatOpt")
    parser.add_argument(
        "--timesteps", type=int, default=100_000, help="Total training timesteps"
    )
    parser.add_argument(
        "--layers", type=int, default=20, help="Number of coating layers"
    )
    parser.add_argument(
        "--materials", type=str, default=None, help="Path to materials JSON"
    )
    parser.add_argument(
        "--save-dir", type=str, default="./morl_output", help="Output directory"
    )
    parser.add_argument(
        "--pop-size", type=int, default=6, help="Population size for MORL/D"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level")
    parser.add_argument(
        "--plot-freq",
        type=int,
        default=10_000,
        help="How often to save plots (timesteps)",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=5_000,
        help="How often to evaluate and update Pareto front (timesteps)",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="morld",
        choices=["morld", "moppo", "manual"],
        help="Training method: 'morld' for MORL/D, 'moppo' for MO-PPO, or 'manual' for exploration",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=4,
        help="Number of parallel environments (for moppo)",
    )
    parser.add_argument(
        "--n-prefs",
        type=int,
        default=10,
        help="Number of preference vectors to try (for moppo)",
    )

    args = parser.parse_args()

    if args.method == "morld":
        train_morld(
            total_timesteps=args.timesteps,
            n_layers=args.layers,
            materials_path=args.materials,
            save_dir=args.save_dir,
            pop_size=args.pop_size,
            seed=args.seed,
            plot_freq=args.plot_freq,
            eval_freq=args.eval_freq,
            verbose=args.verbose,
        )
    elif args.method == "moppo":
        train_nlmoppo(
            total_timesteps=args.timesteps,
            n_layers=args.layers,
            materials_path=args.materials,
            save_dir=args.save_dir,
            n_envs=args.n_envs,
            n_preferences=args.n_prefs,
            seed=args.seed,
            plot_freq=args.plot_freq,
            verbose=args.verbose,
        )
    else:
        train_manual(
            total_timesteps=args.timesteps,
            n_layers=args.layers,
            materials_path=args.materials,
            save_dir=args.save_dir,
            plot_freq=args.plot_freq,
            verbose=args.verbose,
        )
