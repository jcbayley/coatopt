"""Shared callback classes for CoatOpt experiments."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import math
import mlflow

class EntropyAnnealingCallback(BaseCallback):
    """Callback to update entropy coefficient with cosine annealing schedule.

    SB3 doesn't support callable schedules for ent_coef, so we use a callback.
    """

    def __init__(
        self,
        max_ent: float = 0.2,
        min_ent: float = 0.01,
        epochs_per_step: int = None,
        verbose: int = 0,
        adaptive_to_constraints: bool = False,
        constraint_window: int = 50,
    ):
        """Initialize entropy annealing callback.

        Args:
            max_ent: Maximum entropy coefficient (high exploration)
            min_ent: Minimum entropy coefficient (low exploration)
            epochs_per_step: If provided, reset annealing every N episodes (for cycling).
                            If None, anneal once over entire training.
            verbose: Verbosity level
            adaptive_to_constraints: If True, increase entropy when constraint violations are high
            constraint_window: Number of recent episodes to track for constraint violations
        """
        super().__init__(verbose)
        self.max_ent = max_ent
        self.min_ent = min_ent
        self.epochs_per_step = epochs_per_step
        self.episode_count = 0
        self.adaptive_to_constraints = adaptive_to_constraints
        self.constraint_window = constraint_window
        self.recent_constraint_violations = []

    def _on_step(self) -> bool:
        """Update entropy coefficient at each step."""
        # Track episodes and constraint violations
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info and isinstance(info["episode"], dict):
                self.episode_count += 1

                # Track constraint violations if adaptive mode is enabled
                if self.adaptive_to_constraints:
                    # Check if episode had constraint violations
                    constraints = info.get("constraints", {})
                    rewards = info.get("rewards", {})

                    if constraints and rewards:
                        # Calculate total constraint violation
                        violation = 0.0
                        for obj, threshold in constraints.items():
                            reward = rewards.get(obj, 0.0)
                            if reward < threshold:
                                violation += (threshold - reward)

                        self.recent_constraint_violations.append(violation)
                        if len(self.recent_constraint_violations) > self.constraint_window:
                            self.recent_constraint_violations.pop(0)

        # Compute current entropy coefficient
        if self.epochs_per_step is not None:
            # Cycling mode: reset every epochs_per_step episodes
            episode_in_cycle = self.episode_count % self.epochs_per_step
            progress = episode_in_cycle / self.epochs_per_step
        else:
            # Single annealing mode: use training progress
            progress = 1.0 - (self.num_timesteps / self.model._total_timesteps)

        # Cosine annealing formula (base entropy)
        base_ent_coef = self.min_ent + 0.5 * (self.max_ent - self.min_ent) * (
            1 + math.cos(math.pi * progress)
        )

        # Apply adaptive boost based on constraint violations
        ent_coef = base_ent_coef
        if self.adaptive_to_constraints and len(self.recent_constraint_violations) > 0:
            avg_violation = sum(self.recent_constraint_violations) / len(self.recent_constraint_violations)
            # Scale entropy boost by violation magnitude (0 violation = no boost, high violation = up to 2x boost)
            boost_factor = 1.0 + min(avg_violation, 1.0)  # Cap at 2x
            ent_coef = min(base_ent_coef * boost_factor, self.max_ent * 2.0)  # Allow going above max_ent

            if self.verbose > 0 and self.episode_count % 100 == 0:
                print(f"  Avg constraint violation: {avg_violation:.4f}, boost factor: {boost_factor:.2f}x")

        # Update model's entropy coefficient
        self.model.ent_coef = ent_coef

        # Log to tensorboard if available
        if self.verbose > 0 and self.episode_count % 100 == 0:
            print(f"Episode {self.episode_count}: ent_coef = {ent_coef:.4f} (base: {base_ent_coef:.4f})")

        return True


class PlottingCallback(BaseCallback):
    """Callback for plotting training progress and best designs found.

    This is the base callback used by both SB3 simple and discrete training.
    It tracks episode rewards, Pareto fronts, and plots coating stack designs.
    """

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
        env=None,
        plot_freq: int = 5000,
        design_plot_freq: int = 100,
        save_dir: str = ".",
        n_best_designs: int = 5,
        materials: dict = None,
        verbose: int = 0,
        track_action_distributions: bool = False,
    ):
        """Initialize plotting callback.

        Args:
            env: Training environment (used for normalised_reward in discrete)
            plot_freq: Plot training progress every N timesteps
            design_plot_freq: Plot best designs every N episodes
            save_dir: Directory to save plots
            n_best_designs: Number of best designs to track
            materials: Materials dictionary for plotting
            verbose: Verbosity level
            track_action_distributions: Track discrete action choices (for discrete training)
        """
        super().__init__(verbose)
        self.env = env
        self.plot_freq = plot_freq
        self.design_plot_freq = design_plot_freq
        self.save_dir = Path(save_dir)
        self.n_best_designs = n_best_designs
        self.materials = materials or {}
        self.track_action_distributions = track_action_distributions

        # Episode tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []
        self.episode_count = 0

        # Track all episode rewards (normalised) for both objectives
        self.all_episode_rewards = []  # List of (ref_reward, abs_reward) tuples

        # Annealing/schedule tracking
        self.annealing_progress = []
        self.constraint_history = {obj: [] for obj in ["reflectivity", "absorption"]}
        self.target_objective_history = []
        self.phase_history = []
        self.level_history = []
        self.entropy_coef_history = []

        # Action distribution tracking (only for discrete)
        if track_action_distributions:
            self.material_action_history = []
            self.thickness_bin_history = []


    def _on_step(self) -> bool:
        """Called at each step. Collect episode statistics and trigger plotting."""
        infos = self.locals.get("infos", [])
        for info in infos:
            # Track discrete action choices (if enabled)
            if self.track_action_distributions:
                if "material_idx" in info:
                    self.material_action_history.append(info["material_idx"])
                if "thickness_bin" in info:
                    self.thickness_bin_history.append(info["thickness_bin"])

            if "episode" in info:
                if type(info["episode"]) == int:
                    continue
                ep_reward = info["episode"]["r"]
                ep_length = info["episode"]["l"]
                self.episode_rewards.append(ep_reward)
                self.episode_lengths.append(ep_length)
                self.episode_count += 1

                # Track annealing/schedule info
                if "annealing_progress" in info:
                    self.annealing_progress.append(info["annealing_progress"])
                if "target" in info:
                    self.target_objective_history.append(info["target"])
                if "constraints" in info:
                    for obj in ["reflectivity", "absorption"]:
                        if obj in info["constraints"]:
                            self.constraint_history[obj].append(info["constraints"][obj])
                        else:
                            self.constraint_history[obj].append(None)
                if "phase" in info:
                    self.phase_history.append(info["phase"])
                if "level" in info:
                    self.level_history.append(info["level"])

                # Track entropy coefficient
                if hasattr(self.model, 'ent_coef'):
                    self.entropy_coef_history.append(float(self.model.ent_coef))

                # Get vals and rewards from info
                vals = info.get("vals", {})
                rewards = info.get("rewards", {})

                # Log to MLflow
                if mlflow.active_run():
                    mlflow.log_metric("episode_reward", ep_reward, step=self.episode_count)
                    mlflow.log_metric("episode_length", ep_length, step=self.episode_count)
                    if hasattr(self.model, 'ent_coef'):
                        mlflow.log_metric("entropy_coef", float(self.model.ent_coef), step=self.episode_count)
                    if vals:
                        for obj_name, obj_val in vals.items():
                            mlflow.log_metric(f"value_{obj_name}", obj_val, step=self.episode_count)
                    if rewards:
                        for obj_name, obj_val in rewards.items():
                            mlflow.log_metric(f"reward_{obj_name}", obj_val, step=self.episode_count)
                    if "annealing_progress" in info:
                        mlflow.log_metric("annealing_progress", info["annealing_progress"], step=self.episode_count)

                # Track normalized rewards if available in info
                if rewards:
                    ref_reward = rewards.get('reflectivity', 0.0)
                    abs_reward = rewards.get('absorption', 0.0)
                    self.all_episode_rewards.append((ref_reward, abs_reward))

                # Plot designs periodically
                if self.episode_count % self.design_plot_freq == 0:
                    self._plot_best_designs()
                    if hasattr(self, '_plot_alternating_materials'):
                        self._plot_alternating_materials()
                    # Save Pareto front to CSV periodically
                    self.save_pareto_front_to_csv("pareto_front.csv")

        # Collect training loss if available
        if hasattr(self.model, "logger"):
            loss = self.model.logger.name_to_value.get("train/loss", None)
            if loss is not None:
                self.losses.append(loss)

        # Plot training progress periodically
        if self.n_calls % self.plot_freq == 0 and self.episode_rewards:
            self._plot_training_progress()

        return True

    def _plot_training_progress(self):
        """Plot training metrics."""
        n_rows = 3 if self.track_action_distributions else 2
        fig, axs = plt.subplots(n_rows, 4, figsize=(20, 5 * n_rows))

        # Get environment
        env = self.env
        if hasattr(env, 'env'):
            env = env.env

        # Get both Pareto fronts from environment
        pareto_front_rewards = env.get_pareto_front(space="reward") if hasattr(env, 'get_pareto_front') else []
        pareto_front_values = env.get_pareto_front(space="value") if hasattr(env, 'get_pareto_front') else []

        # Process for plotting
        pareto_designs = []
        for i, (reward_vector, state) in enumerate(pareto_front_rewards):
            # Get corresponding value vector
            val_vector = pareto_front_values[i][0] if i < len(pareto_front_values) else reward_vector

            # Build dicts
            vals = {}
            reward_vals = {}
            if hasattr(env, 'optimise_parameters'):
                for j, param_name in enumerate(env.optimise_parameters):
                    if j < len(val_vector):
                        vals[param_name] = val_vector[j]
                    if j < len(reward_vector):
                        reward_vals[param_name] = reward_vector[j]

            # Count active layers
            state_array = state.get_array()
            n_layers = np.sum(state_array[:, 0] > 1e-12)

            pareto_designs.append({
                'vals': vals,
                'reward_vals': reward_vals,
                'n_layers': n_layers
            })

        # Episode rewards
        axs[0, 0].plot(self.episode_rewards, alpha=0.6)
        axs[0, 0].set_title("Episode Rewards")
        axs[0, 0].set_xlabel("Episode")
        axs[0, 0].set_ylabel("Reward")
        if len(self.episode_rewards) > 10:
            window = min(50, len(self.episode_rewards) // 5)
            if window > 1:
                rolling = np.convolve(
                    self.episode_rewards, np.ones(window) / window, mode="valid"
                )
                axs[0, 0].plot(
                    range(window - 1, len(self.episode_rewards)),
                    rolling,
                    "r-",
                    linewidth=2,
                    label=f"Rolling avg ({window})",
                )
                axs[0, 0].legend()

        # Episode lengths
        axs[0, 1].plot(self.episode_lengths, alpha=0.6)
        axs[0, 1].set_title("Episode Lengths (Layers Used)")
        axs[0, 1].set_xlabel("Episode")
        axs[0, 1].set_ylabel("Layers")
        if len(self.episode_lengths) > 10:
            window = min(50, len(self.episode_lengths) // 5)
            if window > 1:
                rolling = np.convolve(
                    self.episode_lengths, np.ones(window) / window, mode="valid"
                )
                axs[0, 1].plot(
                    range(window - 1, len(self.episode_lengths)),
                    rolling,
                    "r-",
                    linewidth=2,
                )

        # Annealing progress
        if self.annealing_progress:
            axs[0, 2].plot(self.annealing_progress)
            axs[0, 2].set_title("Annealing Progress")
            axs[0, 2].set_xlabel("Episode")
            axs[0, 2].set_ylabel("Progress (0=loose, 1=tight)")
            axs[0, 2].set_ylim(-0.05, 1.05)
            axs[0, 2].axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Fully annealed')
            axs[0, 2].legend()
        else:
            axs[0, 2].text(0.5, 0.5, "No annealing data", ha="center", va="center")
            axs[0, 2].set_title("Annealing Progress")

        # Training loss
        if self.losses:
            axs[0, 3].plot(self.losses)
            axs[0, 3].set_title("Training Loss")
            axs[0, 3].set_xlabel("Update Step")
            axs[0, 3].set_ylabel("Loss")
        else:
            axs[0, 3].text(0.5, 0.5, "No loss data", ha="center", va="center")
            axs[0, 3].set_title("Training Loss")

        # Row 2: Constraints and Pareto fronts

        # Constraint thresholds over time
        ax_constraints = axs[1, 0]
        has_constraint_data = False
        if self.constraint_history.get("reflectivity"):
            ref_data = [(i, v) for i, v in enumerate(self.constraint_history["reflectivity"]) if v is not None]
            if ref_data:
                indices, values = zip(*ref_data)
                ax_constraints.scatter(indices, values, label="Reflectivity", color="blue", s=10, alpha=0.6)
                has_constraint_data = True
        if self.constraint_history.get("absorption"):
            abs_data = [(i, v) for i, v in enumerate(self.constraint_history["absorption"]) if v is not None]
            if abs_data:
                indices, values = zip(*abs_data)
                ax2 = ax_constraints.twinx()
                ax2.scatter(indices, values, label="Absorption", color="orange", s=10, alpha=0.6)
                ax2.set_ylabel("Absorption threshold", color="orange")
                ax2.tick_params(axis='y', labelcolor='orange')
        if has_constraint_data:
            ax_constraints.set_title("Constraint Thresholds (alternating)")
            ax_constraints.set_xlabel("Episode")
            ax_constraints.set_ylabel("Reflectivity threshold", color="blue")
            ax_constraints.tick_params(axis='y', labelcolor='blue')
            ax_constraints.legend(loc="upper left")
        else:
            ax_constraints.text(0.5, 0.5, "No constraint data", ha="center", va="center")
            ax_constraints.set_title("Constraint Thresholds")

        # Entropy coefficient over time
        if self.entropy_coef_history:
            axs[1, 1].plot(self.entropy_coef_history, alpha=0.8, linewidth=1.5)
            axs[1, 1].set_title("Entropy Coefficient")
            axs[1, 1].set_xlabel("Episode")
            axs[1, 1].set_ylabel("Entropy Coef")
            axs[1, 1].grid(True, alpha=0.3)
        else:
            axs[1, 1].text(0.5, 0.5, "No entropy data", ha="center", va="center")
            axs[1, 1].set_title("Entropy Coefficient")

        # Pareto front in VALUE space (objective space)
        ax_pareto = axs[1, 2]
        if pareto_designs:
            reflectivities = [d['vals'].get('reflectivity', 0) for d in pareto_designs]
            absorptions = [d['vals'].get('absorption', 0) for d in pareto_designs]
            # Convert to 1-reflectivity (loss)
            reflectivity_loss = [1 - r for r in reflectivities]
            ax_pareto.scatter(absorptions, reflectivity_loss, alpha=0.7, s=50, edgecolor='black')
            ax_pareto.set_xlabel('Absorption (ppm)')
            ax_pareto.set_ylabel('1 - Reflectivity')
            ax_pareto.set_title('Pareto Front (Value Space)')
            ax_pareto.set_xscale('log')
            ax_pareto.set_yscale('log')  # Log scale for 1-reflectivity
            sorted_front = sorted(pareto_designs, key=lambda x: x['vals'].get('absorption', float('inf')))
            sorted_abs = [d['vals'].get('absorption', 0) for d in sorted_front]
            sorted_ref_loss = [1 - d['vals'].get('reflectivity', 0) for d in sorted_front]
            ax_pareto.plot(sorted_abs, sorted_ref_loss, 'r--', alpha=0.5)
        else:
            ax_pareto.text(0.5, 0.5, "No Pareto data", ha="center", va="center")
            ax_pareto.set_title("Pareto Front (Value Space)")

        # Pareto front in REWARD space [0, 1] with all episode points
        ax_pareto_reward = axs[1, 3]
        if self.all_episode_rewards:
            # Plot all episode rewards as background (use subset if too many)
            all_rewards = self.all_episode_rewards
            if len(all_rewards) > 2000:
                indices = np.linspace(0, len(all_rewards) - 1, 2000, dtype=int)
                all_rewards = [all_rewards[i] for i in indices]

            all_ref = [r[0] for r in all_rewards]
            all_abs = [r[1] for r in all_rewards]
            ax_pareto_reward.scatter(all_abs, all_ref, alpha=0.15, s=15, c='gray', label='All episodes')

            # Plot Pareto front in REWARD space
            if pareto_designs:
                pareto_ref = [d['reward_vals'].get('reflectivity', 0) for d in pareto_designs if d['reward_vals']]
                pareto_abs = [d['reward_vals'].get('absorption', 0) for d in pareto_designs if d['reward_vals']]

                if pareto_ref and pareto_abs:
                    ax_pareto_reward.scatter(pareto_abs, pareto_ref, alpha=0.9, s=60, c='red', edgecolor='black', label='Pareto front (reward)')

                    # Sort and draw Pareto front line
                    sorted_indices = np.argsort(pareto_abs)
                    sorted_abs_r = [pareto_abs[i] for i in sorted_indices]
                    sorted_ref_r = [pareto_ref[i] for i in sorted_indices]
                    ax_pareto_reward.plot(sorted_abs_r, sorted_ref_r, 'r-', linewidth=2, alpha=0.7)

            ax_pareto_reward.set_xlabel('Absorption Reward (normalised)')
            ax_pareto_reward.set_ylabel('Reflectivity Reward (normalised)')
            ax_pareto_reward.set_title('Pareto Front (Reward Space)')
            ax_pareto_reward.legend(loc='lower left', fontsize=8)
        else:
            ax_pareto_reward.text(0.5, 0.5, "No episode data", ha="center", va="center")
            ax_pareto_reward.set_title("Pareto Front (Reward Space)")

        # Row 3 (optional): Action distributions for discrete training
        if self.track_action_distributions:
            # Material action distribution
            if hasattr(self, 'material_action_history') and self.material_action_history:
                recent_materials = self.material_action_history[-1000:]
                unique, counts = np.unique(recent_materials, return_counts=True)
                axs[2, 0].bar(unique, counts, edgecolor="black", alpha=0.7)
                axs[2, 0].set_title("Material Selection (recent 1000 steps)")
                axs[2, 0].set_xlabel("Material Index")
                axs[2, 0].set_ylabel("Count")
            else:
                axs[2, 0].text(0.5, 0.5, "No material data", ha="center", va="center")
                axs[2, 0].set_title("Material Selection")

            # Thickness bin distribution
            if hasattr(self, 'thickness_bin_history') and self.thickness_bin_history:
                recent_bins = self.thickness_bin_history[-1000:]
                unique, counts = np.unique(recent_bins, return_counts=True)
                axs[2, 1].bar(unique, counts, edgecolor="black", alpha=0.7)
                axs[2, 1].set_title("Thickness Bin Selection (recent 1000 steps)")
                axs[2, 1].set_xlabel("Thickness Bin Index")
                axs[2, 1].set_ylabel("Count")
            else:
                axs[2, 1].text(0.5, 0.5, "No thickness data", ha="center", va="center")
                axs[2, 1].set_title("Thickness Bin Selection")

            # Material selection over time
            if hasattr(self, 'material_action_history') and len(self.material_action_history) > 100:
                window = 100
                n_windows = len(self.material_action_history) // window
                if n_windows > 1:
                    material_over_time = []
                    for i in range(n_windows):
                        window_data = self.material_action_history[i*window:(i+1)*window]
                        material_over_time.append(np.mean(window_data))
                    axs[2, 2].plot(np.arange(n_windows) * window, material_over_time)
                    axs[2, 2].set_title(f"Avg Material Idx (window={window})")
                    axs[2, 2].set_xlabel("Step")
                    axs[2, 2].set_ylabel("Avg Material Index")
                else:
                    axs[2, 2].text(0.5, 0.5, "Not enough data", ha="center", va="center")
                    axs[2, 2].set_title("Material Over Time")
            else:
                axs[2, 2].text(0.5, 0.5, "Not enough data", ha="center", va="center")
                axs[2, 2].set_title("Material Over Time")

            # Thickness bin over time
            if hasattr(self, 'thickness_bin_history') and len(self.thickness_bin_history) > 100:
                window = 100
                n_windows = len(self.thickness_bin_history) // window
                if n_windows > 1:
                    thickness_over_time = []
                    for i in range(n_windows):
                        window_data = self.thickness_bin_history[i*window:(i+1)*window]
                        thickness_over_time.append(np.mean(window_data))
                    axs[2, 3].plot(np.arange(n_windows) * window, thickness_over_time)
                    axs[2, 3].set_title(f"Avg Thickness Bin (window={window})")
                    axs[2, 3].set_xlabel("Step")
                    axs[2, 3].set_ylabel("Avg Thickness Bin")
                else:
                    axs[2, 3].text(0.5, 0.5, "Not enough data", ha="center", va="center")
                    axs[2, 3].set_title("Thickness Over Time")
            else:
                axs[2, 3].text(0.5, 0.5, "Not enough data", ha="center", va="center")
                axs[2, 3].set_title("Thickness Over Time")

        plt.tight_layout()
        save_path = self.save_dir / "training_progress.png"
        plt.savefig(save_path)
        plt.close(fig)

        # Print summary if verbose
        if self.verbose and pareto_designs:
            sorted_front = sorted(pareto_designs, key=lambda x: x['vals'].get('reflectivity', 0), reverse=True)
            best = sorted_front[0]
            progress = self.annealing_progress[-1] if self.annealing_progress else 0
            phase = self.phase_history[-1] if self.phase_history else "unknown"
            print(
                f"[{self.num_timesteps}] Pareto designs: {len(pareto_designs)}, "
                f"Best: R={best['vals'].get('reflectivity', 0):.6f}, "
                f"A={best['vals'].get('absorption', 0):.2e}, "
                f"layers={best['n_layers']}, anneal={progress:.1%}, phase={phase}"
            )

    def _plot_best_designs(self):
        """Plot coating stack structure for Pareto front designs."""
        # Get environment
        env = self.env
        if hasattr(env, 'env'):
            env = env.env

        # Use value space for plotting (visual diagnostics)
        pareto_front_values = env.get_pareto_front(space="value") if hasattr(env, 'get_pareto_front') else []
        if not pareto_front_values:
            return

        # Convert to design format for plotting
        designs_with_state = []
        for obj_vector, state in pareto_front_values:
            vals = {}
            if hasattr(env, 'optimise_parameters'):
                for i, param_name in enumerate(env.optimise_parameters):
                    if i < len(obj_vector):
                        vals[param_name] = obj_vector[i]

            state_array = state.get_array()
            n_layers = np.sum(state_array[:, 0] > 1e-12)

            design = {
                "vals": vals,
                "state_array": state_array,
                "n_layers": n_layers
            }
            designs_with_state.append(design)

        if not designs_with_state:
            return

        n_designs = len(designs_with_state)
        fig, axs = plt.subplots(1, n_designs, figsize=(4 * n_designs, 6), squeeze=False)

        for i, design in enumerate(designs_with_state):
            ax = axs[0, i]
            state_array = design["state_array"]
            vals = design["vals"]

            self._plot_single_stack(ax, state_array, vals, rank=i + 1)

        plt.tight_layout()
        save_path = self.save_dir / "pareto_front_designs.png"
        plt.savefig(save_path)
        plt.close(fig)

    def _plot_single_stack(self, ax, state_array, vals, rank: int):
        """Plot a single coating stack as vertical bars."""
        # state_array shape: (n_layers, n_materials + 1 + ...)
        # Column 0: thickness, Columns 1+: one-hot material

        # Filter to active layers (thickness > 0)
        thicknesses = state_array[:, 0]
        active_mask = thicknesses > 1e-12
        active_thicknesses = thicknesses[active_mask]

        # Check if using optical thickness (very small values ~0.1-0.5)
        # or physical thickness (larger values ~10e-9 to 500e-9)
        if len(active_thicknesses) > 0 and active_thicknesses.max() < 10:
            # Optical thickness, keep as is
            thickness_label = "Optical Thickness"
        else:
            # Physical thickness, convert to nm
            active_thicknesses = active_thicknesses * 1e9
            thickness_label = "Thickness (nm)"

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
                label=mat_name if layer_idx == 0 or mat_idx not in material_indices[:layer_idx] else "",
            )
            y_pos += thickness

        # Labels and title
        R = vals.get("reflectivity", 0)
        A = vals.get("absorption", 0)
        ax.set_title(f"Rank {rank}\nR={R:.6f}, A={A:.3e}")
        ax.set_ylabel(thickness_label)
        ax.set_xticks([])
        ax.set_xlim(-0.5, 0.5)

        # Legend with unique materials
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label:
            ax.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=8)

    def save_pareto_front_to_csv(self, filename: str = "pareto_front.csv"):
        """Save Pareto front from environment to CSV file.

        Args:
            filename: Name of CSV file to save
        """
        import pandas as pd
        from pathlib import Path

        # Build filepath
        filepath = Path(filename)
        if not filepath.is_absolute():
            filepath = self.save_dir / filename

        # Get environment
        env = self.env
        if hasattr(env, 'env'):
            env = env.env

        # Use value space for CSV output (actual objective values)
        pareto_front_values = env.get_pareto_front(space="value") if hasattr(env, 'get_pareto_front') else []

        if not pareto_front_values:
            print("No Pareto front data to save")
            return

        # Convert to CSV format
        data = []
        for obj_vector, state in pareto_front_values:
            row = {}

            # Add objective values
            if hasattr(env, 'optimise_parameters'):
                for i, param_name in enumerate(env.optimise_parameters):
                    if i < len(obj_vector):
                        row[param_name] = obj_vector[i]

            # Add state information
            state_array = state.get_array()
            thicknesses = state_array[:, 0]
            material_indices = state_array[:, 1].astype(int)

            # Filter active layers
            active_mask = thicknesses > 1e-12
            active_thicknesses = thicknesses[active_mask]
            active_materials = material_indices[active_mask]

            row["n_layers"] = len(active_thicknesses)
            row["thicknesses"] = ",".join(f"{t:.6f}" for t in active_thicknesses)
            row["materials"] = ",".join(map(str, active_materials))

            data.append(row)

        df = pd.DataFrame(data)

        # Sort by absorption if available (lower is better)
        if "absorption" in df.columns:
            df = df.sort_values("absorption", ascending=True)
        elif "reflectivity" in df.columns:
            df = df.sort_values("reflectivity", ascending=False)

        df.to_csv(filepath, index=False)
        print(f"Saved Pareto front ({len(df)} designs) to {filepath}")
