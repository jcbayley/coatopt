"""Shared callback classes for CoatOpt experiments."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


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
    ):
        """Initialize entropy annealing callback.

        Args:
            max_ent: Maximum entropy coefficient (high exploration)
            min_ent: Minimum entropy coefficient (low exploration)
            epochs_per_step: If provided, reset annealing every N episodes (for cycling).
                            If None, anneal once over entire training.
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.max_ent = max_ent
        self.min_ent = min_ent
        self.epochs_per_step = epochs_per_step
        self.episode_count = 0

    def _on_step(self) -> bool:
        """Update entropy coefficient at each step."""
        # Track episodes
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info and isinstance(info["episode"], dict):
                self.episode_count += 1

        # Compute current entropy coefficient
        if self.epochs_per_step is not None:
            # Cycling mode: reset every epochs_per_step episodes
            episode_in_cycle = self.episode_count % self.epochs_per_step
            progress = episode_in_cycle / self.epochs_per_step
        else:
            # Single annealing mode: use training progress
            progress = 1.0 - (self.num_timesteps / self.model._total_timesteps)

        # Cosine annealing formula
        ent_coef = self.min_ent + 0.5 * (self.max_ent - self.min_ent) * (
            1 + math.cos(math.pi * progress)
        )

        # Update model's entropy coefficient
        self.model.ent_coef = ent_coef

        # Log to tensorboard if available
        if self.verbose > 0 and self.episode_count % 100 == 0:
            print(f"Episode {self.episode_count}: ent_coef = {ent_coef:.4f}")

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

        # Pareto front tracking in VALUE space (reflectivity, absorption)
        self.pareto_front_values = []

        # Pareto front tracking in REWARD space (normalized rewards)
        self.pareto_front_rewards = []

        # Track all episode rewards (normalised) for both objectives
        self.all_episode_rewards = []  # List of (ref_reward, abs_reward) tuples

        # Annealing/schedule tracking
        self.annealing_progress = []
        self.constraint_history = {obj: [] for obj in ["reflectivity", "absorption"]}
        self.target_objective_history = []
        self.phase_history = []
        self.level_history = []

        # Action distribution tracking (only for discrete)
        if track_action_distributions:
            self.material_action_history = []
            self.thickness_bin_history = []

    def _is_dominated(self, vals1: dict, vals2: dict) -> bool:
        """Check if vals1 dominates vals2 in multi-objective space.

        For reflectivity: higher is better
        For absorption: lower is better
        """
        better_or_equal = True
        strictly_better = False

        for obj in ["reflectivity", "absorption"]:
            v1 = vals1.get(obj, float('-inf') if obj == "reflectivity" else float('inf'))
            v2 = vals2.get(obj, float('-inf') if obj == "reflectivity" else float('inf'))

            if obj == "reflectivity":
                if v1 < v2:
                    better_or_equal = False
                elif v1 > v2:
                    strictly_better = True
            elif obj == "absorption":
                if v1 > v2:
                    better_or_equal = False
                elif v1 < v2:
                    strictly_better = True

        return better_or_equal and strictly_better

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

                # Update Pareto fronts (both value and reward space)
                vals = info.get("vals", {})
                rewards = info.get("rewards", {})
                state_array = info.get("state_array", None)
                if vals:
                    # Update value space Pareto front
                    self._update_pareto_front_values(vals, ep_length, state_array)

                    # Track normalized rewards if available in info
                    if rewards:
                        ref_reward = rewards.get('reflectivity', 0.0)
                        abs_reward = rewards.get('absorption', 0.0)
                        self.all_episode_rewards.append((ref_reward, abs_reward))

                        # Update reward space Pareto front
                        reward_vals = {'reflectivity': ref_reward, 'absorption': abs_reward}
                        self._update_pareto_front_rewards(reward_vals, vals, ep_length, state_array)

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

    def _update_pareto_front_values(self, vals: dict, n_layers: int, state_array):
        """Update VALUE space Pareto front with new design (no duplicates, only non-dominated)."""
        new_r = vals.get('reflectivity', 0)
        new_a = vals.get('absorption', float('inf'))

        # Check for duplicate (same R and A within tolerance)
        tol = 1e-6
        for d in self.pareto_front_values:
            existing_r = d['vals'].get('reflectivity', 0)
            existing_a = d['vals'].get('absorption', float('inf'))
            if abs(new_r - existing_r) < tol and abs(new_a - existing_a) < tol:
                return  # Duplicate, skip

        design_info = {
            "vals": vals.copy(),
            "n_layers": n_layers,
            "timestep": self.num_timesteps,
            "episode": self.episode_count,
            "state_array": state_array.copy() if state_array is not None else None,
        }

        new_vals = design_info["vals"]

        # Remove points dominated by the new point
        self.pareto_front_values = [d for d in self.pareto_front_values if not self._is_dominated(new_vals, d["vals"])]

        # Add new point if not dominated by any existing point
        if not any(self._is_dominated(d["vals"], new_vals) for d in self.pareto_front_values):
            self.pareto_front_values.append(design_info)

    def _update_pareto_front_rewards(self, reward_vals: dict, original_vals: dict, n_layers: int, state_array):
        """Update REWARD space Pareto front with new design (no duplicates, only non-dominated).

        Args:
            reward_vals: Normalized reward values {'reflectivity': ref_reward, 'absorption': abs_reward}
            original_vals: Original objective values (for storage)
            n_layers: Number of layers
            state_array: State array
        """
        new_r = reward_vals.get('reflectivity', 0)
        new_a = reward_vals.get('absorption', 0)

        # Check for duplicate (same normalized rewards within tolerance)
        tol = 1e-6
        for d in self.pareto_front_rewards:
            existing_r = d['reward_vals'].get('reflectivity', 0)
            existing_a = d['reward_vals'].get('absorption', 0)
            if abs(new_r - existing_r) < tol and abs(new_a - existing_a) < tol:
                return  # Duplicate, skip

        design_info = {
            "reward_vals": reward_vals.copy(),  # Normalized rewards
            "vals": original_vals.copy(),  # Original objective values
            "n_layers": n_layers,
            "timestep": self.num_timesteps,
            "episode": self.episode_count,
            "state_array": state_array.copy() if state_array is not None else None,
        }

        new_rewards = design_info["reward_vals"]

        # For reward space, higher is better for BOTH objectives (they're both normalized to [0,1])
        # So we use a modified dominance check
        def _is_dominated_rewards(vals1, vals2):
            """Check if vals1 dominates vals2 in reward space (higher is better for both)."""
            r1_ref = vals1.get('reflectivity', 0)
            r1_abs = vals1.get('absorption', 0)
            r2_ref = vals2.get('reflectivity', 0)
            r2_abs = vals2.get('absorption', 0)

            # vals1 dominates vals2 if it's >= in both and > in at least one
            better_or_equal = (r1_ref >= r2_ref) and (r1_abs >= r2_abs)
            strictly_better = (r1_ref > r2_ref) or (r1_abs > r2_abs)
            return better_or_equal and strictly_better

        # Remove points dominated by the new point
        self.pareto_front_rewards = [d for d in self.pareto_front_rewards if not _is_dominated_rewards(new_rewards, d["reward_vals"])]

        # Add new point if not dominated by any existing point
        if not any(_is_dominated_rewards(d["reward_vals"], new_rewards) for d in self.pareto_front_rewards):
            self.pareto_front_rewards.append(design_info)

    def _plot_training_progress(self):
        """Plot training metrics."""
        n_rows = 3 if self.track_action_distributions else 2
        fig, axs = plt.subplots(n_rows, 4, figsize=(20, 5 * n_rows))

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

        # Reward distribution (recent episodes)
        if len(self.episode_rewards) > 10:
            recent = self.episode_rewards[-100:]
            axs[1, 1].hist(recent, bins=20, edgecolor="black", alpha=0.7)
            axs[1, 1].set_title(f"Recent Reward Distribution (last {len(recent)})")
            axs[1, 1].set_xlabel("Reward")
            axs[1, 1].set_ylabel("Count")
        else:
            axs[1, 1].text(0.5, 0.5, "Not enough data", ha="center", va="center")
            axs[1, 1].set_title("Reward Distribution")

        # Pareto front in VALUE space (objective space)
        ax_pareto = axs[1, 2]
        if self.pareto_front_values:
            reflectivities = [d['vals'].get('reflectivity', 0) for d in self.pareto_front_values]
            absorptions = [d['vals'].get('absorption', 0) for d in self.pareto_front_values]
            # Convert to 1-reflectivity (loss)
            reflectivity_loss = [1 - r for r in reflectivities]
            ax_pareto.scatter(absorptions, reflectivity_loss, alpha=0.7, s=50, edgecolor='black')
            ax_pareto.set_xlabel('Absorption (ppm)')
            ax_pareto.set_ylabel('1 - Reflectivity')
            ax_pareto.set_title('Pareto Front (Value Space)')
            ax_pareto.set_xscale('log')
            ax_pareto.set_yscale('log')  # Log scale for 1-reflectivity
            sorted_front = sorted(self.pareto_front_values, key=lambda x: x['vals'].get('absorption', float('inf')))
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

            # Plot Pareto front in REWARD space (use reward space Pareto front)
            if self.pareto_front_rewards:
                pareto_ref = [d['reward_vals'].get('reflectivity', 0) for d in self.pareto_front_rewards]
                pareto_abs = [d['reward_vals'].get('absorption', 0) for d in self.pareto_front_rewards]

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
        if self.verbose and self.pareto_front_values:
            sorted_front = sorted(self.pareto_front_values, key=lambda x: x['vals'].get('reflectivity', 0), reverse=True)
            best = sorted_front[0]
            progress = self.annealing_progress[-1] if self.annealing_progress else 0
            phase = self.phase_history[-1] if self.phase_history else "unknown"
            print(
                f"[{self.num_timesteps}] Pareto (val): {len(self.pareto_front_values)}, Pareto (reward): {len(self.pareto_front_rewards)}, "
                f"Best: R={best['vals'].get('reflectivity', 0):.6f}, "
                f"A={best['vals'].get('absorption', 0):.2e}, "
                f"layers={best['n_layers']}, anneal={progress:.1%}, phase={phase}"
            )

    def _plot_best_designs(self):
        """Plot coating stack structure for Pareto front designs (value space)."""
        designs_with_state = [d for d in self.pareto_front_values if d["state_array"] is not None]
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
        if self.pareto_front_values:
            values_filepath = base_dir / f"{base_name}_values.csv"
            data = []
            for design in self.pareto_front_values:
                row = {
                    "episode": design.get("episode", -1),
                    "timestep": design.get("timestep", -1),
                    "n_layers": design.get("n_layers", -1),
                }

                # Add objective values
                vals = design.get("vals", {})
                for key, value in vals.items():
                    row[key] = value

                # Add state array information
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

                data.append(row)

            df = pd.DataFrame(data)
            if "absorption" in df.columns:
                df = df.sort_values("absorption", ascending=True)

            df.to_csv(values_filepath, index=False)
            print(f"Saved VALUE space Pareto front ({len(df)} designs) to {values_filepath}")
            saved_files.append(values_filepath)
        else:
            print("No VALUE space Pareto front data to save")

        # === Save REWARD space Pareto front ===
        if self.pareto_front_rewards:
            rewards_filepath = base_dir / f"{base_name}_rewards.csv"
            data = []
            for design in self.pareto_front_rewards:
                row = {
                    "episode": design.get("episode", -1),
                    "timestep": design.get("timestep", -1),
                    "n_layers": design.get("n_layers", -1),
                }

                # Add reward values
                reward_vals = design.get("reward_vals", {})
                row["reflectivity_reward"] = reward_vals.get("reflectivity", 0)
                row["absorption_reward"] = reward_vals.get("absorption", 0)

                # Also add original objective values for reference
                vals = design.get("vals", {})
                row["reflectivity"] = vals.get("reflectivity", 0)
                row["absorption"] = vals.get("absorption", 0)

                # Add state array information
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

                data.append(row)

            df = pd.DataFrame(data)
            if "absorption_reward" in df.columns:
                df = df.sort_values("absorption_reward", ascending=True)

            df.to_csv(rewards_filepath, index=False)
            print(f"Saved REWARD space Pareto front ({len(df)} designs) to {rewards_filepath}")
            saved_files.append(rewards_filepath)
        else:
            print("No REWARD space Pareto front data to save")

        return saved_files
