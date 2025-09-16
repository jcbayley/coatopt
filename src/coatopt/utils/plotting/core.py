"""
Shared plotting utilities for both CLI and UI training visualization.

This module provides consistent plotting functionality that can be used
by both the command-line interface and the GUI interface.
"""

import os
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


class TrainingPlotManager:
    """Manager for training plots that can be used by both CLI and UI."""

    def __init__(
        self,
        save_plots: bool = True,
        output_dir: str = None,
        ui_mode: bool = False,
        figure_size: Tuple[int, int] = (12, 10),
    ):
        """
        Initialize plot manager.

        Args:
            save_plots: Whether to save plots to disk
            output_dir: Directory to save plots (required if save_plots=True)
            ui_mode: Whether running in UI mode (affects backend and display)
            figure_size: Size of figures (width, height)
        """
        self.save_plots = save_plots
        self.output_dir = output_dir
        self.ui_mode = ui_mode
        self.figure_size = figure_size

        # Data storage
        self.training_data = []
        self.pareto_data = []
        self.eval_data = {}
        self.historical_pareto_data = {}

        # Coating state storage for UI interactivity
        self.coating_states = (
            {}
        )  # Maps episode -> list of coating states for Pareto points
        self.eval_coating_states = []  # Coating states for evaluation Pareto points

        # Objective information
        self.objective_labels = []
        self.objective_scales = []
        self.optimisation_parameters = []
        self.objective_targets = {}
        self.design_criteria = {}

        # Plot update throttling
        self.last_plot_update = 0
        self.plot_update_interval = 40

        # Preference constraints tracking
        self.constraint_data = []

        # Initialize figures
        self._init_figures()

    def _init_figures(self):
        """Initialize matplotlib figures."""
        if not self.ui_mode:
            # For CLI, use Agg backend for file saving
            matplotlib.use("Agg")

        # Create figures
        self.rewards_fig = Figure(figsize=self.figure_size, dpi=100)
        self.values_fig = Figure(figsize=self.figure_size, dpi=100)
        self.pareto_fig = Figure(figsize=(14, 10), dpi=100)
        self.pareto_rewards_fig = Figure(figsize=(14, 10), dpi=100)
        self.constraints_fig = Figure(figsize=(14, 8), dpi=100)

        self._init_subplot_layout()

    def _init_subplot_layout(self):
        """Initialize subplot layouts."""
        # Rewards plot with multiple subplots
        self.rewards_axes = self.rewards_fig.subplots(3, 2)
        self.rewards_axes = self.rewards_axes.flatten()

        rewards_subplot_titles = [
            "Total Reward",
            "Individual Rewards",
            "Entropy Weight (β)",
            "Learning Rates",
            "Objective Weights",
            "Training Losses",
        ]

        for i, ax in enumerate(self.rewards_axes):
            ax.set_title(
                rewards_subplot_titles[i]
                if i < len(rewards_subplot_titles)
                else f"Plot {i+1}"
            )
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("Episode")

        self.rewards_fig.tight_layout()

        # Physical values plot
        self.values_axes = self.values_fig.subplots(2, 2)
        self.values_axes = self.values_axes.flatten()

        values_subplot_titles = [
            "Reflectivity",
            "Thermal Noise",
            "Absorption",
            "Thickness",
        ]

        for i, ax in enumerate(self.values_axes):
            ax.set_title(
                values_subplot_titles[i]
                if i < len(values_subplot_titles)
                else f"Value {i+1}"
            )
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("Episode")

        self.values_fig.tight_layout()

        # Pareto front plots - will be dynamically created
        self._init_pareto_plot()
        self._init_pareto_rewards_plot()

    def _init_pareto_plot(self):
        """Initialize Pareto front plot."""
        self.pareto_fig.clear()
        ax = self.pareto_fig.add_subplot(1, 1, 1)
        ax.text(
            0.5,
            0.5,
            "Load configuration to see Pareto front visualization",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
        )
        ax.set_title("Pareto Front Visualization")
        self.pareto_fig.tight_layout()

    def _init_pareto_rewards_plot(self):
        """Initialize Pareto rewards plot."""
        self.pareto_rewards_fig.clear()
        ax = self.pareto_rewards_fig.add_subplot(1, 1, 1)
        ax.text(
            0.5,
            0.5,
            "Load configuration to see Pareto rewards visualization",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
        )
        ax.set_title("Pareto Front - Objective Rewards")
        self.pareto_rewards_fig.tight_layout()

    def set_objective_info(
        self,
        optimisation_parameters: List[str],
        objective_targets: Optional[List[float]] = None,
        design_criteria: Optional[List[float]] = None,
    ):
        """Set objective information for consistent labeling."""
        self.optimisation_parameters = optimisation_parameters.copy()
        self.objective_targets = objective_targets if objective_targets else {}
        self.design_criteria = design_criteria if design_criteria else {}
        self.objective_labels = self._get_objective_labels()
        self.objective_scales = self._get_objective_scales()

    def _get_objective_labels(self):
        """Get human-readable labels for objectives."""
        if not self.optimisation_parameters:
            return []

        label_mapping = {
            "reflectivity": "1 - Reflectivity",
            "absorption": "Absorption [ppm]",
            "thermal_noise": "Thermal Noise [m/√Hz]",
            "thickness": "Total Thickness [nm]",
        }

        return [
            label_mapping.get(param, param.replace("_", " ").title())
            for param in self.optimisation_parameters
        ]

    def _get_objective_scales(self):
        """Get appropriate scales (linear/log) for each objective."""
        if not self.optimisation_parameters:
            return []

        scale_mapping = {
            "reflectivity": "log",
            "absorption": "log",
            "thermal_noise": "log",
            "thickness": "linear",
        }

        return [
            scale_mapping.get(param, "linear") for param in self.optimisation_parameters
        ]

    def add_training_data(self, episode_data: Dict[str, Any]):
        """Add training data for plotting."""
        self.training_data.append(episode_data)

    def add_pareto_data(self, pareto_data: Dict[str, Any]):
        """Add Pareto front data for plotting."""
        self.pareto_data.append(pareto_data)

        # Store in historical data
        episode = pareto_data.get("episode", 0)
        self.historical_pareto_data[episode] = pareto_data.copy()

        # Store pareto states if provided (these correspond to the Pareto points)
        if "pareto_states" in pareto_data:
            pareto_states = pareto_data["pareto_states"]
            self.coating_states[episode] = pareto_states
        else:
            print(f"Warning: No pareto_states in pareto_data for episode {episode}")

    def add_eval_pareto_data(self, eval_data: Dict[str, Any]):
        """Add evaluation Pareto data for plotting."""
        # If running in UI mode, update plots immediately
        self.eval_data = eval_data

        # Store evaluation pareto states if provided
        if "pareto_states" in eval_data:
            self.eval_coating_states = eval_data["pareto_states"]

    def should_update_plots(self, current_episode: int) -> bool:
        """Check if plots should be updated based on throttling."""
        return current_episode - self.last_plot_update >= self.plot_update_interval

    def load_context_data(self, checkpoint_manager) -> bool:
        """
        Load training context data directly into plot manager.

        Args:
            checkpoint_manager: TrainingCheckpointManager instance

        Returns:
            bool: True if data was successfully loaded, False otherwise
        """
        if not checkpoint_manager:
            return False

        try:
            if not hasattr(checkpoint_manager, "checkpoint_path") or not os.path.exists(
                checkpoint_manager.checkpoint_path
            ):
                print("No checkpoint found for loading context data")
                return False

            # Load context directly
            context = checkpoint_manager.load_context()

            if not context or context.training_metrics.empty:
                print("No context data available to load")
                return False

            # Load training metrics
            for _, row in context.training_metrics.iterrows():
                episode_data = {
                    "episode": int(row.get("episode", 0)),
                    "reward": float(row.get("reward", 0.0)),
                    "metrics": {
                        key: float(val)
                        for key, val in row.items()
                        if key not in ["episode", "reward"] and pd.notna(val)
                    },
                }
                self.add_training_data(episode_data)

            # Load current Pareto front data (only the most recent front, not historical)
            if len(context.pareto_front_values) > 0:
                # Create current Pareto data entry with full context data
                current_pareto_data = {
                    "episode": context.current_episode,
                    "pareto_front_values": context.pareto_front_values,
                    "pareto_front_rewards": context.pareto_front_rewards,
                    "pareto_states": context.pareto_states,
                    "all_values": np.array(context.all_values),
                    "all_rewards": np.array(context.all_rewards),
                }
                self.add_pareto_data(current_pareto_data)

            # Load constraint history for preference-constrained training
            if context.constraint_history:
                print(
                    f"[DEBUG] Loading {len(context.constraint_history)} constraint entries from context"
                )
                for i, entry in enumerate(context.constraint_history):
                    print(
                        f"[DEBUG] Constraint entry {i}: episode={entry.get('episode')}, phase={entry.get('phase')}"
                    )
                    self.add_constraint_data(
                        episode=entry["episode"],
                        phase=entry["phase"],
                        target_objective=entry.get("target_objective"),
                        constraints=entry.get("constraints", {}),
                        reward_bounds=entry.get("reward_bounds", {}),
                        constraint_step=entry.get("constraint_step", 0),
                    )
                print(
                    f"Loaded {len(context.constraint_history)} constraint history entries"
                )
            else:
                print("[DEBUG] No constraint history in context")

            print(
                f"Loaded context data: {len(self.training_data)} episodes, "
                f"Pareto front size: {len(context.pareto_front_values)}"
            )
            return True

        except Exception as e:
            print(f"Warning: Failed to load context data: {e}")
            return False

    def update_all_plots(self, current_episode: int = None):
        """Update all plots."""
        if current_episode and not self.should_update_plots(current_episode):
            return

        self.update_rewards_plot()
        self.update_values_plot()
        self.update_pareto_plot()
        self.update_pareto_rewards_plot()
        self.update_constraints_plot()

        if current_episode:
            self.last_plot_update = current_episode

        if self.save_plots and self.output_dir:
            self.save_plots_to_disk()

    def add_constraint_data(
        self,
        episode: int,
        phase: int,
        target_objective: str = None,
        constraints: Dict = None,
        reward_bounds: Dict = None,
        constraint_step: int = 0,
    ):
        """
        Add constraint tracking data for preference-constrained training.

        Args:
            episode: Current training episode
            phase: Training phase (1 or 2)
            target_objective: Currently optimized objective (Phase 2 only)
            constraints: Active constraint thresholds (Phase 2 only)
            reward_bounds: Current reward min/max bounds
            constraint_step: Current constraint step (Phase 2 only)
        """
        constraint_entry = {
            "episode": episode,
            "phase": phase,
            "target_objective": target_objective,
            "constraints": constraints.copy() if constraints else {},
            "reward_bounds": reward_bounds.copy() if reward_bounds else {},
            "constraint_step": constraint_step,
        }
        self.constraint_data.append(constraint_entry)

    def update_rewards_plot(self):
        """Update the comprehensive rewards plot."""
        if not self.training_data:
            return

        try:
            # Convert training data to DataFrame
            df_data = []
            for d in self.training_data:
                row = {"episode": d["episode"], "reward": d["reward"]}
                if "metrics" in d:
                    row.update(d["metrics"])
                df_data.append(row)

            if not df_data:
                return

            df = pd.DataFrame(df_data)
            episodes = df["episode"].values
            window_size = min(20, len(episodes) // 4) if len(episodes) > 20 else 5

            # Clear all axes
            for ax in self.rewards_axes:
                ax.clear()
                ax.grid(True, alpha=0.3)

            # 1. Total Reward
            self._plot_metric_with_smoothing(
                self.rewards_axes[0], df, "reward", "Total Reward", window_size
            )

            # 2. Individual Reward Components
            optimise_params = (
                self.optimisation_parameters
                if self.optimisation_parameters
                else ["reflectivity", "thermal_noise", "absorption", "thickness"]
            )

            reward_components = []
            for param in optimise_params:
                possible_names = [
                    f"{param}_reward",
                    f'{param.replace("thermal_noise", "thermalnoise")}_reward',
                ]

                for name in possible_names:
                    if name in df.columns:
                        reward_components.append(name)
                        break

            for component in reward_components:
                if component in df.columns:
                    label = (
                        component.replace("_reward", "")
                        .replace("thermalnoise", "thermal_noise")
                        .replace("_", " ")
                        .title()
                    )
                    self.rewards_axes[1].plot(
                        episodes, df[component], alpha=0.6, label=label
                    )

            self.rewards_axes[1].set_title("Individual Reward Components")
            self.rewards_axes[1].set_ylabel("Reward")
            self.rewards_axes[1].legend(fontsize=8)

            # 3. Entropy Weights (Beta)
            entropy_plotted = False

            # Check for separate discrete and continuous entropy weights
            if "beta_discrete" in df.columns and "beta_continuous" in df.columns:
                self.rewards_axes[2].plot(
                    episodes,
                    df["beta_discrete"],
                    "blue",
                    linewidth=2,
                    label="Discrete",
                    alpha=0.8,
                )
                self.rewards_axes[2].plot(
                    episodes,
                    df["beta_continuous"],
                    "red",
                    linewidth=2,
                    label="Continuous",
                    alpha=0.8,
                )
                self.rewards_axes[2].set_title("Entropy Weights (β)")
                self.rewards_axes[2].set_ylabel("Beta")
                self.rewards_axes[2].legend(fontsize=8)
                entropy_plotted = True
            elif "beta" in df.columns:
                # Fallback to original single entropy weight
                self.rewards_axes[2].plot(
                    episodes, df["beta"], "purple", linewidth=2, label="Shared"
                )
                self.rewards_axes[2].set_title("Entropy Weight (β)")
                self.rewards_axes[2].set_ylabel("Beta")
                entropy_plotted = True

            # If neither are available, try to detect individual entropy columns
            if not entropy_plotted:
                entropy_found = False
                if "entropy_discrete" in df.columns:
                    self.rewards_axes[2].plot(
                        episodes,
                        df["entropy_discrete"],
                        "blue",
                        linewidth=2,
                        label="Discrete",
                        alpha=0.8,
                    )
                    entropy_found = True
                if "entropy_continuous" in df.columns:
                    self.rewards_axes[2].plot(
                        episodes,
                        df["entropy_continuous"],
                        "red",
                        linewidth=2,
                        label="Continuous",
                        alpha=0.8,
                    )
                    entropy_found = True

                if entropy_found:
                    self.rewards_axes[2].set_title("Entropy Weights (β)")
                    self.rewards_axes[2].set_ylabel("Beta")
                    self.rewards_axes[2].legend(fontsize=8)

            # 4. Learning Rates
            lr_components = ["lr_discrete", "lr_continuous", "lr_value"]
            for lr_comp in lr_components:
                if lr_comp in df.columns:
                    self.rewards_axes[3].plot(
                        episodes,
                        df[lr_comp],
                        alpha=0.8,
                        label=lr_comp.replace("lr_", "").title(),
                    )
            self.rewards_axes[3].set_title("Learning Rates")
            self.rewards_axes[3].set_ylabel("Learning Rate")
            self.rewards_axes[3].legend(fontsize=8)
            self.rewards_axes[3].set_yscale("log")

            # 5. Objective Weights
            weight_components = []
            for param in optimise_params:
                possible_names = [
                    f"{param}_reward_weights",
                    f"{param}_weights",
                    f'{param.replace("_", "")}_reward_weights',
                    f'{param.replace("thermal_noise", "thermalnoise")}_reward_weights',
                ]

                for name in possible_names:
                    if name in df.columns:
                        weight_components.append((name, param))
                        break

            for weight_comp, param in weight_components:
                label = param.replace("_", " ").title()
                self.rewards_axes[4].plot(
                    episodes, df[weight_comp], alpha=0.8, label=label
                )

            self.rewards_axes[4].set_title("Objective Weights")
            self.rewards_axes[4].set_ylabel("Weight")
            self.rewards_axes[4].legend(fontsize=8)

            # 6. Training Losses
            loss_components = [
                "loss_policy_discrete",
                "loss_policy_continuous",
                "loss_value",
            ]
            for loss_comp in loss_components:
                if loss_comp in df.columns:
                    valid_losses = df[loss_comp].dropna()
                    if not valid_losses.empty:
                        label = (
                            loss_comp.replace("loss_policy_", "")
                            .replace("loss_", "")
                            .replace("_", " ")
                            .title()
                        )
                        self.rewards_axes[5].plot(
                            episodes[: len(valid_losses)],
                            valid_losses,
                            alpha=0.8,
                            label=label,
                        )
            self.rewards_axes[5].set_title("Training Losses")
            self.rewards_axes[5].set_ylabel("Loss")
            self.rewards_axes[5].legend(fontsize=8)

            # Set x-labels for all subplots
            for ax in self.rewards_axes:
                ax.set_xlabel("Episode")

            self.rewards_fig.tight_layout()

        except Exception as e:
            print(f"Error updating rewards plot: {e}")

    def update_values_plot(self):
        """Update the physical values plot."""
        if not self.training_data:
            return

        try:
            # Convert training data to DataFrame
            df_data = []
            for d in self.training_data:
                row = {"episode": d["episode"], "reward": d["reward"]}
                if "metrics" in d:
                    row.update(d["metrics"])
                df_data.append(row)

            if not df_data:
                return

            df = pd.DataFrame(df_data)
            episodes = df["episode"].values
            window_size = min(20, len(episodes) // 4) if len(episodes) > 20 else 5

            # Clear all axes
            for ax in self.values_axes:
                ax.clear()
                ax.grid(True, alpha=0.3)

            # Physical values components
            physical_components = [
                ("reflectivity", "Reflectivity", False),
                ("thermal_noise", "Thermal Noise", True),
                ("absorption", "Absorption [ppm]", True),
                ("thickness", "Thickness", False),
            ]

            for i, (component, title, use_log) in enumerate(physical_components):
                if component in df.columns and i < len(self.values_axes):
                    values = df[component].values

                    # Plot raw data
                    self.values_axes[i].plot(
                        episodes, values, alpha=0.4, color="blue", linewidth=0.8
                    )

                    # Plot smoothed data
                    if len(values) > window_size:
                        smoothed = (
                            pd.Series(values)
                            .rolling(window=window_size, center=False)
                            .median()
                        )
                        self.values_axes[i].plot(
                            episodes,
                            smoothed,
                            linewidth=2,
                            color="red",
                            label="Smoothed",
                        )
                        self.values_axes[i].legend(fontsize=8)

                    self.values_axes[i].set_title(title)
                    self.values_axes[i].set_ylabel(title)
                    self.values_axes[i].set_xlabel("Episode")

                    # Use log scale for thermal noise and absorption
                    if use_log:
                        non_zero_values = (
                            values[values > 0]
                            if len(values[values > 0]) > 0
                            else [1e-10]
                        )
                        if len(non_zero_values) > 0:
                            self.values_axes[i].set_yscale("log")

            self.values_fig.tight_layout()

        except Exception as e:
            print(f"Error updating values plot: {e}")

    def update_constraints_plot(self):
        """Update the preference constraints tracking plot."""

        if not self.constraint_data:
            print("[DEBUG] No constraint data available, skipping constraints plot")
            return

        try:
            self.constraints_fig.clear()

            # Convert constraint data to more workable format
            episodes = [entry["episode"] for entry in self.constraint_data]
            phases = [entry["phase"] for entry in self.constraint_data]

            # Get all objective parameters
            all_objectives = set()
            for entry in self.constraint_data:
                all_objectives.update(entry["reward_bounds"].keys())
                all_objectives.update(entry["constraints"].keys())
            all_objectives = sorted(list(all_objectives))

            if not all_objectives:
                return

            # Create subplots: 2x2 layout
            # Top row: Phase tracking and constraint steps
            # Bottom row: Reward bounds evolution and active constraints
            axes = self.constraints_fig.subplots(2, 2)

            # Plot 1: Training phases
            axes[0, 0].plot(
                episodes, phases, "b-", linewidth=2, marker="o", markersize=3
            )
            axes[0, 0].set_title("Training Phases")
            axes[0, 0].set_ylabel("Phase")
            axes[0, 0].set_ylim(0.5, 2.5)
            axes[0, 0].set_yticks([1, 2])
            axes[0, 0].set_yticklabels(
                ["Phase 1\n(Exploration)", "Phase 2\n(Constrained)"]
            )
            axes[0, 0].grid(True, alpha=0.3)

            # Plot 2: Constraint steps (Phase 2 only)
            constraint_steps = [
                entry["constraint_step"] for entry in self.constraint_data
            ]
            phase2_episodes = [ep for ep, phase in zip(episodes, phases) if phase == 2]
            phase2_steps = [
                step for step, phase in zip(constraint_steps, phases) if phase == 2
            ]

            if phase2_episodes:
                axes[0, 1].plot(
                    phase2_episodes,
                    phase2_steps,
                    "r-",
                    linewidth=2,
                    marker="s",
                    markersize=3,
                )
                axes[0, 1].set_title("Constraint Steps (Phase 2)")
                axes[0, 1].set_ylabel("Constraint Step")
                axes[0, 1].grid(True, alpha=0.3)
            else:
                axes[0, 1].text(
                    0.5,
                    0.5,
                    "Phase 2 not started",
                    transform=axes[0, 1].transAxes,
                    ha="center",
                    va="center",
                )
                axes[0, 1].set_title("Constraint Steps (Phase 2)")

            # Plot 3: Reward bounds evolution
            colors = plt.cm.Set1(np.linspace(0, 1, len(all_objectives)))
            for i, obj in enumerate(all_objectives):
                obj_mins = []
                obj_maxs = []
                obj_episodes = []

                for entry in self.constraint_data:
                    if obj in entry["reward_bounds"]:
                        bounds = entry["reward_bounds"][obj]
                        if "min" in bounds and "max" in bounds:
                            obj_episodes.append(entry["episode"])
                            obj_mins.append(bounds["min"])
                            obj_maxs.append(bounds["max"])

                if obj_episodes:
                    # Plot min and max bounds
                    axes[1, 0].plot(
                        obj_episodes,
                        obj_mins,
                        "--",
                        color=colors[i],
                        alpha=0.7,
                        label=f"{obj} min",
                    )
                    axes[1, 0].plot(
                        obj_episodes,
                        obj_maxs,
                        "-",
                        color=colors[i],
                        alpha=0.9,
                        label=f"{obj} max",
                    )

            axes[1, 0].set_title("Reward Bounds Evolution")
            axes[1, 0].set_ylabel("Reward Value")
            axes[1, 0].legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc="upper left")
            axes[1, 0].grid(True, alpha=0.3)

            # Plot 4: Active constraints (Phase 2)
            for i, obj in enumerate(all_objectives):
                constraint_values = []
                constraint_episodes = []

                for entry in self.constraint_data:
                    if entry["phase"] == 2 and obj in entry["constraints"]:
                        constraint_episodes.append(entry["episode"])
                        constraint_values.append(entry["constraints"][obj])

                if constraint_episodes:
                    axes[1, 1].plot(
                        constraint_episodes,
                        constraint_values,
                        "o-",
                        color=colors[i],
                        alpha=0.8,
                        label=f"{obj} threshold",
                        linewidth=2,
                        markersize=4,
                    )

            axes[1, 1].set_title("Active Constraint Thresholds (Phase 2)")
            axes[1, 1].set_ylabel("Constraint Threshold")
            if len(all_objectives) <= 5:  # Only show legend if not too many objectives
                axes[1, 1].legend(fontsize=8)
            axes[1, 1].grid(True, alpha=0.3)

            # Set x-labels for bottom plots
            axes[1, 0].set_xlabel("Episode")
            axes[1, 1].set_xlabel("Episode")

            # Add phase transition annotations
            phase_transitions = []
            current_phase = None
            for episode, phase in zip(episodes, phases):
                if phase != current_phase:
                    phase_transitions.append((episode, phase))
                    current_phase = phase

            # Annotate phase transitions on all plots
            for ax in axes.flatten():
                for transition_ep, new_phase in phase_transitions:
                    if new_phase == 2:  # Transition to Phase 2
                        ax.axvline(
                            x=transition_ep,
                            color="red",
                            linestyle="--",
                            alpha=0.5,
                            linewidth=1,
                        )

            self.constraints_fig.tight_layout()

        except Exception as e:
            print(f"Error updating constraints plot: {e}")
            traceback.print_exc()

    def update_pareto_plot(self, episode: int = None):
        """Update the Pareto front plot using unified Pareto tracking data."""
        if not self.pareto_data:
            print("Warning: No pareto_data available")
            return None

        try:
            self.pareto_fig.clear()

            # Get the latest data or specific episode data
            if episode and episode in self.historical_pareto_data:
                latest_data = self.historical_pareto_data[episode]
            else:
                latest_data = self.pareto_data[-1]

            # Use unified keys - handle both numpy arrays and lists
            pareto_front_values_raw = latest_data.get("pareto_front_values", [])
            all_values_raw = latest_data.get("all_values", [])
            pareto_states = latest_data.get("pareto_states", [])
            current_episode = latest_data.get("episode", "unknown")
            eval_points = self.eval_data.get("eval_points", None)

            # Convert to numpy arrays if needed
            if (
                isinstance(pareto_front_values_raw, list)
                and len(pareto_front_values_raw) > 0
            ):
                pareto_front_values = np.array(pareto_front_values_raw)
            elif (
                isinstance(pareto_front_values_raw, np.ndarray)
                and pareto_front_values_raw.size > 0
            ):
                pareto_front_values = pareto_front_values_raw
            else:
                pareto_front_values = np.array([])

            if isinstance(all_values_raw, list) and len(all_values_raw) > 0:
                all_values = np.array(all_values_raw)
            elif isinstance(all_values_raw, np.ndarray) and all_values_raw.size > 0:
                all_values = all_values_raw
            else:
                all_values = np.array([])

            # Check if we have any data to plot
            if pareto_front_values.size == 0 and all_values.size == 0:
                ax = self.pareto_fig.add_subplot(1, 1, 1)
                ax.text(
                    0.5,
                    0.5,
                    f"No Pareto data available for episode {current_episode}",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=12,
                )
                ax.set_title("Pareto Front Visualization")
                self.pareto_fig.tight_layout()
                return None

            # Transform for plotting (reflectivity -> 1-reflectivity)
            def transform_points(values):
                if values.size == 0:
                    return np.array([])

                # Ensure 2D array
                if values.ndim == 1:
                    values = values.reshape(1, -1)

                points = []
                for vals in values:
                    point = []
                    for j, param in enumerate(self.optimisation_parameters):
                        if j < len(vals):
                            if param == "reflectivity":
                                point.append(1 - vals[j])
                            else:
                                point.append(vals[j])
                    if len(point) > 0:
                        points.append(point)
                return np.array(points) if points else np.array([])

            pareto_front = transform_points(pareto_front_values)
            best_points = transform_points(all_values)

            # Set optimization parameters if not set
            if not self.optimisation_parameters:
                if pareto_front.size > 0:
                    n_objectives = pareto_front.shape[1]
                elif best_points.size > 0:
                    n_objectives = best_points.shape[1]
                else:
                    n_objectives = 2  # Default

                self.optimisation_parameters = [
                    f"objective_{i}" for i in range(n_objectives)
                ]
                self.objective_labels = self._get_objective_labels()
                self.objective_scales = self._get_objective_scales()

            # Ensure objective labels match data
            if pareto_front.size > 0:
                data_objectives = pareto_front.shape[1]
            elif best_points.size > 0:
                data_objectives = best_points.shape[1]
            else:
                data_objectives = len(self.optimisation_parameters)

            if (
                not self.objective_labels
                or len(self.objective_labels) != data_objectives
            ):
                self._update_objective_labels_from_data(data_objectives)

            n_objectives = len(self.objective_labels)

            if n_objectives < 2:
                ax = self.pareto_fig.add_subplot(1, 1, 1)
                ax.text(
                    0.5,
                    0.5,
                    "Need at least 2 objectives for Pareto visualization",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                )
                return None

            # Plot based on number of objectives
            click_handler = None
            if n_objectives == 2:
                click_handler = self._plot_2d_pareto(
                    pareto_front, best_points, current_episode, eval_points
                )
            else:
                self._plot_multi_objective_pareto(
                    pareto_front, best_points, current_episode
                )

            self.pareto_fig.tight_layout()
            return click_handler

        except Exception as e:
            print(f"Error updating Pareto plot: {e}")
            import traceback

            print(f"Traceback: {traceback.format_exc()}")
            return None

    def update_pareto_rewards_plot(self, episode: int = None):
        """Update the Pareto rewards plot using objective rewards data."""
        if not self.pareto_data:
            print("Warning: No pareto_data available for rewards plot")
            return None

        try:
            self.pareto_rewards_fig.clear()

            # Get the latest data or specific episode data
            if episode and episode in self.historical_pareto_data:
                latest_data = self.historical_pareto_data[episode]
            else:
                latest_data = self.pareto_data[-1]

            # Use reward keys instead of value keys
            pareto_front_rewards_raw = latest_data.get("pareto_front_rewards", [])
            all_rewards_raw = latest_data.get("all_rewards", [])
            pareto_states = latest_data.get("pareto_states", [])
            current_episode = latest_data.get("episode", "unknown")
            eval_points = self.eval_data.get("eval_rewards", None)

            # Convert to numpy arrays if needed
            if (
                isinstance(pareto_front_rewards_raw, list)
                and len(pareto_front_rewards_raw) > 0
            ):
                pareto_front_rewards = np.array(pareto_front_rewards_raw)
            elif (
                isinstance(pareto_front_rewards_raw, np.ndarray)
                and pareto_front_rewards_raw.size > 0
            ):
                pareto_front_rewards = pareto_front_rewards_raw
            else:
                pareto_front_rewards = np.array([])

            if isinstance(all_rewards_raw, list) and len(all_rewards_raw) > 0:
                all_rewards = np.array(all_rewards_raw)
            elif isinstance(all_rewards_raw, np.ndarray) and all_rewards_raw.size > 0:
                all_rewards = all_rewards_raw
            else:
                all_rewards = np.array([])

            # Check if we have any data to plot
            if pareto_front_rewards.size == 0 and all_rewards.size == 0:
                ax = self.pareto_rewards_fig.add_subplot(1, 1, 1)
                ax.text(
                    0.5,
                    0.5,
                    f"No Pareto rewards data available for episode {current_episode}",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=12,
                )
                ax.set_title("Pareto Front - Objective Rewards")
                self.pareto_rewards_fig.tight_layout()
                return None

            # No transformation needed for rewards - they're already in the correct space
            pareto_front = pareto_front_rewards
            best_points = all_rewards

            # Set optimization parameters if not set
            if not self.optimisation_parameters:
                if pareto_front.size > 0:
                    n_objectives = pareto_front.shape[1] if pareto_front.ndim > 1 else 1
                elif best_points.size > 0:
                    n_objectives = best_points.shape[1] if best_points.ndim > 1 else 1
                else:
                    n_objectives = 2  # Default

                self.optimisation_parameters = [
                    f"objective_{i}" for i in range(n_objectives)
                ]

            # Create reward-specific labels
            reward_labels = [
                f'{param.replace("_", " ").title()} Reward'
                for param in self.optimisation_parameters
            ]

            # Ensure we have at least 2 objectives for plotting
            if len(reward_labels) < 2:
                ax = self.pareto_rewards_fig.add_subplot(1, 1, 1)
                ax.text(
                    0.5,
                    0.5,
                    "Need at least 2 objectives for Pareto rewards visualization",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                )
                return None

            # Plot based on number of objectives
            if len(reward_labels) == 2:
                self._plot_2d_pareto_rewards(
                    pareto_front,
                    best_points,
                    current_episode,
                    eval_points,
                    reward_labels,
                )
            else:
                self._plot_multi_objective_pareto_rewards(
                    pareto_front, best_points, current_episode, reward_labels
                )

            self.pareto_rewards_fig.tight_layout()
            return None

        except Exception as e:
            print(f"Error updating Pareto rewards plot: {e}")
            import traceback

            print(f"Traceback: {traceback.format_exc()}")
            return None

    def _plot_2d_pareto_rewards(
        self, pareto_front, best_points, episode, eval_points, reward_labels
    ):
        """Plot 2D Pareto front for rewards."""
        ax = self.pareto_rewards_fig.add_subplot(1, 1, 1)

        # Plot background points with training progression gradient
        if best_points is not None and len(best_points) > 0:
            try:
                best_points_array = np.array(best_points)
                if (
                    len(best_points_array.shape) == 2
                    and best_points_array.shape[1] >= 2
                ):
                    # Create color gradient based on order (training progression)
                    n_points = len(best_points_array)
                    if n_points > 1:
                        # Create colormap from blue (early) to red (late)
                        colors = plt.cm.coolwarm(np.linspace(0, 1, n_points))
                        ax.scatter(
                            best_points_array[:, 0],
                            best_points_array[:, 1],
                            c=colors,
                            alpha=0.6,
                            s=8,
                            label="All Solutions (Blue→Red: Training Progression)",
                        )
                    else:
                        # Single point case
                        ax.scatter(
                            best_points_array[:, 0],
                            best_points_array[:, 1],
                            c="lightblue",
                            alpha=0.3,
                            s=10,
                            label="All Solutions",
                        )
            except Exception as e:
                print(f"Error plotting background reward points: {e}")

        # Plot Pareto front
        if pareto_front.size > 0:
            # Ensure 2D array
            if pareto_front.ndim == 1:
                pareto_front = pareto_front.reshape(1, -1)

            ax.scatter(
                pareto_front[:, 0],
                pareto_front[:, 1],
                c="red",
                s=40,
                alpha=0.8,
                label="Pareto Front",
                edgecolors="black",
                linewidths=1,
            )

            # Connect Pareto points if not too many
            if len(pareto_front) < 20:
                try:
                    sorted_indices = np.argsort(pareto_front[:, 0])
                    sorted_front_x = pareto_front[sorted_indices, 0]
                    sorted_front_y = pareto_front[sorted_indices, 1]
                    ax.plot(
                        sorted_front_x, sorted_front_y, "r-", alpha=0.5, linewidth=1
                    )
                except:
                    pass

        # Add evaluation points if available
        if eval_points is not None and len(eval_points) > 0:
            eval_points_array = np.array(eval_points)
            if eval_points_array.shape[1] >= 2:
                ax.scatter(
                    eval_points_array[:, 0],
                    eval_points_array[:, 1],
                    c="orange",
                    s=10,
                    alpha=0.5,
                    label="Evaluation Points",
                )

        # Configure axes
        ax.set_xlabel(reward_labels[0])
        ax.set_ylabel(reward_labels[1])
        ax.set_title(f"Pareto Front - Objective Rewards - Episode {episode}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    def _plot_multi_objective_pareto_rewards(
        self, pareto_front, best_points, episode, reward_labels
    ):
        """Plot multi-objective Pareto front for rewards using 2D pairs."""
        n_objectives = len(reward_labels)
        n_pairs = min(
            4, n_objectives * (n_objectives - 1) // 2
        )  # Limit to 4 pairs for readability

        n_cols = 2
        n_rows = (n_pairs + n_cols - 1) // n_cols

        pair_idx = 0
        for i in range(n_objectives):
            for j in range(i + 1, n_objectives):
                if pair_idx >= n_pairs:
                    break

                ax = self.pareto_rewards_fig.add_subplot(n_rows, n_cols, pair_idx + 1)

                # Plot background points with training progression gradient
                if best_points is not None and len(best_points) > 0:
                    try:
                        best_points_array = np.array(best_points)
                        if best_points_array.shape[1] > max(i, j):
                            # Create color gradient based on order (training progression)
                            n_points = len(best_points_array)
                            if n_points > 1:
                                # Create colormap from blue (early) to red (late)
                                colors = plt.cm.coolwarm(np.linspace(0, 1, n_points))
                                ax.scatter(
                                    best_points_array[:, i],
                                    best_points_array[:, j],
                                    c=colors,
                                    alpha=0.6,
                                    s=4,
                                    label="All Solutions (Blue→Red: Training)",
                                )
                            else:
                                # Single point case
                                ax.scatter(
                                    best_points_array[:, i],
                                    best_points_array[:, j],
                                    c="lightblue",
                                    alpha=0.3,
                                    s=5,
                                    label="All Solutions",
                                )
                    except:
                        pass

                # Plot Pareto front
                if pareto_front.size > 0:
                    # Ensure 2D array
                    if pareto_front.ndim == 1:
                        pareto_front = pareto_front.reshape(1, -1)

                    ax.scatter(
                        pareto_front[:, i],
                        pareto_front[:, j],
                        c="red",
                        s=20,
                        alpha=0.8,
                        label="Pareto Front",
                        edgecolors="black",
                        linewidths=0.5,
                    )

                # Set labels
                ax.set_xlabel(reward_labels[i])
                ax.set_ylabel(reward_labels[j])

                ax.grid(True, alpha=0.3)
                if pair_idx == 0:  # Only show legend on first subplot
                    ax.legend(fontsize=6)

                pair_idx += 1

        # Add overall title
        self.pareto_rewards_fig.suptitle(
            f"Multi-Objective Pareto Front - Objective Rewards - Episode {episode}"
        )

    def _update_objective_labels_from_data(self, n_objectives: int):
        """Update objective labels based on data dimensions."""
        if (
            self.optimisation_parameters
            and len(self.optimisation_parameters) == n_objectives
        ):
            self.objective_labels = self._get_objective_labels()
            self.objective_scales = self._get_objective_scales()
        else:
            # Fallback to generic labels
            self.objective_labels = [f"Objective {i+1}" for i in range(n_objectives)]
            self.objective_scales = ["linear"] * n_objectives

    def _plot_2d_pareto(self, pareto_front, best_points, episode, eval_points=None):
        """Plot 2D Pareto front."""
        if self.ui_mode:
            # Enhanced UI mode with coating stack visualization
            return self._plot_2d_pareto_with_coating(
                pareto_front, best_points, episode, eval_points
            )
        else:
            # Standard CLI mode plotting
            return self._plot_2d_pareto_standard(
                pareto_front, best_points, episode, eval_points
            )

    def _plot_2d_pareto_standard(
        self, pareto_front, best_points, episode, eval_points=None
    ):
        """Plot standard 2D Pareto front for CLI mode."""
        ax = self.pareto_fig.add_subplot(1, 1, 1)

        # Plot background points with training progression gradient
        if best_points is not None and len(best_points) > 0:
            try:
                best_points_array = np.array(best_points)
                if (
                    len(best_points_array.shape) == 2
                    and best_points_array.shape[1] >= 2
                ):
                    # Create color gradient based on order (training progression)
                    n_points = len(best_points_array)
                    if n_points > 1:
                        # Create colormap from blue (early) to red (late)
                        colors = plt.cm.coolwarm(np.linspace(0, 1, n_points))
                        ax.scatter(
                            best_points_array[:, 0],
                            best_points_array[:, 1],
                            c=colors,
                            alpha=0.6,
                            s=8,
                            label="All Solutions (Blue→Red: Training Progression)",
                        )
                    else:
                        # Single point case
                        ax.scatter(
                            best_points_array[:, 0],
                            best_points_array[:, 1],
                            c="lightblue",
                            alpha=0.3,
                            s=10,
                            label="All Solutions",
                        )
            except Exception as e:
                print(f"Error plotting background points: {e}")

        # Plot Pareto front
        ax.scatter(
            pareto_front[:, 0],
            pareto_front[:, 1],
            c="red",
            s=40,
            alpha=0.8,
            label="Pareto Front",
            edgecolors="black",
            linewidths=1,
        )

        # add evaluation points as well as training points
        if eval_points is not None and len(eval_points) > 0:
            eval_points_array = np.array(eval_points)
            if eval_points_array.shape[1] >= 2:
                ax.scatter(
                    eval_points_array[:, 0],
                    eval_points_array[:, 1],
                    c="orange",
                    s=10,
                    alpha=0.5,
                    label="Evaluation Points",
                )

        # Connect Pareto points if not too many
        if len(pareto_front) < 20:
            try:
                sorted_indices = np.argsort(pareto_front[:, 0])
                sorted_front_x = pareto_front[sorted_indices, 0]
                sorted_front_y = pareto_front[sorted_indices, 1]
                ax.plot(sorted_front_x, sorted_front_y, "r-", alpha=0.5, linewidth=1)
            except:
                pass

        # self._add_objective_targets(ax)
        self._add_design_criteria(ax)
        self._configure_axis_labels_and_scales(ax, 0, 1)
        ax.set_title(f"Pareto Front - Episode {episode}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    def _plot_2d_pareto_with_coating(
        self, pareto_front, best_points, episode, eval_points=None
    ):
        """Plot 2D Pareto front with interactive coating stack visualization for UI mode."""
        # Create subplot layout: Pareto plot on left, coating stack on right
        gs = self.pareto_fig.add_gridspec(1, 2, width_ratios=[2, 1])
        pareto_ax = self.pareto_fig.add_subplot(gs[0])
        coating_ax = self.pareto_fig.add_subplot(gs[1])

        # Store axes for click handling
        self.pareto_ax = pareto_ax
        self.coating_ax = coating_ax

        # Plot background points with training progression gradient
        if best_points is not None and len(best_points) > 0:
            try:
                best_points_array = np.array(best_points)
                if (
                    len(best_points_array.shape) == 2
                    and best_points_array.shape[1] >= 2
                ):
                    # Create color gradient based on order (training progression)
                    n_points = len(best_points_array)
                    if n_points > 1:
                        # Create colormap from blue (early) to red (late)
                        colors = plt.cm.coolwarm(np.linspace(0, 1, n_points))
                        pareto_ax.scatter(
                            best_points_array[:, 0],
                            best_points_array[:, 1],
                            c=colors,
                            alpha=0.6,
                            s=8,
                            label="All Solutions (Blue→Red: Training Progression)",
                        )
                    else:
                        # Single point case
                        pareto_ax.scatter(
                            best_points_array[:, 0],
                            best_points_array[:, 1],
                            c="lightblue",
                            alpha=0.3,
                            s=10,
                            label="All Solutions",
                        )
            except Exception as e:
                print(f"Error plotting background points: {e}")

        # Plot Pareto front
        pareto_scatter = pareto_ax.scatter(
            pareto_front[:, 0],
            pareto_front[:, 1],
            c="red",
            s=40,
            alpha=0.8,
            label="Pareto Front",
            edgecolors="black",
            linewidths=1,
            picker=True,
        )

        # Store data for click handling
        self.current_pareto_front = pareto_front
        self.current_episode = episode

        # add evaluation points as well as training points
        if eval_points is not None and len(eval_points) > 0:
            eval_points_array = np.array(eval_points)
            if eval_points_array.shape[1] >= 2:
                pareto_ax.scatter(
                    eval_points_array[:, 0],
                    eval_points_array[:, 1],
                    c="orange",
                    s=10,
                    alpha=0.5,
                    label="Evaluation Points",
                )

        # Connect Pareto points if not too many
        if len(pareto_front) < 20:
            try:
                sorted_indices = np.argsort(pareto_front[:, 0])
                sorted_front_x = pareto_front[sorted_indices, 0]
                sorted_front_y = pareto_front[sorted_indices, 1]
                pareto_ax.plot(
                    sorted_front_x, sorted_front_y, "r-", alpha=0.5, linewidth=1
                )
            except:
                pass

        # Configure Pareto plot
        # self._add_objective_targets(pareto_ax)
        self._add_design_criteria(pareto_ax)
        self._configure_axis_labels_and_scales(pareto_ax, 0, 1)
        pareto_ax.set_title(f"Pareto Front - Episode {episode}")
        pareto_ax.grid(True, alpha=0.3)
        pareto_ax.legend(fontsize=8)

        # Initialize coating plot
        coating_ax.text(
            0.5,
            0.5,
            "Click a Pareto point\nto view coating stack",
            transform=coating_ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.5),
        )
        coating_ax.set_title("Coating Stack")
        coating_ax.set_xlabel("Layer Index")
        coating_ax.set_ylabel("Optical Thickness")

        # Return callback function for click handling
        click_handler = self._create_pareto_click_handler()

        return click_handler

    def _create_pareto_click_handler(self):
        """Create a click handler function for Pareto front interaction."""

        def on_pareto_click(event):

            # Check if we have the required attributes
            if not hasattr(self, "pareto_ax"):
                print(f"[DEBUG] No pareto_ax attribute found")
                return

            if not hasattr(self, "current_pareto_front"):
                print(f"[DEBUG] No current_pareto_front attribute found")
                return

            if event.inaxes == self.pareto_ax and hasattr(self, "current_pareto_front"):

                if event.xdata is not None and event.ydata is not None:

                    # Find closest Pareto point
                    click_x, click_y = event.xdata, event.ydata

                    pareto_front = self.current_pareto_front
                    # Handle log scale distances if necessary
                    if (
                        len(self.objective_scales) >= 2
                        and self.objective_scales[0] == "log"
                        and self.objective_scales[1] == "log"
                    ):
                        log_click_x = np.log10(max(click_x, 1e-10))
                        log_click_y = np.log10(max(click_y, 1e-10))
                        log_front_x = np.log10(np.maximum(pareto_front[:, 0], 1e-10))
                        log_front_y = np.log10(np.maximum(pareto_front[:, 1], 1e-10))
                        distances = np.sqrt(
                            (log_front_x - log_click_x) ** 2
                            + (log_front_y - log_click_y) ** 2
                        )
                    else:
                        distances = np.sqrt(
                            (pareto_front[:, 0] - click_x) ** 2
                            + (pareto_front[:, 1] - click_y) ** 2
                        )

                    closest_idx = np.argmin(distances)
                    closest_point = pareto_front[closest_idx]

                    # Update coating stack visualization
                    self._plot_coating_stack_for_point(closest_idx, closest_point)
                else:
                    print(
                        f"[DEBUG] Click coordinates are None: x={event.xdata}, y={event.ydata}"
                    )
            else:
                print(
                    f"[DEBUG] Click not on pareto axis or no pareto front data available"
                )
                print(f"[DEBUG] Available axes in event.inaxes: {event.inaxes}")

        return on_pareto_click

    def _plot_coating_stack_for_point(self, pareto_idx: int, pareto_point: np.ndarray):
        """Plot coating stack for a selected Pareto point."""
        print(
            f"[DEBUG] _plot_coating_stack_for_point called with idx={pareto_idx}, point={pareto_point}"
        )

        if not hasattr(self, "coating_ax"):
            print(f"[DEBUG] No coating_ax available")
            return

        try:
            self.coating_ax.clear()

            # Try to get coating state
            coating_state = None

            # First try current episode training data
            if (
                hasattr(self, "current_episode")
                and self.current_episode in self.coating_states
            ):
                states = self.coating_states[self.current_episode]
                if pareto_idx < len(states):
                    coating_state = states[pareto_idx]

            # If no training data, try evaluation data
            if coating_state is None and pareto_idx < len(self.eval_coating_states):
                coating_state = self.eval_coating_states[pareto_idx]

            if coating_state is not None:
                layers = self.parse_coating_state(coating_state)

                if layers:
                    # Material colors
                    material_colors = {
                        0: "lightgray",  # Air
                        1: "blue",  # Substrate/SiO2
                        2: "green",  # Ta2O5
                        3: "red",  # TiO2
                        4: "purple",  # Additional materials
                        5: "orange",
                        6: "brown",
                    }

                    material_names = {
                        0: "Air",
                        1: "SiO₂",
                        2: "Ta₂O₅",
                        3: "TiO₂",
                        4: "Mat4",
                        5: "Mat5",
                        6: "Mat6",
                    }

                    # Plot each layer as a bar
                    x_positions = range(len(layers))
                    thicknesses = [layer[1] for layer in layers]
                    materials = [layer[0] for layer in layers]

                    bars = self.coating_ax.bar(
                        x_positions,
                        thicknesses,
                        color=[material_colors.get(mat, "gray") for mat in materials],
                        alpha=0.8,
                        edgecolor="black",
                        linewidth=1,
                    )

                    # Add material labels
                    for i, (material, thickness) in enumerate(layers):
                        if thickness > 0.01:  # Only label if thick enough
                            self.coating_ax.text(
                                i,
                                thickness / 2,
                                material_names.get(material, f"M{material}"),
                                ha="center",
                                va="center",
                                fontsize=8,
                                fontweight="bold",
                            )

                    # Create objective info string
                    obj_info = []
                    for i, val in enumerate(
                        pareto_point[
                            : min(len(pareto_point), len(self.objective_labels))
                        ]
                    ):
                        if i < len(self.objective_labels):
                            obj_info.append(f"{self.objective_labels[i]}: {val:.2e}")

                    title = f"Coating Stack\n" + "\n".join(obj_info)
                    self.coating_ax.set_title(title, fontsize=10)
                    self.coating_ax.set_xlabel("Layer Index")
                    self.coating_ax.set_ylabel("Optical Thickness")
                    self.coating_ax.grid(True, alpha=0.3)

                else:
                    self.coating_ax.text(
                        0.5,
                        0.5,
                        "Could not parse\ncoating state",
                        transform=self.coating_ax.transAxes,
                        ha="center",
                        va="center",
                    )
                    print(f"[DEBUG] Could not parse coating state")
            else:
                self.coating_ax.text(
                    0.5,
                    0.5,
                    "Coating state\nnot available",
                    transform=self.coating_ax.transAxes,
                    ha="center",
                    va="center",
                )
                print(f"[DEBUG] No coating state available")

            # Redraw the canvas if available
            if hasattr(self, "canvas_draw_callback") and self.canvas_draw_callback:
                self.canvas_draw_callback()
            else:
                print(f"[DEBUG] No canvas draw callback available")

        except Exception as e:
            print(f"Error plotting coating stack: {e}")
            if hasattr(self, "coating_ax"):
                self.coating_ax.text(
                    0.5,
                    0.5,
                    f"Error: {str(e)}",
                    transform=self.coating_ax.transAxes,
                    ha="center",
                    va="center",
                )

    def _add_objective_targets(self, ax):
        """Add objective target lines to plot."""
        # Add objective target lines if available
        if hasattr(self, "objective_targets") and self.objective_targets:
            try:
                # Draw vertical line for first objective target
                if (
                    len(self.objective_targets) > 0
                    and self.objective_targets[0] is not None
                ):
                    target_x = (
                        1 - self.design_criteria[0]
                        if self.objective_labels[0] == "1 - Reflectivity"
                        else self.design_criteria[0]
                    )
                    ax.axvline(
                        x=target_x,
                        color="green",
                        linestyle="--",
                        alpha=0.7,
                        linewidth=2,
                        label=f'Target {self.objective_labels[0] if len(self.objective_labels) > 0 else "Obj 1"}',
                    )

                # Draw horizontal line for second objective target
                if (
                    len(self.objective_targets) > 1
                    and self.objective_targets[1] is not None
                ):
                    target_y = (
                        1 - self.design_criteria[1]
                        if self.objective_labels[1] == "1 - Reflectivity"
                        else self.design_criteria[1]
                    )
                    ax.axhline(
                        y=target_y,
                        color="orange",
                        linestyle="--",
                        alpha=0.7,
                        linewidth=2,
                        label=f'Target {self.objective_labels[1] if len(self.objective_labels) > 1 else "Obj 2"}',
                    )
            except Exception as e:
                print(
                    f"Error plotting objective targets: {e} traceback: {e.__traceback__}"
                )
        else:
            print(f"No objective targets set for Pareto plot.")

    def _add_design_criteria(self, ax):
        """Add design critera lines to plot."""
        # Add objective target lines if available
        if hasattr(self, "design_criteria") and self.design_criteria:
            try:

                # Draw vertical line for first objective target
                if (
                    len(self.design_criteria) > 0
                    and self.design_criteria[0] is not None
                ):
                    target_x = (
                        1 - self.design_criteria[0]
                        if self.objective_labels[0] == "1 - Reflectivity"
                        else self.design_criteria[0]
                    )
                    ax.axvline(
                        x=target_x,
                        color="k",
                        linestyle="--",
                        alpha=0.7,
                        linewidth=2,
                        label=f'Target {self.objective_labels[0] if len(self.objective_labels) > 0 else "Obj 1"}',
                    )

                # Draw horizontal line for second objective target
                if (
                    len(self.design_criteria) > 1
                    and self.design_criteria[1] is not None
                ):
                    target_y = (
                        1 - self.design_criteria[1]
                        if self.objective_labels[1] == "1 - Reflectivity"
                        else self.design_criteria[1]
                    )
                    ax.axhline(
                        y=target_y,
                        color="k",
                        linestyle="--",
                        alpha=0.7,
                        linewidth=2,
                        label=f'Target {self.objective_labels[1] if len(self.objective_labels) > 1 else "Obj 2"}',
                    )
            except Exception as e:
                print(
                    f"Error plotting objective targets: {e} traceback: {e.__traceback__}"
                )
        else:
            print(f"No objective targets set for design criteria.")

    def _configure_axis_labels_and_scales(self, ax, i, j):
        """Configure axis labels and scales for objectives i and j."""
        # Set labels and scales
        ax.set_xlabel(
            self.objective_labels[i]
            if i < len(self.objective_labels)
            else f"Objective {i+1}"
        )
        ax.set_ylabel(
            self.objective_labels[j]
            if j < len(self.objective_labels)
            else f"Objective {j+1}"
        )

        if i < len(self.objective_scales) and self.objective_scales[i] == "log":
            ax.set_xscale("log")
        if j < len(self.objective_scales) and self.objective_scales[j] == "log":
            ax.set_yscale("log")

    def set_canvas_draw_callback(self, callback):
        """Set callback function for canvas redraw (used by UI)."""
        self.canvas_draw_callback = callback

    def _plot_multi_objective_pareto(self, pareto_front, best_points, episode):
        """Plot multi-objective Pareto front using 2D pairs."""
        n_objectives = len(self.objective_labels)
        n_pairs = min(
            4, n_objectives * (n_objectives - 1) // 2
        )  # Limit to 4 pairs for readability

        n_cols = 2
        n_rows = (n_pairs + n_cols - 1) // n_cols

        pair_idx = 0
        for i in range(n_objectives):
            for j in range(i + 1, n_objectives):
                if pair_idx >= n_pairs:
                    break

                ax = self.pareto_fig.add_subplot(n_rows, n_cols, pair_idx + 1)

                # Plot background points with training progression gradient
                if best_points is not None and len(best_points) > 0:
                    try:
                        best_points_array = np.array(best_points)
                        if best_points_array.shape[1] > max(i, j):
                            # Create color gradient based on order (training progression)
                            n_points = len(best_points_array)
                            if n_points > 1:
                                # Create colormap from blue (early) to red (late)
                                colors = plt.cm.coolwarm(np.linspace(0, 1, n_points))
                                ax.scatter(
                                    best_points_array[:, i],
                                    best_points_array[:, j],
                                    c=colors,
                                    alpha=0.6,
                                    s=4,
                                    label="All Solutions (Blue→Red: Training)",
                                )
                            else:
                                # Single point case
                                ax.scatter(
                                    best_points_array[:, i],
                                    best_points_array[:, j],
                                    c="lightblue",
                                    alpha=0.3,
                                    s=5,
                                    label="All Solutions",
                                )
                    except:
                        pass

                # Plot Pareto front
                ax.scatter(
                    pareto_front[:, i],
                    pareto_front[:, j],
                    c="red",
                    s=20,
                    alpha=0.8,
                    label="Pareto Front",
                    edgecolors="black",
                    linewidths=0.5,
                )

                # Set labels and scales
                ax.set_xlabel(
                    self.objective_labels[i]
                    if i < len(self.objective_labels)
                    else f"Obj {i+1}"
                )
                ax.set_ylabel(
                    self.objective_labels[j]
                    if j < len(self.objective_labels)
                    else f"Obj {j+1}"
                )

                if i < len(self.objective_scales) and self.objective_scales[i] == "log":
                    ax.set_xscale("log")
                if j < len(self.objective_scales) and self.objective_scales[j] == "log":
                    ax.set_yscale("log")

                ax.grid(True, alpha=0.3)
                if pair_idx == 0:  # Only show legend on first subplot
                    ax.legend(fontsize=6)

                pair_idx += 1

        # Add overall title
        self.pareto_fig.suptitle(f"Multi-Objective Pareto Front - Episode {episode}")

    def _plot_metric_with_smoothing(self, ax, df, column, title, window_size):
        """Plot metric with smoothing."""
        if column not in df.columns:
            ax.set_title(f"{title} (No Data)")
            return

        episodes = df["episode"].values
        values = df[column].values

        # Plot raw data
        ax.plot(episodes, values, alpha=0.3, color="blue")

        # Plot smoothed data if enough points
        if len(values) > window_size:
            smoothed = (
                pd.Series(values).rolling(window=window_size, center=False).median()
            )
            ax.plot(episodes, smoothed, linewidth=2, color="red", label="Smoothed")
            ax.legend(fontsize=8)

        ax.set_title(title)
        ax.set_ylabel(column.replace("_", " ").title())

    def save_plots_to_disk(self):
        """Save all plots to disk."""
        if not self.save_plots or not self.output_dir:
            return

        try:
            os.makedirs(self.output_dir, exist_ok=True)

            # Save plots
            self.rewards_fig.savefig(
                os.path.join(self.output_dir, "rewards_plot.png"),
                dpi=150,
                bbox_inches="tight",
            )
            self.values_fig.savefig(
                os.path.join(self.output_dir, "values_plot.png"),
                dpi=150,
                bbox_inches="tight",
            )
            self.pareto_fig.savefig(
                os.path.join(self.output_dir, "pareto_plot.png"),
                dpi=150,
                bbox_inches="tight",
            )
            self.pareto_rewards_fig.savefig(
                os.path.join(self.output_dir, "pareto_rewards_plot.png"),
                dpi=150,
                bbox_inches="tight",
            )
            self.constraints_fig.savefig(
                os.path.join(self.output_dir, "constraints_plot.png"),
                dpi=150,
                bbox_inches="tight",
            )

        except Exception as e:
            print(f"Warning: Failed to save plots to disk: {e}")

    def get_figures(self):
        """Get matplotlib figures for UI display."""
        return (
            self.rewards_fig,
            self.values_fig,
            self.pareto_fig,
            self.pareto_rewards_fig,
        )

    def clear_data(self):
        """Clear all stored data."""
        self.training_data.clear()
        self.pareto_data.clear()
        self.historical_pareto_data.clear()
        self.coating_states.clear()
        self.eval_coating_states.clear()
        self.last_plot_update = 0

    def get_coating_state(self, episode: int, pareto_idx: int) -> Optional[np.ndarray]:
        """Get coating state for a specific Pareto point at a given episode."""
        if episode in self.coating_states:
            states = self.coating_states[episode]
            if pareto_idx < len(states):
                return states[pareto_idx]
        return None

    def get_eval_coating_state(self, pareto_idx: int) -> Optional[np.ndarray]:
        """Get coating state for a specific evaluation Pareto point."""
        if pareto_idx < len(self.eval_coating_states):
            return self.eval_coating_states[pareto_idx]
        return None

    def parse_coating_state(self, state: np.ndarray) -> List[Tuple[int, float]]:
        """Parse a coating state array into (material, thickness) pairs."""
        try:
            if state is None or len(state) == 0:
                return []

            state_array = np.array(state)
            layers = []

            for layer_idx in range(len(state_array)):
                layer = state_array[layer_idx]
                if len(layer) > 1:
                    thickness = layer[0]
                    if (
                        thickness > 1e-6
                    ):  # Only include layers with significant thickness
                        # Find which material (one-hot encoded)
                        material_probs = layer[1:]
                        material = (
                            np.argmax(material_probs) if len(material_probs) > 0 else 0
                        )
                        layers.append((material, thickness))

            return layers

        except Exception as e:
            print(f"Error parsing coating state: {e}")
            return []
