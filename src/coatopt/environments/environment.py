"""
Simplified unified coating environment.

Reads configuration directly from config object to avoid parameter passing errors.
Uses existing physics modules (coating_utils, EFI_tmm, YAM_CoatingBrownian)
without modification.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ..environments.state import CoatingState
from ..environments.utils import coating_utils, state_utils


class CoatingEnvironment:
    """
    Unified coating optimization environment. Reads directly from config.
    """

    def __init__(self, config, materials: Dict[int, Dict]):
        """
        Initialize from config object.

        Args:
            config: CoatingOptimisationConfig object
            materials: Dict mapping material index to material properties
        """
        data = config.data
        training = config.training

        # Materials
        self.materials = materials or {}
        self.n_materials = len(self.materials)

        # Core parameters from config.data
        self.max_layers = data.n_layers
        self.min_thickness = data.min_thickness
        self.max_thickness = data.max_thickness

        # Material indices (could be in config, using defaults for now)
        self.air_material_index = 0
        self.substrate_material_index = 1

        # Physics parameters (could add to config if needed)
        self.light_wavelength = 1064e-9
        self.frequency = 100.0
        self.wBeam = 0.062
        self.Temp = 293.0
        self.use_optical_thickness = getattr(data, "use_optical_thickness", False)

        # Optimization parameters - strip direction suffixes
        raw_params = data.optimise_parameters or ["reflectivity"]
        self.optimise_parameters = [
            p.split(":")[0].strip() if ":" in p else p for p in raw_params
        ]
        self.optimise_targets = data.optimise_targets or {}
        self.optimise_weight_ranges = getattr(data, "optimise_weight_ranges", {}) or {}
        self.design_criteria = getattr(data, "design_criteria", {}) or {}

        # Objective directions: True = maximize, False = minimize
        self.objective_directions = {
            "reflectivity": True,   # Higher is better
            "absorption": False,    # Lower is better
            "thermal_noise": False, # Lower is better
        }

        # Action space constraints
        self.ignore_air_option = getattr(data, "ignore_air_option", False)
        self.ignore_substrate_option = getattr(data, "ignore_substrate_option", False)

        # Multi-objective settings
        self.multi_objective = len(self.optimise_parameters) > 1
        self.cycle_weights = getattr(training, "cycle_weights", "random")
        self.objective_weights = {param: 1.0 for param in self.optimise_parameters}

        # Reward configuration
        self.use_intermediate_reward = getattr(data, "use_intermediate_reward", False)
        self.combine = getattr(data, "combine", "sum")

        # Reward normalization bounds (for algorithms that need it)
        self.objective_bounds = getattr(data, "objective_bounds", {}) or {}
        self.reward_bounds = {}  # {objective: [min_reward, max_reward]}
        self._initialize_reward_bounds()

        # Warmup tracking (for constrained training)
        self.warmup_best_rewards = {obj: 0.0 for obj in self.optimise_parameters}
        self.observed_value_bounds = {
            obj: {"min": np.inf, "max": -np.inf} for obj in self.optimise_parameters
        }

        # Constrained training state
        self.use_constrained_training = False  # Set by wrapper if needed
        self.episode_count = 0
        self.is_warmup = True
        self.target_objective = None
        self.constraints = {}
        self.constraint_penalty = 10.0

        # Pareto dominance reward bonus
        self.pareto_dominance_bonus = 0.0  # Bonus reward for dominating pareto front
        self.use_pareto_bonus = False  # Enable pareto dominance bonus

        # Environment state
        self.current_state = None
        self.current_index = 0
        self.done = False

        # Multi-objective tracking
        # IMPORTANT: Reward space Pareto front is used for all calculations
        # Value space Pareto front is only for visual diagnostics
        self.pareto_front_rewards = []  # List of (reward_vector, state) - used for calculations
        self.pareto_front_values = []   # List of (value_vector, state) - used for plotting
        self.all_points = []

        # Observation space shape
        features_per_layer = 1 + self.n_materials + 2
        self.obs_space_shape = (self.max_layers, features_per_layer)

    def reset(self) -> CoatingState:
        """Reset environment to initial state."""
        self.current_state = CoatingState(
            max_layers=self.max_layers,
            n_materials=self.n_materials,
            air_material_index=self.air_material_index,
            substrate_material_index=self.substrate_material_index,
            materials=self.materials,
        )
        self.current_index = 0
        self.done = False
        return self.current_state

    def step(
        self,
        action: np.ndarray,
        objective_weights: Optional[Dict[str, float]] = None,
        pc_tracker=None,
        pareto_tracker=None,
        phase_info=None,
        **kwargs,
    ) -> Tuple:
        """Take an action in the environment.

        Returns:
            Tuple of (new_state, rewards, terminated, finished, reward, full_action, vals)
            matching base environment interface.
        """
        if self.done:
            raise RuntimeError("Episode is done. Call reset() to start new episode.")

        if objective_weights is None:
            objective_weights = self.objective_weights

        # Extract action - base environment format is [material, thickness]
        # but agent outputs [thickness, material_probs...]
        if len(action) > 2:
            # Agent format: [thickness, material_probs...]
            thickness = float(action[0])
            material_idx = int(np.argmax(action[1:]))
        else:
            # Base environment format: [material, thickness]
            material_idx = int(action[0])
            thickness = float(action[1])

        # Clamp thickness
        thickness = np.clip(thickness, self.min_thickness, self.max_thickness)

        # Create full_action in base environment format
        full_action = [material_idx, thickness]

        # Update state
        self.current_state.set_layer(self.current_index, thickness, material_idx)
        self.current_index += 1

        # Check termination conditions
        terminated = False
        finished = False
        if (
            self.current_index >= self.max_layers
            or material_idx == self.air_material_index
        ):
            finished = True
            self.done = True

        # Calculate reward
        if finished or self.use_intermediate_reward:
            total_reward, vals, reward_components = self.compute_training_reward(
                self.current_state,
                objective_weights=objective_weights,
                pc_tracker=pc_tracker,
                phase_info=phase_info
            )

            # Always update pareto front when episode finishes (for tracking and optional bonus)
            if finished and self.multi_objective:
                self.update_pareto_front(vals, self.current_state)
        else:
            total_reward = 0.0
            vals = {}
            reward_components = {}

        # Use reward_components as rewards dict (includes PC metadata)
        rewards = reward_components

        return (
            self.current_state,
            rewards,
            terminated,
            finished,
            total_reward,
            full_action,
            vals,
        )

    def compute_reward(
        self,
        state,  # Can be CoatingState or numpy array
        normalised: bool = True,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Compute base rewards for all objectives (no modifiers).

        This is the core reward function that computes rewards for each objective.
        Use this for evaluation or when you want base rewards without training modifiers.

        Args:
            state: CoatingState or numpy array
            normalised: If True, scale rewards to [0, 1]. If False, return raw log-based rewards.

        Returns:
            Tuple of (individual_rewards, vals)
            - individual_rewards: Dict mapping objective names to their rewards
            - vals: Dict of physics values (reflectivity, thermal_noise, etc.)
        """
        # Convert numpy array to CoatingState if needed
        if isinstance(state, np.ndarray):
            state = CoatingState.from_array(
                state,
                self.n_materials,
                self.air_material_index,
                self.substrate_material_index,
                self.materials,
            )

        # Get physics values
        reflectivity, thermal_noise, absorption, total_thickness = (
            self.compute_state_value(state)
        )

        vals = {
            "reflectivity": reflectivity,
            "thermal_noise": thermal_noise,
            "thickness": total_thickness,
            "absorption": absorption,
        }

        # Compute base rewards for all objectives
        individual_rewards = self.compute_objective_rewards(vals, normalised=normalised)

        return individual_rewards, vals

    def _initialize_reward_bounds(self):
        """Initialize reward bounds from objective_bounds config."""
        for obj in self.optimise_parameters:
            if obj in self.objective_bounds:
                bounds = self.objective_bounds[obj]
                if isinstance(bounds, (list, tuple)) and len(bounds) >= 2:
                    min_val, max_val = float(bounds[0]), float(bounds[1])
                    # Compute raw rewards at bounds
                    target = self.optimise_targets.get(obj, 0.0)
                    min_reward = -np.log(np.abs(min_val - target) + 1e-30)
                    max_reward = -np.log(np.abs(max_val - target) + 1e-30)
                    self.reward_bounds[obj] = [
                        min(min_reward, max_reward),
                        max(min_reward, max_reward)
                    ]

    def update_observed_bounds(self, vals: dict):
        """Update observed value bounds during training."""
        for obj in self.optimise_parameters:
            if obj in vals and vals[obj] is not None:
                val = float(vals[obj])
                if not np.isnan(val):
                    self.observed_value_bounds[obj]["min"] = min(
                        self.observed_value_bounds[obj]["min"], val
                    )
                    self.observed_value_bounds[obj]["max"] = max(
                        self.observed_value_bounds[obj]["max"], val
                    )

    def update_warmup_best(self, objective: str, normalised_reward: float):
        """Update best normalised reward seen during warmup."""
        old_best = self.warmup_best_rewards[objective]
        if normalised_reward > old_best:
            print(f"    WARMUP: New best {objective}={normalised_reward:.4f} (was {old_best:.4f})")
        self.warmup_best_rewards[objective] = max(old_best, normalised_reward)

    def enable_constrained_training(
        self,
        warmup_episodes_per_objective: int = 200,
        steps_per_objective: int = 10,
        epochs_per_step: int = 200,
        constraint_penalty: float = 10.0,
    ):
        """Enable two-phase constrained training.

        Phase 1 (Warmup): Optimize each objective individually
        Phase 2 (Constrained): Cycle through objectives with constraints
        """
        self.use_constrained_training = True
        self.warmup_episodes_per_objective = warmup_episodes_per_objective
        self.total_warmup_episodes = warmup_episodes_per_objective * len(self.optimise_parameters)
        self.steps_per_objective = steps_per_objective
        self.epochs_per_step = epochs_per_step
        self.constraint_penalty = constraint_penalty
        self.total_levels = steps_per_objective
        self.total_phases = self.total_levels * len(self.optimise_parameters)

    def enable_pareto_bonus(self, bonus: float = 1.0):
        """Enable pareto dominance bonus reward.

        Args:
            bonus: Reward bonus per dominated pareto front point
        """
        self.use_pareto_bonus = True
        self.pareto_dominance_bonus = bonus

    # ========================================================================
    # REWARD COMPUTATION
    # ========================================================================

    def compute_objective_rewards(
        self, vals: dict, normalised: bool = True
    ) -> Dict[str, float]:
        """Compute base rewards for all objectives.

        This is the main reward function that computes rewards for each objective.

        Args:
            vals: Dictionary of objective values (reflectivity, thermal_noise, etc.)
            normalised: If True, scale rewards to [0, 1]. If False, return raw log-based rewards.

        Returns:
            Dictionary mapping objective names to their rewards
        """
        rewards = {}

        for objective in self.optimise_parameters:
            val = vals.get(objective)
            if val is None or np.isnan(val):
                rewards[objective] = 0.0
                continue

            # Compute raw log-based reward
            target = self.optimise_targets.get(objective, 0.0)
            raw_reward = -np.log(np.abs(val - target) + 1e-30)

            if normalised:
                # Scale to [0, 1] using reward bounds
                bounds = self.reward_bounds.get(objective, [-100, 0])
                min_reward, max_reward = bounds[0], bounds[1]

                if max_reward <= min_reward:
                    rewards[objective] = 0.5
                else:
                    rewards[objective] = (raw_reward - min_reward) / (max_reward - min_reward)
            else:
                rewards[objective] = raw_reward

        return rewards

    # ========================================================================
    # REWARD MODIFIERS (applied on top of base rewards)
    # ========================================================================

    def _compute_constraint_penalty(self, vals: dict, base_rewards: Dict[str, float]) -> float:
        """Compute constraint violation penalty.

        Args:
            vals: Dictionary of objective values
            base_rewards: Dictionary of normalised base rewards for each objective

        Returns:
            Penalty value (positive number to subtract from reward)
        """
        penalty = 0.0

        for obj, threshold in self.constraints.items():
            val = vals.get(obj)
            if val is None or np.isnan(val):
                penalty += 1.0
                continue

            norm_reward = base_rewards.get(obj, 0.0)

            if norm_reward < threshold:
                violation = threshold - norm_reward
                penalty += violation * self.constraint_penalty

        return penalty

    def _compute_pareto_dominance_bonus(self, vals: dict) -> float:
        """Compute pareto dominance bonus.

        This is a reward modifier that gives extra reward if the current point
        dominates points on the pareto front.

        IMPORTANT: Uses REWARD space Pareto front for calculations.

        Args:
            vals: Dictionary of objective values

        Returns:
            Bonus reward (number of dominated points * bonus weight)
        """
        if not self.use_pareto_bonus or not self.multi_objective:
            return 0.0

        # Build reward vector for current point (normalised rewards)
        reward_dict = self.compute_objective_rewards(vals, normalised=True)
        current_reward = [reward_dict.get(param, 0.0) for param in self.optimise_parameters]

        # Count how many pareto front points are dominated (in reward space)
        dominated_count = 0
        for pareto_reward, _ in self.pareto_front_rewards:
            if self._dominates(current_reward, pareto_reward):
                dominated_count += 1

        return dominated_count * self.pareto_dominance_bonus

    def compute_training_reward(
        self,
        state,  # Can be CoatingState or numpy array
        objective_weights: Optional[Dict[str, float]] = None,
        pc_tracker=None,
        pareto_tracker=None,
        phase_info=None,
    ) -> Tuple[float, Dict[str, float], Dict[str, float]]:
        """Compute reward for training (base rewards + modifiers).

        This method:
        1. Calls compute_reward() to get base rewards
        2. Computes total reward based on training mode
        3. Adds modifiers: constraint penalty, pareto bonus

        Args:
            state: CoatingState or numpy array
            objective_weights: Weights for each objective (used in standard mode)
            pc_tracker: Optional progress tracker
            pareto_tracker: Optional pareto front tracker
            phase_info: Optional phase information

        Returns:
            Tuple of (total_reward, vals, individual_rewards)
        """
        # Get base rewards (normalised for constrained training, raw otherwise)
        normalised = self.use_constrained_training
        individual_rewards, vals = self.compute_reward(state, normalised=normalised)

        # Update observed bounds (for constrained training)
        if self.use_constrained_training:
            self.update_observed_bounds(vals)

        # Compute total reward based on training mode
        if self.use_constrained_training:
            # Constrained training mode
            if self.is_warmup:
                # Phase 1 (Warmup): Optimize single objective
                total_reward = individual_rewards.get(self.target_objective, 0.0)
                self.update_warmup_best(self.target_objective, total_reward)
            else:
                # Phase 2 (Constrained): Optimize target objective with constraints
                total_reward = individual_rewards.get(self.target_objective, 0.0)

                # Add constraint penalty modifier
                penalty = self._compute_constraint_penalty(vals, individual_rewards)
                total_reward -= penalty
                individual_rewards["constraint_penalty"] = -penalty
        else:
            # Standard mode: weighted sum of base rewards
            if objective_weights is None:
                objective_weights = self.objective_weights

            total_reward = sum(
                individual_rewards.get(param, 0.0) * objective_weights.get(param, 1.0)
                for param in self.optimise_parameters
            )

        # Add pareto dominance bonus modifier (for both modes)
        if self.use_pareto_bonus:
            pareto_bonus = self._compute_pareto_dominance_bonus(vals)
            total_reward += pareto_bonus
            individual_rewards["pareto_bonus"] = pareto_bonus

        return total_reward, vals, individual_rewards

    def compute_state_value(
        self, state: CoatingState, return_field_data: bool = False
    ) -> Tuple:
        """
        Compute physics values using coating_utils.merit_function.

        This is the ONLY interface to physics modules - does not modify them.
        """
        # Get state array - use get_array() to match base_environment behavior
        state_array = state.get_array()

        # Trim out air layers and reverse order (as base_environment does)
        state_trim = state_utils.trim_state(state_array)
        state_trim = state_trim[::-1]

        # Check if state is empty (all air layers)
        if len(state_trim) == 0:
            # Return default values for empty coating
            if return_field_data:
                return (0.0, None, 0.0, 0.0, None)
            else:
                return (0.0, None, 0.0, 0.0)

        # Call existing physics code
        result = coating_utils.merit_function(
            np.array(state_trim),
            self.materials,
            light_wavelength=self.light_wavelength,
            frequency=self.frequency,
            wBeam=self.wBeam,
            Temp=self.Temp,
            substrate_index=self.substrate_material_index,
            air_index=self.air_material_index,
            use_optical_thickness=self.use_optical_thickness,
            return_field_data=return_field_data,
        )

        if return_field_data:
            return result  # (r, thermal, absorption, thickness, field_data)
        else:
            return result  # (r, thermal, absorption, thickness)

    def sample_action_space(self) -> np.ndarray:
        """Sample random action."""
        action = np.zeros(self.n_materials + 1)

        if self.use_optical_thickness:
            action[0] = np.random.uniform(0.01, 1.0)
        else:
            action[0] = np.random.uniform(self.min_thickness, self.max_thickness)

        valid_materials = list(range(self.n_materials))
        if self.ignore_air_option:
            valid_materials = [
                m for m in valid_materials if m != self.air_material_index
            ]
        if self.ignore_substrate_option:
            valid_materials = [
                m for m in valid_materials if m != self.substrate_material_index
            ]

        material_idx = np.random.choice(valid_materials)
        action[material_idx + 1] = 1.0
        return action

    def update_pareto_front(self, objectives: Dict[str, float], state: CoatingState):
        """Update both reward and value space Pareto fronts.

        IMPORTANT: Dominance checks use REWARD space, not value space.
        Value space is only for visual diagnostics.

        Args:
            objectives: Dictionary of objective values (reflectivity, absorption, etc.)
            state: Current coating state
        """
        if not self.multi_objective:
            return

        # Get value vector
        val_vector = [objectives.get(param, 0.0) for param in self.optimise_parameters]

        # Get reward vector (normalised rewards)
        reward_dict = self.compute_objective_rewards(objectives, normalised=True)
        reward_vector = [reward_dict.get(param, 0.0) for param in self.optimise_parameters]

        # Check dominance in REWARD space
        dominated = False
        for existing_reward, _ in self.pareto_front_rewards:
            if self._dominates(existing_reward, reward_vector):
                dominated = True
                break

        if not dominated:
            # Find indices of dominated points in reward space
            dominated_indices = []
            for i, (reward, _) in enumerate(self.pareto_front_rewards):
                if self._dominates(reward_vector, reward):
                    dominated_indices.append(i)

            # Remove dominated points from BOTH fronts (keep them in sync)
            self.pareto_front_rewards = [
                (reward, s) for i, (reward, s) in enumerate(self.pareto_front_rewards)
                if i not in dominated_indices
            ]
            self.pareto_front_values = [
                (val, s) for i, (val, s) in enumerate(self.pareto_front_values)
                if i not in dominated_indices
            ]

            # Add new point to BOTH fronts
            self.pareto_front_rewards.append((reward_vector, state.copy()))
            self.pareto_front_values.append((val_vector, state.copy()))

    def _dominates(self, obj1: List[float], obj2: List[float]) -> bool:
        """Check if obj1 Pareto dominates obj2.

        Takes into account objective directions (maximize vs minimize).
        obj1 dominates obj2 if it's better or equal in all objectives
        and strictly better in at least one.

        Args:
            obj1: First objective vector [val1, val2, ...]
            obj2: Second objective vector [val1, val2, ...]

        Returns:
            True if obj1 dominates obj2
        """
        if len(obj1) != len(obj2) or len(obj1) != len(self.optimise_parameters):
            return False

        better_or_equal = True
        strictly_better = False

        for i, param_name in enumerate(self.optimise_parameters):
            maximize = self.objective_directions.get(param_name, True)

            if maximize:
                # Higher is better
                if obj1[i] < obj2[i]:
                    better_or_equal = False
                    break
                elif obj1[i] > obj2[i]:
                    strictly_better = True
            else:
                # Lower is better
                if obj1[i] > obj2[i]:
                    better_or_equal = False
                    break
                elif obj1[i] < obj2[i]:
                    strictly_better = True

        return better_or_equal and strictly_better

    def get_state(self) -> CoatingState:
        """Get current state."""
        return self.current_state

    def set_state(self, state: CoatingState):
        """Set current state."""
        self.current_state = state.copy()

    def get_pareto_front(self, space: str = "reward") -> List[Tuple[List[float], CoatingState]]:
        """Get Pareto front.

        Args:
            space: "reward" for reward space (used for calculations), "value" for value space (visual diagnostics)

        Returns:
            List of (vector, state) tuples
        """
        if space == "value":
            return self.pareto_front_values.copy()
        else:
            return self.pareto_front_rewards.copy()

    def get_parameter_names(self) -> List[str]:
        """Get list of optimization parameter names."""
        return self.optimise_parameters

    def __repr__(self) -> str:
        return (
            f"CoatingEnvironment(max_layers={self.max_layers}, "
            f"n_materials={self.n_materials}, multi_objective={self.multi_objective}, "
            f"objectives={self.optimise_parameters})"
        )
