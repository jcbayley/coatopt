"""
Simplified unified coating environment.

Reads configuration directly from config object to avoid parameter passing errors.
Uses existing physics modules (coating_utils, EFI_tmm, YAM_CoatingBrownian)
without modification.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ..environments.core.state import CoatingState
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

        # Environment state
        self.current_state = None
        self.current_index = 0
        self.done = False

        # Multi-objective tracking
        self.pareto_front = []
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
            total_reward, vals, reward_components = self.compute_reward(
                self.current_state,
                objective_weights=objective_weights,
                pc_tracker=pc_tracker,
                phase_info=phase_info
            )
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
        objective_weights: Optional[Dict[str, float]] = None,
        pc_tracker=None,
        pareto_tracker=None,
        phase_info=None,
    ) -> Tuple[float, Dict, Dict]:
        """Main reward computation - selects appropriate reward function based on training mode.

        Returns:
            Tuple of (total_reward, vals, individual_rewards)
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

        # Compute individual rewards (normalised in constrained mode, raw otherwise)
        individual_rewards = {}
        for param in self.optimise_parameters:
            val = vals.get(param)
            if val is not None:
                if self.use_constrained_training:
                    # Use normalised rewards [0, 1] for constrained training
                    individual_rewards[param] = self._compute_normalised_reward(param, val)
                else:
                    # Use raw rewards for standard training
                    individual_rewards[param] = self._compute_raw_reward(param, val)

        # Compute total reward based on training mode
        if self.use_constrained_training:
            # Constrained training mode (warmup or constrained phase)
            total_reward = self._compute_training_reward(vals)
        else:
            # Standard mode: weighted sum of raw rewards
            if objective_weights is None:
                objective_weights = self.objective_weights
            total_reward = sum(
                individual_rewards.get(param, 0.0) * objective_weights.get(param, 1.0)
                for param in self.optimise_parameters
            )

        return total_reward, vals, individual_rewards

    def _initialize_reward_bounds(self):
        """Initialize reward bounds from objective_bounds config."""
        for obj in self.optimise_parameters:
            if obj in self.objective_bounds:
                bounds = self.objective_bounds[obj]
                if isinstance(bounds, (list, tuple)) and len(bounds) >= 2:
                    min_val, max_val = float(bounds[0]), float(bounds[1])
                    # Compute rewards at bounds
                    min_reward = self._compute_raw_reward(obj, min_val)
                    max_reward = self._compute_raw_reward(obj, max_val)
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
        self.warmup_best_rewards[objective] = max(
            self.warmup_best_rewards[objective], normalised_reward
        )

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

    # ========================================================================
    # REWARD FUNCTIONS (3 types: raw, normalised, constrained)
    # ========================================================================

    def _compute_raw_reward(self, objective: str, value: float) -> float:
        """1. Raw reward: log-based, unbounded."""
        target = self.optimise_targets.get(objective, 0.0)
        return -np.log(np.abs(value - target) + 1e-30)
        #return 1./np.abs(value - target)

    def _compute_normalised_reward(self, objective: str, value: float) -> float:
        """2. normalised reward: scales raw reward to [0, 1]."""
        raw_reward = self._compute_raw_reward(objective, value)
        bounds = self.reward_bounds.get(objective, [-100, 0])
        min_reward, max_reward = bounds[0], bounds[1]

        if max_reward <= min_reward:
            return 0.5

        normalised = (raw_reward - min_reward) / (max_reward - min_reward)
        return normalised

    def _compute_constrained_reward(self, vals: dict) -> float:
        """3. Constrained reward: normalised reward with constraint penalties."""
        target_val = vals.get(self.target_objective)
        if target_val is None or np.isnan(target_val):
            return 0.0

        # Get normalised reward for target objective
        reward = self._compute_normalised_reward(self.target_objective, target_val)

        # Apply constraint penalties
        penalty = 0.0
        for obj, threshold in self.constraints.items():
            val = vals.get(obj)
            if val is None or np.isnan(val):
                penalty += 1.0
                continue

            norm_reward = self._compute_normalised_reward(obj, val)

            if norm_reward < threshold:
                violation = threshold - norm_reward
                penalty += violation * self.constraint_penalty

        return reward - penalty

    def _compute_training_reward(self, vals: dict) -> float:
        """Compute reward based on training phase (warmup or constrained)."""
        # Update observed bounds
        self.update_observed_bounds(vals)

        if self.is_warmup:
            # Phase 1 (Warmup): Optimize single objective, track best
            target_val = vals.get(self.target_objective)
            if target_val is None or np.isnan(target_val):
                return 0.0

            reward = self._compute_normalised_reward(self.target_objective, target_val)
            self.update_warmup_best(self.target_objective, reward)
            return reward
        else:
            # Phase 2 (Constrained): Optimize target with constraints
            return self._compute_constrained_reward(vals)

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
        """Update Pareto front (for multi-objective)."""
        if not self.multi_objective:
            return

        obj_vector = [objectives.get(param, 0.0) for param in self.optimise_parameters]

        dominated = False
        for existing_obj, _ in self.pareto_front:
            if self._dominates(existing_obj, obj_vector):
                dominated = True
                break

        if not dominated:
            self.pareto_front = [
                (obj, s)
                for obj, s in self.pareto_front
                if not self._dominates(obj_vector, obj)
            ]
            self.pareto_front.append((obj_vector, state.copy()))

    def _dominates(self, obj1: List[float], obj2: List[float]) -> bool:
        """Check Pareto dominance."""
        better_in_all = all(o1 >= o2 for o1, o2 in zip(obj1, obj2))
        better_in_one = any(o1 > o2 for o1, o2 in zip(obj1, obj2))
        return better_in_all and better_in_one

    def get_state(self) -> CoatingState:
        """Get current state."""
        return self.current_state

    def set_state(self, state: CoatingState):
        """Set current state."""
        self.current_state = state.copy()

    def get_pareto_front(self) -> List[Tuple[List[float], CoatingState]]:
        """Get Pareto front."""
        return self.pareto_front.copy()

    def get_parameter_names(self) -> List[str]:
        """Get list of optimization parameter names."""
        return self.optimise_parameters

    def __repr__(self) -> str:
        return (
            f"CoatingEnvironment(max_layers={self.max_layers}, "
            f"n_materials={self.n_materials}, multi_objective={self.multi_objective}, "
            f"objectives={self.optimise_parameters})"
        )
