"""
Pareto optimization coating environment.
Extends the base CoatingStack with multi-objective optimization capabilities.
"""

import logging
import random
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from ..config.structured_config import CoatingOptimisationConfig
from .core.base_environment import BaseCoatingEnvironment
from .core.state import CoatingState
from .hppo_environment import HPPOEnvironment
from .reward_functions.reward_system import RewardCalculator
from .utils import coating_utils, state_utils


class MultiObjectiveEnvironment(HPPOEnvironment):
    """
    Coating environment with Pareto multi-objective optimization.
    This extends the base functionality to support multi-objective optimization.
    """

    def __init__(self, config: Optional[CoatingOptimisationConfig] = None, **kwargs):
        """Initialize Pareto environment."""
        super().__init__(config, **kwargs)

        # Parse optimization directions from optimise_parameters
        self.optimization_directions = self._parse_optimization_parameters()

        # Enable multi-objective optimization
        self.multi_objective = True
        self.pareto_objectives = ["reflectivity", "thermal_noise", "absorption"]

        # Override reward calculator with normalisation support
        reward_type = (
            "default" if self.reward_function is None else str(self.reward_function)
        )

        self.reward_calculator = RewardCalculator(
            reward_type=reward_type,
            optimise_parameters=self.get_parameter_names(),  # Use clean parameter names
            optimise_targets=self.optimise_targets,
            combine=self.combine,
            env=self,
            # Reward normalisation system
            use_reward_normalisation=self.use_reward_normalisation,
            reward_normalisation_mode=self.reward_normalisation_mode,
            reward_normalisation_ranges=self.reward_normalisation_ranges,
            reward_normalisation_alpha=self.reward_normalisation_alpha,
            reward_normalisation_apply_clipping=self.reward_normalisation_apply_clipping,
            # Addon system
            apply_boundary_penalties=self.apply_boundary_penalties,
            apply_divergence_penalty=self.apply_divergence_penalty,
            apply_air_penalty=self.apply_air_penalty,
            apply_pareto_improvement=self.apply_pareto_improvement,
            air_penalty_weight=self.air_penalty_weight,
            divergence_penalty_weight=self.divergence_penalty_weight,
            pareto_improvement_weight=self.pareto_improvement_weight,
        )

        # Pareto tracking is now handled by the trainer - remove local tracking
        # Keep pareto_front attribute for backward compatibility (will be synced
        # from trainer)
        self.pareto_front = []

    def setup_multiobjective_specific_attributes(self, **kwargs):
        """Setup multi-objective specific attributes."""
        # Enable multi-objective optimization
        self.multi_objective = True
        self.pareto_objectives = ["reflectivity", "thermal_noise", "absorption"]
        # pareto_front is maintained for compatibility but updated by trainer
        self.pareto_front = []
        self.all_points = []
        self.all_vals = []

    def _parse_optimization_parameters(self):
        """
        Parse optimization parameters and directions from config.

        Supports formats like:
        - ["reflectivity:max", "absorption:min"]
        - ["reflectivity", "absorption"] (fallback to defaults)

        Returns:
            dict: {parameter_name: direction} where direction is 'max' or 'min'
        """
        optimization_directions = {}

        for param in self.optimise_parameters:
            if isinstance(param, str) and ":" in param:
                # New format: "parameter:direction"
                param_name, direction = param.split(":", 1)
                param_name = param_name.strip()
                direction = direction.strip().lower()

                if direction in ["max", "maximize", "maximum"]:
                    optimization_directions[param_name] = "max"
                elif direction in ["min", "minimize", "minimum"]:
                    optimization_directions[param_name] = "min"
                else:
                    print(
                        f"Warning: Unknown optimization direction '{direction}' for {param_name}, defaulting to 'min'"
                    )
                    optimization_directions[param_name] = "min"
            else:
                # Legacy format: just parameter name, use defaults
                param_name = param if isinstance(param, str) else str(param)
                if param_name.lower() == "reflectivity":
                    optimization_directions[param_name] = (
                        "max"  # Default: maximize reflectivity
                    )
                else:
                    optimization_directions[param_name] = (
                        "min"  # Default: minimize others
                    )

        return optimization_directions

    def compute_reward(
        self,
        new_state,
        max_value=0.0,
        target_reflectivity=1.0,
        objective_weights=None,
        pc_tracker=None,
        pareto_tracker=None,
        phase_info=None,
    ):
        """Compute reward for the given state.

        Args:
            new_state: The coating state to evaluate
            max_value: Maximum value (unused, kept for compatibility)
            target_reflectivity: Target reflectivity (unused, kept for compatibility)
            objective_weights: Weights for multi-objective optimization
            pc_tracker: Preference constrained tracker (passed through)
            pareto_tracker: Pareto tracker for hypervolume calculations (passed through)
            phase_info: Phase information for preference constrained optimization (passed through)

        Returns:
            tuple: (total_reward, vals_dict, rewards_dict)
        """

        new_reflectivity, new_thermal_noise, new_E_integrated, new_total_thickness = (
            self.compute_state_value(new_state, return_separate=True)
        )

        if objective_weights is not None:
            weights = {
                key: objective_weights[i]
                for i, key in enumerate(self.get_parameter_names())
            }
        else:
            weights = None

        total_reward, vals, rewards = self.reward_calculator.calculate(
            reflectivity=new_reflectivity,
            thermal_noise=new_thermal_noise,
            thickness=new_total_thickness,
            absorption=new_E_integrated,
            weights=weights,  # Pass weights directly to calculator
            expert_constraints=getattr(self, "current_expert_constraints", None),
            env=self,
            pc_tracker=pc_tracker,
            pareto_tracker=pareto_tracker,
            phase_info=phase_info,
        )

        return total_reward, vals, rewards

    def step(
        self,
        action,
        max_state=0,
        verbose=False,
        state=None,
        layer_index=None,
        always_return_value=False,
        objective_weights=None,
        pc_tracker=None,
        phase_info=None,
        **kwargs,
    ):
        """Step function simplified to work directly with CoatingState objects."""

        # Extract pareto_tracker from kwargs
        pareto_tracker = kwargs.get("pareto_tracker", None)

        # Use current state if none provided
        if state is None:
            state = self.current_state
        else:
            self.current_state = state

        # Use current index if none provided
        if layer_index is None:
            layer_index = self.current_index
        else:
            self.current_index = layer_index

        # Extract action parameters (now action is a simple list)
        material = int(action[0])
        thickness = float(action[1])

        # Use the base class update_state method
        self.current_state, new_layer = self.update_state(
            self.current_state, thickness, material
        )

        # Initialize default values
        neg_reward = -1000
        reward = neg_reward
        terminated = False
        finished = False
        full_action = None

        rewards = {
            "reflectivity": 0,
            "thermal_noise": 0,
            "thickness": 0,
            "absorption": 0,
            "total_reward": 0,
        }
        vals = {"reflectivity": 0, "thermal_noise": 0, "thickness": 0, "absorption": 0}

        # Check termination conditions
        if (
            self.min_thickness > thickness
            or thickness > self.max_thickness
            or not np.isfinite(thickness)
        ):
            print("out of thickness bounds")
        elif (
            self.current_index == self.max_layers - 1
            or material == self.air_material_index
        ):
            # Episode finished
            finished = True
            reward, vals, rewards = self.compute_reward(
                self.current_state,
                max_state,
                objective_weights=objective_weights,
                pc_tracker=pc_tracker,
                pareto_tracker=pareto_tracker,
                phase_info=phase_info,
            )
        elif self.use_intermediate_reward:
            # Intermediate reward calculation
            reward, vals, rewards = self.compute_reward(
                self.current_state,
                max_state,
                objective_weights=objective_weights,
                pc_tracker=pc_tracker,
                pareto_tracker=pareto_tracker,
                phase_info=phase_info,
            )

        # Check for invalid states using CoatingState validation
        if not self.current_state.is_valid() or np.isnan(reward) or np.isinf(reward):
            reward = neg_reward
            terminated = True

        # Pareto front updates are now handled by the trainer
        # Remove all Pareto update logic that was previously here

        # Update tracking variables
        self.previous_material = material
        self.length += 1
        self.current_index += 1

        return (
            self.current_state,
            rewards,
            terminated,
            finished,
            reward,
            full_action,
            vals,
        )
