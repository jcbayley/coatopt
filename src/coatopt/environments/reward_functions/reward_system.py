"""
Plugin-based reward system that automatically discovers reward functions.
Makes it easy to add new reward functions without modifying selectors.
"""

import copy
import importlib
import inspect
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# Import addon functions from separate module
from .reward_addons import (
    apply_air_penalty_addon,
    apply_boundary_penalties,
    apply_divergence_penalty,
    apply_pareto_improvement_addon,
    apply_preference_constraints_addon,
    reward_function_plugin,
)

# Optional import for hypervolume calculation
try:
    from pymoo.indicators.hv import HV

    PYMOO_AVAILABLE = True
except ImportError:
    PYMOO_AVAILABLE = False
    print("Warning: pymoo not available. Hypervolume calculation will not work.")


class RewardRegistry:
    """Registry for reward functions using automatic discovery."""

    def __init__(self):
        self._functions = {}
        self._discover_reward_functions()

    def _discover_reward_functions(self):
        """Automatically discover all reward functions."""
        try:
            # Import the reward function module
            from . import coating_reward_function as reward_module

            # Get all functions that start with 'reward_function'
            for name, func in inspect.getmembers(reward_module, inspect.isfunction):
                if name.startswith("reward_function"):
                    # Use the part after 'reward_function_' as the key
                    key = name.replace("reward_function_", "") or "default"
                    self._functions[key] = func

        except ImportError:
            print("Warning: Could not import coating_reward_function module")

    def register(self, name: str, func: Callable):
        """Manually register a reward function."""
        self._functions[name] = func

    def get_function(self, name: str) -> Callable:
        """Get a reward function by name."""
        if name not in self._functions:
            raise ValueError(
                f"Reward function '{name}' not found. Available: {list(self._functions.keys())}"
            )
        return self._functions[name]

    def list_functions(self) -> list:
        """List all available reward functions."""
        return list(self._functions.keys())


class RewardCalculator:
    """
    Reward calculator with plugin system and configurable addon functions.

    This class provides a flexible reward calculation system that:
    1. Automatically discovers reward functions using a plugin registry
    2. Applies configurable addon functions (penalties, improvements, etc.)
    3. Supports various reward combination methods
    4. Handles reward normalisation for multi-objective optimization
    """

    def __init__(
        self,
        reward_type="default",
        optimise_parameters=None,
        optimise_targets=None,
        # Reward normalization system
        use_reward_normalisation=True,
        reward_normalisation_mode="fixed",
        reward_normalisation_ranges=None,
        reward_normalisation_alpha=0.1,
        reward_normalisation_apply_clipping=True,
        reward_history_size=1000,
        # Addon configuration
        apply_boundary_penalties=False,
        apply_divergence_penalty=False,
        apply_air_penalty=False,
        apply_pareto_improvement=False,
        apply_preference_constraints=False,
        # Addon weights
        air_penalty_weight=1.0,
        divergence_penalty_weight=1.0,
        pareto_improvement_weight=1.0,
        preference_constraint_weight=1.0,
        # Combination method
        target_mapping=None,
        combine: str = "sum",
        **kwargs,
    ):
        """
        Initialize reward calculator with configuration.

        Args:
            reward_type: Name of base reward function to use
            optimise_parameters: List of parameters to optimize
            optimise_targets: Dict of target values for each parameter

            # Reward normalisation system
            use_reward_normalisation: Whether to normalise individual rewards before weighting
            reward_normalisation_mode: "fixed" (use provided ranges) or "adaptive" (learn from history)
            reward_normalisation_ranges: Dict mapping parameter names to [min, max] ranges for normalisation
            reward_normalisation_alpha: Learning rate for adaptive range updates
            reward_history_size: Number of recent rewards to keep for adaptive normalisation

            # Addons (applied automatically based on these flags)
            apply_boundary_penalties: Apply boundary penalty addon
            apply_divergence_penalty: Apply divergence penalty addon
            apply_air_penalty: Apply air penalty addon
            apply_pareto_improvement: Apply Pareto improvement addon
            apply_preference_constraints: Apply preference constraints addon

            # Addon weights
            air_penalty_weight: Weight for air penalty
            divergence_penalty_weight: Weight for divergence penalty
            pareto_improvement_weight: Weight for Pareto improvement
            preference_constraint_weight: Weight for preference constraints

            # Combination
            target_mapping: Parameter scaling types ("log-", "linear-", etc.)
            combine: How to combine rewards ("sum", "product", "logproduct", "hypervolume")
            **kwargs: Additional parameters passed to reward functions
        """
        # Core setup
        self.registry = RewardRegistry()
        self.reward_type = reward_type
        self.reward_function = self.registry.get_function(self.reward_type)
        self.kwargs = kwargs

        # Reward normalisation setup
        self.use_reward_normalisation = use_reward_normalisation
        self.reward_normalisation_mode = reward_normalisation_mode
        self.reward_normalisation_ranges = reward_normalisation_ranges or {}
        self.reward_normalisation_alpha = reward_normalisation_alpha
        self.reward_normalisation_apply_clipping = reward_normalisation_apply_clipping

        # History tracking for adaptive normalisation
        self.reward_history_size = reward_history_size
        if use_reward_normalisation:
            from collections import deque

            self.reward_history = {
                param: deque(maxlen=reward_history_size)
                for param in (optimise_parameters or [])
            }
            self.adaptive_ranges = {
                param: {"min": float("inf"), "max": float("-inf")}
                for param in (optimise_parameters or [])
            }

        # Optimization parameters
        self._setup_optimization_parameters(
            optimise_parameters, optimise_targets, target_mapping
        )

        # Addon configuration
        self._setup_addons(
            apply_boundary_penalties,
            apply_divergence_penalty,
            apply_air_penalty,
            apply_pareto_improvement,
            apply_preference_constraints,
            air_penalty_weight,
            divergence_penalty_weight,
            pareto_improvement_weight,
            preference_constraint_weight,
        )

        # Combination method
        self.combine = combine
        self.multi_value_rewards = kwargs.get("multi_value_rewards", False)

    def _setup_optimization_parameters(
        self, optimise_parameters, optimise_targets, target_mapping
    ):
        """Setup optimization parameters with defaults."""
        self.optimise_parameters = optimise_parameters or [
            "reflectivity",
            "thermal_noise",
            "absorption",
            "thickness",
        ]
        self.optimise_targets = optimise_targets or {
            "reflectivity": 0.99999,
            "thermal_noise": 5.394480540642821e-21,
            "absorption": 0.01,
            "thickness": 0.1,
        }
        self.target_mapping = target_mapping or {
            "reflectivity": "log-",
            "thermal_noise": "log-",
            "thickness": "linear-",
            "absorption": "log-",
        }

    def _setup_addons(
        self,
        boundary,
        divergence,
        air,
        pareto,
        preference,
        air_weight,
        div_weight,
        pareto_weight,
        pref_weight,
    ):
        """Setup addon configuration."""
        # Addon flags
        self.apply_boundary_penalties = boundary
        self.apply_divergence_penalty = divergence
        self.apply_air_penalty = air
        self.apply_pareto_improvement = pareto
        self.apply_preference_constraints = preference

        # Addon weights
        self.air_penalty_weight = air_weight
        self.divergence_penalty_weight = div_weight
        self.pareto_improvement_weight = pareto_weight
        self.preference_constraint_weight = pref_weight

    # ===== MAIN CALCULATION METHODS =====

    def calculate(
        self,
        reflectivity,
        thermal_noise,
        thickness,
        absorption,
        env=None,
        weights=None,
        expert_constraints=None,
        constraint_penalty_weight=100.0,
        pc_tracker=None,
        phase_info=None,
        **extra_kwargs,
    ):
        """
        Main reward calculation method with addon functions applied automatically.

        This method combines base reward calculation with configured addon functions
        to provide a comprehensive reward signal for optimization.

        Args:
            reflectivity: Current reflectivity value
            thermal_noise: Current thermal noise value
            thickness: Current coating thickness
            absorption: Current absorption value
            env: Environment object
            weights: Dict of weights for each parameter
            expert_constraints: Dict mapping parameter names to constraint target reward values
            constraint_penalty_weight: Weight for constraint violation penalties
            pc_tracker: PreferenceConstrainedTracker for constrained optimization
            phase_info: Phase information for preference-constrained training
            **extra_kwargs: Additional parameters

        Returns:
            tuple: (total_reward, vals_dict, rewards_dict)
        """
        # Get base reward calculation
        total_reward, vals, rewards = self.calculate_base(
            reflectivity,
            thermal_noise,
            thickness,
            absorption,
            env=env,
            weights=weights,
            expert_constraints=expert_constraints,
            constraint_penalty_weight=constraint_penalty_weight,
            **extra_kwargs,
        )

        # print("Base rewards:", rewards)

        # Apply configured addon functions
        final_reward, final_vals, final_rewards = self.apply_addon_functions(
            total_reward,
            vals,
            rewards,
            env=env,
            weights=weights,
            use_boundary_penalties=self.apply_boundary_penalties,
            use_divergence_penalty=self.apply_divergence_penalty,
            use_air_penalty=self.apply_air_penalty,
            use_pareto_improvement=self.apply_pareto_improvement,
            use_preference_constraints=self.apply_preference_constraints,
            air_penalty_weight=self.air_penalty_weight,
            divergence_penalty_weight=self.divergence_penalty_weight,
            pareto_improvement_weight=self.pareto_improvement_weight,
            constraint_penalty_weight=self.preference_constraint_weight,
            target_mapping=self.target_mapping,
            pc_tracker=pc_tracker,
            phase_info=phase_info,
            **extra_kwargs,
        )
        # print("Final rewards after addons:", final_rewards)

        # Combine rewards using configured method
        final_reward = self.combine_rewards(
            final_rewards, objective_weights=weights, env=env, vals=vals, **extra_kwargs
        )
        final_rewards["total_reward"] = final_reward

        return final_reward, final_vals, final_rewards

    # ===== MAIN CALCULATION METHODS =====

    def calculate_base(
        self,
        reflectivity,
        thermal_noise,
        thickness,
        absorption,
        env=None,
        weights=None,
        expert_constraints=None,
        constraint_penalty_weight=100.0,
        **extra_kwargs,
    ):
        """
        Calculate base reward using the selected function, with old-style normalisation if enabled.

        Args:
            expert_constraints: Dict mapping parameter names to constraint target reward values
            constraint_penalty_weight: Weight for constraint violation penalties
        """
        # Merge initialization kwargs with call-time kwargs
        call_kwargs = {**self.kwargs, **extra_kwargs}

        # Always try to pass env if available - most functions can use it
        if env is not None:

            # Auto-compute reward ranges from objective bounds if not provided
            if self.use_reward_normalisation and not self.reward_normalisation_ranges:
                computed_ranges = self._compute_reward_ranges_from_objective_bounds(env)
                if computed_ranges:
                    self.reward_normalisation_ranges = computed_ranges
                    print(
                        f"Auto-computed reward normalisation ranges: {computed_ranges}"
                    )

        # Get the original reward calculation
        total_reward, vals, rewards = self.reward_function(
            reflectivity=reflectivity,
            thermal_noise=thermal_noise,
            total_thickness=thickness,
            absorption=absorption,
            optimise_parameters=self.optimise_parameters,
            optimise_targets=self.optimise_targets,
            weights=weights,
            **self.kwargs,
        )

        # Apply reward normalisation (if enabled)
        normalised_rewards = {}
        if self.use_reward_normalisation:
            # Extract individual rewards from the rewards dict
            individual_rewards = {
                param: rewards.get(param, 0.0) for param in self.optimise_parameters
            }

            # normalise individual rewards
            normalised_rewards = self._normalise_rewards(individual_rewards)

            # Update rewards dict with normalised values (for debugging/logging)
            for param in self.optimise_parameters:
                rewards[f"{param}_normalised"] = normalised_rewards[param]
                rewards[f"{param}"] = normalised_rewards[param]
        else:
            # No normalisation - use original rewards
            normalised_rewards = {
                param: rewards.get(param, 0.0) for param in self.optimise_parameters
            }

        # Apply constraints if provided (to normalised rewards)
        if expert_constraints:
            total_reward, rewards = self._apply_expert_constraints(
                total_reward,
                rewards,
                expert_constraints,
                constraint_penalty_weight,
                normalised_rewards,
            )
        else:
            # No constraints - use standard weighted sum of normalised rewards (if
            # weights provided)
            if weights is not None and self.use_reward_normalisation:
                total_reward = sum(
                    weights.get(param, 0.0) * normalised_rewards.get(param, 0.0)
                    for param in self.optimise_parameters
                )

        return total_reward, vals, rewards

    # ===== REWARD COMBINATION METHODS =====

    def combine_rewards(
        self,
        rewards: Dict[str, float],
        objective_weights: Dict[str, float] = None,
        env=None,
        vals: Dict[str, float] = None,
        **kwargs,
    ) -> float:
        """
        Combine individual rewards into a single total reward.

        Args:
            rewards: Dict of individual rewards
            objective_weights: Weights for each objective
            env: Environment object (needed for hypervolume calculation)
            vals: Current objective values (needed for hypervolume calculation)

        Returns:
            Combined total reward
        """
        # Simple sum of all rewards as an example
        if objective_weights is None:
            objective_weights = {param: 1.0 for param in self.optimise_parameters}

        # Combine the rewards
        if self.combine == "sum":
            total_reward = np.sum(
                [
                    rewards[key] * objective_weights[key]
                    for key in self.optimise_parameters
                ]
            )
        elif self.combine == "product":
            total_reward = np.prod(
                [
                    rewards[key] * objective_weights[key]
                    for key in self.optimise_parameters
                ]
            )
        elif self.combine == "logproduct":
            total_reward = np.log(
                np.prod(
                    [
                        rewards[key] * objective_weights[key]
                        for key in self.optimise_parameters
                    ]
                )
            )
        elif self.combine == "hypervolume":
            total_reward = self._calculate_hypervolume_reward(
                env, vals, rewards, **kwargs
            )
        else:
            raise ValueError(
                f"combine must be either 'sum', 'product', 'logproduct', or 'hypervolume', not {self.combine}"
            )

        # apply addons
        for key in rewards.keys():
            if "addon" in key:
                total_reward += rewards[key]
                rewards["total_reward"] = total_reward

        return total_reward

        # ===== ADDON APPLICATION METHODS =====

    def apply_addon_functions(
        self,
        total_reward: float,
        vals: Dict[str, float],
        rewards: Dict[str, float],
        env=None,
        weights: Dict[str, float] = None,
        use_boundary_penalties: bool = False,
        use_divergence_penalty: bool = False,
        use_air_penalty: bool = False,
        use_pareto_improvement: bool = False,
        use_preference_constraints: bool = False,
        air_penalty_weight: float = 1.0,
        divergence_penalty_weight: float = 1.0,
        pareto_improvement_weight: float = 1.0,
        pc_tracker=None,
        phase_info=None,
        constraint_penalty_weight: float = 1.0,
        target_mapping: Dict[str, str] = None,
        **kwargs,
    ) -> Tuple[float, Dict[str, float], Dict[str, float]]:
        """
        Apply addon functions to reward calculation results.

        Args:
            total_reward: Current total reward
            vals: Current objective values
            rewards: Individual rewards dict
            env: Environment object
            weights: Parameter weights
            use_boundary_penalties: Whether to apply boundary penalties
            use_divergence_penalty: Whether to apply divergence penalty
            use_air_penalty: Whether to apply air penalty
            use_pareto_improvement: Whether to apply Pareto improvement reward
            use_preference_constraints: Whether to apply preference constraints addon
            air_penalty_weight: Weight for air penalty
            divergence_penalty_weight: Weight for divergence penalty
            pareto_improvement_weight: Weight for Pareto improvement reward
            pc_tracker: PreferenceConstrainedTracker instance
            phase_info: Phase information from preference-constrained training
            target_mapping: Mapping of parameter names to scaling types
            **kwargs: Additional parameters for addon functions

        Returns:
            Tuple of (updated_total_reward, updated_vals, updated_rewards)
        """
        updated_total_reward = total_reward
        updated_vals = vals.copy()
        updated_rewards = rewards.copy()

        # Apply boundary penalties
        if use_boundary_penalties and env is not None:
            updated_rewards = apply_boundary_penalties(
                updated_rewards,
                vals,
                self.optimise_parameters,
                env,
                target_mapping,
            )

        # Apply divergence penalty
        if use_divergence_penalty:
            updated_rewards = apply_divergence_penalty(
                updated_rewards,
                self.optimise_parameters,
                weights,
                divergence_penalty_weight,
                self.multi_value_rewards,
            )

        # Apply air penalty addon
        if use_air_penalty:
            updated_rewards = apply_air_penalty_addon(
                updated_total_reward,
                updated_rewards,
                vals,
                env,
                self.optimise_parameters,
                air_penalty_weight,
                self.multi_value_rewards,
                **kwargs,
            )

        # Apply Pareto improvement addon
        if use_pareto_improvement:
            updated_rewards = apply_pareto_improvement_addon(
                updated_total_reward,
                updated_rewards,
                vals,
                env,
                self.optimise_parameters,
                pareto_improvement_weight,
                self.multi_value_rewards,
                **kwargs,
            )

        # Apply preference constraints addon
        if use_preference_constraints:
            updated_total_reward, updated_rewards = apply_preference_constraints_addon(
                updated_total_reward,
                updated_rewards,
                vals,
                env,
                self.optimise_parameters,
                pc_tracker,
                phase_info,
                constraint_penalty_weight,
                **kwargs,
            )

        return updated_total_reward, updated_vals, updated_rewards

        # ===== REWARD NORMALIZATION METHODS =====

    def _compute_reward_ranges_from_objective_bounds(
        self, env
    ) -> Dict[str, List[float]]:
        """
        Compute expected reward ranges by evaluating the reward function at objective bounds.

        Args:
            env: Environment instance with objective_bounds attribute

        Returns:
            Dict mapping parameter names to [min_reward, max_reward] ranges
        """
        if not hasattr(env, "objective_bounds"):
            return {}

        bounds = env.objective_bounds
        if not bounds:
            return {}

        ranges = {}

        # Dummy values for parameters not being tested
        dummy_values = {
            "reflectivity": 0.999,
            "absorption": 1e-6,
            "thermal_noise": 1e-20,
            "thickness": 5.0,
        }

        # For each parameter we want to optimize
        for param in self.optimise_parameters:
            if param not in bounds:
                continue

            # Expect bounds in [min, max] format from config
            if not isinstance(bounds[param], (list, tuple)) or len(bounds[param]) != 2:
                continue

            min_bound, max_bound = bounds[param]

            # Ensure bounds are numeric
            try:
                min_bound = float(min_bound)
                max_bound = float(max_bound)
            except (ValueError, TypeError):
                continue

            reward_values = []

            # Test at min and max bounds only (rewards are monotonic)
            for test_value in [min_bound, max_bound]:
                # Set up test parameters
                test_params = dummy_values.copy()
                test_params[param] = test_value

                try:
                    # Call the reward function
                    _, _, rewards = self.reward_function(
                        reflectivity=test_params["reflectivity"],
                        thermal_noise=test_params["thermal_noise"],
                        total_thickness=test_params["thickness"],
                        absorption=test_params["absorption"],
                        optimise_parameters=[param],
                        optimise_targets=self.optimise_targets,
                        **self.kwargs,
                    )

                    if param in rewards:
                        reward_values.append(rewards[param])

                except Exception as e:
                    print(
                        f"Warning: Error evaluating reward function at {param}={test_value}: {e}"
                    )
                    continue

            if len(reward_values) >= 2:
                ranges[param] = [min(reward_values), max(reward_values)]

        return ranges

    def _normalise_rewards(
        self, individual_rewards: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Normalise individual rewards to [0, 1] range.

        This preserves relative reward magnitudes while scaling to a standard range.

        Args:
            individual_rewards: Dict mapping parameter names to their individual reward values

        Returns:
            Dict mapping parameter names to normalised rewards [0, 1]
        """
        if not self.use_reward_normalisation:
            return individual_rewards

        normalised_rewards = {}

        for param, reward in individual_rewards.items():
            if param not in self.optimise_parameters:
                normalised_rewards[param] = reward
                continue

            # Update history for adaptive mode
            if hasattr(self, "reward_history"):
                self.reward_history[param].append(reward)
            # Get normalisation range
            if (
                self.reward_normalisation_mode == "fixed"
                and param in self.reward_normalisation_ranges
            ):
                min_val, max_val = self.reward_normalisation_ranges[param]
            elif (
                self.reward_normalisation_mode == "adaptive"
                and hasattr(self, "reward_history")
                and len(self.reward_history[param]) > 10
            ):
                # Use moving average for adaptive range
                history = list(self.reward_history[param])
                current_min, current_max = min(history), max(history)

                # Update adaptive range with exponential moving average
                if self.adaptive_ranges[param]["min"] == float("inf"):
                    self.adaptive_ranges[param]["min"] = current_min
                    self.adaptive_ranges[param]["max"] = current_max
                else:
                    alpha = self.reward_normalisation_alpha
                    self.adaptive_ranges[param]["min"] = (
                        1 - alpha
                    ) * self.adaptive_ranges[param]["min"] + alpha * current_min
                    self.adaptive_ranges[param]["max"] = (
                        1 - alpha
                    ) * self.adaptive_ranges[param]["max"] + alpha * current_max

                min_val = self.adaptive_ranges[param]["min"]
                max_val = self.adaptive_ranges[param]["max"]
            else:
                # No normalisation available yet
                normalised_rewards[param] = reward
                continue

            # normalise to [0, 1]
            if max_val > min_val:
                normalised_rewards[param] = (reward - min_val) / (max_val - min_val)
            else:
                normalised_rewards[param] = reward

            if self.reward_normalisation_apply_clipping:
                normalised_rewards[param] = np.clip(normalised_rewards[param], 0.0, 1.0)

            # print(param, reward, normalised_rewards[param], min_val, max_val)
        return normalised_rewards

    def _calculate_hypervolume_reward(
        self, env, vals: Dict[str, float], rewards: Dict[str, float], **kwargs
    ) -> float:
        """
        Calculate reward based on hypervolume of Pareto front including the new point.

        Args:
            env: Environment object containing pareto_tracker
            vals: Current objective values
            rewards: Individual rewards dict (for fallback)

        Returns:
            Hypervolume-based reward
        """
        if not PYMOO_AVAILABLE:
            print(
                "Warning: pymoo not available for hypervolume calculation. Falling back to sum."
            )
            return np.sum([rewards[key] for key in self.optimise_parameters])

        # Check if pareto tracker is provided via kwargs first, then try env
        pareto_tracker = kwargs.get("pareto_tracker")
        if pareto_tracker is None:
            if (
                env is None
                or not hasattr(env, "pareto_tracker")
                or env.pareto_tracker is None
            ):
                raise Exception(
                    "Warning: No pareto_tracker available in kwargs or environment. Falling back to sum."
                )
            pareto_tracker = env.pareto_tracker

        try:
            # Get current Pareto front
            # pareto_tracker is now obtained from above logic

            # Extract current objective values for the optimized parameters
            current_point = np.array(
                [rewards.get(param, 0.0) for param in self.optimise_parameters]
            )
            current_val = np.array(
                [vals.get(param, 0.0) for param in self.optimise_parameters]
            )
            current_state = env.current_state.get_array()

            # Create a temporary copy of the tracker to test with the new point
            import copy

            temp_tracker = copy.deepcopy(pareto_tracker)
            temp_tracker.add_point(
                current_point, current_val, current_state, force_update=True
            )

            # Get the Pareto front points (including the new point if it's
            # non-dominated)
            pareto_points = temp_tracker.get_front()

            if len(pareto_points) == 0:
                print("Warning: Empty Pareto front. Falling back to sum.")
                return np.sum([rewards[key] for key in self.optimise_parameters])

            # Convert to numpy array if needed
            if not isinstance(pareto_points, np.ndarray):
                pareto_points = np.array(pareto_points)

            # Ensure we have a 2D array
            if pareto_points.ndim == 1:
                pareto_points = pareto_points.reshape(1, -1)

            # Define reference point for hypervolume calculation
            # Use the worst values from objective bounds or current worst in Pareto set
            reference_point = self._get_reference_point(env, pareto_points)

            # Calculate hypervolume using pymoo
            hv_indicator = HV(ref_point=reference_point)

            # For minimization problems, we need to negate the objectives
            # Check optimization direction from target_mapping or assume minimization
            pareto_for_hv = self._prepare_objectives_for_hypervolume(pareto_points)

            hypervolume = hv_indicator(pareto_for_hv)

            # Store hypervolume info in rewards dict for debugging
            rewards["hypervolume_value"] = hypervolume
            rewards["pareto_front_size"] = len(pareto_points)

            return hypervolume

        except Exception as e:
            print(f"Warning: Error calculating hypervolume: {e}. Falling back to sum.")
            return np.sum([rewards[key] for key in self.optimise_parameters])

    # ===== HYPERVOLUME CALCULATION HELPERS =====

    def _get_reference_point(self, env, pareto_points: np.ndarray) -> np.ndarray:
        """
        Get reference point for hypervolume calculation.

        Args:
            env: Environment object
            pareto_points: Current Pareto front points

        Returns:
            Reference point for hypervolume calculation
        """
        # Method 1: Use objective bounds if available
        if hasattr(env, "objective_bounds") and env.objective_bounds:
            reference_point = []
            for param in self.optimise_parameters:
                if param in env.objective_bounds:
                    bounds = env.objective_bounds[param]
                    if isinstance(bounds, list):
                        # Use the worse bound based on optimization direction
                        if self.target_mapping.get(param, "").endswith("-"):
                            # For maximization, use minimum as reference
                            reference_point.append(bounds[0])
                        else:
                            # For minimization, use maximum as reference
                            reference_point.append(bounds[1])
                    else:
                        # Fallback: use worst point in current front + margin
                        reference_point.append(
                            np.max(pareto_points[:, len(reference_point)]) * 1.1
                        )
                else:
                    # Fallback: use worst point in current front + margin
                    reference_point.append(
                        np.max(pareto_points[:, len(reference_point)]) * 1.1
                    )
            return np.array(reference_point)

        # Method 2: Use worst points in current front + margin
        return np.max(pareto_points, axis=0) * 1.1

    def _prepare_objectives_for_hypervolume(
        self, pareto_points: np.ndarray
    ) -> np.ndarray:
        """
        Prepare objectives for hypervolume calculation by handling maximization/minimization.

        Args:
            pareto_points: Pareto front points

        Returns:
            Transformed points for hypervolume calculation (pymoo expects minimization)
        """
        transformed_points = pareto_points.copy()

        # Transform maximization objectives to minimization by negating
        for i, param in enumerate(self.optimise_parameters):
            if i < transformed_points.shape[1]:
                # Check if this is a maximization objective
                if self.target_mapping.get(param, "").endswith("-"):
                    # This is maximization, negate for pymoo (which expects
                    # minimization)
                    transformed_points[:, i] = -transformed_points[:, i]

        return transformed_points

    def _apply_expert_constraints(
        self,
        total_reward: float,
        rewards: Dict[str, float],
        expert_constraints: Dict[str, float],
        constraint_penalty_weight: float = 100.0,
        normalised_rewards: Dict[str, float] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Apply expert constraints to modify reward calculation.

        For constrained objectives: Apply large penalty for deviation from target reward
        For unconstrained objectives: Use normalised reward contribution

        Args:
            total_reward: Original total reward
            rewards: Dict of individual parameter rewards
            expert_constraints: Dict mapping parameter names to target reward values
            constraint_penalty_weight: Weight for constraint violation penalties
            normalised_rewards: Dict of normalised reward values to use for unconstrained params

        Returns:
            Tuple of (modified_total_reward, updated_rewards_dict)
        """
        if normalised_rewards is None:
            normalised_rewards = {
                param: rewards.get(param, 0.0) for param in self.optimise_parameters
            }

        constrained_reward = 0.0
        updated_rewards = rewards.copy()

        for param in self.optimise_parameters:
            normalised_reward = normalised_rewards.get(param, 0.0)

            if param in expert_constraints:
                # This parameter is constrained - apply penalty for deviation from
                # target
                target_reward = expert_constraints[param]
                constraint_penalty = (
                    -abs(normalised_reward - target_reward) * constraint_penalty_weight
                )
                updated_rewards[f"{param}_constraint_penalty"] = constraint_penalty
                constrained_reward += constraint_penalty
            else:
                # This parameter is unconstrained - use normalised reward
                constrained_reward += normalised_reward

        return constrained_reward, updated_rewards


# Example of how to add new reward functions easily:
@reward_function_plugin("example_new")
def reward_function_example_new(
    reflectivity,
    thermal_noise,
    total_thickness,
    absorption,
    optimise_parameters,
    optimise_targets,
    env=None,
    **kwargs,
):
    """
    Example of how to add a new reward function.
    Just define it with this decorator and it's automatically available.
    All reward functions should accept env as a parameter to access environment state.
    """
    # Simple example: negative sum of all parameters
    return -(reflectivity + thermal_noise + total_thickness + absorption), {}, {}
