"""
Plugin-based reward system that automatically discovers reward functions.
Makes it easy to add new reward functions without modifying selectors.
"""
import importlib
import inspect
from typing import Dict, Callable, Any, List, Optional, Tuple
import numpy as np
from collections import deque
import torch

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
                if name.startswith('reward_function'):
                    # Use the part after 'reward_function_' as the key
                    key = name.replace('reward_function_', '') or 'default'
                    self._functions[key] = func
                    
        except ImportError:
            print("Warning: Could not import coating_reward_function module")
    
    def register(self, name: str, func: Callable):
        """Manually register a reward function."""
        self._functions[name] = func
    
    def get_function(self, name: str) -> Callable:
        """Get a reward function by name."""
        if name not in self._functions:
            raise ValueError(f"Reward function '{name}' not found. Available: {list(self._functions.keys())}")
        return self._functions[name]
    
    def list_functions(self) -> list:
        """List all available reward functions."""
        return list(self._functions.keys())


class RewardCalculator:
    """Simplified reward calculator with automatic function discovery and optional reward normalization."""
    
    def __init__(self, reward_type="default", optimise_parameters=None, optimise_targets=None, 
                 use_reward_normalization=False, reward_normalization_mode="fixed",
                 reward_normalization_ranges=None, reward_normalization_alpha=0.1,
                 reward_history_size=1000, 
                 # Addon configuration parameters
                 apply_normalization=False, apply_boundary_penalties=False,
                 apply_divergence_penalty=False, apply_air_penalty=False,
                 apply_pareto_improvement=False, apply_preference_constraints=False,
                 air_penalty_weight=1.0, divergence_penalty_weight=1.0,
                 pareto_improvement_weight=1.0, preference_constraint_weight=1.0,
                 target_mapping=None, combine: str = "sum", **kwargs):
        """
        Initialize with basic parameters and addon configuration.
        
        Args:
            reward_type: Name of reward function to use
            optimise_parameters: Parameters to optimize
            optimise_targets: Target values
            use_reward_normalization: Whether to normalize individual rewards before weighting
            reward_normalization_mode: "fixed" (use provided ranges) or "adaptive" (learn from history)
            reward_normalization_ranges: Dict mapping parameter names to [min, max] ranges for normalization
            reward_normalization_alpha: Learning rate for adaptive range updates
            reward_history_size: Number of recent rewards to keep for adaptive normalization
            # Addon configuration
            apply_normalization: Whether to apply normalization addon by default
            apply_boundary_penalties: Whether to apply boundary penalties by default
            apply_divergence_penalty: Whether to apply divergence penalty by default
            apply_air_penalty: Whether to apply air penalty by default
            apply_pareto_improvement: Whether to apply Pareto improvement reward by default
            air_penalty_weight: Weight for air penalty
            divergence_penalty_weight: Weight for divergence penalty
            pareto_improvement_weight: Weight for Pareto improvement reward
            target_mapping: Mapping of parameter names to scaling types
            combine: Method to combine rewards - "sum", "product", "logproduct", or "hypervolume"
            **kwargs: Additional parameters passed to reward function
        """
        self.registry = RewardRegistry()
        self.reward_type = reward_type
        self.optimise_parameters = optimise_parameters or ["reflectivity", "thermal_noise", "absorption", "thickness"]
        self.optimise_targets = optimise_targets or {
            "reflectivity": 0.99999,
            "thermal_noise": 5.394480540642821e-21,
            "absorption": 0.01,
            "thickness": 0.1
        }
        self.kwargs = kwargs
        
        # Reward normalization setup
        self.use_reward_normalization = use_reward_normalization
        self.reward_normalization_mode = reward_normalization_mode
        self.reward_normalization_ranges = reward_normalization_ranges or {}
        self.reward_normalization_alpha = reward_normalization_alpha
    
        self.combine = combine  # "sum", "product", "logproduct", or "hypervolume"

        self.multi_value_rewards = kwargs.get('multi_value_rewards', False)
        
        # History tracking for adaptive normalization
        self.reward_history_size = reward_history_size
        self.reward_history = {param: deque(maxlen=reward_history_size) for param in self.optimise_parameters}
        self.adaptive_ranges = {param: {"min": float('inf'), "max": float('-inf')} for param in self.optimise_parameters}
        
        # Addon configuration
        self.apply_normalization = apply_normalization
        self.apply_boundary_penalties = apply_boundary_penalties
        self.apply_divergence_penalty = apply_divergence_penalty
        self.apply_air_penalty = apply_air_penalty
        self.apply_pareto_improvement = apply_pareto_improvement
        self.apply_preference_constraints = apply_preference_constraints
        print("Preference constraint:", self.apply_preference_constraints)
        self.air_penalty_weight = air_penalty_weight
        self.divergence_penalty_weight = divergence_penalty_weight
        self.pareto_improvement_weight = pareto_improvement_weight
        self.preference_constraint_weight = preference_constraint_weight
        self.target_mapping = target_mapping or {
            "reflectivity": "log-",
            "thermal_noise": "log-",
            "thickness": "linear-",
            "absorption": "log-"
        }
        
        # Get the reward function once at initialization
        self.reward_function = self.registry.get_function(self.reward_type)
    
    def _compute_reward_ranges_from_objective_bounds(self, env) -> Dict[str, List[float]]:
        """
        Compute expected reward ranges by evaluating the reward function at objective bounds.
        
        Args:
            env: Environment instance with objective_bounds attribute
            
        Returns:
            Dict mapping parameter names to [min_reward, max_reward] ranges
        """
        if not hasattr(env, 'objective_bounds'):
            return {}
            
        bounds = env.objective_bounds
        if not bounds:
            return {}
            
        ranges = {}
        
        # Dummy values for parameters not being tested
        dummy_values = {
            'reflectivity': 0.999,
            'absorption': 1e-6,
            'thermal_noise': 1e-20,
            'thickness': 5.0,
        }
        
        # For each parameter we want to optimize
        for param in self.optimise_parameters:
            if param not in bounds:
                continue
                
            # Expect bounds in [min, max] format from config
            if not isinstance(bounds[param], (list, tuple)) or len(bounds[param]) != 2:
                print(f"Warning: Expected [min, max] format for {param}, got: {bounds[param]}")
                continue
                
            min_bound, max_bound = bounds[param]
            
            # Ensure bounds are numeric
            try:
                min_bound = float(min_bound)
                max_bound = float(max_bound)
            except (ValueError, TypeError):
                print(f"Warning: Non-numeric bounds for {param}: min={min_bound}, max={max_bound}")
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
                        reflectivity=test_params['reflectivity'],
                        thermal_noise=test_params['thermal_noise'], 
                        total_thickness=test_params['thickness'],
                        absorption=test_params['absorption'],
                        optimise_parameters=[param],
                        optimise_targets={param: self.optimise_targets.get(param, test_value)},
                        weights={param: 1.0},
                        env=env,
                        **{k: v for k, v in self.kwargs.items() if k != 'env'}  # Exclude 'env' from kwargs
                    )
                    
                    if param in rewards:
                        reward_values.append(rewards[param])
                        
                except Exception as e:
                    print(f"Warning: Could not evaluate reward function for {param} at {test_value}: {e}")
                    continue
            
            # Set range based on min/max rewards
            if reward_values:
                ranges[param] = [min(reward_values), max(reward_values)]
                
        return ranges
    
    def _normalize_rewards(self, individual_rewards: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize individual rewards to [0, 1] range.
        
        Args:
            individual_rewards: Dict mapping parameter names to their individual reward values
            
        Returns:
            Dict mapping parameter names to normalized rewards [0, 1]
        """
        if not self.use_reward_normalization:
            return individual_rewards
            
        normalized_rewards = {}
        
        for param, reward in individual_rewards.items():
            if param not in self.optimise_parameters:
                normalized_rewards[param] = reward
                continue
                
            # Update history for adaptive mode
            self.reward_history[param].append(reward)
            
            # Get normalization range
            if self.reward_normalization_mode == "fixed" and param in self.reward_normalization_ranges:
                min_val, max_val = self.reward_normalization_ranges[param]
            elif self.reward_normalization_mode == "adaptive" and len(self.reward_history[param]) > 10:
                # Use moving average for adaptive range
                history = list(self.reward_history[param])
                current_min, current_max = min(history), max(history)
                
                # Update adaptive range with exponential moving average
                if self.adaptive_ranges[param]["min"] == float('inf'):
                    self.adaptive_ranges[param]["min"] = current_min
                    self.adaptive_ranges[param]["max"] = current_max
                else:
                    self.adaptive_ranges[param]["min"] = (1 - self.reward_normalization_alpha) * self.adaptive_ranges[param]["min"] + self.reward_normalization_alpha * current_min
                    self.adaptive_ranges[param]["max"] = (1 - self.reward_normalization_alpha) * self.adaptive_ranges[param]["max"] + self.reward_normalization_alpha * current_max
                
                min_val = self.adaptive_ranges[param]["min"]
                max_val = self.adaptive_ranges[param]["max"]
            else:
                # No normalization available yet
                normalized_rewards[param] = reward
                continue
            
            # Normalize to [0, 1]
            if max_val > min_val:
                normalized_rewards[param] = (reward - min_val) / (max_val - min_val)
            else:
                normalized_rewards[param] = reward
                
        return normalized_rewards

    def calculate_base(self, reflectivity, thermal_noise, thickness, absorption, env=None, weights=None, 
                      expert_constraints=None, constraint_penalty_weight=100.0, **extra_kwargs):
        """
        Calculate base reward using the selected function without addons (legacy method).
        
        Args:
            expert_constraints: Dict mapping parameter names to constraint target reward values
            constraint_penalty_weight: Weight for constraint violation penalties
        """
        # Merge initialization kwargs with call-time kwargs
        call_kwargs = {**self.kwargs, **extra_kwargs}
        
        # Always try to pass env if available - most functions can use it
        if env is not None:
            call_kwargs['env'] = env
            
            # Auto-compute reward ranges from objective bounds if not provided
            if self.use_reward_normalization and not self.reward_normalization_ranges:
                computed_ranges = self._compute_reward_ranges_from_objective_bounds(env)
                if computed_ranges:
                    self.reward_normalization_ranges = computed_ranges
                    print(f"Auto-computed reward normalization ranges: {computed_ranges}")

        # Get the original reward calculation
        total_reward, vals, rewards = self.reward_function(
            reflectivity=reflectivity,
            thermal_noise=thermal_noise,
            total_thickness=thickness,
            absorption=absorption,
            optimise_parameters=self.optimise_parameters,
            optimise_targets=self.optimise_targets,
            weights=weights,
            **call_kwargs
        )
        
        """
        # Apply normalization first (always if enabled)
        normalized_rewards = {}
        if self.use_reward_normalization and weights is not None:
            # Extract individual rewards from the rewards dict
            individual_rewards = {param: rewards.get(param, 0.0) for param in self.optimise_parameters}
            
            # Normalize individual rewards
            normalized_rewards = self._normalize_rewards(individual_rewards)
            
            # Update rewards dict with normalized values (for debugging/logging)
            for param in self.optimise_parameters:
                rewards[f"{param}_normalized"] = normalized_rewards[param]
        else:
            # No normalization - use original rewards
            normalized_rewards = {param: rewards.get(param, 0.0) for param in self.optimise_parameters}
        
        # Apply constraints if provided (to normalized rewards)
        if expert_constraints:
            total_reward, rewards = self._apply_expert_constraints(
                total_reward, rewards, expert_constraints, constraint_penalty_weight, normalized_rewards
            )
        else:
            # No constraints - use standard weighted sum of normalized rewards
            if weights is not None:
                total_reward = sum(weights.get(param, 0.0) * normalized_rewards.get(param, 0.0) 
                                 for param in self.optimise_parameters)
        """
        return total_reward, vals, rewards
    
    def calculate(self, reflectivity, thermal_noise, thickness, absorption, env=None, weights=None, 
                  expert_constraints=None, constraint_penalty_weight=100.0, **extra_kwargs):
        """
        Calculate reward with addons automatically applied based on initialization config.
        
        This is now the main calculate method that applies addons based on the configuration
        provided during initialization.
        
        Args:
            reflectivity: Current reflectivity value
            thermal_noise: Current thermal noise value  
            thickness: Current coating thickness
            absorption: Current absorption value
            env: Environment object
            weights: Dict of weights for each parameter
            expert_constraints: Dict mapping parameter names to constraint target reward values
            constraint_penalty_weight: Weight for constraint violation penalties
            **extra_kwargs: Additional parameters
            
        Returns:
            tuple: (total_reward, vals_dict, rewards_dict)
        """
        # Get base reward calculation
        total_reward, vals, rewards = self.calculate_base(
            reflectivity, thermal_noise, thickness, absorption, env=env,
            weights=weights, expert_constraints=expert_constraints,
            constraint_penalty_weight=constraint_penalty_weight, **extra_kwargs
        )
        
        # Apply addons based on initialization config
        final_reward, final_vals, final_rewards = self.apply_addon_functions(
            total_reward, vals, rewards, env=env, weights=weights,
            use_normalization=self.apply_normalization, 
            use_boundary_penalties=self.apply_boundary_penalties,
            use_divergence_penalty=self.apply_divergence_penalty, 
            use_air_penalty=self.apply_air_penalty,
            use_pareto_improvement=self.apply_pareto_improvement,
            air_penalty_weight=self.air_penalty_weight, 
            use_preference_constraints=self.apply_preference_constraints,
            divergence_penalty_weight=self.divergence_penalty_weight,
            pareto_improvement_weight=self.pareto_improvement_weight,
            constraint_penalty_weight=self.preference_constraint_weight,
            target_mapping=self.target_mapping, **extra_kwargs
        )

        final_reward = self.combine_rewards(final_rewards, objective_weights=weights, env=env, vals=vals)
        final_rewards["total_reward"] = final_reward
        
        return final_reward, final_vals, final_rewards

    def combine_rewards(self, rewards: Dict[str, float], objective_weights: Dict[str, float] = None, 
                       env=None, vals: Dict[str, float] = None) -> float:
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
            total_reward = np.sum([rewards[key] * objective_weights[key] for key in self.optimise_parameters])
        elif self.combine == "product":
            total_reward = np.prod([rewards[key] * objective_weights[key] for key in self.optimise_parameters])
        elif self.combine == "logproduct":
            total_reward = np.log(np.prod([rewards[key] * objective_weights[key] for key in self.optimise_parameters]))
        elif self.combine == "hypervolume":
            total_reward = self._calculate_hypervolume_reward(env, vals, rewards)
        else:
            raise ValueError(f"combine must be either 'sum', 'product', 'logproduct', or 'hypervolume', not {self.combine}")
        
        #apply addons
        for key in rewards.keys():
            if "addon" in key:
                total_reward += rewards[key]
                rewards["total_reward"] = total_reward

        return total_reward
    
    def _calculate_hypervolume_reward(self, env, vals: Dict[str, float], rewards: Dict[str, float]) -> float:
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
            print("Warning: pymoo not available for hypervolume calculation. Falling back to sum.")
            return np.sum([rewards[key] for key in self.optimise_parameters])
        
        # Check if environment has a pareto tracker
        if env is None or not hasattr(env, 'pareto_tracker') or env.pareto_tracker is None:
            raise Exception("Warning: No pareto_tracker available in environment. Falling back to sum.", env)
            return np.sum([rewards[key] for key in self.optimise_parameters])
        
        try:
            # Get current Pareto front
            pareto_tracker = env.pareto_tracker
            
            # Extract current objective values for the optimized parameters
            current_point = np.array([rewards.get(param, 0.0) for param in self.optimise_parameters])
            current_val = np.array([vals.get(param, 0.0) for param in self.optimise_parameters])
            current_state = env.current_state.get_array()
            
            # Create a temporary copy of the tracker to test with the new point
            import copy
            temp_tracker = copy.deepcopy(pareto_tracker)
            temp_tracker.add_point(current_point, current_val, current_state, force_update=True)
            
            # Get the Pareto front points (including the new point if it's non-dominated)
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
            raise Exception(f"Warning: Error calculating hypervolume: {e}. Falling back to sum. traceback: {e.__traceback__}")
            return np.sum([rewards[key] for key in self.optimise_parameters])
    
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
        if hasattr(env, 'objective_bounds') and env.objective_bounds:
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
                        reference_point.append(np.max(pareto_points[:, len(reference_point)]) * 1.1)
                else:
                    # Fallback: use worst point in current front + margin
                    reference_point.append(np.max(pareto_points[:, len(reference_point)]) * 1.1)
            return np.array(reference_point)
        
        # Method 2: Use worst points in current front + margin
        return np.max(pareto_points, axis=0) * 1.1
    
    def _prepare_objectives_for_hypervolume(self, pareto_points: np.ndarray) -> np.ndarray:
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
                    # This is maximization, negate for pymoo (which expects minimization)
                    transformed_points[:, i] = -transformed_points[:, i]
        
        return transformed_points
    
    def apply_addon_functions(self, total_reward: float, vals: Dict[str, float], 
                            rewards: Dict[str, float], env=None, weights: Dict[str, float] = None,
                            use_normalization: bool = False, use_boundary_penalties: bool = False,
                            use_divergence_penalty: bool = False, use_air_penalty: bool = False,
                            use_pareto_improvement: bool = False, use_preference_constraints: bool = False,
                            air_penalty_weight: float = 1.0, divergence_penalty_weight: float = 1.0,
                            pareto_improvement_weight: float = 1.0, pc_tracker=None, phase_info=None,
                            constraint_penalty_weight: float = 1.0,
                            target_mapping: Dict[str, str] = None, **kwargs) -> Tuple[float, Dict[str, float], Dict[str, float]]:
        """
        Apply addon functions to reward calculation results.
        
        Args:
            total_reward: Current total reward
            vals: Current objective values
            rewards: Individual rewards dict
            env: Environment object
            weights: Parameter weights
            use_normalization: Whether to apply normalization addon
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
        
        # Apply normalization addon
        if use_normalization and env is not None:
            updated_rewards = apply_normalization_addon(
                updated_rewards, vals, self.optimise_parameters, 
                self.optimise_targets, env, target_mapping
            )
        
        # Apply boundary penalties
        if use_boundary_penalties and env is not None:
            updated_rewards = apply_boundary_penalties(
                updated_rewards, vals, self.optimise_parameters, env, target_mapping,
            )
        
        # Apply divergence penalty
        if use_divergence_penalty:
            updated_rewards = apply_divergence_penalty(
                updated_rewards, self.optimise_parameters, weights, divergence_penalty_weight,
                self.multi_value_rewards
            )
        
        # Apply air penalty addon  
        if use_air_penalty:
            updated_rewards = apply_air_penalty_addon(
                updated_total_reward, updated_rewards, vals, env, 
                self.optimise_parameters, air_penalty_weight, self.multi_value_rewards, **kwargs
            )
        
        # Apply Pareto improvement addon
        if use_pareto_improvement:
            updated_rewards = apply_pareto_improvement_addon(
                updated_total_reward, updated_rewards, vals, env,
                self.optimise_parameters, pareto_improvement_weight, self.multi_value_rewards, **kwargs
            )
        
        # Apply preference constraints addon
        if use_preference_constraints:
            updated_total_reward, updated_rewards = apply_preference_constraints_addon(
                updated_total_reward, updated_rewards, vals, env,
                self.optimise_parameters, pc_tracker, phase_info, constraint_penalty_weight, **kwargs
            )
        

        return updated_total_reward, updated_vals, updated_rewards
    
    def _apply_expert_constraints(self, total_reward: float, rewards: Dict[str, float], 
                                  expert_constraints: Dict[str, float], 
                                  constraint_penalty_weight: float = 100.0,
                                  normalized_rewards: Dict[str, float] = None) -> Tuple[float, Dict[str, float]]:
        """
        Apply expert constraints to modify reward calculation.
        
        For constrained objectives: Apply large penalty for deviation from target reward
        For unconstrained objectives: Use normalized reward contribution
        
        Args:
            total_reward: Original total reward
            rewards: Dict of individual parameter rewards
            expert_constraints: Dict mapping parameter names to target reward values
            constraint_penalty_weight: Weight for constraint violation penalties
            normalized_rewards: Dict of normalized reward values to use for unconstrained params
            
        Returns:
            Tuple of (modified_total_reward, updated_rewards_dict)
        """
        if normalized_rewards is None:
            normalized_rewards = {param: rewards.get(param, 0.0) for param in self.optimise_parameters}
            
        constrained_reward = 0.0
        updated_rewards = rewards.copy()
        
        for param in self.optimise_parameters:
            normalized_reward = normalized_rewards.get(param, 0.0)
            
            if param in expert_constraints:
                # This parameter is constrained - apply penalty for deviation from target
                target_reward = expert_constraints[param]
                constraint_penalty = -abs(normalized_reward - target_reward) * constraint_penalty_weight
                updated_rewards[f"{param}_constraint_penalty"] = constraint_penalty
                constrained_reward += constraint_penalty
            else:
                # This parameter is unconstrained - use normalized reward
                constrained_reward += normalized_reward
        
        return constrained_reward, updated_rewards


# Addon reward functions that can be applied to any reward function shape
def apply_boundary_penalties(rewards: Dict[str, float], vals: Dict[str, float], 
                           optimise_parameters: List[str], env, 
                           target_mapping: Dict[str, str] = None) -> Dict[str, float]:
    """
    Apply boundary penalties for out-of-bounds values.
    
    Args:
        rewards: Current reward values
        vals: Current objective values
        optimise_parameters: List of parameters being optimized
        env: Environment object containing objective bounds
        target_mapping: Mapping of parameter names to scaling types
        
    Returns:
        Updated rewards dict with boundary penalties applied
    """
    if not hasattr(env, 'objective_bounds'):
        return rewards
    
    if target_mapping is None:
        target_mapping = {
            "reflectivity": "log-",
            "thermal_noise": "log-", 
            "thickness": "linear-",
            "absorption": "log-"
        }
    
    updated_rewards = rewards.copy()
    
    for key in vals.keys():
        if key in optimise_parameters and key in env.objective_bounds:
            bounds = env.objective_bounds[key]
            # Handle both list [min, max] and dict {'min': x, 'max': y} formats
            if isinstance(bounds, list):
                min_bound, max_bound = bounds
            elif isinstance(bounds, dict):
                min_bound, max_bound = bounds['min'], bounds['max']
            else:
                continue
                
            # Apply boundary penalties for out-of-bounds values
            penalty = 0.0
            if key == "reflectivity":
                if vals[key] < min_bound:
                    penalty += 10  # Large penalty for below range reflectivity
            else:
                if vals[key] > max_bound:
                    penalty += 10  # Large penalty for out-of-bounds values
            
            if penalty > 0:
                updated_rewards[key] = updated_rewards.get(key, 0.0) - penalty
                updated_rewards[f"{key}_boundary_penalty"] = -penalty
    
    return updated_rewards


def apply_normalization_addon(rewards: Dict[str, float], vals: Dict[str, float],
                            optimise_parameters: List[str], optimise_targets: Dict[str, float],
                            env, target_mapping: Dict[str, str] = None) -> Dict[str, float]:
    """
    Apply normalization based on objective bounds and target mapping.
    
    Args:
        rewards: Current reward values
        vals: Current objective values  
        optimise_parameters: List of parameters being optimized
        optimise_targets: Target values for each parameter
        env: Environment object containing objective bounds
        target_mapping: Mapping of parameter names to scaling types
        
    Returns:
        Updated rewards dict with normalization applied
    """
    if not hasattr(env, 'objective_bounds'):
        return rewards
        
    if target_mapping is None:
        target_mapping = {
            "reflectivity": "log-",
            "thermal_noise": "log-",
            "thickness": "linear-", 
            "absorption": "log-"
        }
    
    updated_rewards = rewards.copy()
    
    for key in vals.keys():
        if key in optimise_parameters and key in env.objective_bounds:
            bounds = env.objective_bounds[key]
            # Handle both list [min, max] and dict {'min': x, 'max': y} formats
            if isinstance(bounds, list):
                min_bound, max_bound = bounds
            elif isinstance(bounds, dict):
                min_bound, max_bound = bounds['min'], bounds['max']
            else:
                continue
                
            # Calculate absolute difference from target
            diff = np.abs(vals[key] - optimise_targets.get(key, vals[key]))
            
            if target_mapping[key][:-1] == "log":
                # For log scaling: transform difference to log space and normalize
                # Define minimum detectable difference (prevents log(0))
                min_diff = (max_bound - min_bound) * 1e-12  # Very small fraction of range
                max_diff = max_bound - min_bound  # Maximum possible difference
                
                # Clamp difference to valid range
                diff_clamped = np.clip(diff, min_diff, max_diff)
                
                # Log transform the difference (smaller differences -> more negative)
                log_diff = np.log10(diff_clamped)
                log_min_diff = np.log10(min_diff)
                log_max_diff = np.log10(max_diff)
                
                # Normalize to [0,1] where 0 = perfect match, 1 = maximum difference
                normed_val = (log_diff - log_min_diff) / (log_max_diff - log_min_diff)

            elif target_mapping[key][:-1] == "linear":
                # For linear scaling: normalize difference directly
                max_diff = max_bound - min_bound
                normed_val = np.clip(diff / max_diff, 0, 1)
            else:
                continue

            # Invert if we want smaller differences to give higher rewards
            if target_mapping[key].endswith("-"):
                normed_val = 1.0 - normed_val
            
            # Set the rewards dictionary with the normalized values
            updated_rewards[key] = normed_val
            #updated_rewards[f"{key}_normalized"] = normed_val
    
    return updated_rewards


def apply_divergence_penalty(rewards: Dict[str, float], optimise_parameters: List[str],
                           weights: Dict[str, float] = None, 
                           divergence_penalty_weight: float = 1.0, multi_value_rewards: bool = False) -> Tuple[float, Dict[str, float]]:
    """
    Apply divergence penalty to encourage both rewards to be high simultaneously.
    Only applied for 2-objective optimization problems.
    
    Args:
        rewards: Individual parameter rewards
        optimise_parameters: List of parameters being optimized
        weights: Weights for each parameter
        divergence_penalty_weight: Weight for divergence penalty
        
    Returns:
        Tuple of (divergence_penalty_value, updated_rewards_dict)
    """
    divergence_penalty = 0.0
    updated_rewards = rewards.copy()
    
    if weights is None:
        weights = {key: 1.0 for key in optimise_parameters}
    
    # Only apply divergence penalty for 2-objective optimization
    if len(optimise_parameters) == 2:
        reward_values = [rewards.get(key, 0.0) for key in optimise_parameters]
        param_weights = [weights.get(key, 1.0) for key in optimise_parameters]
        
        # Only apply divergence penalty if both weights are significant (> 0.01)
        # This prevents penalty when one objective has zero or very low weight
        min_weight = min(param_weights)
        if min_weight > 0.01:
            # Calculate divergence as difference between individual rewards
            max_reward = max(reward_values)
            min_reward = min(reward_values)
            divergence = max_reward - min_reward
            
            # Scale penalty by minimum weight - less penalty when one objective is less important
            # This ensures that when weights are [1,0] or close to it, penalty is minimal
            divergence_penalty = -divergence * min_weight * 0.5 * divergence_penalty_weight
            
            # Additional penalty if both rewards are very low (quality control)
            if min_reward < 0.1:
                divergence_penalty -= (0.1 - min_reward) * min_weight * divergence_penalty_weight
            
            # Distribute divergence penalty across individual objective rewards
            if len(optimise_parameters) > 0 and multi_value_rewards:
                penalty_per_objective = divergence_penalty 
                for param in optimise_parameters:
                    updated_rewards[param] = updated_rewards.get(param, 0.0) + penalty_per_objective
        else:
            # No penalty when one weight is effectively zero
            divergence_penalty = 0.0
    
    updated_rewards["divergence_addon"] = divergence_penalty * divergence_penalty_weight
    updated_rewards["total_reward"] += updated_rewards["divergence_addon"]
    return updated_rewards


def calculate_air_penalty_reward_new(state, air_material_index: int = 0, design_criteria: Dict = None,
                                    current_vals: Dict = None, optimise_parameters: List[str] = None,
                                    penalty_strength: float = 1.0, reward_strength: float = 1.0,
                                    min_real_layers: int = 5) -> float:
    """
    Calculate penalty for air-only coatings using new CoatingState methods.
    
    Args:
        state: Current coating state (CoatingState object or array for backward compatibility)
        air_material_index: Index of air material (usually 0)
        design_criteria: Dict of design criteria thresholds
        current_vals: Dict of current objective values
        optimise_parameters: List of parameters being optimized
        penalty_strength: How strong the penalty should be
        reward_strength: How strong the reward should be for air layers when criteria are met
        min_real_layers: Minimum number of non-air layers (default 5)
    
    Returns:
        Air penalty/reward value (negative = penalty, positive = reward)
    """
    if state is None:
        return -penalty_strength
    
    # Handle both CoatingState objects and legacy arrays
    if hasattr(state, 'get_layer_count'):
        # New CoatingState object - use its methods
        total_layers = state.get_num_active_layers()
        if total_layers == 0:
            return -penalty_strength
            
        # Count non-air layers using CoatingState methods
        non_air_layers = total_layers - state.get_layer_count(material_index=air_material_index)
        air_fraction = state.get_layer_count(material_index=air_material_index) / total_layers if total_layers > 0 else 1.0
        
    else:
        # Legacy array format - use original method
        if len(state) == 0:
            return -penalty_strength
            
        # Count non-air layers using air material column
        if isinstance(state, torch.Tensor):
            state = state.numpy()
            
        # Get the air material column (1 for air, 0 for non-air)
        air_column = state[:, air_material_index + 1]
        # Count non-air layers (where air_column is 0)
        non_air_layers = np.sum(air_column == 0)
        
        # Calculate air fraction
        total_layers = len(state)
        air_fraction = 1.0 - (non_air_layers / total_layers) if total_layers > 0 else 1.0
    
    # Check if design criteria are met
    criteria_met = {key:False for key in optimise_parameters}
    if design_criteria is not None and current_vals is not None:
        for key, threshold in design_criteria.items():
            if key in current_vals and key in optimise_parameters:
                val = current_vals[key]
                # !!!!!!!!!!! This should be changed to handle the direction of optimization properly !!!!!!!!!!!!!!!!!!
                if key in ["reflectivity"]:
                    if val > threshold:
                        criteria_met[key] = True
                        break
                else:
                    if val < threshold:
                        criteria_met[key] = True
                        break
    criteria_met = all(criteria_met.values())

    #print(f"Penalty strength: {penalty_strength}, Reward strength: {reward_strength}, Cirteria met: {criteria_met}, Non-air layers: {non_air_layers}, Air fraction: {air_fraction:.3f}")
    if criteria_met:
        # Design criteria met - small reward for more air layers
        return reward_strength * air_fraction
    else:
        # Design criteria not met
        if non_air_layers < min_real_layers:
            # Large penalty for too few real layers
            return -penalty_strength
        else:
            # Fractional penalty for excess air beyond minimum
            return 0#-penalty_strength * air_fraction * 0.1


def apply_air_penalty_addon(total_reward: float, rewards: Dict[str, float], vals: Dict[str, float],
                          env, optimise_parameters: List[str] = None, 
                          air_penalty_weight: float = 1.0, multi_value_rewards: bool = False, **kwargs) -> Tuple[float, Dict[str, float]]:
    """
    Apply air penalty addon to reward calculation.
    
    Args:
        total_reward: Current total reward
        rewards: Individual rewards dict
        vals: Current objective values
        env: Environment object
        optimise_parameters: List of parameters being optimized
        air_penalty_weight: Weight for air penalty
        **kwargs: Additional parameters for air penalty calculation
        
    Returns:
        Tuple of (updated_total_reward, updated_rewards_dict)
    """
    updated_rewards = rewards.copy()
    air_penalty = 0.0
    if env is not None:
        # Get current coating state and design criteria from env
        current_state = getattr(env, 'current_state', None)
        design_criteria = getattr(env, 'design_criteria', None)
        
        air_penalty = calculate_air_penalty_reward_new(
            state=current_state,
            air_material_index=0,
            design_criteria=design_criteria,
            current_vals=vals,
            optimise_parameters=optimise_parameters,
            **kwargs
        )
        
        # Apply air penalty to total reward
        air_penalty_scaled = air_penalty_weight * air_penalty
        updated_total_reward = total_reward + air_penalty_scaled
        updated_rewards["total_reward"] = updated_total_reward
        
        # Distribute air penalty across individual objective rewards
        if optimise_parameters and len(optimise_parameters) > 0 and multi_value_rewards:
            penalty_per_objective = air_penalty_scaled 
            for param in optimise_parameters:
                updated_rewards[param] = updated_rewards.get(param, 0.0) + penalty_per_objective
        
        # Store original air penalty for debugging
        updated_rewards["air_addon"] = air_penalty_scaled
    else:
        updated_total_reward = total_reward
    
    return updated_rewards


def apply_pareto_improvement_addon(total_reward: float, rewards: Dict[str, float], vals: Dict[str, float],
                                 env, optimise_parameters: List[str] = None, 
                                 pareto_improvement_weight: float = 1.0, multi_value_rewards: bool = False, **kwargs) -> Tuple[float, Dict[str, float]]:
    """
    Apply Pareto front improvement reward addon to reward calculation.
    
    This addon provides additional reward when a new point causes changes to the Pareto front,
    indicating that the solution is improving the multi-objective optimization frontier.
    
    Args:
        total_reward: Current total reward
        rewards: Individual rewards dict
        vals: Current objective values
        env: Environment object (must have pareto_tracker)
        optimise_parameters: List of parameters being optimized
        pareto_improvement_weight: Weight for Pareto improvement reward
        **kwargs: Additional parameters
        
    Returns:
        Tuple of (updated_total_reward, updated_rewards_dict)
    """
    updated_rewards = rewards.copy()
    pareto_improvement_reward = 0.0
    
    # Only apply for environments with Pareto trackers and multi-objective optimization
    if (env is not None and hasattr(env, 'pareto_tracker') and env.pareto_tracker is not None 
        and optimise_parameters is not None and len(optimise_parameters) >= 2):
        
        try:
            # Extract objective values for the optimized parameters
            new_point = np.array([rewards.get(param, 0.0) for param in optimise_parameters])
            new_value = np.array([vals.get(param, 0.0) for param in optimise_parameters])
            new_state = getattr(env, 'current_state', None).get_array()
            
            # Create a copy of the tracker to test without actually updating
            import copy
            test_tracker = copy.deepcopy(env.pareto_tracker)
            
            # Check if adding this point would change the Pareto front
            _, was_updated = test_tracker.add_point(new_point, new_value, new_state, force_update=True)
            
            if was_updated:
                pareto_improvement_reward = 1.0
                updated_rewards["pareto_front_changed"] = True
            else:
                updated_rewards["pareto_front_changed"] = False
                
        except Exception as e:
            # Handle any errors gracefully
            print(f"Warning: Error in Pareto improvement addon: {e}")
            updated_rewards["pareto_front_changed"] = False
    else:
        # Not applicable for this environment
        updated_rewards["pareto_front_changed"] = False
    
    # Apply the reward to total and distribute across individual objectives
    pareto_reward_scaled = pareto_improvement_weight * pareto_improvement_reward
    updated_total_reward = total_reward + pareto_reward_scaled
    updated_rewards["total_reward"] = updated_total_reward
    
    # Distribute Pareto improvement reward across individual objective rewards
    if optimise_parameters and len(optimise_parameters) > 0 and multi_value_rewards:
        reward_per_objective = pareto_reward_scaled 
        for param in optimise_parameters:
            updated_rewards[param] = updated_rewards.get(param, 0.0) + reward_per_objective
    
    # Store original Pareto improvement reward for debugging
    updated_rewards["pareto_improvement_addon"] = pareto_improvement_reward * pareto_improvement_weight
    
    return updated_rewards


# Decorator for easy reward function registration
def reward_function_plugin(name: str = None):
    """Decorator to register reward functions."""
    def decorator(func):
        registry = RewardRegistry()
        plugin_name = name or func.__name__.replace('reward_function_', '') or 'default'
        registry.register(plugin_name, func)
        return func
    return decorator


def apply_preference_constraints_addon(total_reward: float, rewards: Dict[str, float], 
                                     vals: Dict[str, float], env, 
                                     optimise_parameters: List[str], pc_tracker=None, 
                                     phase_info=None, constraint_penalty_weight=1.0, **kwargs) -> Tuple[float, Dict[str, float]]:
    """
    Apply preference constraints addon for constrained multi-objective optimization.
    
    Phase 1: Updates reward bounds with current observations
    Phase 2: Applies constraint penalties to non-target objectives
    
    Args:
        total_reward: Current total reward
        rewards: Individual rewards dict
        vals: Current objective values
        env: Environment object
        optimise_parameters: List of parameters being optimized
        pc_tracker: PreferenceConstrainedTracker instance
        phase_info: Phase information from preference-constrained training
        **kwargs: Additional parameters
        
    Returns:
        Tuple of (updated_total_reward, updated_rewards_dict)
    """
    updated_rewards = rewards.copy()
    updated_total_reward = total_reward
    print(pc_tracker, phase_info)
    if pc_tracker is None or phase_info is None:
        # No preference constraints active
        updated_rewards["pc_phase"] = 0
        updated_rewards["pc_constraints_active"] = {}
        return updated_total_reward, updated_rewards
    
    # Update reward bounds with current observations
    print("Updating PC reward bounds with current observations")
    pc_tracker.update_reward_bounds(rewards)
    
    # Get current phase information
    current_phase = phase_info.get("phase", 1)
    updated_rewards["pc_phase"] = current_phase
    print(f"Current PC phase: {current_phase}")
    if current_phase == 2:
        # Phase 2: Apply constraint penalties
        constraints = phase_info.get("constraints", {})
        print("Constraints", constraints)
        if constraints:
            # Apply constraint penalties
            constraint_penalty = pc_tracker.apply_constraint_penalties(rewards, constraints)
            print(f"Constraint penalty applied: {constraint_penalty}")
            
            # Add constraint info to rewards dict for debugging
            updated_rewards["pc_constraint_penalty"] = constraint_penalty
            updated_rewards["pc_constraints_active"] = constraints
            updated_rewards["pc_target_objective"] = phase_info.get("target_objective", "unknown")
        else:
            updated_rewards["pc_constraints_active"] = {}
    else:
        # Phase 1: No constraints, just track bounds
        updated_rewards["pc_constraints_active"] = {}
        constraint_penalty = 0.0
    
    # Update total reward in rewards dict
    updated_total_reward = total_reward - constraint_penalty
    updated_rewards["total_reward"] = updated_total_reward
    print(constraint_penalty)
    updated_rewards["pc_penalty_addon"] = constraint_penalty
    
    return updated_total_reward, updated_rewards


# Example of how to add new reward functions easily:
@reward_function_plugin("example_new")
def reward_function_example_new(reflectivity, thermal_noise, total_thickness, absorption, 
                               optimise_parameters, optimise_targets, env=None, **kwargs):
    """
    Example of how to add a new reward function.
    Just define it with this decorator and it's automatically available.
    All reward functions should accept env as a parameter to access environment state.
    """
    # Simple example: negative sum of all parameters
    return -(reflectivity + thermal_noise + total_thickness + absorption)
