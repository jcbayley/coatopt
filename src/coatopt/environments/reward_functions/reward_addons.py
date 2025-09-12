"""
Reward addon functions for enhancing base reward calculations.

This module contains addon functions that can be applied to any base reward function
to add additional behaviors like boundary penalties, air penalties, Pareto improvements, etc.
These functions are modular and can be selectively applied based on configuration.
"""
import numpy as np
import torch
from typing import Dict, List, Tuple, Callable, Any
from collections import deque


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


def apply_normalisation_addon(rewards: Dict[str, float], vals: Dict[str, float],
                            optimise_parameters: List[str], optimise_targets: Dict[str, float],
                            env, target_mapping: Dict[str, str] = None) -> Dict[str, float]:
    """
    Apply normalisation based on objective bounds and target mapping.
    
    Args:
        rewards: Current reward values
        vals: Current objective values  
        optimise_parameters: List of parameters being optimized
        optimise_targets: Target values for each parameter
        env: Environment object containing objective bounds
        target_mapping: Mapping of parameter names to scaling types
        
    Returns:
        Updated rewards dict with normalisation applied
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
                # For log scaling: transform difference to log space and normalise
                # Define minimum detectable difference (prevents log(0))
                min_diff = (max_bound - min_bound) * 1e-12  # Very small fraction of range
                max_diff = max_bound - min_bound  # Maximum possible difference
                
                # Clamp difference to valid range
                diff_clamped = np.clip(diff, min_diff, max_diff)

                # Log transform the difference (smaller differences -> more negative)
                log_diff = np.log(diff_clamped)
                log_min_diff = np.log(min_diff)
                log_max_diff = np.log(max_diff)
                
                # normalise to [0,1] where 0 = perfect match, 1 = maximum difference
                normed_val = (log_diff - log_min_diff) / (log_max_diff - log_min_diff)

            elif target_mapping[key][:-1] == "linear":
                # For linear scaling: normalise difference directly
                max_diff = max_bound - min_bound
                normed_val = np.clip(diff / max_diff, 0, 1)
            else:
                continue

            # Invert if we want smaller differences to give higher rewards
            if target_mapping[key].endswith("-"):
                normed_val = 1.0 - normed_val
            
            # Set the rewards dictionary with the normalised values
            updated_rewards[key] = normed_val
    
    return updated_rewards


def apply_divergence_penalty(rewards: Dict[str, float], optimise_parameters: List[str],
                           weights: Dict[str, float] = None, 
                           divergence_penalty_weight: float = 1.0, 
                           multi_value_rewards: bool = False) -> Dict[str, float]:
    """
    Apply divergence penalty to encourage both rewards to be high simultaneously.
    Only applied for 2-objective optimization problems.
    
    Args:
        rewards: Individual parameter rewards
        optimise_parameters: List of parameters being optimized
        weights: Weights for each parameter
        divergence_penalty_weight: Weight for divergence penalty
        multi_value_rewards: Whether to distribute penalty across individual rewards
        
    Returns:
        Updated rewards dict with divergence penalty applied
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
    updated_rewards["total_reward"] = updated_rewards.get("total_reward", 0.0) + updated_rewards["divergence_addon"]
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
    criteria_met = {key: False for key in optimise_parameters}
    if design_criteria is not None and current_vals is not None:
        for key, threshold in design_criteria.items():
            if key in current_vals and key in optimise_parameters:
                val = current_vals[key]
                # TODO: This should be changed to handle the direction of optimization properly
                if key in ["reflectivity"]:
                    if val > threshold:
                        criteria_met[key] = True
                        break
                else:
                    if val < threshold:
                        criteria_met[key] = True
                        break
    criteria_met = all(criteria_met.values())

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
            return 0  # -penalty_strength * air_fraction * 0.1


def apply_air_penalty_addon(total_reward: float, rewards: Dict[str, float], vals: Dict[str, float],
                          env, optimise_parameters: List[str] = None, 
                          air_penalty_weight: float = 1.0, multi_value_rewards: bool = False, 
                          **kwargs) -> Dict[str, float]:
    """
    Apply air penalty addon to reward calculation.
    
    Args:
        total_reward: Current total reward
        rewards: Individual rewards dict
        vals: Current objective values
        env: Environment object
        optimise_parameters: List of parameters being optimized
        air_penalty_weight: Weight for air penalty
        multi_value_rewards: Whether to distribute penalty across individual rewards
        **kwargs: Additional parameters for air penalty calculation
        
    Returns:
        Updated rewards dict with air penalty applied
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
    
    return updated_rewards


def apply_pareto_improvement_addon(total_reward: float, rewards: Dict[str, float], vals: Dict[str, float],
                                 env, optimise_parameters: List[str] = None, 
                                 pareto_improvement_weight: float = 1.0, multi_value_rewards: bool = False, 
                                 **kwargs) -> Dict[str, float]:
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
        multi_value_rewards: Whether to distribute reward across individual rewards
        **kwargs: Additional parameters
        
    Returns:
        Updated rewards dict with Pareto improvement reward applied
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


def apply_preference_constraints_addon(total_reward: float, rewards: Dict[str, float], 
                                     vals: Dict[str, float], env, 
                                     optimise_parameters: List[str], pc_tracker=None, 
                                     phase_info=None, constraint_penalty_weight=1.0, 
                                     **kwargs) -> Tuple[float, Dict[str, float]]:
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
        constraint_penalty_weight: Weight for constraint penalties
        **kwargs: Additional parameters
        
    Returns:
        Tuple of (updated_total_reward, updated_rewards_dict)
    """
    updated_rewards = rewards.copy()
    updated_total_reward = total_reward
    
    if pc_tracker is None or phase_info is None:
        raise Exception("PreferenceConstrainedTracker and phase_info must be provided for preference constraints addon.")
    
    # Update reward bounds with current observations
    pc_tracker.update_reward_bounds(rewards)
    
    # Get current phase information
    current_phase = phase_info.get("phase", 1)
    updated_rewards["pc_phase"] = current_phase
    
    if current_phase == 2:
        # Phase 2: Apply constraint penalties
        constraints = phase_info.get("constraints", {})
        if constraints:
            # Apply constraint penalties
            constraint_penalty = pc_tracker.apply_constraint_penalties(rewards, constraints)
            
            # Add constraint info to rewards dict for debugging
            updated_rewards["pc_constraint_penalty"] = constraint_penalty
            updated_rewards["pc_constraints_active"] = constraints
            updated_rewards["pc_target_objective"] = phase_info.get("target_objective", "unknown")
        else:
            updated_rewards["pc_constraints_active"] = {}
            constraint_penalty = 0.0
    else:
        # Phase 1: No constraints, just track bounds
        updated_rewards["pc_constraints_active"] = {}
        constraint_penalty = 0.0
    
    # Update total reward in rewards dict
    updated_total_reward = total_reward - constraint_penalty
    updated_rewards["total_reward"] = updated_total_reward
    updated_rewards["pc_penalty_addon"] = -constraint_penalty
    
    return updated_total_reward, updated_rewards


# Decorator for easy reward function registration
def reward_function_plugin(name: str = None):
    """
    Decorator to register reward functions automatically.
    
    Args:
        name: Optional name for the reward function. If not provided,
              will use the function name with 'reward_function_' prefix removed.
    """
    def decorator(func):
        # Import here to avoid circular imports
        from .reward_system import RewardRegistry
        registry = RewardRegistry()
        plugin_name = name or func.__name__.replace('reward_function_', '') or 'default'
        registry.register(plugin_name, func)
        return func
    return decorator
