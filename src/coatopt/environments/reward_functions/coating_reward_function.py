import numpy as np
from pymoo.indicators.hv import HV
import copy

def reward_function_log_targets(reflectivity, thermal_noise, total_thickness, absorption, 
                                     optimise_parameters, optimise_targets, env=None, 
                                     combine="product", neg_reward=-1e3, weights=None, **kwargs):
    """
    Simple reward function with log-based targeting - core logic only.
    
    This is the simplified version that focuses on the core reward calculation.
    Addon functions (normalization, boundary penalties, divergence penalty, air penalty) 
    should be applied separately using the reward_system addon functions.
    
    Args:
        reflectivity: Current reflectivity value
        thermal_noise: Current thermal noise value  
        total_thickness: Current coating thickness
        absorption: Current absorption value
        optimise_parameters: List of parameters to optimize
        optimise_targets: Dict of target values for each parameter
        env: Environment object
        combine: How to combine rewards ('sum', 'product', 'logproduct')
        neg_reward: Reward value for invalid/NaN results
        weights: Dict of weights for each parameter
        
    Returns:
        tuple: (total_reward, vals_dict, rewards_dict)
    """
    vals = {
        "reflectivity": reflectivity,
        "thermal_noise": thermal_noise,
        "thickness": total_thickness,
        "absorption": absorption
    }
    
    if weights is None:
        weights = {key: 1 for key in vals}

    rewards = {key: 0 for key in vals}

    # Compute the log target rewards (core logic) with small offset to avoid log(0)
    if "reflectivity" in optimise_parameters:
        log_reflect = -np.log(np.abs(reflectivity - optimise_targets["reflectivity"]) + 1e-30)
        rewards["reflectivity"] = log_reflect 

    if "thermal_noise" in optimise_parameters and thermal_noise is not None:
        log_therm = -np.log(np.abs(thermal_noise - optimise_targets["thermal_noise"]) + 1e-30)
        rewards["thermal_noise"] = log_therm 

    if "thickness" in optimise_parameters:
        rewards["thickness"] = -total_thickness  # Fixed typo from "thickeness"
    
    if "absorption" in optimise_parameters:
        log_absorption = -np.log(np.abs(absorption - optimise_targets["absorption"]) + 1e-30)
        rewards["absorption"] = log_absorption
    
    # Combine the rewards 
    if combine == "sum":
        total_reward = np.sum([rewards[key] * weights[key] for key in optimise_parameters])
    elif combine == "product":
        total_reward = np.prod([rewards[key] for key in optimise_parameters])
    elif combine == "logproduct":
        total_reward = np.log(np.prod([rewards[key] for key in optimise_parameters]))
    else:
        raise ValueError(f"combine must be either 'sum', 'product', or 'logproduct', not {combine}")

    if np.isnan(total_reward) or np.isinf(total_reward):
        total_reward = neg_reward

    rewards["total_reward"] = total_reward
    return total_reward, vals, rewards


# Create an alias for the default function
def reward_function_default(reflectivity, thermal_noise, total_thickness, absorption, 
                           optimise_parameters, optimise_targets, env=None, **kwargs):
    """Default reward function - uses simple log targets."""
    return reward_function_log_targets(
        reflectivity, thermal_noise, total_thickness, absorption,
        optimise_parameters, optimise_targets, env, **kwargs
    )

import numpy as np
from pymoo.indicators.hv import HV
import copy

# Import the updated air penalty function from reward_system
from .reward_system import calculate_air_penalty_reward_new as calculate_air_penalty_reward


def reward_function_log_targets_limits(reflectivity, thermal_noise, total_thickness, absorption, optimise_parameters, optimise_targets, env, combine="product", neg_reward=-1e3, weights=None, use_air_penalty=True, air_penalty_weight=1.0, use_divergence_penalty=False, divergence_penalty_weight=1.0, **kwargs):
    """Reward function with log-based normalization and optional divergence penalty.
    
    This function normalizes objective values based on their bounds and applies log scaling
    for better numerical behavior. It supports an optional divergence penalty that encourages
    both objectives to be high simultaneously, with penalty strength adjusted by the weight balance.

    Args:
        reflectivity: Current reflectivity value
        thermal_noise: Current thermal noise value  
        total_thickness: Current coating thickness
        absorption: Current absorption value
        optimise_parameters: List of parameters to optimize
        optimise_targets: Dict of target values for each parameter
        env: Environment object containing objective bounds
        combine: How to combine rewards ('sum', 'product', 'logproduct')
        neg_reward: Reward value for invalid/NaN results
        weights: Dict of weights for each parameter
        use_air_penalty: Whether to apply air layer penalties
        air_penalty_weight: Weight for air penalty
        use_divergence_penalty: Whether to apply divergence penalty (for 2-objective problems)
        divergence_penalty_weight: Weight for divergence penalty

    Returns:
        tuple: (total_reward, vals_dict, rewards_dict)
    """

    vals = {
        "reflectivity": reflectivity,
        "thermal_noise": thermal_noise,
        "thickness": total_thickness,
        "absorption": absorption
    }
    if weights is None:
        weights = {key:1 for key in vals}

    rewards = {key:0 for key in vals}

    if not hasattr(env, 'objective_bounds'):
        env.objective_bounds = {
            'reflectivity': [1-1e-1, 1-1e-6],      # Typical coating values
            'absorption': [1e-4, 1000.0],         # Physical bounds
            'thermal_noise': [1e-25, 1e-15],   # Typical noise values  
            'thickness': [100, 50000]          # nm range
        }

    target_mapping = {
        "reflectivity": "log-",
        "thermal_noise": "log-",
        "thickness": "linear-",
        "absorption": "log-"
    }

    ###########################
    # Compute the log target rewards
    ##########################3
    if "reflectivity" in optimise_parameters:
        #log_reflect = np.log(1/np.abs(reflectivity - 1)+1)
        #target_log_reflect = np.log(1/np.abs(optimise_targets["reflectivity"] - 1)+1)
        log_reflect = -np.log(np.abs(reflectivity-optimise_targets["reflectivity"]))
        rewards["reflectivity"] = log_reflect 

    if "thermal_noise" in optimise_parameters and thermal_noise is not None:
        log_therm = -np.log(np.abs(thermal_noise-optimise_targets["thermal_noise"]))
        rewards["thermal_noise"] = log_therm 

    if "thickness" in optimise_parameters:
        rewards["thickeness"] = -total_thickness
    
    if "absorption" in optimise_parameters:
        log_absorption = (-np.log(np.abs(absorption-optimise_targets["absorption"])))
        rewards["absorption"] = log_absorption
    
    #############################
    # Now apply normalization and boundary penalties
    ######################################
    normed_vals = {}
    for key in vals.keys():
        if key in optimise_parameters:
            bounds = env.objective_bounds[key]
            # Handle both list [min, max] and dict {'min': x, 'max': y} formats
            if isinstance(bounds, list):
                min_bound, max_bound = bounds
            elif isinstance(bounds, dict):
                min_bound, max_bound = bounds['min'], bounds['max']
            else:
                raise ValueError(f"Unexpected bounds format for {key}: {bounds}")
                
            # Calculate absolute difference from target
            diff = np.abs(vals[key] - optimise_targets[key])
            
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
                normed_vals[key] = (log_diff - log_min_diff) / (log_max_diff - log_min_diff)

            elif target_mapping[key][:-1] == "linear":
                # For linear scaling: normalize difference directly
                max_diff = max_bound - min_bound
                normed_vals[key] = np.clip(diff / max_diff, 0, 1)
            else:
                raise ValueError(f"Unknown target mapping for {key}: {target_mapping[key]}")

            # Apply boundary penalties for out-of-bounds values
            if key == "reflectivity":
                if vals[key] < min_bound:
                    normed_vals[key] += 10  # Large penalty for below range reflectivity
            else:
                if vals[key] > max_bound:
                    normed_vals[key] += 10  # Large penalty for out-of-bounds values

            # Invert if we want smaller differences to give higher rewards
            if target_mapping[key].endswith("-"):
                normed_vals[key] = 1.0 - normed_vals[key]
            
            # Set the rewards dictionary with the normalized values
            rewards[key] = normed_vals[key]
    
    ###############################
    # Calcualte a reward divergence penalty
    ###################################

    # Calculate divergence penalty to encourage both rewards to be high
    divergence_penalty = 0.0
    if use_divergence_penalty and len(optimise_parameters) == 2:
        # Only apply divergence penalty for 2-objective optimization
        reward_values = [rewards[key] for key in optimise_parameters]
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
        else:
            # No penalty when one weight is effectively zero
            divergence_penalty = 0.0
    
    #######################
    # Combine the rewards 
    #########################
    if combine=="sum":
        total_reward = np.sum([rewards[key]*weights[key] for key in optimise_parameters])
    elif combine=="product":
        total_reward = np.prod([rewards[key] for key in optimise_parameters])
    elif combine=="logproduct":
        total_reward = np.log(np.prod([rewards[key] for key in optimise_parameters]) )
    else:
        raise ValueError(f"combine must be either 'sum' or 'product' not {combine}")

    # Apply divergence penalty
    total_reward += divergence_penalty

    ############################
    # Add a penalty for the amount of air layers in the coating (dependent on design criteria)
    #############################
    # Add air penalty if enabled
    air_penalty = 0.0
    if use_air_penalty and env is not None:
        # Get current coating state and design criteria from env
        current_state = getattr(env, 'current_state', None)
        design_criteria = getattr(env, 'design_criteria', None)
        
        air_penalty = calculate_air_penalty_reward(
            state=current_state,
            air_material_index=0,
            design_criteria=design_criteria,
            current_vals=vals,
            optimise_parameters=optimise_parameters,
            **kwargs
        )
        total_reward += air_penalty_weight * air_penalty

    if np.isnan(total_reward) or np.isinf(total_reward):
        #rewards["total_reward"] = neg_reward
        total_reward = neg_reward

    # Store additional info in rewards dict for debugging
    if use_divergence_penalty and len(optimise_parameters) == 2:
        rewards["divergence_penalty"] = divergence_penalty
        
    if use_air_penalty:
        rewards["air_penalty"] = air_penalty
    
    rewards["total_reward"] = total_reward

    return total_reward, vals, rewards

