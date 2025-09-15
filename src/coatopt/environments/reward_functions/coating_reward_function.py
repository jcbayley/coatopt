import numpy as np
from pymoo.indicators.hv import HV
import copy

def reward_function_log_targets(reflectivity, thermal_noise, total_thickness, absorption, 
                                     optimise_parameters, optimise_targets, env=None, 
                                     combine="product", neg_reward=-1e3, weights=None, **kwargs):
    """
    Simple reward function with log-based targeting - core logic only.
    
    This is the simplified version that focuses on the core reward calculation.
    Addon functions (normalisation, boundary penalties, divergence penalty, air penalty) 
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

