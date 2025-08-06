import numpy as np
from pymoo.indicators.hv import HV
import copy

def sigmoid(x, mean=0.5, a=0.01):
    return 1/(1+np.exp(-a*(x-mean)))

def inv_sigmoid(x, mean=0.5, a=0.01):
    return -np.log((1/x) - 1)/a + mean

def reward_function(reflectivity, thermal_noise, total_thickness, absorption, optimise_parameters, optimise_targets, combine="product", neg_reward=-1e3, weights=None):
    """_summary_

    Args:
        reflectivity (_type_): _description_
        thermal_noise (_type_): _description_
        total_thickness (_type_): _description_
        absorption (_type_): _description_
        optimise_parameters (_type_): _description_
        optimise_targets (_type_): _description_

    Returns:
        _type_: _description_
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

    if "reflectivity" in optimise_parameters:
        #log_reflect = np.log(1/np.abs(reflectivity - 1)+1)
        #target_log_reflect = np.log(1/np.abs(optimise_targets["reflectivity"] - 1)+1)
        log_reflect = -np.log10(1-reflectivity)
        target_log_reflect = -np.log10(1-optimise_targets["reflectivity"])
        rewards["reflectivity"] = log_reflect * sigmoid(log_reflect, mean=target_log_reflect, a=10)

    if "thermal_noise" in optimise_parameters and thermal_noise is not None:
        log_therm = -np.log10(thermal_noise)/10 
        target_log_therm = -np.log10(optimise_targets["thermal_noise"])/10 
        rewards["thermal_noise"] = log_therm * sigmoid(log_therm, mean=target_log_therm, a=10)

    if "thickness" in optimise_parameters:
        rewards["thickeness"] = -total_thickness
    
    if "absorption" in optimise_parameters:
        #if absorption == 0:
        #    log_absorption = -10
        #else:
        log_absorption = -np.log10(absorption) + 10
        target_log_absorption = -np.log10((optimise_targets["absorption"])) + 10
        rewards["absorption"] = log_absorption * sigmoid(log_absorption, mean=target_log_absorption, a=1)


    if combine=="sum":
        total_reward = np.sum([rewards[key]*weights[key] for key in optimise_parameters])
    elif combine=="product":
        total_reward = np.prod([rewards[key] for key in optimise_parameters])
    elif combine=="logproduct":
        total_reward = np.log(np.prod([rewards[key] for key in optimise_parameters]) )
    else:
        raise ValueError(f"combine must be either 'sum' or 'product' not {combine}")

    if np.isnan(total_reward) or np.isinf(total_reward):
        #rewards["total_reward"] = neg_reward
        total_reward = neg_reward

    rewards["total_reward"] = total_reward

    return total_reward, vals, rewards



def inv_reward_function(reflectivity, thermal_noise, total_thickness, absorption, optimise_parameters, optimise_targets, combine="product", neg_reward=-1e3, weights=None):
    """_summary_

    Args:
        reflectivity (_type_): _description_
        thermal_noise (_type_): _description_
        total_thickness (_type_): _description_
        absorption (_type_): _description_
        optimise_parameters (_type_): _description_
        optimise_targets (_type_): _description_

    Returns:
        _type_: _description_
    """

    rewards = {
        "reflectivity": reflectivity,
        "thermal_noise": thermal_noise,
        "thickness": total_thickness,
        "absorption": absorption
    }
    if weights is None:
        weights = {key:1 for key in rewards}

    vals = {key:0 for key in rewards}

    if "reflectivity" in optimise_parameters:
        target_log_reflect = -np.log10(1-optimise_targets["reflectivity"])
        inv_sig = inv_sigmoid(reflectivity, mean=target_log_reflect, a=10)
        vals["reflectivity"] = 1 - 10**(-reflectivity/inv_sig)

    if "thermal_noise" in optimise_parameters and thermal_noise is not None:
        target_log_therm = -np.log10(optimise_targets["thermal_noise"])/10
        inv_sig = inv_sigmoid(thermal_noise, mean=target_log_therm, a=10)
        vals["thermal_noise"] = 10**(-10*thermal_noise/inv_sig)

    if "thickness" in optimise_parameters:
        vals["thickeness"] = -total_thickness
    
    if "absorption" in optimise_parameters:
        target_log_absorption = -np.log10(optimise_targets["absorption"]) + 10
        inv_sig = inv_sigmoid(absorption, mean=target_log_absorption, a=10)
        vals["absorption"] = 10**(-(absorption/inv_sig-10))


    return None, vals, rewards


def reward_function_norm(reflectivity, thermal_noise, total_thickness, absorption, optimise_parameters, optimise_targets, combine="product", neg_reward=-1e3, weights=None):
    """_summary_

    Args:
        reflectivity (_type_): _description_
        thermal_noise (_type_): _description_
        total_thickness (_type_): _description_
        absorption (_type_): _description_
        optimise_parameters (_type_): _description_
        optimise_targets (_type_): _description_

    Returns:
        _type_: _description_
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

    if "reflectivity" in optimise_parameters:
        #log_reflect = np.log(1/np.abs(reflectivity - 1)+1)
        #target_log_reflect = np.log(1/np.abs(optimise_targets["reflectivity"] - 1)+1)
        log_reflect = -np.log10(1-reflectivity)
        target_log_reflect = -np.log10(1-optimise_targets["reflectivity"])
        rewards["reflectivity"] = log_reflect/np.abs(target_log_reflect) * sigmoid(log_reflect, mean=target_log_reflect, a=10)

    if "thermal_noise" in optimise_parameters and thermal_noise is not None:
        log_therm = -np.log10(thermal_noise)
        target_log_therm = -np.log10(optimise_targets["thermal_noise"]) 
        rewards["thermal_noise"] = log_therm/np.abs(target_log_therm) * sigmoid(log_therm, mean=target_log_therm, a=10)

    if "thickness" in optimise_parameters:
        rewards["thickeness"] = -total_thickness
    
    if "absorption" in optimise_parameters:
        log_absorption = -np.log10(absorption) 
        target_log_absorption = -np.log10((optimise_targets["absorption"])) 
        rewards["absorption"] = log_absorption/np.abs(target_log_absorption) * sigmoid(log_absorption, mean=target_log_absorption, a=10)


    if combine=="sum":
        total_reward = np.sum([rewards[key]*weights[key] for key in optimise_parameters])
    elif combine=="product":
        total_reward = np.prod([rewards[key] for key in optimise_parameters])
    elif combine=="logproduct":
        total_reward = np.log(np.prod([rewards[key] for key in optimise_parameters]) )
    else:
        raise ValueError(f"combine must be either 'sum' or 'product' not {combine}")

    if np.isnan(total_reward) or np.isinf(total_reward):
        #rewards["total_reward"] = neg_reward
        total_reward = neg_reward

    rewards["total_reward"] = total_reward

    return total_reward, vals, rewards

def reward_function_target(reflectivity, thermal_noise, total_thickness, absorption, optimise_parameters, optimise_targets, combine="product", neg_reward=-1e3, weights=None):
    """_summary_

    Args:
        reflectivity (_type_): _description_
        thermal_noise (_type_): _description_
        total_thickness (_type_): _description_
        absorption (_type_): _description_
        optimise_parameters (_type_): _description_
        optimise_targets (_type_): _description_

    Returns:
        _type_: _description_
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

    if "reflectivity" in optimise_parameters:
        #log_reflect = np.log(1/np.abs(reflectivity - 1)+1)
        #target_log_reflect = np.log(1/np.abs(optimise_targets["reflectivity"] - 1)+1)
        log_reflect = np.log(1./np.abs(reflectivity-optimise_targets["reflectivity"]))
        rewards["reflectivity"] = log_reflect 

    if "thermal_noise" in optimise_parameters and thermal_noise is not None:
        log_therm = np.log(1./np.abs(thermal_noise-optimise_targets["thermal_noise"]))
        rewards["thermal_noise"] = log_therm 

    if "thickness" in optimise_parameters:
        rewards["thickeness"] = -total_thickness
    
    if "absorption" in optimise_parameters:
        log_absorption = (np.log(1./np.abs(absorption-optimise_targets["absorption"]))+10)*10
        rewards["absorption"] = log_absorption


    if combine=="sum":
        total_reward = np.sum([rewards[key]*weights[key] for key in optimise_parameters])
    elif combine=="product":
        total_reward = np.prod([rewards[key] for key in optimise_parameters])
    elif combine=="logproduct":
        total_reward = np.log(np.prod([rewards[key] for key in optimise_parameters]) )
    else:
        raise ValueError(f"combine must be either 'sum' or 'product' not {combine}")

    if np.isnan(total_reward) or np.isinf(total_reward):
        #rewards["total_reward"] = neg_reward
        total_reward = neg_reward

    rewards["total_reward"] = total_reward

    return total_reward, vals, rewards


def reward_function_raw(reflectivity, thermal_noise, total_thickness, absorption, optimise_parameters, optimise_targets, combine="product", neg_reward=-1e3, weights=None):
    """_summary_

    Args:
        reflectivity (_type_): _description_
        thermal_noise (_type_): _description_
        total_thickness (_type_): _description_
        absorption (_type_): _description_
        optimise_parameters (_type_): _description_
        optimise_targets (_type_): _description_

    Returns:
        _type_: _description_
    """

    vals = {
        "reflectivity": reflectivity,
        "thermal_noise": thermal_noise,
        "thickness": total_thickness,
        "absorption": absorption
    }

    rewards = vals
    if weights is None:
        weights = {key:1 for key in vals}

    if combine=="sum":
        total_reward = np.sum([rewards[key]*weights[key] for key in optimise_parameters])
    elif combine=="product":
        total_reward = np.prod([rewards[key] for key in optimise_parameters])
    elif combine=="logproduct":
        total_reward = np.log(np.prod([rewards[key] for key in optimise_parameters]) )
    else:
        raise ValueError(f"combine must be either 'sum' or 'product' not {combine}")

    if np.isnan(total_reward) or np.isinf(total_reward):
        #rewards["total_reward"] = neg_reward
        total_reward = neg_reward

    rewards["total_reward"] = total_reward

    return total_reward, vals, rewards


def reward_function_log_minimise(reflectivity, thermal_noise, total_thickness, absorption, optimise_parameters, optimise_targets, combine="product", neg_reward=-1e3, weights=None):
    """_summary_

    Args:
        reflectivity (_type_): _description_
        thermal_noise (_type_): _description_
        total_thickness (_type_): _description_
        absorption (_type_): _description_
        optimise_parameters (_type_): _description_
        optimise_targets (_type_): _description_

    Returns:
        _type_: _description_
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

    if "reflectivity" in optimise_parameters:
        #log_reflect = np.log(1/np.abs(reflectivity - 1)+1)
        #target_log_reflect = np.log(1/np.abs(optimise_targets["reflectivity"] - 1)+1)
        log_reflect = -np.log(np.abs(reflectivity-optimise_targets["reflectivity"])) + 12
        rewards["reflectivity"] = log_reflect 

    if "thermal_noise" in optimise_parameters and thermal_noise is not None:
        log_therm = np.log(1./np.abs(thermal_noise-optimise_targets["thermal_noise"]))
        rewards["thermal_noise"] = log_therm 

    if "thickness" in optimise_parameters:
        rewards["thickeness"] = -total_thickness
    
    if "absorption" in optimise_parameters:
        log_absorption = (-np.log(np.abs(absorption-optimise_targets["absorption"])) + 12) * 3
        rewards["absorption"] = log_absorption


    if combine=="sum":
        total_reward = np.sum([rewards[key]*weights[key] for key in optimise_parameters])
    elif combine=="product":
        total_reward = np.prod([rewards[key] for key in optimise_parameters])
    elif combine=="logproduct":
        total_reward = np.log(np.prod([rewards[key] for key in optimise_parameters]) )
    else:
        raise ValueError(f"combine must be either 'sum' or 'product' not {combine}")

    if np.isnan(total_reward) or np.isinf(total_reward):
        #rewards["total_reward"] = neg_reward
        total_reward = neg_reward

    rewards["total_reward"] = total_reward

    return total_reward, vals, rewards



def reward_function_area(reflectivity, thermal_noise, total_thickness, absorption, optimise_parameters, optimise_targets, env, combine="product", neg_reward=-1e3, weights=None):
    """Area-based reward function that considers Pareto front diversity and domination.

    Args:
        reflectivity (_type_): _description_
        thermal_noise (_type_): _description_
        total_thickness (_type_): _description_
        absorption (_type_): _description_
        optimise_parameters (_type_): _description_
        optimise_targets (_type_): _description_
        env (_type_): environment containing pareto front and reference point
        combine (str): how to combine rewards ("sum", "product", "logproduct")
        neg_reward (float): penalty for invalid values
        weights (_type_): weights for each parameter

    Returns:
        tuple: (total_reward, vals, rewards)
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

    # Create new point from current values
    new_point = np.array([[vals[param] for param in optimise_parameters]])
    
    updated_pareto_front, front_updated = env.update_pareto_front(copy.copy(env.pareto_front), copy.copy(new_point))
    #updated_pareto_front = env.compute_pareto_front(points)       


    # Compute diversity reward based on spread of Pareto front
    diversity_reward = 0
    area_reward_val = 0
    
    if len(updated_pareto_front) > 1:
        # Sort points by first objective for area calculation
        sorted_front = updated_pareto_front[np.argsort(updated_pareto_front[:, 0])]
        
        # Calculate area under front using trapezoidal rule
        area_under_front = np.trapz(sorted_front[:, 1], sorted_front[:, 0])
        
        # Normalize area by width to avoid penalizing wider fronts
        front_width = np.max(sorted_front[:, 0]) - np.min(sorted_front[:, 0])
        normalized_area = area_under_front / (front_width + 1e-6)  # Add small epsilon to avoid division by zero
        
        # Reward lower normalized area (closer to origin)
        area_reward_val = -normalized_area * 5  # Negative because smaller area is better
        
        # Diversity reward based on spread
        front_range = np.ptp(updated_pareto_front, axis=0)  # Range in each dimension
        diversity_reward = np.sum(front_range) * 3  # Reward wider spread
        
        # Additional reward for front size
        diversity_reward += len(updated_pareto_front) * 2

    # Domination reward - how much better this point is than reference
    domination_reward = 0
    if len(env.reference_point) > 0:
        # Distance from reference point (negative because we want to minimize)
        ref_distance = np.linalg.norm(new_point - env.reference_point)
        domination_reward = -ref_distance * 10


    # Add area-based rewards
    total_reward = diversity_reward + area_reward_val + domination_reward

    if np.isnan(total_reward) or np.isinf(total_reward):
        total_reward = neg_reward

    rewards["total_reward"] = total_reward
    rewards["diversity_reward"] = diversity_reward
    rewards["area_reward"] = area_reward_val
    rewards["domination_reward"] = domination_reward
    rewards["reflectivity"] = 0
    rewards["thermal_noise"] = 0
    rewards["thickness"] = 0
    rewards["absorption"] = 0

    return total_reward, vals, rewards


def reward_function_hypervolume(reflectivity, thermal_noise, total_thickness, absorption, 
                               optimise_parameters, optimise_targets, env, 
                               combine="product", neg_reward=-1e3, weights=None, 
                               ref_point=None, adaptive_ref=True):
    """Hypervolume-based reward function that rewards both Pareto front quality and diversity.
    
    Hypervolume measures the area/volume above the Pareto front, which naturally combines
    both front quality (closer to origin is better) and diversity (wider spread is better).

    Args:
        reflectivity: Reflectivity value (typically 1-R)
        thermal_noise: Thermal noise value
        total_thickness: Total coating thickness
        absorption: Absorption value
        optimise_parameters: List of parameters being optimized
        optimise_targets: Target values for optimization
        env: Environment containing pareto front and reference point
        combine: How to combine rewards (not used in hypervolume approach)
        neg_reward: Penalty for invalid values
        weights: Parameter weights (not used in hypervolume approach)
        ref_point: Reference point for hypervolume calculation. If None, will be computed adaptively
        adaptive_ref: Whether to use adaptive reference point based on current front

    Returns:
        tuple: (total_reward, vals, rewards)
    """
    
    vals = {
        "reflectivity": reflectivity,
        "thermal_noise": thermal_noise,
        "thickness": total_thickness,
        "absorption": absorption
    }
    
    rewards = {key: 0 for key in vals}
    
    # Check for invalid values
    if any(np.isnan(val) or np.isinf(val) for val in vals.values() if val is not None):
        total_reward = neg_reward
        rewards["total_reward"] = total_reward
        return total_reward, vals, rewards
    
    # Create new point from current values
    new_point = np.array([vals[param] for param in optimise_parameters])
    
    # Update Pareto front with new point
    updated_pareto_front, front_updated = env.update_pareto_front(
        copy.copy(env.pareto_front), 
        copy.copy(new_point.reshape(1, -1))
    )
    
    # Calculate hypervolume
    hypervolume_reward = 0
    
    if len(updated_pareto_front) > 0:
        try:
            # Determine reference point
            if ref_point is None:
                if adaptive_ref:
                    # Use adaptive reference point: slightly worse than worst point in each objective
                    max_vals = np.max(updated_pareto_front, axis=0)
                    # Add 10% margin to ensure reference point dominates all points
                    ref_point_calc = max_vals * 1.1
                    
                    # For objectives that should be minimized, ensure ref point is positive
                    # and reasonable (e.g., for log-scale values like 1-R)
                    for i, param in enumerate(optimise_parameters):
                        if param == "reflectivity":
                            # For 1-R values, cap reference point at reasonable value
                            ref_point_calc[i] = min(ref_point_calc[i], 1e-3)
                        elif param == "absorption":
                            # For absorption, reasonable upper bound
                            ref_point_calc[i] = min(ref_point_calc[i], 1.0)
                        elif param == "thermal_noise":
                            # For thermal noise, use reasonable upper bound
                            ref_point_calc[i] = min(ref_point_calc[i], 1e-15)
                        elif param == "thickness":
                            # For thickness, reasonable upper bound
                            ref_point_calc[i] = min(ref_point_calc[i], 1000.0)
                    
                    ref_point_calc = np.maximum(ref_point_calc, np.max(updated_pareto_front, axis=0) * 1.01)
                else:
                    # Use fixed reference point based on targets
                    ref_point_calc = np.array([optimise_targets.get(param, 1.0) for param in optimise_parameters])
                    ref_point_calc *= 2  # Make it clearly dominated
            else:
                ref_point_calc = np.array(ref_point)
            
            # Ensure reference point dominates all points
            ref_point_calc = np.maximum(ref_point_calc, np.max(updated_pareto_front, axis=0) * 1.001)
            
            # Calculate hypervolume using pymoo's HV indicator
            hv_indicator = HV(ref_point=ref_point_calc)
            hypervolume = hv_indicator(updated_pareto_front)
            
            # Scale hypervolume to reasonable reward range
            # Normalize by the maximum possible hypervolume (reference point volume)
            max_volume = np.prod(ref_point_calc) if len(ref_point_calc) > 0 else 1.0
            normalized_hypervolume = hypervolume / (max_volume + 1e-12)
            
            # Scale to reward range (multiply by large factor to make it significant)
            hypervolume_reward = normalized_hypervolume * 1000
            
        except Exception as e:
            print(f"Error calculating hypervolume: {e}")
            hypervolume_reward = neg_reward / 10  # Small penalty, not full negative reward
    
    # Additional small reward for improving the front
    front_improvement_reward = 0
    if front_updated:
        front_improvement_reward = 50  # Bonus for adding to Pareto front
    
    # Total reward is hypervolume + improvement bonus
    total_reward = hypervolume_reward + front_improvement_reward
    
    if np.isnan(total_reward) or np.isinf(total_reward):
        total_reward = neg_reward
    
    # Store individual rewards for analysis
    rewards["total_reward"] = total_reward
    rewards["hypervolume_reward"] = hypervolume_reward
    rewards["front_improvement_reward"] = front_improvement_reward
    rewards["reflectivity"] = 0  # Not used in hypervolume approach
    rewards["thermal_noise"] = 0  # Not used in hypervolume approach  
    rewards["thickness"] = 0  # Not used in hypervolume approach
    rewards["absorption"] = 0  # Not used in hypervolume approach
    
    return total_reward, vals, rewards