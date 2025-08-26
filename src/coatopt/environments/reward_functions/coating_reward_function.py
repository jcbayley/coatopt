import numpy as np
from pymoo.indicators.hv import HV
import copy

def calculate_air_penalty_reward(state, air_material_index=0, design_criteria=None, 
                                current_vals=None, penalty_strength=1.0, reward_strength=0.5, 
                                min_real_layers=2):
    """
    Calculate penalty for air-only coatings when design criteria are NOT met,
    or reward for more air layers when design criteria ARE met.
    
    Args:
        state: Current coating state
        air_material_index: Index of air material (usually 0)
        design_criteria: Dict of design criteria thresholds
        current_vals: Dict of current objective values
        penalty_strength: How strong the penalty should be for air-only when criteria not met
        reward_strength: How strong the reward should be for air layers when criteria are met
        min_real_layers: Minimum number of non-air layers
    
    Returns:
        Air penalty/reward value (negative = penalty, positive = reward)
    """
    if state is None or len(state) == 0:
        return -penalty_strength  # Maximum penalty for empty state
    
    # Count non-air layers with significant thickness
    non_air_layers = 0
    total_non_air_thickness = 0
    total_layers = 0
    
    for i, layer in enumerate(state):
        if len(layer) <= 1:
            continue
            
        thickness = layer[0] if len(layer) > 0 else 0
        materials = layer[1:] if len(layer) > 1 else []
        
        if len(materials) == 0:
            continue
            
        material_idx = np.argmax(materials)
        
        # Only consider layers with meaningful thickness
        if thickness > 1e-9:  # 1 nm threshold
            total_layers += 1
            if material_idx != air_material_index:
                non_air_layers += 1
                total_non_air_thickness += thickness
    
    if total_layers == 0:
        return -penalty_strength  # Maximum penalty for no layers
    
    # Check if design criteria are met
    criteria_met = True
    if design_criteria is not None and current_vals is not None:
        for key, threshold in design_criteria.items():
            if key in current_vals:
                val = current_vals[key]
                if key in ["reflectivity"]:
                    # For reflectivity, higher is better
                    if val < threshold:
                        criteria_met = False
                        break
                elif key in ["thermal_noise", "absorption"]:
                    # For thermal noise and absorption, lower is better
                    if val > threshold:
                        criteria_met = False
                        break
    
    # Calculate air fraction
    air_fraction = 1.0 - (non_air_layers / total_layers) if total_layers > 0 else 1.0
    
    if criteria_met:
        # Design criteria are met - reward having more air (fewer layers)
        # More air = simpler coating = better
        if non_air_layers >= min_real_layers:
            # We have enough real layers and meet criteria, so reward air
            air_reward = reward_strength * air_fraction
            return air_reward
        else:
            # Still need more real layers even though criteria are met
            return 0
    else:
        # Design criteria not met - penalize air-heavy coatings
        if non_air_layers == 0:
            # Pure air stack - maximum penalty
            return -penalty_strength
        elif non_air_layers < min_real_layers:
            # Too few real layers - scaled penalty
            penalty = penalty_strength * (1.0 - non_air_layers / min_real_layers) * 0.8
            return -penalty
        else:
            # Have enough real layers but criteria not met - small penalty for excess air
            penalty = penalty_strength * air_fraction * 0.2
            return -penalty

def sigmoid(x, mean=0.5, a=0.01):
    return 1/(1+np.exp(-a*(x-mean)))

def inv_sigmoid(x, mean=0.5, a=0.01):
    return -np.log((1/x) - 1)/a + mean


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


def reward_function_normalise_log_targets(reflectivity, thermal_noise, total_thickness, absorption, optimise_parameters, optimise_targets, env, combine="product", neg_reward=-1e3, weights=None):
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

    if not hasattr(env, 'objective_bounds'):
        env.objective_bounds = {
            'reflectivity': {'min': 1e-6, 'max': 1e-1},      # Typical coating values
            'absorption': {'min': 1e-4, 'max': 1000.0},         # Physical bounds
            'thermal_noise': {'min': 1e-25, 'max': 1e-15},   # Typical noise values  
            'thickness': {'min': 100, 'max': 50000}          # nm range
        }

    normed_vals = {}
    normed_targets = {}
    for key in vals.keys():
        if key in optimise_parameters:
            normed_vals[key] = (vals[key] - env.objective_bounds[key]['min']) / (env.objective_bounds[key]['max'] - env.objective_bounds[key]['min'])
            normed_targets[key] = (optimise_targets[key] - env.objective_bounds[key]['min']) / (env.objective_bounds[key]['max'] - env.objective_bounds[key]['min'])

    if "reflectivity" in optimise_parameters:
        log_reflect = -np.log(np.abs(normed_vals["reflectivity"]-normed_targets["reflectivity"])) + 12
        rewards["reflectivity"] = log_reflect 

    if "thermal_noise" in optimise_parameters and thermal_noise is not None:
        log_therm = np.log(1./np.abs(normed_vals["thermal_noise"]-normed_targets["thermal_noise"]))
        rewards["thermal_noise"] = log_therm 

    if "thickness" in optimise_parameters:
        rewards["thickeness"] = -normed_vals["thickness"]
    
    if "absorption" in optimise_parameters:
        log_absorption = (-np.log(np.abs(normed_vals["absorption"]-normed_targets["absorption"])) + 12) 
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


def reward_function_normalise_log_targets_with_air_management(reflectivity, thermal_noise, total_thickness, absorption, 
                                                             optimise_parameters, optimise_targets, env, 
                                                             combine="product", neg_reward=-1e3, weights=None,
                                                             design_criteria=None, air_penalty_strength=1.0, 
                                                             air_reward_strength=0.5, min_real_layers=2):
    """
    Normalized log targets reward function with intelligent air management.
    
    - Penalizes air-only coatings when design criteria are NOT met
    - Rewards air layers (simpler coatings) when design criteria ARE met
    
    Args:
        reflectivity, thermal_noise, total_thickness, absorption: Current objective values
        optimise_parameters: Parameters being optimized
        optimise_targets: Target values for optimization 
        env: Environment instance (needed to access current state and bounds)
        combine: How to combine rewards ("sum", "product", "logproduct")
        neg_reward: Negative reward for invalid states
        weights: Optional weight dictionary
        design_criteria: Dict of design criteria thresholds (e.g., {"reflectivity": 0.99999, "thermal_noise": 5e-21, "absorption": 0.01})
        air_penalty_strength: Strength of penalty for air-only when criteria not met
        air_reward_strength: Strength of reward for air layers when criteria are met  
        min_real_layers: Minimum number of real layers before air management kicks in
        
    Returns:
        Tuple of (total_reward, vals, rewards)
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
            'reflectivity': {'min': 1e-6, 'max': 1e-1},      # Typical coating values
            'absorption': {'min': 1e-4, 'max': 1000.0},      # Physical bounds
            'thermal_noise': {'min': 1e-25, 'max': 1e-15},   # Typical noise values  
            'thickness': {'min': 100, 'max': 50000}          # nm range
        }

    # Calculate normalized values as in original function
    normed_vals = {}
    normed_targets = {}
    for key in vals.keys():
        if key in optimise_parameters:
            normed_vals[key] = (vals[key] - env.objective_bounds[key]['min']) / (env.objective_bounds[key]['max'] - env.objective_bounds[key]['min'])
            normed_targets[key] = (optimise_targets[key] - env.objective_bounds[key]['min']) / (env.objective_bounds[key]['max'] - env.objective_bounds[key]['min'])

    # Calculate standard reward components (same as original)
    if "reflectivity" in optimise_parameters:
        log_reflect = -np.log(np.abs(normed_vals["reflectivity"]-normed_targets["reflectivity"])) + 12
        rewards["reflectivity"] = log_reflect 

    if "thermal_noise" in optimise_parameters and thermal_noise is not None:
        log_therm = np.log(1./np.abs(normed_vals["thermal_noise"]-normed_targets["thermal_noise"]))
        rewards["thermal_noise"] = log_therm 

    if "thickness" in optimise_parameters:
        rewards["thickness"] = -normed_vals["thickness"]
    
    if "absorption" in optimise_parameters:
        log_absorption = (-np.log(np.abs(normed_vals["absorption"]-normed_targets["absorption"])) + 12) 
        rewards["absorption"] = log_absorption

    # Calculate air penalty/reward
    air_adjustment = 0
    if env is not None and hasattr(env, 'current_state'):
        air_adjustment = calculate_air_penalty_reward(
            env.current_state, 
            air_material_index=getattr(env, 'air_material_index', 0),
            design_criteria=design_criteria,
            current_vals=vals,
            penalty_strength=air_penalty_strength,
            reward_strength=air_reward_strength,
            min_real_layers=min_real_layers
        )
    
    # Combine base rewards  
    if combine=="sum":
        base_reward = np.sum([rewards[key]*weights[key] for key in optimise_parameters])
    elif combine=="product":
        base_reward = np.prod([rewards[key] for key in optimise_parameters])
    elif combine=="logproduct":
        base_reward = np.log(np.prod([rewards[key] for key in optimise_parameters]))
    else:
        raise ValueError(f"combine must be either 'sum' or 'product' not {combine}")
    
    # Add air adjustment to total reward
    total_reward = base_reward + air_adjustment

    if np.isnan(total_reward) or np.isinf(total_reward):
        total_reward = neg_reward

    rewards["total_reward"] = total_reward
    rewards["base_reward"] = base_reward
    rewards["air_adjustment"] = air_adjustment
    rewards["air_penalty_active"] = air_adjustment < 0
    rewards["air_reward_active"] = air_adjustment > 0

    return total_reward, vals, rewards

def reward_function_log_normalise_targets(reflectivity, thermal_noise, total_thickness, absorption, optimise_parameters, optimise_targets, env, combine="product", neg_reward=-1e3, weights=None):
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

    if not hasattr(env, 'objective_bounds'):
        env.objective_bounds = {
            'reflectivity': {'min': 1e-6, 'max': 1e-1},      # Typical coating values
            'absorption': {'min': 1e-4, 'max': 1000.0},         # Physical bounds
            'thermal_noise': {'min': 1e-25, 'max': 1e-15},   # Typical noise values  
            'thickness': {'min': 100, 'max': 50000}          # nm range
        }

    target_mapping = {
        "reflectivity": "log-",
        "thermal_noise": "log-",
        "thickness": "linear-",
        "absorption": "log-"
    }

    normed_vals = {}
    for key in vals.keys():
        if key in optimise_parameters:
            if target_mapping[key][:-1] == "log":
                normed_vals[key] = np.log(np.abs(vals[key] - optimise_targets[key])) / (np.log(env.objective_bounds[key]['max']) - np.log(env.objective_bounds[key]['min']))

            elif target_mapping[key][:-1] == "linear":
                normed_vals[key] = np.abs(vals[key] - optimise_targets[key]) / (env.objective_bounds[key]['max'] - env.objective_bounds[key]['min']) 
            else:
                raise ValueError(f"Unknown target mapping for {key}: {target_mapping[key]}")

            if target_mapping[key].endswith("-"):
                normed_vals[key] = -normed_vals[key]
            
            # Set the rewards dictionary with the normalized values
            rewards[key] = normed_vals[key]



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


def reward_function_normalise_log(reflectivity, thermal_noise, total_thickness, absorption, optimise_parameters, optimise_targets, env, combine="product", neg_reward=-1e3, weights=None):
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

    if not hasattr(env, 'objective_bounds'):
        env.objective_bounds = {
            'reflectivity': {'min': 1e-6, 'max': 1e-1},      # Typical coating values
            'absorption': {'min': 1e-4, 'max': 1000.0},         # Physical bounds
            'thermal_noise': {'min': 1e-25, 'max': 1e-15},   # Typical noise values  
            'thickness': {'min': 100, 'max': 50000}          # nm range
        }

    normed_vals = {}
    normed_targets = {}
    normed_log_vals = {}
    for key in vals.keys():
        if key in optimise_parameters:
            normed_vals[key] = (vals[key] - env.objective_bounds[key]['min']) / (env.objective_bounds[key]['max'] - env.objective_bounds[key]['min'])
            normed_targets[key] = (optimise_targets[key] - env.objective_bounds[key]['min']) / (env.objective_bounds[key]['max'] - env.objective_bounds[key]['min'])
            normed_log_vals[key] = (np.log(vals[key]) - np.log(env.objective_bounds[key]['min'])) / (np.log(env.objective_bounds[key]['max']) - np.log(env.objective_bounds[key]['min']))
    
    if "reflectivity" in optimise_parameters:
        log_reflect = normed_log_vals["reflectivity"] 
        rewards["reflectivity"] = log_reflect 

    if "thermal_noise" in optimise_parameters and thermal_noise is not None:
        log_therm = -normed_vals["thermal_noise"] 
        rewards["thermal_noise"] = log_therm 

    if "thickness" in optimise_parameters:
        rewards["thickeness"] = -normed_vals["thickness"]
    
    if "absorption" in optimise_parameters:
        log_absorption = -normed_log_vals["absorption"]
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


def reward_function_hypervolume(reflectivity, thermal_noise, total_thickness, absorption, 
                               optimise_parameters, optimise_targets, env, 
                               combine="product", neg_reward=-1e3, weights=None, 
                               ref_point=None, adaptive_ref=True):
    """Hypervolume-based reward function with log-space normalization.
    
    This version transforms objectives to log-space and normalizes them to handle
    different scales properly, then computes hypervolume for multi-objective optimisation.

    Args:
        reflectivity: Reflectivity value (typically 1-R)
        thermal_noise: Thermal noise value
        total_thickness: Total coating thickness
        absorption: Absorption value
        optimise_parameters: List of parameters being optimized
        optimise_targets: Target values for optimisation
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
    if any(np.isnan(val) or np.isinf(val) or val is None for val in vals.values() if val is not None):
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
            # Initialize objective bounds if not present
            if not hasattr(env, 'objective_bounds'):
                env.objective_bounds = {
                    'reflectivity': {'min': 1e-8, 'max': 1e-2},      # Typical coating values
                    'absorption': {'min': 1e-6, 'max': 1000.0},         # Physical bounds
                    'thermal_noise': {'min': 1e-25, 'max': 1e-15},   # Typical noise values  
                    'thickness': {'min': 100, 'max': 50000}          # nm range
                }
            
            # Update bounds based on current data (with safety margins)
            pareto_array = np.array(updated_pareto_front)
            for i, param in enumerate(optimise_parameters):
                current_min = np.min(pareto_array[:, i])
                current_max = np.max(pareto_array[:, i])
                
                bounds = env.objective_bounds[param]
                bounds['min'] = min(bounds['min'], current_min * 0.5)  # Expand with safety margin
                bounds['max'] = max(bounds['max'], current_max * 2.0)  # Expand with safety margin
                
                # Ensure positive bounds for log transformation
                bounds['min'] = max(bounds['min'], 1e-12)
            
            # Transform to log space and normalize
            log_normalized_front = []
            
            for point in updated_pareto_front:
                log_normalized_point = []
                for i, param in enumerate(optimise_parameters):
                    bounds = env.objective_bounds[param]
                    if param == "reflectivity":
                        # Reflectivity is typically 1-R, so we want to transform it to log space
                        value = 1 - point[i]
                    else:
                        value = point[i]
                    
                    # Ensure value is within bounds and positive for log
                    value = max(value, bounds['min'])
                    value = min(value, bounds['max'])
                    
                    # Transform to log space
                    log_value = np.log10(value)
                    log_min = np.log10(bounds['min'])
                    log_max = np.log10(bounds['max'])
                    
                    # Normalize to [0,1] in log space
                    if log_max > log_min:
                        normalized_val = (log_value - log_min) / (log_max - log_min)
                    else:
                        normalized_val = 0.5  # Default if no range
                    
                    # Clip to ensure [0,1] range
                    normalized_val = np.clip(normalized_val, 0, 1)
                    log_normalized_point.append(normalized_val)
                
                log_normalized_front.append(log_normalized_point)
            
            log_normalized_front = np.array(log_normalized_front)
            
            # Reference point in normalized log space (slightly worse than worst point)
            ref_point_normalized = np.ones(len(optimise_parameters)) * 1.1
            
            # Calculate hypervolume in normalized log space
            hv_indicator = HV(ref_point=ref_point_normalized)
            hypervolume = hv_indicator(log_normalized_front)
            
            # Scale hypervolume to reward range
            base_hypervolume_reward = hypervolume * 1000
            
            # Additional reward components
            diversity_bonus = 0
            if len(updated_pareto_front) > 1:
                # Reward front diversity (spread in each dimension)
                front_ranges = np.ptp(log_normalized_front, axis=0)  # Range in each normalized dimension
                diversity_bonus = np.sum(front_ranges) * 100  # Bonus for diversity
            
            # Size bonus for larger fronts
            size_bonus = len(updated_pareto_front) * 10
            
            hypervolume_reward = base_hypervolume_reward + diversity_bonus + size_bonus
            
        except Exception as e:
            print(f"Error calculating hypervolume: {e}")
            import traceback
            traceback.print_exc()
            hypervolume_reward = neg_reward / 10  # Small penalty, not full negative reward
    
    # Additional reward for improving the front
    front_improvement_reward = 0
    if front_updated:
        front_improvement_reward = 100  # Bonus for adding to Pareto front
    
    # Total reward
    total_reward = hypervolume_reward + front_improvement_reward
    
    if np.isnan(total_reward) or np.isinf(total_reward):
        total_reward = neg_reward
    
    # Store individual rewards for analysis
    rewards["total_reward"] = total_reward
    rewards["hypervolume_reward"] = hypervolume_reward
    rewards["front_improvement_reward"] = front_improvement_reward
    rewards["diversity_bonus"] = diversity_bonus if 'diversity_bonus' in locals() else 0
    rewards["size_bonus"] = size_bonus if 'size_bonus' in locals() else 0
    rewards["reflectivity"] = 0  # Not used in hypervolume approach
    rewards["thermal_noise"] = 0  # Not used in hypervolume approach  
    rewards["thickness"] = 0  # Not used in hypervolume approach
    rewards["absorption"] = 0  # Not used in hypervolume approach
    
    return total_reward, vals, rewards