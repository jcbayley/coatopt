import numpy as np
from pymoo.indicators.hv import HV
import copy

def calculate_air_penalty_reward(state, air_material_index=0, design_criteria=None, 
                                current_vals=None, optimise_parameters=None, penalty_strength=20.0, reward_strength=0.5, 
                                min_real_layers=5):
    """
    Calculate penalty for air-only coatings when design criteria are NOT met,
    or reward for more air layers when design criteria ARE met.
    
    Args:
        state: Current coating state
        air_material_index: Index of air material (usually 0)
        design_criteria: Dict of design criteria thresholds
        current_vals: Dict of current objective values
        penalty_strength: How strong the penalty should be
        reward_strength: How strong the reward should be for air layers when criteria are met
        min_real_layers: Minimum number of non-air layers (default 5)
    
    Returns:
        Air penalty/reward value (negative = penalty, positive = reward)
    """
    if state is None or len(state) == 0:
        return -penalty_strength
    
    # Count non-air layers using air material column
    if len(state) > 0:
        # Get the air material column (1 for air, 0 for non-air)
        air_column = state[:, air_material_index + 1]
        # Count non-air layers (where air_column is 0)
        non_air_layers = np.sum(air_column == 0)
    else:
        non_air_layers = 0
    
    # Calculate air fraction
    total_layers = len(state)
    air_fraction = 1.0 - (non_air_layers / total_layers) if total_layers > 0 else 1.0
    
    # Check if design criteria are met
    criteria_met = True if design_criteria is None else False
    if design_criteria is not None and current_vals is not None:
        for key, threshold in design_criteria.items():
            if key in current_vals and key in optimise_parameters:
                val = current_vals[key]
                if key in ["reflectivity"]:
                    if val < threshold:
                        criteria_met = False
                        break
                elif key in ["thermal_noise", "absorption"]:
                    if val > threshold:
                        criteria_met = False
                        break
    
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
            return -penalty_strength * air_fraction * 0.5

def sigmoid(x, mean=0.5, a=0.01):
    return 1/(1+np.exp(-a*(x-mean)))

def inv_sigmoid(x, mean=0.5, a=0.01):
    return -np.log((1/x) - 1)/a + mean


def reward_function_target(reflectivity, thermal_noise, total_thickness, absorption, optimise_parameters, optimise_targets, combine="product", neg_reward=-1e3, weights=None, env=None, **kwargs):
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


def reward_function_raw(reflectivity, thermal_noise, total_thickness, absorption, optimise_parameters, optimise_targets, combine="product", neg_reward=-1e3, weights=None, env=None, **kwargs):
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


def reward_function_log_minimise(reflectivity, thermal_noise, total_thickness, absorption, optimise_parameters, optimise_targets, combine="product", neg_reward=-1e3, weights=None, env=None, **kwargs):
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


def reward_function_normalise_log_targets(reflectivity, thermal_noise, total_thickness, absorption, optimise_parameters, optimise_targets, env, combine="product", neg_reward=-1e3, weights=None, **kwargs):
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
            'reflectivity': [1e-6, 1e-1],      # Typical coating values
            'absorption': [1e-4, 1000.0],      # Physical bounds
            'thermal_noise': [1e-25, 1e-15],   # Typical noise values  
            'thickness': [100, 50000]          # nm range
        }

    normed_vals = {}
    normed_targets = {}
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
                
            normed_vals[key] = (vals[key] - min_bound) / (max_bound - min_bound)
            normed_targets[key] = (optimise_targets[key] - min_bound) / (max_bound - min_bound)

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
                                                             design_criteria=None, air_penalty_strength=50.0, 
                                                             air_reward_strength=1, min_real_layers=2):
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
            'reflectivity': [1e-6, 1e-1],      # Typical coating values
            'absorption': [1e-4, 1000.0],      # Physical bounds
            'thermal_noise': [1e-25, 1e-15],   # Typical noise values  
            'thickness': [100, 50000]          # nm range
        }

    # Calculate normalized values as in original function
    normed_vals = {}
    normed_targets = {}
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
            
            normed_vals[key] = (vals[key] - min_bound) / (max_bound - min_bound)
            normed_targets[key] = (optimise_targets[key] - min_bound) / (max_bound - min_bound)

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
            design_criteria=getattr(env, 'design_criteria'),
            optimise_parameters=optimise_parameters,
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

def reward_function_log_normalise_targets(reflectivity, thermal_noise, total_thickness, absorption, optimise_parameters, optimise_targets, env, combine="product", neg_reward=-1e3, weights=None, **kwargs):
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
            'reflectivity': [1e-6, 1e-1],      # Typical coating values
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
                
            if target_mapping[key][:-1] == "log":
                normed_vals[key] = np.log(np.abs(vals[key] - optimise_targets[key])) / (np.log(max_bound) - np.log(min_bound))

            elif target_mapping[key][:-1] == "linear":
                normed_vals[key] = np.abs(vals[key] - optimise_targets[key]) / (max_bound - min_bound) 
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


def reward_function_log_targets_limits(reflectivity, thermal_noise, total_thickness, absorption, optimise_parameters, optimise_targets, env, combine="product", neg_reward=-1e3, weights=None, use_air_penalty=True, air_penalty_weight=1.0, use_divergence_penalty=True, divergence_penalty_weight=1.0, **kwargs):
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
                    normed_vals[key] += 10  # Large penalty for impossible reflectivity
            else:
                if vals[key] > max_bound:
                    normed_vals[key] += 10  # Large penalty for out-of-bounds values

            # Invert if we want smaller differences to give higher rewards
            if target_mapping[key].endswith("-"):
                normed_vals[key] = 1.0 - normed_vals[key]
            
            # Set the rewards dictionary with the normalized values
            rewards[key] = normed_vals[key]

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

def reward_function_normalise_log(reflectivity, thermal_noise, total_thickness, absorption, optimise_parameters, optimise_targets, env, combine="product", neg_reward=-1e3, weights=None, **kwargs):
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
            'reflectivity': [1e-6, 1e-1],      # Typical coating values
            'absorption': [1e-4, 1000.0],         # Physical bounds
            'thermal_noise': [1e-25, 1e-15],   # Typical noise values  
            'thickness': [100, 50000]          # nm range
        }

    normed_vals = {}
    normed_targets = {}
    normed_log_vals = {}
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
                
            normed_vals[key] = (vals[key] - min_bound) / (max_bound - min_bound)
            normed_targets[key] = (optimise_targets[key] - min_bound) / (max_bound - min_bound)
            normed_log_vals[key] = (np.log(vals[key]) - np.log(min_bound)) / (np.log(max_bound) - np.log(min_bound))
    
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


def reward_function_novelty_search(reflectivity, thermal_noise, total_thickness, absorption, 
                                  optimise_parameters, optimise_targets, env=None,
                                  combine="sum", neg_reward=-1e3, weights=None,
                                  novelty_weight=0.5, archive_size=1000, k_nearest=10,
                                  use_air_penalty=True, air_penalty_weight=1.0, **kwargs):
    """
    Novelty search reward that encourages exploration of underexplored regions.
    Combines standard objective rewards with novelty bonus based on distance to archive.
    
    Args:
        novelty_weight: Weight for novelty component (0.0 = pure objectives, 1.0 = pure novelty)
        archive_size: Maximum size of archive storing explored solutions
        k_nearest: Number of nearest neighbors for novelty calculation
        use_air_penalty: Whether to apply air penalty/reward
        air_penalty_weight: Weight for air penalty component
    """
    vals = {
        "reflectivity": reflectivity,
        "thermal_noise": thermal_noise,
        "thickness": total_thickness,
        "absorption": absorption
    }
    
    if weights is None:
        weights = {key: 1 for key in optimise_parameters}
    
    rewards = {key: 0 for key in vals}
    
    # Calculate base objective rewards (logarithmic for better scaling)
    if "reflectivity" in optimise_parameters:
        rewards["reflectivity"] = -np.log10(max(1 - reflectivity, 1e-15))
    
    if "absorption" in optimise_parameters:
        rewards["absorption"] = -np.log10(max(absorption, 1e-10))
        
    if "thermal_noise" in optimise_parameters and thermal_noise is not None:
        rewards["thermal_noise"] = -np.log10(max(thermal_noise, 1e-25))
        
    if "thickness" in optimise_parameters:
        rewards["thickness"] = -total_thickness
    
    # Calculate base objective reward
    base_reward = sum(rewards[key] * weights[key] for key in optimise_parameters)
    
    # Calculate novelty component
    novelty_reward = 0.0
    if env is not None:
        # Initialize archive if not exists
        if not hasattr(env, 'novelty_archive'):
            env.novelty_archive = []
            
        # Current solution in objective space (normalized)
        current_objectives = []
        if "reflectivity" in optimise_parameters:
            current_objectives.append(-np.log10(max(1 - reflectivity, 1e-15)))
        if "absorption" in optimise_parameters:
            current_objectives.append(-np.log10(max(absorption, 1e-10)))
            
        current_objectives = np.array(current_objectives)
        
        if len(env.novelty_archive) < k_nearest:
            # High novelty for early solutions
            novelty_reward = 1.0
        else:
            # Calculate distances to k-nearest neighbors
            distances = []
            for archived_obj in env.novelty_archive:
                if len(archived_obj) == len(current_objectives):
                    dist = np.linalg.norm(current_objectives - archived_obj)
                    distances.append(dist)
            
            if distances:
                distances.sort()
                k_distances = distances[:min(k_nearest, len(distances))]
                novelty_reward = np.mean(k_distances)
        
        # Update archive
        env.novelty_archive.append(current_objectives.copy())
        if len(env.novelty_archive) > archive_size:
            # Remove random old solution
            import random
            env.novelty_archive.pop(random.randint(0, len(env.novelty_archive)-1))
    
    # Combine base reward and novelty
    total_reward = (1 - novelty_weight) * base_reward + novelty_weight * novelty_reward
    
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
        total_reward = neg_reward
    
    # Store rewards for analysis
    rewards["total_reward"] = total_reward
    rewards["base_reward"] = base_reward
    rewards["novelty_reward"] = novelty_reward
    rewards["air_penalty"] = air_penalty
    
    return total_reward, vals, rewards


def reward_function_pareto_dominance(reflectivity, thermal_noise, total_thickness, absorption,
                                   optimise_parameters, optimise_targets, env=None,
                                   combine="sum", neg_reward=-1e3, weights=None,
                                   reference_set_size=500, crowding_weight=0.3,
                                   use_air_penalty=True, air_penalty_weight=1.0, **kwargs):
    """
    Pareto dominance-based reward that encourages solutions along the Pareto front.
    Minimizes domination count and maximizes crowding distance for diversity.
    
    Args:
        reference_set_size: Size of reference set for dominance comparison
        crowding_weight: Weight for crowding distance component
        use_air_penalty: Whether to apply air penalty/reward
        air_penalty_weight: Weight for air penalty component
    """
    vals = {
        "reflectivity": reflectivity,
        "thermal_noise": thermal_noise,
        "thickness": total_thickness,
        "absorption": absorption
    }
    
    if weights is None:
        weights = {key: 1 for key in optimise_parameters}
    
    rewards = {key: 0 for key in vals}
    
    # Convert to minimization objectives (lower is better)
    objectives = []
    if "reflectivity" in optimise_parameters:
        obj_refl = np.log10(1 - reflectivity)  # minimize (1 - reflectivity)
        objectives.append(obj_refl)
        rewards["reflectivity"] = -obj_refl
        
    if "absorption" in optimise_parameters:
        if absorption == 0:
            obj_abs = 10
        else:
            obj_abs = np.log10(absorption)  # minimize absorption
        objectives.append(obj_abs)
        rewards["absorption"] = -obj_abs
        
    if "thermal_noise" in optimise_parameters and thermal_noise is not None:
        obj_therm = np.log10(thermal_noise)  # minimize thermal noise
        objectives.append(obj_therm)
        rewards["thermal_noise"] = -obj_therm
        
    if "thickness" in optimise_parameters:
        obj_thick = total_thickness  # minimize thickness
        objectives.append(obj_thick)
        rewards["thickness"] = -obj_thick
    
    current_objectives = np.array(objectives)
    
    # Initialize or access reference set
    if env is not None:
        if not hasattr(env, 'pareto_reference_set'):
            env.pareto_reference_set = []
            
        reference_set = env.pareto_reference_set
        
        # Calculate domination count
        domination_count = 0
        for ref_obj in reference_set:
            if len(ref_obj) == len(current_objectives):
                # Check if reference solution dominates current solution
                if (np.all(ref_obj <= current_objectives) and 
                    np.any(ref_obj < current_objectives)):
                    domination_count += 1
        
        # Calculate crowding distance for diversity
        crowding_distance = 0.0
        if len(reference_set) > 2:
            try:
                # Find distances to nearest neighbors in objective space
                distances = []
                for ref_obj in reference_set:
                    if len(ref_obj) == len(current_objectives):
                        dist = np.linalg.norm(current_objectives - ref_obj)
                        if dist > 0:  # Avoid identical solutions
                            distances.append(dist)
                
                if distances:
                    # Crowding distance is minimum distance to existing solutions
                    crowding_distance = min(distances)
            except:
                crowding_distance = 1.0
        else:
            crowding_distance = 1.0  # High diversity for early solutions
        
        # Update reference set
        reference_set.append(current_objectives.copy())
        if len(reference_set) > reference_set_size:
            # Remove a random solution (could be improved with better selection)
            import random
            reference_set.pop(random.randint(0, len(reference_set)-1))
        
                # Calculate reward: minimize domination, maximize diversity
        domination_penalty = -domination_count * 10  # Large penalty for being dominated
        diversity_bonus = crowding_weight * crowding_distance
        
        total_reward = domination_penalty + diversity_bonus
    else:
        # Fallback: simple weighted sum
        total_reward = sum(rewards[key] * weights[key] for key in optimise_parameters)
    
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
        total_reward = neg_reward
    
    # Store rewards for analysis
    rewards["total_reward"] = total_reward
    rewards["domination_penalty"] = domination_penalty if 'domination_penalty' in locals() else 0
    rewards["diversity_bonus"] = diversity_bonus if 'diversity_bonus' in locals() else 0
    rewards["domination_count"] = domination_count if 'domination_count' in locals() else 0
    rewards["crowding_distance"] = crowding_distance if 'crowding_distance' in locals() else 0
    rewards["air_penalty"] = air_penalty
    
    return total_reward, vals, rewards