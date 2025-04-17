import numpy as np

def sigmoid(x, mean=0.5, a=0.01):
        return 1/(1+np.exp(-a*(x-mean)))

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
        if absorption == 0:
            log_absorption = -10
        else:
            log_absorption = -np.log10(absorption) + 10
        target_log_absorption = -np.log10((optimise_targets["absorption"])) + 10
        rewards["absorption"] = log_absorption * sigmoid(log_absorption, mean=target_log_absorption, a=10)


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

def reward_function_n2(reflectivity, thermal_noise, total_thickness, absorption, optimise_parameters, optimise_targets, combine="product"):
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

    rewards = {key:0 for key in vals}

    if "reflectivity" in optimise_parameters:
        #log_reflect = np.log(1/np.abs(reflectivity - 1)+1)
        #target_log_reflect = np.log(1/np.abs(optimise_targets["reflectivity"] - 1)+1)
        log_reflect = -np.log10(1-reflectivity) 
        target_log_reflect = -np.log10(1-optimise_targets["reflectivity"])
        rewards["reflectivity"] = log_reflect * sigmoid(log_reflect, mean=target_log_reflect, a=10) 
        """
        if type(reflectivity) == np.ndarray:
            rewards["reflectivity"][reflectivity<0.8] -= 10
        else:
            if reflectivity < 0.8:
                rewards["reflectivity"] -= 10
        """
        #if reflectivity < 0:
        #    rewards["reflectivity"] = 0

    if "thermal_noise" in optimise_parameters and thermal_noise is not None:
        log_therm = -np.log10(thermal_noise)/10 
        target_log_therm = -np.log10(optimise_targets["thermal_noise"])/10 
        rewards["thermal_noise"] = log_therm * sigmoid(log_therm, mean=target_log_therm, a=10)
        if rewards["thermal_noise"] < 0:
            rewards["thermal_noise"] = 0


    if "thickness" in optimise_parameters:
        rewards["thickeness"] = 1/total_thickness
        if total_thickness < 0:
            rewards["thickeness"] = 0
    
    if "absorption" in optimise_parameters:
        log_absorption = (1./absorption)
        target_log_absorption = (1./optimise_targets["absorption"])
        rewards["absorption"] = log_absorption * sigmoid(log_absorption, mean=target_log_absorption, a=1) 


    if combine=="sum":
        total_reward = np.sum([rewards[key] for key in optimise_parameters])
    elif combine=="product":
        total_reward = np.prod([rewards[key] for key in optimise_parameters])
    else:
        raise ValueError(f"combine must be either 'sum' or 'product' not {combine}")


    rewards["total_reward"] = total_reward

    return total_reward, vals, rewards