import torch
import numpy as np
from coatopt.environments.coating_utils import getCoatAbsorption, getCoatNoise2, getCoatRefl2, merit_function_2
import time
from tmm import coh_tmm
import copy
import matplotlib.pyplot as plt
from coatopt.environments.thermal_noise_environment import CoatingStack

class MCMCCoatingStack(CoatingStack):

    def __init__(
            self, 
            max_layers, 
            min_thickness, 
            max_thickness, 
            materials, 
            air_material_index=0,
            substrate_material_index=1,
            variable_layers=False,
            opt_init=False,
            reflectivity_reward_shape="none",
            thermal_reward_shape="log_thermal_noise",
            absorption_reward_shape="log_absorption",
            use_intermediate_reward=False,
            ignore_air_option=False,
            ignore_substrate_option=False,
            use_ligo_reward=False,
            optimise_parameters = ["reflectivity", "thermal_noise", "absorption","thickness"],
            optimise_targets = {"reflectivity":0.99999, "thermal_noise":5.394480540642821e-21, "absorption":0.01, "thickness":0.1},
            light_wavelength=1064e-9,
            include_random_rare_state=False,
            use_optical_thickness=True,
            thickness_sigma=0.1):
        """_summary_

        Args:
            max_layers (_type_): _description_
            min_thickness (_type_): _description_
            max_thickness (_type_): _description_
            materials (_type_): _description_
            air_material (_type_): _description_
            thickness_options (list, optional): _description_. Defaults to [0.1,1,10].
            variable_layers (bool, optional): _description_. Defaults to False.
        """
        self.thickness_sigma = thickness_sigma
        super(MCMCCoatingStack, self).__init__(
            max_layers, 
            min_thickness, 
            max_thickness, 
            materials, 
            air_material_index=air_material_index,
            substrate_material_index=substrate_material_index,
            variable_layers=variable_layers,
            opt_init=opt_init,
            reflectivity_reward_shape=reflectivity_reward_shape,
            thermal_reward_shape=thermal_reward_shape,
            absorption_reward_shape=absorption_reward_shape,
            use_intermediate_reward=use_intermediate_reward,
            ignore_air_option=ignore_air_option,
            ignore_substrate_option=ignore_substrate_option,
            use_ligo_reward=use_ligo_reward,
            optimise_parameters = optimise_parameters,
            optimise_targets = optimise_targets,
            light_wavelength=light_wavelength,
            include_random_rare_state=include_random_rare_state,
            use_optical_thickness=use_optical_thickness)

    
    def sample_state_space(self, ):
        """return air with a thickness of 1

        Returns:
            _type_: _description_
        """

        layers = np.zeros((self.max_layers, self.n_materials + 1))
        reach_end = False
        for i in range(self.max_layers):
            material = np.random.randint(0, self.n_materials)
            if material == self.air_material_index:
                reach_end = True
            if reach_end:
                layers[i,self.air_material_index+1] = 1
            else:
                layers[i,material+1] = 1

        layers[:,0] = np.random.uniform(self.min_thickness, self.max_thickness, size=len(layers[:,0]))

        if np.any(layers[:,0] < 0):
            print(f"state and thickness: sample")
            print(layers)

        return layers

    def sample_prior(self, current_state):
        """sample from the available state space

        Returns:
            _type_: _description_
        """
        maxind = 0
        for i,current_layer in enumerate(current_state):
            maxind = i
            if current_layer[1] == 1:
                break
        if maxind == 0:
            layer_ind = 0
        else:
            layer_ind = np.random.randint(maxind+1)
        new_material = torch.nn.functional.one_hot(torch.from_numpy(np.array(np.random.randint(self.n_materials))), num_classes=self.n_materials)

        thickness_change = torch.randn(1)*self.thickness_sigma
        new_thickness = current_state[layer_ind, 0] + thickness_change

        while new_thickness < self.min_thickness or new_thickness > self.max_thickness:
            thickness_change = torch.randn(1)*self.thickness_sigma
            new_thickness = current_state[layer_ind, 0] + thickness_change

        if new_thickness < 0:
            print(new_thickness)

        return new_material, thickness_change, layer_ind
    
    def convert_params_to_state(self, params):
        materials, thicknesses = np.split(params, [self.max_layers])
        materials = np.round(materials).astype(int)
        materials = torch.nn.functional.one_hot(torch.from_numpy(materials), num_classes=self.n_materials)
        state = np.concatenate((thicknesses[:, np.newaxis], materials), axis=1)
        return state
    
    def convert_state_to_params(self, state):
        thicknesses = state[:,0]
        materials = np.argmax(state[:,1:], axis=1)
        return np.hstack((materials, thicknesses))
    
    def sigmoid(self, x, mean=0.5, a=0.01):
        return 1/(1+np.exp(-a*(x-mean)))
    
    def compute_reward(self, new_state, max_value=0.0, target_reflectivity=1.0):
        """reward is the improvement of the state over the previous one

        Args:
            state (_type_): _description_
            action (_type_): _description_
        """

        new_reflectivity, new_thermal_noise, new_absorption, new_total_thickness = self.compute_state_value(new_state, return_separate=True)

        vals = {
            "reflectivity": new_reflectivity,
            "thermal_noise": new_thermal_noise,
            "thickness": new_total_thickness,
            "absorption": new_absorption
        }

        rewards = {key:0 for key in vals}

        #new_value = np.log(new_value/(1-new_value))
        #old_value = self.compute_state_value(old_state) + 5
        #reward_diff = 0.01/(new_value - target_reflectivity)**2
        #reward_diff = (new_value - target_reflectivity)**2
        #reward_diff = np.log(reward_diff/(1-reward_diff))
        
        if "reflectivity" in self.optimise_parameters:
            log_reflect = np.log(1/np.abs(new_reflectivity - 1)+1)
            target_log_reflect = np.log(1/np.abs(self.optimise_targets["reflectivity"] - 1)+1)
            rewards["reflectivity"] = -np.log(1-new_reflectivity)#log_reflect * self.sigmoid(log_reflect, mean=target_log_reflect, a=0.1)
 

        if "thermal_noise" in self.optimise_parameters and new_thermal_noise is not None:
            log_therm = -np.log(new_thermal_noise)/10
            target_log_therm = -np.log(self.optimise_targets["thermal_noise"])/10
            rewards["thermal_noise"] = log_therm * self.sigmoid(log_therm, mean=target_log_therm, a=0.1)



        if "thickness" in self.optimise_parameters:
            rewards["thickeness"] = -new_total_thickness
        
        if "absorption" in self.optimise_parameters:
            log_absorption = -np.log(new_absorption)
            target_log_absorption = -np.log(self.optimise_targets["absorption"])
            rewards["absorption"] = log_absorption * self.sigmoid(log_absorption, mean=target_log_absorption, a=0.1)
   

        total_reward = np.sum([rewards[key] for key in self.optimise_parameters])


        rewards["total_reward"] = total_reward

        return total_reward, vals, rewards

    def log_probability(self, params):
        # Compute the mean and standard deviation of the state values

        materials, thicknesses = np.split(params, [self.max_layers])

        if np.any(materials > (self.n_materials+0.49 - 1)) or np.any(materials < 0.5):
            return -np.inf
        
        if np.any(thicknesses < self.min_thickness) or np.any(thicknesses > self.max_thickness):
            return -np.inf
        
        
        state = self.convert_params_to_state(params)
        reward, vals, rewards = self.compute_reward(state)
        
        std_dev = 0.1  # Set the standard deviation as desired


        return reward
        

    



