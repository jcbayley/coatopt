import torch
import numpy as np
from coatopt.environments.coating_utils import getCoatAbsorption, getCoatNoise2, getCoatRefl2, merit_function_2
import time
from tmm import coh_tmm
import copy
import matplotlib.pyplot as plt
from coatopt.environments.thermal_noise_environment import CoatingStack

class GeneticCoatingStack(CoatingStack):

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
            reward_func="default",
            use_intermediate_reward=False,
            ignore_air_option=False,
            ignore_substrate_option=False,
            use_ligo_reward=False,
            optimise_parameters = ["reflectivity", "thermal_noise", "absorption","thickness"],
            optimise_targets = {"reflectivity":0.99999, "thermal_noise":5.394480540642821e-21, "absorption":0.01, "thickness":0.1},
            light_wavelength=1064e-9,
            include_random_rare_state=False,
            use_optical_thickness=True,
            thickness_sigma=1e-4,
            combine="logproduct"):
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
        super(GeneticCoatingStack, self).__init__(
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
            use_optical_thickness=use_optical_thickness,
            combine=combine,
            reward_func=reward_func,)

    
    def sample_state_space(self, ):
        """return air with a thickness of 1

        Returns:
            _type_: _description_
        """

        layers = np.zeros((self.max_layers, self.n_materials + 1))
        reach_end = False
        for i in range(self.max_layers):
            material = np.random.randint(1, self.n_materials)
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

    def sample_action_space(self, current_state):
        """sample from the available state space

        Returns:
            _type_: _description_
        """
        maxind = 0
        for i,current_layer in enumerate(current_state):
            maxind = i
            if current_layer[self.air_material_index] == 1:
                break
        if maxind == 0:
            layer_ind = 0
        else:   
            layer_ind = np.random.randint(maxind+1)

        if self.ignore_air_option:
            new_material = torch.nn.functional.one_hot(torch.from_numpy(np.array(np.random.randint(self.n_materials-1) + 1)), num_classes=self.n_materials)
        else:
            new_material = torch.nn.functional.one_hot(torch.from_numpy(np.array(np.random.randint(self.n_materials))), num_classes=self.n_materials)

        thickness_change = torch.randn(1)*self.thickness_sigma
        new_thickness = current_state[layer_ind, 0] + thickness_change

        while new_thickness < self.min_thickness or new_thickness > self.max_thickness:
            thickness_change = torch.randn(1)*self.thickness_sigma
            new_thickness = current_state[layer_ind, 0] + thickness_change

        if new_thickness < 0:
            print(new_thickness)

        return np.argmax(new_material), new_thickness[0], layer_ind

    '''
    def compute_reward(self, new_state, max_value=0.0, target_reflectivity=1.0):
        """reward is the improvement of the state over the previous one

        Args:
            state (_type_): _description_
            action (_type_): _description_
        """

        state_trim = []
        set_reward = False
        for c_layer in new_state:
            if np.argmax(c_layer[1:]) == self.air_material_index:
                break
            else:
                state_trim.append(c_layer)
            
            if c_layer[0] < self.min_thickness or  c_layer[0] > self.max_thickness:
                set_reward = True


        new_value = self.compute_state_value(state_trim)
        #new_value = np.log(new_value/(1-new_value))
        #old_value = self.compute_state_value(old_state) + 5
        #reward_diff = 0.01/(new_value - target_reflectivity)**2
        #reward_diff = (new_value - target_reflectivity)**2
        #reward_diff = np.log(reward_diff/(1-reward_diff))
        reward = new_value
        if set_reward:
            reward += -0.1
        #reward_diff = new_value - max_value

 
        return reward
    '''



    def step(self, action, max_state=0, verbose=False, state=None, layer_index=None, always_return_value=False):
        """action[0] - thickness
           action[1:N] - material probability

        Args:
            action (_type_): _description_
        """
        
        if state is None:
            state = self.current_state
        else:
            self.current_state = state

        if layer_index is None:
            layer_index = self.current_index
        else:
            self.current_index = layer_index

        material = action[0]
        thickness = action[1] #* self.light_wavelength /(4*self.materials[material]["n"])
        new_state, full_action = self.update_state(np.copy(state), thickness, material)


        neg_reward = -1000
        reward = neg_reward

        terminated = False
        finished = False

        reward, vals, rewards = self.compute_reward(new_state, max_state)
     

        if np.any(np.isinf(new_state)) or np.any(np.isnan(new_state)) or np.isnan(reward) or np.isinf(reward) or self.min_thickness > thickness or thickness > self.max_thickness or not np.isfinite(thickness):
            #rewards["total_reward"] = neg_reward
            reward = neg_reward
            terminated = True
            new_value = neg_reward

        self.previous_material = material
        #print(new_value)

        self.length += 1
        self.current_index += 1


        return new_state, rewards, terminated, finished, reward, full_action
    



