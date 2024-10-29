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
            thickness_sigma=1e-4,
            substrate_material_index=1,
            variable_layers=False,
            opt_init=False):
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
        super(MCMCCoatingStack, self).__init__(max_layers, min_thickness, max_thickness, materials, air_material_index, substrate_material_index, variable_layers, opt_init)

    
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

    def log_probability(self, params):
        # Compute the mean and standard deviation of the state values

        materials, thicknesses = np.split(params, [self.max_layers])

        if np.any(materials > (self.n_materials+0.49 - 1)) or np.any(materials < 0.5):
            return -np.inf
        
        if np.any(thicknesses < self.min_thickness) or np.any(thicknesses > self.max_thickness):
            return -np.inf
        
        
        state = self.convert_params_to_state(params)
        mean = self.compute_state_value(state)


        
        std_dev = 0.1  # Set the standard deviation as desired

        # Compute the likelihood using the Gaussian distribution
        likelihood = mean/(1-mean)

        return likelihood
        

    



