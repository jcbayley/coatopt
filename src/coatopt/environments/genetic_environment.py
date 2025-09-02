import torch
import numpy as np
import time
from tmm import coh_tmm
import copy
import matplotlib.pyplot as plt
from typing import Optional, TYPE_CHECKING
from coatopt.environments.core.base_environment import BaseCoatingEnvironment

from coatopt.config.structured_config import CoatingOptimisationConfig


class GeneticCoatingStack(BaseCoatingEnvironment):
    """
    Genetic algorithm coating environment with dual initialization support.
    """

    def __init__(self, config: Optional[CoatingOptimisationConfig] = None, **kwargs):
        """
        Initialize genetic environment with dual initialization support.
        
        Args:
            config: CoatingOptimisationConfig object (new approach)
            **kwargs: Individual parameters (legacy approach), including:
                     thickness_sigma: Standard deviation for thickness mutations
        """
        # Extract genetic-specific parameters before calling super()
        genetic_params = {}
        if 'thickness_sigma' in kwargs:
            genetic_params['thickness_sigma'] = kwargs.pop('thickness_sigma')
        
        # Initialize base environment with all standard parameters
        super().__init__(config, **kwargs)
        
        # Genetic-specific initialization
        self._setup_genetic_specific_attributes(**genetic_params)

    def _setup_genetic_specific_attributes(self, **kwargs):
        """Setup genetic algorithm specific attributes."""
        # Extract genetic-specific parameters
        self.thickness_sigma = kwargs.get('thickness_sigma', 1e-4)

    
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
    



