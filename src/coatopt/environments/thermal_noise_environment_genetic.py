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
        super(GeneticCoatingStack, self).__init__(max_layers, min_thickness, max_thickness, materials, air_material_index, substrate_material_index, variable_layers, opt_init)

    
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

    def sample_action_space(self, current_state):
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
    


    def step(self, state, action, max_state=0, verbose=False):
        """action[0] - thickness
           action[1:N] - material probability

        Args:
            action (_type_): _description_
        """
        layer = action[2]
        thickness_change = action[1]
        material_layer = action[0]
        material = np.argmax(material_layer)
        #new_state, full_action = self.update_state(state, thickness_change, material)

        new_state = copy.copy(state)
        new_state[layer][0] += thickness_change
        if material == self.air_material_index:
            repmat = np.tile(material_layer, (len(new_state) - layer,1))
            new_state[layer:, 1:] = repmat
        else:
            new_state[layer][1:] = material_layer

        thickness = new_state[layer][0]
        

        reward = 0
        neg_reward = -1.0
        new_value = 0

        terminated = False
        finished = False
        reward = self.compute_reward(new_state, max_state)

        if thickness < 0:
            print(action)
            print(state)
            print(new_state)
            sys.exit()

        #print(torch.any((self.current_state[0] + actions[2]) < self.min_thickness))
        if self.min_thickness > thickness or thickness > self.max_thickness or not np.isfinite(thickness):
            terminated=True
            reward = neg_reward
            self.current_state = new_state
            new_value = neg_reward
        elif self.current_index == self.max_layers-1 or material == self.air_material_index:
         #print("out of thickness bounds")
            finished = True
            self.current_state = new_state
            #reward_diff, reward, new_value = self.compute_reward(new_state, max_state)
        #elif action[1][0] == self.previous_material:
        #    terminated = True
        #    reward = -0.01
        else:
            self.current_state = new_state
            #reward_diff, reward, new_value = self.compute_reward(new_state, max_state)
            #self.current_state_value = reward
            reward = 0.0
        
        # was for adding an extra layer at start and end
        #if new_state[0, 4] == 1 and new_state[-1, 4] and np.all(new_state[:,4]) == False:
        #    reward += 1

        if np.any(np.isinf(new_state)) or np.any(np.isnan(new_state)) or np.isnan(reward):
            reward = 0.0#neg_reward-10
            terminated = True
            new_value = neg_reward

        self.previous_material = material
        #print(new_value)

        self.length += 1
        self.current_index += 1

        #print("cind:", self.current_index)
        #print(new_state)


        return new_state, reward
    



