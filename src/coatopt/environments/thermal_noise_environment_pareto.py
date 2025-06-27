import torch
import numpy as np
from coatopt.environments.coating_utils import getCoatAbsorption, getCoatNoise2, getCoatRefl2, merit_function_2
import time
from tmm import coh_tmm
import copy
import matplotlib.pyplot as plt
from coatopt.environments.thermal_noise_environment import CoatingStack
from coatopt.environments.coating_reward_function import reward_function_log_minimise, reward_function_raw, reward_function_target, reward_function
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.indicators.hv import HV


class ParetoCoatingStack(CoatingStack):

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
            optimise_weight_ranges = {"reflectivity":[0,1], "thermal_noise":[0,1], "absorption":[0,1], "thickness":[0,1]},
            light_wavelength=1064e-9,
            include_random_rare_state=False,
            use_optical_thickness=True,
            thickness_sigma=1e-4,
            combine="logproduct",
            final_weight_epoch = 1,
            start_weight_alpha = 1.0,
            final_weight_alpha = 1.0,
            cycle_weights=False,
            n_weight_cycles=2):
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
        super(ParetoCoatingStack, self).__init__(
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
            optimise_weight_ranges=optimise_weight_ranges,
            light_wavelength=light_wavelength,
            include_random_rare_state=include_random_rare_state,
            use_optical_thickness=use_optical_thickness,
            combine=combine,
            reward_func=reward_func,
            final_weight_epoch=final_weight_epoch,
            start_weight_alpha=start_weight_alpha,
            final_weight_alpha=final_weight_alpha,
            cycle_weights=cycle_weights,
            n_weight_cycles=n_weight_cycles)
        
        self.pareto_front = []
        self.reference_point = []
        self.saved_points = []
        self.saved_data = []

    def update_pareto_front(self, pareto_front, new_point):
        """
        Update the Pareto front with a new point and check if it updates the front.

        Parameters:
            pareto_front (numpy.ndarray): Current Pareto front points (shape: [n_points, n_dimensions]).
            new_point (numpy.ndarray): New point to be added (shape: [n_dimensions]).

        Returns:
            numpy.ndarray: Updated Pareto front.
            bool: Whether the Pareto front was updated or not.
        """
        # Combine the current Pareto front with the new point
        combined_points = np.vstack([pareto_front, new_point])

        # Perform non-dominated sorting
        nds = NonDominatedSorting()
        fronts = nds.do(combined_points)

        # Extract the updated Pareto front (the first front)
        updated_pareto_front = combined_points[fronts[0]]

        # Check if the Pareto front was updated
        pareto_updated = not np.array_equal(updated_pareto_front, pareto_front)

        return updated_pareto_front, pareto_updated

    def compute_hv_reward(self,Y_old, Y_new):
        Y_all = np.vstack([Y_old, Y_new])
        #ref_point = np.max(Y_all, axis=0) * 1.1
        hv = HV(ref_point=self.reference_point)
        return hv(Y_new) - hv(Y_old)
    
    def check_air_position(self, state):
        """check at which index the first air layer is present
        """
        pos = len(state) - sum(state[:,self.air_material_index+1])
   
        return pos


    
    def compute_reward(self, new_state, max_value=0.0, target_reflectivity=1.0, objective_weights=None):
        """reward is the improvement of the state over the previous one

        Args:
            state (_type_): _description_
            action (_type_): _description_
        """
        air_index = self.check_air_position(new_state)  

        new_reflectivity, new_thermal_noise, new_E_integrated, new_total_thickness = self.compute_state_value(new_state, return_separate=True)
        vals = {"reflectivity": new_reflectivity, "thermal_noise": new_thermal_noise, "absorption": new_E_integrated, "thickness": new_total_thickness}
        new_point = np.array([new_reflectivity, new_E_integrated])

        new_point = np.zeros((len(self.optimise_parameters),))

        if objective_weights is not None:
            weights = {
                key:objective_weights[i] for i,key in enumerate(self.optimise_parameters)
            }
        else:
            weights=None

        
        total_reward, vals, rewards = self.select_reward(
            new_reflectivity, 
            new_thermal_noise, 
            new_total_thickness, 
            new_E_integrated, 
            weights=weights)

        
        i = 0
        for key in self.optimise_parameters:
            new_point[i] = rewards[key]
            i += 1

        #total_reward = self.compute_hv_reward(self.pareto_front, new_point) if len(self.pareto_front) > 0 else 0

        #rewards = vals
        #rewards["total_reward"] = total_reward
        """
        i=0
        for key in self.optimise_parameters:
            if key == "reflectivity":
                new_point[i] = np.log(1-new_reflectivity)
            elif key == "thermal_noise":
                new_point[i] = np.log(new_thermal_noise)
            elif key == "absorption":
                new_point[i] = np.log(new_E_integrated) - 3
            elif key == "thickness":
                new_point[i] = new_total_thickness
            i += 1
        """
        #volume = self.compute_hv_reward(copy.copy(self.pareto_front), new_point) if len(self.pareto_front) > 0 else 0
        
        updated_pareto_front, front_updated = self.update_pareto_front(copy.copy(self.pareto_front), copy.copy(new_point))

        #minpareto = np.min(self.reference_point, axis=0)
        if air_index <= 10:
            total_reward -= 50

        if front_updated:
            #total_reward = np.abs(np.sqrt(np.sum((new_point - self.pareto_front)**2)))/10 + 10
            total_reward = total_reward
            #total_reward = np.sum((minpareto - new_point)/minpareto) 
            total_reward = total_reward + 10 #+ volume*1e-4
            rewards["total_reward"] = total_reward
            # Calculate the rewards based on the updated Pareto front
            #rewards = {"reflectivity": 1, "thermal_noise": 1, "absorption": 1, "thickness": 1, "total_reward": total_reward}
            self.pareto_front = updated_pareto_front
            self.saved_points.append(new_point)
            self.saved_data.append([new_reflectivity, new_thermal_noise, new_E_integrated, new_total_thickness])
        else:
            # No change in Pareto front, return zero rewards
            #total_reward = -np.abs(np.sqrt(np.sum((new_point - self.pareto_front)**2)))/10
            #total_reward = np.sum((minpareto - new_point)/minpareto)
            total_reward = total_reward #+ volume*1e-4
            rewards["total_reward"] = total_reward 
            #rewards = {"total_reward": 0, "reflectivity": 0, "thermal_noise": 0, "absorption": 0, "thickness": 0, "total_reward": total_reward}
        
        rewards["updated_pareto_front"] = updated_pareto_front
        rewards["front_updated"] = front_updated
            
        return total_reward, vals, rewards

    
    def step(self, action, max_state=0, verbose=False, state=None, layer_index=None, always_return_value=False, objective_weights=None):
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
        new_value = 0
        rewards = {
            "reflectivity": 0,
            "thermal_noise": 0,
            "thickness": 0,
            "absorption": 0,
            "total_reward": 0
        }
        vals = {
            "reflectivity": 0,
            "thermal_noise": 0,
            "thickness": 0,
            "absorption": 0,}

        terminated = False
        finished = False
        front_updated=False
        

        #print(torch.any((self.current_state[0] + actions[2]) < self.min_thickness))
        if self.min_thickness > thickness or thickness > self.max_thickness or not np.isfinite(thickness):
            #terminated=True
            #reward += neg_reward
            self.current_state = new_state
            #new_value = neg_reward
            print("out of thickness bounds")
        #elif material == self.air_material_index and self.ignore_air_option == False:
        #    terminated=True
        #    reward += neg_reward - 10
        #    self.current_state = new_state
        elif self.current_index == self.max_layers-1 or material == self.air_material_index:
         #print("out of thickness bounds")
            finished = True
            self.current_state = new_state
            reward, vals, rewards = self.compute_reward(new_state, max_state, objective_weights=objective_weights)
            #print("finished")
            #reward_diff, reward, new_value = self.compute_reward(new_state, max_state)
        #elif material == self.previous_material:
        #    terminated = True
            #reward = neg_reward
            #self.current_state = new_state
            #reward += neg_reward
            #print("same material")
        else:
            self.current_state = new_state
            #reward_diff, reward, new_value = self.compute_reward(new_state, max_state)
            #self.current_state_value = reward
            if self.use_intermediate_reward:
                reward, vals, rewards = self.compute_reward(new_state, max_state, objective_weights=objective_weights)
        
        # was for adding an extra layer at start and end
        #if new_state[0, 4] == 1 and new_state[-1, 4] and np.all(new_state[:,4]) == False:
        #    reward += 1

        if np.any(np.isinf(new_state)) or np.any(np.isnan(new_state)) or np.isnan(reward) or np.isinf(reward):
            #rewards["total_reward"] = neg_reward
            reward = neg_reward
            terminated = True
            new_value = neg_reward

        if finished and rewards["front_updated"]:
            self.pareto_front = rewards["updated_pareto_front"]
            #print("pareto front updated")
    
        self.previous_material = material
        #print(new_value)

        self.length += 1
        self.current_index += 1




        return new_state, rewards, terminated, finished, reward, full_action, vals


