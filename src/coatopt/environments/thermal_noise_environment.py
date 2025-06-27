import torch
import numpy as np
from .coating_utils import getCoatAbsorption, getCoatNoise2, getCoatRefl2, merit_function, merit_function_2
from .coating_reward_function import reward_function, reward_function_target, reward_function_raw, reward_function_log_minimise
import time
import scipy
from tmm import coh_tmm
import matplotlib.pyplot as plt

class CoatingStack():

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
            final_weight_epoch = 1,
            start_weight_alpha = 1.0,
            final_weight_alpha = 1.0,
            cycle_weights=False,
            n_weight_cycles=2,
            combine="product"):
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
        self.variable_layers = variable_layers
        self.max_layers = max_layers
        self.min_thickness = min_thickness
        self.max_thickness = max_thickness
        self.materials = materials
        self.n_materials = len(materials) 
        self.n_material_options = self.n_materials#len(materials) - 1 if ignore_air_option else len(materials)
        self.air_material_index = air_material_index
        self.substrate_material_index = substrate_material_index
        self.combine=combine
        self.optimise_weight_ranges = optimise_weight_ranges
        self.reward_func = reward_func
        self.final_weight_epoch = final_weight_epoch
        self.start_weight_alpha = start_weight_alpha
        self.final_weight_alpha = final_weight_alpha
        self.cycle_weights = cycle_weights
        self.n_weight_cycles = n_weight_cycles

        self.opt_init = opt_init
        self.reflectivity_reward_shape = reflectivity_reward_shape
        self.thermal_reward_shape = thermal_reward_shape
        self.absorption_reward_shape = absorption_reward_shape
        self.use_intermediate_reward = use_intermediate_reward
        self.ignore_air_option = ignore_air_option
        self.ignore_substrate_option = ignore_substrate_option
        self.use_ligo_reward = use_ligo_reward
        self.optimise_parameters = optimise_parameters
        self.optimise_targets = optimise_targets
        self.use_optical_thickness = use_optical_thickness
        self.include_random_rare_state = include_random_rare_state

        # state space size is index for each material (onehot encoded) plus thickness of each material
        self.state_space_size = self.max_layers*self.n_materials + self.max_layers
        self.state_space_shape = (self.max_layers, self.n_materials + 1)
        self.obs_space_size = self.max_layers*2
        self.obs_space_shape = (self.max_layers, 2)

        self.length = 0
        self.current_state = self.sample_state_space()
        self.current_index = self.max_layers - 1
        self.previous_material = self.substrate_material_index

        self.light_wavelength = light_wavelength


    def reset(self,):
        """reset the state space and length
        """
        self.length = 0
        self.current_state = self.sample_state_space()
        self.current_index = 0#self.max_layers - 1

        return self.current_state
    
    def print_state(self,):
        for i in range(len(self.current_state)):
            print(self.current_state[i])

    def get_optimal_state(self, reverse=False, inds_alternate=[1,2]):

        layers = []
        if self.use_optical_thickness:
            thickness1 = 1/4
            thickness2 = 1/4
        else:
            thickness1 = 1064e-9 /(4*self.materials[inds_alternate[0]]["n"])
            thickness2 = 1064e-9 /(4*self.materials[inds_alternate[1]]["n"])
        #print("thickness", thickness1, thickness2)
        
        opt_state2 = []
        if reverse:
            material = inds_alternate[1]
        else:
            material = inds_alternate[0]
        for i in range(self.max_layers):
            current_material = inds_alternate[1] if material == inds_alternate[0] else inds_alternate[0]
            if current_material == inds_alternate[0]:
                thickness = thickness1
            elif current_material == inds_alternate[1]:
                thickness = thickness2
            l_state = [0,]*(self.n_materials+1)
            l_state[0] = thickness
            l_state[current_material+1] = 1

            opt_state2.append(l_state)
            material = current_material

        opt_state2 = np.array(opt_state2)

        if self.include_random_rare_state:
            mean1 = self.min_thickness + (self.max_thickness - self.min_thickness)/10
            mean2 = self.min_thickness + (self.max_thickness - self.min_thickness)/2
            opt_state2[1] = [mean1, 0,0,0,1]
            opt_state2[-2] = [mean2, 0,0,0,1]

        return opt_state2

    def get_optimal_state_2mat(self, reverse=False, inds_alternate=[1,2,3]):

        layers = []
        if self.use_optical_thickness:
            thicknesses = [1/4,1/4,1/4]
        else:
            thicknesses = [1064e-9 /(4*self.materials[inds_alternate[0]]["n"]),
                            1064e-9 /(4*self.materials[inds_alternate[1]]["n"]),
                            1064e-9 /(4*self.materials[inds_alternate[2]]["n"])]
        #print("thickness", thickness1, thickness2)
        pairs1 = [0,1]
        pairs2 = [0,2]
        opt_state2 = []
        pairs = pairs1
        pind = pairs[1]

        for i in range(self.max_layers):
            if i >= int(0.5*self.max_layers):
                pairs=pairs2
            current_ind = pairs[1] if pind == pairs[0] else pairs[0] 
            current_material = inds_alternate[current_ind]
            thickness = thicknesses[current_ind]
            l_state = [0,]*(self.n_materials+1)
            l_state[0] = thickness
            l_state[current_material+1] = 1

            opt_state2.append(l_state)
            material = current_material
            pind = current_ind

        opt_state2 = np.array(opt_state2)

        if self.include_random_rare_state:
            mean1 = self.min_thickness + (self.max_thickness - self.min_thickness)/10
            mean2 = self.min_thickness + (self.max_thickness - self.min_thickness)/2
            opt_state2[1] = [mean1, 0,0,0,1]
            opt_state2[-2] = [mean2, 0,0,0,1]

        return opt_state2
    
    def get_air_only_state(self,n_layers = None):
        """return air with a thickness of 1

        Returns:
            _type_: _description_
        """
        if n_layers is None:
            n_layers = self.max_layers
        layers = np.zeros((n_layers, self.n_materials + 1))
        layers[:,self.air_material_index+1] = 1
        layers[:,0] = self.min_thickness*np.ones(len(layers[:,0]))
        return layers

    def annealed_dirichlet_weights(self, epoch, total_epochs, base_alpha=0.05, final_alpha=1.0, num_samples=10):
        """
        Sample preference weights from an annealed Dirichlet distribution.
        
        Parameters:
        - epoch: current training epoch
        - total_epochs: total number of epochs
        - base_alpha: initial concentration (low -> extreme weights)
        - final_alpha: final concentration (higher -> uniform weights)
        - num_samples: number of weight vectors to sample
        
        Returns:
        - weights: List of sampled 2D weight vectors summing to 1
        """
        # Annealing factor: linear schedule (can be nonlinear if desired)
        progress = np.min([epoch / total_epochs, 1])
        alpha = base_alpha + (final_alpha - base_alpha) * progress

        # Dirichlet concentration vector: same alpha for both objectives
        concentration = [alpha, alpha]
        
        weights = np.random.dirichlet(concentration, size=num_samples)
        # Replace rows with NaN or Inf with [1, 0] or [0, 1] randomly
        mask = np.isnan(weights).any(axis=1) | np.isinf(weights).any(axis=1)
        random_choices = np.random.choice([0, 1], size=mask.sum())
        weights[mask] = np.column_stack((random_choices, 1 - random_choices))
        return weights
    
    def smooth_cycle_weights(self, t, N, T_cycle, T_hold, total_steps, random_anneal = True):
        """
        Generate smooth cyclic weights for N classes.
        At each cycle, the weight is held at [1, 0, ..., 0], [0, 1, ..., 0], etc. for T_hold steps.
        In between, smooth transitions are applied over the remaining steps.
        
        Args:
            N: Number of classes
            T_cycle: Total steps for one full cycle through all classes (T_hold * N + transition steps)
            T_hold: Number of steps to hold each one-hot vector
            total_steps: Total steps to generate

        Returns:
            weights: A (total_steps x N) numpy array of weights over time
        """
        weights = np.zeros(N)
        phase_steps = T_cycle // N  # steps per class phase
        T_transition = phase_steps - T_hold  # transition steps between classes

        if t < total_steps:
            cycle_pos = t % T_cycle
            class_idx = cycle_pos // phase_steps
            pos_in_phase = cycle_pos % phase_steps

            if pos_in_phase < T_hold:
                # Hold one-hot
                weights[class_idx] = 1.0
            else:
                # Transition between class_idx and next class
                next_idx = (class_idx + 1) % N
                alpha = (pos_in_phase - T_hold) / T_transition  # [0,1]
                weights[class_idx] = 1.0 - alpha
                weights[next_idx] = alpha

        if t >= total_steps:
            if random_anneal:
                weights = self.annealed_dirichlet_weights(self.final_weight_epoch, self.final_weight_epoch, base_alpha=self.start_weight_alpha, final_alpha=self.final_weight_alpha, num_samples=1)
            else:
                weights = np.ones(N)/N

        return weights
    
    def sample_reward_weights(self,num_samples=1, epoch=None):
        """sample weights for the reward function

        Args:
            n_weights (_type_): _description_

        Returns:
            _type_: _description_
        """
        N = len(self.optimise_parameters)
        if self.cycle_weights == "step":
            if epoch is not None:
                progress = np.min([epoch / self.final_weight_epoch, 1])
                if progress < 0.25:
                    weights = np.array([[1, 0]] * num_samples)
                elif progress < 0.5:
                    weights = np.array([[0, 1]] * num_samples)
                elif progress < 0.75:
                    weights = np.array([[1, 0]] * num_samples)
                elif progress < 1.0:
                    weights = np.array([[0, 1]] * num_samples)
                else:
                    weights = self.annealed_dirichlet_weights(epoch, 2*self.final_weight_epoch, base_alpha=self.start_weight_alpha, final_alpha=self.final_weight_alpha, num_samples=num_samples)
            else:
                weights = np.array([[0.5,0.5]] * num_samples)
        elif self.cycle_weights == "smooth":
            #T_hold = int(0.75*self.final_weight_epoch // (N*self.n_weight_cycles))
            #T_cycle = self.final_weight_epoch//(N*self.n_weight_cycles)
            T_hold = 0.75*self.final_weight_epoch/(N*self.n_weight_cycles)
            T_cycle = self.final_weight_epoch//(self.n_weight_cycles)
            weights = self.smooth_cycle_weights(epoch, N=2, T_cycle=T_cycle, T_hold=T_hold, total_steps=self.final_weight_epoch)
            weights = np.tile(weights, (num_samples, 1))
        else:
            if epoch is not None:
                weights = self.annealed_dirichlet_weights(epoch, self.final_weight_epoch, base_alpha=self.start_weight_alpha, final_alpha=self.final_weight_alpha, num_samples=10)
            else:
                weights = np.random.dirichlet(alpha=np.ones(N), size=num_samples)
        #weights2 = []
        #for key in self.optimise_parameters:
        #    weights2.append(np.random.uniform(self.optimise_weight_ranges[key][0],self.optimise_weight_ranges[key][1]))

        #print(weights, weights2)
        return weights[0]#/np.sum(weights)
    
    def sample_state_space(self, random_material=False):
        """return air with a thickness of 1

        Returns:
            _type_: _description_
        """
        if self.opt_init:
            layers = self.get_optimal_state()
        else:
            layers = np.zeros((self.max_layers, self.n_materials + 1))
            layers[:,self.air_material_index+1] = 1
            layers[:,0] = np.random.uniform(self.min_thickness, self.max_thickness, size=len(layers[:,0]))

        if random_material:
            for layer_ind in range(len(layers)):
                material = np.random.randint(1, self.n_materials)
                layers[layer_ind][material+1] = 1
                layers[layer_ind][self.air_material_index+1] = 0
        return layers

    def sample_action_space(self, ):
        """sample from the available state space

        Returns:
            _type_: _description_
        """

        new_layer_material = torch.nn.functional.one_hot(torch.from_numpy(np.array(np.random.randint(self.n_material_options))), num_classes=self.n_material_options)
        new_layer_thickness = torch.rand(1)*(self.max_thickness - self.min_thickness) +  self.min_thickness
        new_layer = torch.cat([new_layer_thickness, new_layer_material])

        return new_layer

    
        
    def sample_action_space2(self, ):
        """sample from the action space

        Returns:
            _type_: _description_
        """

        #action = self.numpy_onehot(np.random.randint(0,self.n_actions), num_classes=self.n_actions)
        material = torch.random.randint(0,self.n_materials)
        thickness = torch.random.uniform(self.min_thickness, self.max_thickness)
        return thickness, material
    
    def trim_state(self, state):
        """trim the state to remove duplicate air layers and inverse order

        Args:
            state (_type_): _description_

        Returns:
            _type_: _description_
        """
        # trim out the duplicate air layers and inverse order
        state_trim = self.get_air_only_state(n_layers = len(state))
        trimind = -1
        for ind, layer in enumerate(state):
            material_ind = np.argmax(layer[1:])
            if material_ind == self.air_material_index:
                trimind = ind
                break
            else:
                state_trim[ind] = layer
        
        state_trim = state_trim[:trimind]
        if len(state_trim) == 0:
            state_trim = np.array([[self.min_thickness, 0, 1, 0], ])

        return state_trim
    
    def compute_state_value_ligo(
            self, 
            state, 
            material_sub=1, 
            light_wavelength=1064E-9, 
            frequency=100, 
            wBeam=0.062, 
            Temp=293,
            return_separate = False):
        """_summary_

        Args:
            state (_type_): _description_
            material_sub (int, optional): Substrate material type 
            
            lambda_ (int, optional): laser wavelength (m)
            f (int, optional): frequency of interest (Hz)
            wBeam (int, optional): laser beam radius on optic(m)
            Temp (int, optional): detector temperature (deg)

        Returns:
            _type_: stuff
        """

        
        # trim out the duplicate air layers and inverse order
        state_trim = self.trim_state(state)
        # reverse state
        state_trim = state_trim[::-1]
    

        r, thermal_noise, e_integrated, total_thickness = merit_function(
            np.array(state_trim),
            self.materials,
            light_wavelength=light_wavelength,
            frequency=frequency,
            wBeam=wBeam,
            Temp=Temp,
            substrate_index = self.substrate_material_index,
            air_index = self.air_material_index,
            use_optical_thickness=self.use_optical_thickness
            )
        

        if return_separate:
            return r, thermal_noise, e_integrated, total_thickness
        else:
            return r, thermal_noise
    
    def compute_reflectivity(
            self, 
            state, 
            material_sub=1, 
            light_wavelength=1064e-9, 
            frequency=100, 
            wBeam=0.062, 
            Temp=293):
        
        n_list = [self.materials[1]["n"], ]
        d_list = [np.inf, ]
        for i,layer in enumerate(state):
            d_list.append(layer[0])
            matind = np.argmax(layer[1:])
            n_list.append(self.materials[matind]["n"])

        n_list.append(self.materials[0]["n"], )
        d_list.append(np.inf)

        theta = 0

        ref = coh_tmm('s', n_list, d_list, theta, light_wavelength)['R']
        #refs = coh_tmm('s', n_list, d_list, theta, light_wavelength)['R']

        return ref
    
    def compute_state_value(self, state, return_separate=False):
        if self.use_ligo_reward:
            reflected_power, scaled_thermal_noise, E_integrated, total_thickness = self.compute_state_value_ligo(state, return_separate=return_separate)
            return reflected_power, scaled_thermal_noise , E_integrated, total_thickness
        else:
            return self.compute_reflectivity(state), None, None, None

    def inv_sigmoid(self, val):
        return np.log(val/(1-val))

    def smooth_reward_function(self, vals, a=0.1, b=10):
        """
        Smooth reward function that mimics a piecewise linear+asymptotic behavior.

        Args:
            vals (np.ndarray): Array of values in range (0, 1).
            a (float): Steepness control for the transition at 0.5 (default 10).
            b (float): Steepness control for the asymptotic approach to 1 (default 10).

        Returns:
            np.ndarray: Computed rewards.
        """
        linear_term = (2 * vals) 
        asymptotic_term = a/np.abs(vals - 1)
        return linear_term + asymptotic_term
    
    def smooth_reflectivity_function(self, vals, a=0.1, b=10):
        """
        Smooth reward function that mimics a piecewise linear+asymptotic behavior.

        Args:
            vals (np.ndarray): Array of values in range (0, 1).
            a (float): Steepness control for the transition at 0.5 (default 10).
            b (float): Steepness control for the asymptotic approach to 1 (default 10).

        Returns:
            np.ndarray: Computed rewards.
        """
        linear_term = (2 * vals) 
        asymptotic_term = np.log(a/np.abs(vals - 1)+1)
        both_terms = linear_term + asymptotic_term
        if self.optimise_targets is not None:
            if vals > self.optimise_targets["reflectivity"]:
                both_terms = 5 + np.log(a/np.abs(self.optimise_targets["reflectivity"] - 1) + 1) + vals

        
            
        return both_terms
    
    def smooth_thermal_reward(self, thermal_noise, a=0.1, b=10):
        """
        Smooth reward function that mimics a piecewise linear+asymptotic behavior.

        Args:
            vals (np.ndarray): Array of values in range (0, 1).
            a (float): Steepness control for the transition at 0.5 (default 10).
            b (float): Steepness control for the asymptotic approach to 1 (default 10).

        Returns:
            np.ndarray: Computed rewards.
        """

        scaled_thermal_noise = thermal_noise/5.9e-21
        return -scaled_thermal_noise
    
    def sigmoid(self, x, mean=0.5, a=0.01):
        return 1/(1+np.exp(-a*(x-mean)))
    
    def select_reward(self, new_reflectivity, new_thermal_noise, new_total_thickness, new_E_integrated, weights=None):
        if self.reward_func == "default":
            total_reward, vals, rewards = reward_function(
                new_reflectivity, 
                new_thermal_noise, 
                new_total_thickness, 
                new_E_integrated, 
                self.optimise_parameters, 
                self.optimise_targets, 
                combine=self.combine, 
                weights=weights)
        elif self.reward_func == "target":
            total_reward, vals, rewards = reward_function_target(
                new_reflectivity, 
                new_thermal_noise, 
                new_total_thickness, 
                new_E_integrated, 
                self.optimise_parameters, 
                self.optimise_targets, 
                combine=self.combine, 
                weights=weights)
        elif self.reward_func == "raw":
            total_reward, vals, rewards = reward_function_raw(
                new_reflectivity, 
                new_thermal_noise, 
                new_total_thickness, 
                new_E_integrated, 
                self.optimise_parameters, 
                self.optimise_targets, 
                combine=self.combine, 
                weights=weights)
        elif self.reward_func == "log_targets":
            total_reward, vals, rewards = reward_function_log_minimise(
                new_reflectivity, 
                new_thermal_noise, 
                new_total_thickness, 
                new_E_integrated, 
                self.optimise_parameters, 
                self.optimise_targets, 
                combine=self.combine, 
                weights=weights)
        else:
            raise Exception(f"Unknown reward function type {self.reward_func}")
        
        return total_reward, vals, rewards
        
    def compute_reward(self, new_state, max_value=0.0, target_reflectivity=1.0, objective_weights=None):
        """reward is the improvement of the state over the previous one

        Args:
            state (_type_): _description_
            action (_type_): _description_
        """

        new_reflectivity, new_thermal_noise, new_E_integrated, new_total_thickness = self.compute_state_value(new_state, return_separate=True)
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

        return total_reward, vals, rewards
    
    def include_random_rare_state(self, new_state):
        rstate = 2
        reward = 0
        correct_states = []
        for i in range(len(new_state)):
            if i == 1 or i == len(new_state)-2:
                continue
            if new_state[i][rstate] == 1:
                correct_states.append(True)
            else:
                correct_states.append(False)
            if rstate == 2:
                rstate = 3
            else:
                rstate = 2

        correct_state = np.all(correct_states)
        if new_state[1][4] == 1 and new_state[-2][4] == 1 and correct_state:
            mean1 = self.min_thickness + (self.max_thickness - self.min_thickness)/10
            mean2 = self.min_thickness + (self.max_thickness - self.min_thickness)/2
            var1 = (self.max_thickness - self.min_thickness)/10
            reward += 100*scipy.stats.norm(mean1,var1).pdf(new_state[1][0]) * np.sqrt(2*np.pi*var1**2)
            reward += 100*scipy.stats.norm(mean2,var1).pdf(new_state[-2][0]) * np.sqrt(2*np.pi*var1**2)

        return reward
    
    def update_state(self, current_state, thickness, material):
        """new state is the current action choice

        Args:
            action (_type_): _description_

        Returns:
            _type_: _description_
        """
        material = torch.nn.functional.one_hot(torch.from_numpy(np.array([material]).astype(int)), num_classes=self.n_material_options)[0]
        thickness = torch.from_numpy(np.array([thickness]))
        new_layer = torch.cat([thickness, material])
        current_state[self.current_index][0] = thickness
        #if self.ignore_air_option:
        #    current_state[self.current_index][2:] = material
        #else:
        current_state[self.current_index][1:] = material

        #if material[0] == 1 and self.current_index != self.max_layers - 1:
        #    current_state[self.current_index:] = new_layer.repeat((self.max_layers-self.current_index, 1))

        return current_state, new_layer
    
    def get_observation_from_state(self, state):

        observation = []
        for st in state:
            mind = np.argmax(st[1:])
            n = self.materials[mind]["n"]
            k = self.materials[mind]["k"]
            observation.append([st[0], n, k])

        return np.array(observation)

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

    
        self.previous_material = material
        #print(new_value)

        self.length += 1
        self.current_index += 1




        return new_state, rewards, terminated, finished, reward, full_action, vals

    def plot_stack(self, data):
        #data = self.current_state

        # Extract the layers and their properties
        L = data.shape[0]
        thickness = data[:, 0]
        colors = []
        nmats = data.shape[1] - 1

        # Define colors for m1, m2, and m3
        color_map = {
            0: 'gray',    # air
            1: 'blue',    # m1 - substrate
            2: 'green',   # m2
            3: 'red',      # m3
            4: 'black',
            5: 'yellow',
            6: 'orange',
            7: 'purple',
            8: 'cyan',
        }

        labels = []
        for row in data:
            row = np.argmax(row[1:])
            if row == 0:
                colors.append(color_map[0])  # m1
                labels.append(f"{self.materials[0]['name']}")
            else:
                colors.append(color_map[row])  # m2
                labels.append(f"{self.materials[row]['name']} (1/4 wave{1064e-9 /(4*self.materials[row]['n'])})")



        # Create a bar plot
        fig, ax = plt.subplots(figsize=(10, 6))
        #bars = ax.bar(range(L), thickness, color=colors)
        bars = [ax.bar(x, thickness[x], color=colors[x], label=labels[x]) for x in range(L)]


        # Add labels and title
        ax.set_xlabel('Layer Index')
        ax.set_ylabel('Thickness')
        reward, vals, rewards = self.compute_reward(data)
        ax.set_title(f'TR: {reward}, R: {vals["reflectivity"]:.8f}, T: {vals["thermal_noise"]:.8e}, A: {vals["absorption"]:.8e}')
        ax.set_xticks(range(L), [f'Layer {i + 1}' for i in range(L)])  # X-axis labels

        #  Show thickness values on top of bars
        for x, bar in enumerate(bars):
            yval = thickness[x]
            ax.text(bar[0].get_x() + bar[0].get_width() / 2, yval, f'{yval:.1f}', ha='center', va='bottom')


        ax.set_ylim(0, np.max(thickness)*(1.1) )  # Set Y-axis limit
        ax.grid(axis='y', linestyle='--')

        unique_labels = dict(zip(labels, colors))
        handles = [plt.Line2D([0], [0], marker='o', color='w', label=label, markersize=10, markerfacecolor=color)
               for label, color in unique_labels.items()]
        ax.legend(handles=handles, title="Materials")

        return fig, ax  

if __name__ == "__main__":
    
    max_layers = 5
    min_thickness = 0.1
    max_thickness = 1
    materials = {
        1: {
            'name': 'SiO2',
            'n': 1.44,
            'a': 0,
            'alpha': 0.51e-6,
            'beta': 8e-6,
            'kappa': 1.38,
            'C': 1.64e6,
            'Y': 72e9,
            'prat': 0.17,
            'phiM': 4.6e-5
        },
        2: {
            'name': 'ta2o5',
            'n': 2.07,
            'a': 2,
            'alpha': 3.6e-6,
            'beta': 14e-6,
            'kappa': 33,
            'C': 2.1e6,
            'Y': 140e9,
            'prat': 0.23,
            'phiM': 2.44e-4
        },
    }
    
    cs = CoatingStack(max_layers, min_thickness, max_thickness, materials, thickness_options=[0.1,1,10], variable_layers=False)

    state = cs.sample_state_space()

    val = cs.compute_state_value(state)

    ### TEST vaild arangements

    def generate_valid_arrangements(materials, num_layers):
        def backtrack(arrangement):
            if len(arrangement) == num_layers:
                valid_arrangements.append(arrangement)
                return
            for material in materials:
                if not arrangement or arrangement[-1] != material:
                    backtrack(arrangement + [material])

        valid_arrangements = []
        backtrack([])
        return valid_arrangements

    # Define materials and number of layers
    materials = ['0','1','2']  # Replace '...' with the rest of your materials
    num_layers = 20

    # Generate valid arrangements
    st = time.time()
    valid_arrangements = generate_valid_arrangements(materials, num_layers)
    print("time: ", time.time() - st)
    print(np.shape(valid_arrangements))
    # Print a sample of valid arrangements (to avoid printing too many)
    for arrangement in valid_arrangements[:10]:
        print(arrangement)

