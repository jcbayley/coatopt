import torch
import numpy as np
from .coating_utils import getCoatAbsorption, getCoatNoise2, getCoatRefl2, merit_function_2
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
            reward_shape="none",
            thermal_reward_shape="scaled_thermal_noise",
            use_intermediate_reward=False,
            ignore_air_option=False,
            use_ligo_reward=False,
            use_ligo_thermal_noise=False,
            light_wavelength=1064e-9,
            include_random_rare_state=False,
            use_design_requirements=False):
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
        self.opt_init = opt_init
        self.reward_shape = reward_shape
        self.thermal_reward_shape = thermal_reward_shape
        self.use_intermediate_reward = use_intermediate_reward
        self.ignore_air_option = ignore_air_option
        self.use_ligo_reward = use_ligo_reward
        self.use_ligo_thermal_noise = use_ligo_thermal_noise
        self.include_random_rare_state = include_random_rare_state
        self.use_design_requirements = use_design_requirements

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

        self.design_requirements = {
            "reflectivity": 0.99999,
            "thermal_noise": 5.9e-21
        }
    

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

    def get_optimal_state(self, reverse=False):

        layers = []
        thickness1 = 1064e-9 /(4*self.materials[1]["n"])
        thickness2 = 1064e-9 /(4*self.materials[2]["n"])
        #print("thickness", thickness1, thickness2)
        
        opt_state2 = []
        if reverse:
            material = 2
        else:
            material = 1
        for i in range(self.max_layers):
            current_material = 2 if material == 1 else 1
            if current_material == 1:
                thickness = thickness1
            elif current_material == 2:
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

    def sample_state_space(self, ):
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

        
        # trim out the duplicate air layers
        state_trim = []
        for layer in state[::-1]:
            material_ind = np.argmax(layer[1:])
            if material_ind == 0:
                continue
            else:
                state_trim.append(layer)
        
        if len(state_trim) == 0:
            state_trim = np.array([[self.min_thickness, 0, 1, 0], ])


        r, thermal_noise = merit_function_2(
            np.array(state_trim),
            self.materials,
            light_wavelength=light_wavelength,
            frequency=frequency,
            wBeam=wBeam,
            Temp=Temp,
            substrate_index = self.substrate_material_index,
            air_index = self.air_material_index
            )
        
        R = np.abs(r)**2

        #scaled_thermal_noise = -np.log(thermal_noise)/10
     
        
        #print(R, np.log(thermal_noise))
        if return_separate:
            return r, thermal_noise
        else:
            return R, thermal_noise
    
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
    
    def compute_state_value(self, state):
        if self.use_ligo_reward:
            reflected_power, scaled_thermal_noise = self.compute_state_value_ligo(state)
            return reflected_power, scaled_thermal_noise 
        else:
            return self.compute_reflectivity(state), 0

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
        return linear_term + asymptotic_term
    
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
    
    def compute_reward(self, new_state, max_value=0.0, target_reflectivity=1.0):
        """reward is the improvement of the state over the previous one

        Args:
            state (_type_): _description_
            action (_type_): _description_
        """

        new_reflectivity, new_thermal_noise = self.compute_state_value(new_state)
       
       # Reflectivity reward shaping
        if self.reward_shape == "inv_sigmoid_cut":
            if new_reflectivity > 0.5:
                reflectivity_reward = self.inv_sigmoid(new_reflectivity) + 0.5
            else:
                reflectivity_reward = new_reflectivity
        elif self.reward_shape == "inv_diff":
            reflectivity_reward = 0.01/np.abs(new_reflectivity - target_reflectivity)
        elif self.reward_shape == "smooth_asymptote":
            #reward = self.smooth_reward_function(new_reflectivity, a=0.01)
            reflectivity_reward = self.smooth_reflectivity_function(new_reflectivity, a=0.01)
        elif self.reward_shape == "none":
            reflectivity_reward = new_reflectivity
        else:
            raise Exception(f"reward shape not supported {self.reward_shape}")
        #reward_diff = new_value - max_value

        # thermal noise reward shaping
        if self.use_ligo_thermal_noise and new_thermal_noise is not None:
            if self.thermal_reward_shape == "scaled_thermal_noise":
                thermal_reward = self.smooth_thermal_reward(new_thermal_noise)
            elif self.thermal_reward_shape == "log_thermal_noise":
                thermal_reward = -np.log(new_thermal_noise)
            else:
                raise Exception(f"thermal noise reward shape not supported {self.thermal_reward_shape}")


        reward = reflectivity_reward + thermal_reward

        # design requirements shaping
        if self.use_design_requirements:
            if new_thermal_noise < self.design_requirements["thermal_noise"]:
                reward += 10
            
            if new_reflectivity > self.design_requirements["reflectivity"]:
                reward += 10

        # rare state shaping (not to be used outside of testing)
        if self.include_random_rare_state:
            rstate = 2
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

        return reward, new_reflectivity, new_thermal_noise, reflectivity_reward, thermal_reward
    
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
            observation.append([st[0], n])

        return np.array(observation)

    def step(self, action, max_state=0, verbose=False):
        """action[0] - thickness
           action[1:N] - material probability

        Args:
            action (_type_): _description_
        """
        
        material = action[0]
        thickness = action[1] #* self.light_wavelength /(4*self.materials[material]["n"])

        new_state, full_action = self.update_state(np.copy(self.current_state), thickness, material)

        reward = 0
        neg_reward = -1.0
        new_value = 0

        terminated = False
        finished = False
        reward, new_reflectivity, new_thermal_noise, reflectivity_reward, thermal_reward = self.compute_reward(new_state, max_state)

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
                reward = reward
            else:
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

        rewards = {
            "total_reward": reward,
            "reflectivity_reward": reflectivity_reward,
            "thermal_reward": thermal_reward
        }


        return new_state, rewards, terminated, finished, new_value, full_action

    def plot_stack(self, data):
        #data = self.current_state

        # Extract the layers and their properties
        L = data.shape[0]
        thickness = data[:, 0]
        colors = []

        # Define colors for m1, m2, and m3
        color_map = {
            0: 'gray',    # No active material
            1: 'blue',    # m1
            2: 'green',   # m2
            3: 'red'      # m3
        }

        labels = []
        for row in data:
            if row[1] == 1:
                colors.append(color_map[0])  # m1
                labels.append(f"{self.materials[0]['name']}")
            elif row[2] == 1:
                colors.append(color_map[1])  # m2
                labels.append(f"{self.materials[1]['name']} (1/4 wave{1064e-9 /(4*self.materials[1]['n'])})")
            elif row[3] == 1:
                colors.append(color_map[2])  # m3
                labels.append(f"{self.materials[2]['name']} (1/4 wave{1064e-9 /(4*self.materials[2]['n'])})")
            elif row[4] == 1:
                colors.append(color_map[3])  # No active material
                labels.append(f"{self.materials[3]['name']} (1/4 wave{1064e-9 /(4*self.materials[3]['n'])})")
            else:
                pass


        # Create a bar plot
        fig, ax = plt.subplots(figsize=(10, 6))
        #bars = ax.bar(range(L), thickness, color=colors)
        bars = [ax.bar(x, thickness[x], color=colors[x], label=labels[x]) for x in range(L)]


        # Add labels and title
        ax.set_xlabel('Layer Index')
        ax.set_ylabel('Thickness')
        ax.set_title('Layer Thickness Visualization')
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

