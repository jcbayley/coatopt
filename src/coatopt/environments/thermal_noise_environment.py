import torch
import numpy as np
from .coating_utils import getCoatAbsorption, getCoatNoise2, getCoatRefl2, merit_function_2
import time
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
        self.variable_layers = variable_layers
        self.max_layers = max_layers
        self.min_thickness = min_thickness
        self.max_thickness = max_thickness
        self.materials = materials
        self.n_materials = len(materials)
        self.air_material_index = air_material_index
        self.substrate_material_index = substrate_material_index
        self.opt_init = opt_init

        # state space size is index for each material (onehot encoded) plus thickness of each material
        self.state_space_size = self.max_layers*self.n_materials + self.max_layers
        self.obs_space_size = self.max_layers*2
        self.obs_space_shape = (self.max_layers, 2)

        self.length = 0
        self.current_state = self.sample_state_space()
        self.current_index = self.max_layers - 1
        self.previous_material = -1
    

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

    def get_optimal_state(self, ):

        layers = []
        thickness1 = 1064e-9 /(4*self.materials[1]["n"])
        thickness2 = 1064e-9 /(4*self.materials[2]["n"])
        #print("thickness", thickness1, thickness2)
        
        opt_state2 = []
        material = 1
        for i in range(self.max_layers):
            if material == 1:
                thickness = thickness1
            elif material == 2:
                thickness = thickness2
            l_state = [0,]*(self.n_materials+1)
            l_state[0] = thickness
            l_state[material+1] = 1

            opt_state2.append(l_state)
            if material == 1: material = 2
            elif material == 2: material = 1

        return np.array(opt_state2)

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

        new_layer_material = torch.nn.functional.one_hot(torch.from_numpy(np.array(np.random.randint(self.n_materials))), num_classes=self.n_materials)
        new_layer_thickness = torch.random.uniform(self.min_thickness, self.max_thickness)
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
            Temp=293):
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


        R, thermal_noise = merit_function_2(
            np.array(state_trim),
            self.materials,
            light_wavelength=light_wavelength,
            frequency=frequency,
            wBeam=wBeam,
            Temp=Temp,
            substrate_index = self.substrate_material_index,
            air_index = self.air_material_index
            )
        
     
        rtn = R #+ np.log(thermal_noise)/50
        #print(R, np.log(thermal_noise))
        return rtn
    
    def compute_state_value(
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

        ref = coh_tmm('p', n_list, d_list, theta, light_wavelength)['R']
        #refs = coh_tmm('s', n_list, d_list, theta, light_wavelength)['R']

        return ref

    def inv_sigmoid(self, val):
        return np.log(val/(1-val))

    def compute_reward(self, new_state, max_value=0.0, target_reflectivity=1.0):
        """reward is the improvement of the state over the previous one

        Args:
            state (_type_): _description_
            action (_type_): _description_
        """

        new_value = self.compute_state_value(new_state)
        #new_value = np.log(new_value/(1-new_value))
        #old_value = self.compute_state_value(old_state) + 5
        #reward_diff = 0.01/(new_value - target_reflectivity)**2
        #reward_diff = (new_value - target_reflectivity)**2
        #reward_diff = np.log(reward_diff/(1-reward_diff))
        reward_diff = self.inv_sigmoid(new_value)
        #reward_diff = new_value - max_value

        if reward_diff > 0:
            #print(reward_diff)
            reward = reward_diff
        else:
            reward = reward_diff
 
        return reward_diff, reward, new_value 
    
    def update_state(self, current_state, thickness, material):
        """new state is the current action choice

        Args:
            action (_type_): _description_

        Returns:
            _type_: _description_
        """
        material = torch.nn.functional.one_hot(torch.from_numpy(np.array([material]).astype(int)), num_classes=self.n_materials)[0]
        thickness = torch.from_numpy(np.array([thickness]))
        new_layer = torch.cat([thickness, material])
        current_state[self.current_index] = new_layer

        if material[0] == 1 and self.current_index != self.max_layers - 1:
            current_state[self.current_index:] = new_layer.repeat((self.max_layers-self.current_index, 1))

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
        
        thickness = action[1]
        material = action[0]

        new_state, full_action = self.update_state(np.copy(self.current_state), thickness, material)

        reward = 0
        neg_reward = -1.0
        new_value = 0

        terminated = False
        finished = False
        reward_diff, reward, new_value = self.compute_reward(new_state, max_state)

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


        return new_state, reward, terminated, finished, new_value, full_action

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

