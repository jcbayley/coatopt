import torch
import numpy as np
import time
from tmm import coh_tmm
import matplotlib.pyplot as plt
from coatopt.environments.thermal_noise_environment import CoatingStack

class DiscreteCoatingStack(CoatingStack):

    def __init__(
            self, 
            max_layers, 
            min_thickness, 
            max_thickness, 
            materials, 
            air_material_index=0,
            substrate_material_index=1,
            thickness_options=None,
            variable_layers=False,
            opt_init=False,
            use_intermediate_reward=False,
            use_inv_sigmoid=False,
            ignore_air_option=False,
            use_ligo_reward=False,
            use_ligo_thermal_noise=False):
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
        super().__init__(
            max_layers, 
            min_thickness, 
            max_thickness, 
            materials, 
            air_material_index=air_material_index,
            substrate_material_index=substrate_material_index,
            variable_layers=variable_layers,
            opt_init=opt_init,
            use_inv_sigmoid=use_inv_sigmoid,
            use_intermediate_reward=use_intermediate_reward,
            ignore_air_option=ignore_air_option,
            use_ligo_reward=use_ligo_reward,
            use_ligo_thermal_noise=use_ligo_thermal_noise)
        #self.variable_layers = variable_layers
        self.thickness_options=thickness_options
        #self.max_layers = max_layers
        #self.min_thickness = min_thickness
        #self.max_thickness = max_thickness
        #self.materials = materials
        #self.n_materials = len(materials) - 1 if ignore_air else len(materials)
        #self.air_material_index = air_material_index
        #self.substrate_material_index = substrate_material_index
        #self.opt_init = opt_init

        """
        # state space size is index for each material (onehot encoded) plus thickness of each material
        self.state_space_size = self.max_layers*self.n_materials + self.max_layers
        self.obs_space_size = self.max_layers*2
        self.obs_space_shape = (self.max_layers, 2)

        self.length = 0
        self.current_state = self.sample_state_space()
        self.current_index = self.max_layers - 1
        self.previous_material = -1
        """

    def sample_action_space(self, ):
        """sample from the available state space

        Returns:
            _type_: _description_
        """

        mat_ind = np.random.randint(self.n_material_options)
        new_layer_material = torch.nn.functional.one_hot(torch.from_numpy(np.array(mat_ind)), num_classes=self.n_material_options)
        if self.thickness_options is not None:
            new_layer_thickness = 1064e-9 /(4*self.materials[1]["n"])*np.random.choice(self.thickness_options)
        else:
            new_layer_thickness = 1064e-9 /(4*self.materials[1]["n"])

        new_layer = torch.cat([new_layer_thickness, new_layer_material])

        return new_layer


    def compute_reward(self, new_state, max_value=0.0, target_reflectivity=1.0):
        """reward is the improvement of the state over the previous one

        Args:
            state (_type_): _description_
            action (_type_): _description_
        """

        new_value = self.compute_state_value(new_state)
        #new_value = np.log(new_value/(1-new_value))
        #old_value = self.compute_state_value(old_state) + 5
        reward_diff = new_value#0.01/(new_value - target_reflectivity)**2
        #reward_diff = new_value
        #reward_diff = np.log(reward_diff/(1-reward_diff))
        #reward_diff = new_value
        #reward_diff = new_value - max_value

        if reward_diff > 0:
            #print(reward_diff)
            reward = reward_diff
        else:
            reward = reward_diff
 
        return reward_diff, reward, new_value 
    
    def step(self, action, max_state=0, verbose=False):
        """action[0] - thickness
           action[1:N] - material probability

        Args:
            action (_type_): _description_
        """
        
        material = action[0] 

        if self.thickness_options is None:
            thickness = 1064e-9 /(4*self.materials[material]["n"])
        else:
            thickness = 1064e-9 /(4*self.materials[material]["n"])*action[1]

        new_state, full_action = self.update_state(np.copy(self.current_state), thickness, material)

        reward = 0
        neg_reward = -1
        new_value = 0

        terminated = False
        finished = False
        reward_diff, reward, new_value = self.compute_reward(new_state, max_state)

        if self.min_thickness > thickness or thickness > self.max_thickness or not np.isfinite(thickness):
            terminated=True
            reward = neg_reward
            self.current_state = new_state
            new_value = neg_reward
            #print("out of thickness bounds")
        elif material == self.air_material_index and self.ignore_air_option == False:
            terminated=True
            reward = neg_reward
            self.current_state = new_state
        elif self.current_index == self.max_layers-1:
         #print("out of thickness bounds")
            finished = True
            self.current_state = new_state
            #print("finished")
            #reward_diff, reward, new_value = self.compute_reward(new_state, max_state)
        elif material == self.previous_material:
        #    terminated = True
            #reward = neg_reward
            self.current_state = new_state
            reward = neg_reward
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

        for row in data:
            if row[1] == 1:
                colors.append(color_map[0])  # m1
            elif row[2] == 1:
                colors.append(color_map[1])  # m2
            elif row[3] == 1:
                colors.append(color_map[2])  # m3
            elif row[4] == 1:
                colors.append(color_map[3])  # No active material
            else:
                pass

        # Create a bar plot
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(range(L), thickness, color=colors)

        # Add labels and title
        ax.set_xlabel('Layer Index')
        ax.set_ylabel('Thickness')
        ax.set_title('Layer Thickness Visualization')
        ax.set_xticks(range(L), [f'Layer {i + 1}' for i in range(L)])  # X-axis labels

        # Show thickness values on top of bars
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.1f}', ha='center', va='bottom')

        ax.set_ylim(0, np.max(thickness)*(1.1) )  # Set Y-axis limit
        ax.grid(axis='y', linestyle='--')

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

