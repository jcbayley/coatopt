import torch
import numpy as np
from typing import Tuple, Dict, Any, Optional, List, TYPE_CHECKING
from ..utils import coating_utils 
from ..utils import state_utils
from ..reward_functions.reward_system import RewardCalculator
import time
import scipy
from tmm import coh_tmm
import matplotlib.pyplot as plt

from ...config.structured_config import CoatingOptimisationConfig


class BaseCoatingEnvironment:

    def __init__(self, config: Optional[CoatingOptimisationConfig] = None, **kwargs):
        """
        Initialize with either new config system or individual parameters for backward compatibility.
        
        Args:
            config: CoatingOptimisationConfig object (new approach)
            **kwargs: Individual parameters (legacy approach)
                     Note: materials should be passed in kwargs even when using config
        """
        if config is not None:
            # New config-based initialization
            # Extract materials from kwargs before passing to config init
            self.materials = kwargs.pop('materials', [])
            self.air_material_index = kwargs.pop('air_material_index', 0)
            self.substrate_material_index = kwargs.pop('substrate_material_index', 1)
            self.light_wavelength = kwargs.pop('light_wavelength', 1064e-9)
            self.variable_layers = kwargs.pop('variable_layers', False)
            self.opt_init = kwargs.pop('opt_init', False)
            
            self._init_from_config(config)
            
            # Apply any remaining kwargs as overrides
            for key, value in kwargs.items():
                if hasattr(self, key):
                    setattr(self, key, value)
        else:
            # Legacy initialization for backward compatibility
            self._init_from_legacy_params(**kwargs)
        
        # Common initialization regardless of how parameters were provided
        self._setup_common_attributes()
        
        # Expert constraints for adaptive constraint specialization
        self.current_expert_constraints = None

    def _init_from_config(self, config: CoatingOptimisationConfig):
        """Initialize from structured configuration object."""
        # Basic validation
        if config.data is None:
            raise ValueError("DataConfig is required")
        
        # Physics parameters from DataConfig
        self.max_layers = config.data.n_layers
        self.min_thickness = config.data.min_thickness
        self.max_thickness = config.data.max_thickness
        self.use_optical_thickness = config.data.use_optical_thickness
        
        # Materials should already be set in __init__ from kwargs
        # Validate materials were provided
        if not hasattr(self, 'materials') or not self.materials:
            raise ValueError("Materials must be provided when using structured config")
        
        # Optimization parameters from DataConfig
        self.optimise_parameters = config.data.optimise_parameters
        
        # Extract clean parameter names (removing direction suffixes like ":max")
        self.optimise_parameter_names = self._extract_parameter_names(self.optimise_parameters)
        
        self.optimise_targets = config.data.optimise_targets
        self.optimise_weight_ranges = config.data.optimise_weight_ranges
        self.design_criteria = config.data.design_criteria
        
        # Objective bounds for reward normalization (if provided in config)
        if hasattr(config.data, 'objective_bounds') and config.data.objective_bounds:
            self.objective_bounds = config.data.objective_bounds
        
        # Reward parameters from DataConfig
        self.reward_function = config.data.reward_function
        self.use_intermediate_reward = config.data.use_intermediate_reward
        self.combine = config.data.combine
        self.reflectivity_reward_shape = config.data.reflectivity_reward_shape
        self.thermal_reward_shape = config.data.thermal_reward_shape
        self.absorption_reward_shape = config.data.absorption_reward_shape
        self.use_ligo_reward = config.data.use_ligo_reward
        
        # Training parameters from DataConfig
        self.ignore_air_option = config.data.ignore_air_option
        self.ignore_substrate_option = config.data.ignore_substrate_option
        self.include_random_rare_state = config.data.include_random_rare_state
        
        # Training parameters from TrainingConfig (if available)
        if config.training is not None:
            self.final_weight_epoch = config.training.final_weight_epoch
            self.start_weight_alpha = config.training.start_weight_alpha
            self.final_weight_alpha = config.training.final_weight_alpha
            self.cycle_weights = config.training.cycle_weights
            self.n_weight_cycles = config.training.n_weight_cycles
        else:
            # Set defaults if TrainingConfig not available
            self.final_weight_epoch = 1
            self.start_weight_alpha = 1.0
            self.final_weight_alpha = 1.0
            self.cycle_weights = False
            self.n_weight_cycles = 2

    def _extract_parameter_names(self, param_list):
        """
        Extract clean parameter names from potentially suffixed parameter list.
        
        Args:
            param_list: List that may contain "parameter:direction" format
            
        Returns:
            list: Clean parameter names without direction suffixes
        """
        clean_names = []
        for param in param_list:
            if isinstance(param, str) and ':' in param:
                # New format: "parameter:direction" -> extract just parameter
                param_name, _ = param.split(':', 1)
                clean_names.append(param_name.strip())
            else:
                # Legacy format: just parameter name
                param_name = param if isinstance(param, str) else str(param)
                clean_names.append(param_name)
        return clean_names
        
    def get_parameter_names(self):
        """Get clean parameter names for compatibility."""
        if hasattr(self, 'optimise_parameter_names'):
            return self.optimise_parameter_names
        else:
            return self._extract_parameter_names(getattr(self, 'optimise_parameters', []))

    def _init_from_legacy_params(self, 
                                 max_layers=20, 
                                 min_thickness=1e-9, 
                                 max_thickness=1e-6, 
                                 materials=[], 
                                 air_material_index=0, 
                                 substrate_material_index=1,
                                 variable_layers=False, 
                                 opt_init=False,  
                                 reward_function=None,
                                 use_intermediate_reward=False, 
                                 ignore_air_option=False,
                                 ignore_substrate_option=False, 
                                 optimise_parameters=["reflectivity", "thermal_noise", "absorption","thickness"],
                                 optimise_targets={"reflectivity":0.99999, "thermal_noise":5.394480540642821e-21, 
                                                   "absorption":0.01, "thickness":0.1},
                                 optimise_weight_ranges={"reflectivity":[0,1], "thermal_noise":[0,1], 
                                                         "absorption":[0,1], "thickness":[0,1]},
                                 design_criteria={"reflectivity":0.99999, "thermal_noise":5.394480540642821e-21, 
                                                  "absorption":0.01},
                                 light_wavelength=1064e-9, 
                                 use_optical_thickness=True, 
                                 final_weight_epoch=1, 
                                 start_weight_alpha=1.0, 
                                 final_weight_alpha=1.0, 
                                 cycle_weights=False, 
                                 n_weight_cycles=2, 
                                 combine="product",
                                 objective_bounds=None):
        """Legacy parameter initialization for backward compatibility."""
        
        self.max_layers = max_layers
        self.min_thickness = min_thickness
        self.max_thickness = max_thickness
        self.materials = materials
        self.air_material_index = air_material_index
        self.substrate_material_index = substrate_material_index
        self.variable_layers = variable_layers
        self.opt_init = opt_init
        self.reward_function = reward_function
        self.use_intermediate_reward = use_intermediate_reward
        self.ignore_air_option = ignore_air_option
        self.ignore_substrate_option = ignore_substrate_option
        self.optimise_parameters = optimise_parameters
        self.optimise_targets = optimise_targets
        self.optimise_weight_ranges = optimise_weight_ranges
        self.design_criteria = design_criteria
        self.light_wavelength = light_wavelength
        self.use_optical_thickness = use_optical_thickness
        self.final_weight_epoch = final_weight_epoch
        self.start_weight_alpha = start_weight_alpha
        self.final_weight_alpha = final_weight_alpha
        self.cycle_weights = cycle_weights
        self.n_weight_cycles = n_weight_cycles
        self.combine = combine
        
        # Objective bounds for reward normalization (if provided)
        if objective_bounds is not None:
            self.objective_bounds = objective_bounds

    def _setup_common_attributes(self):
        """Setup attributes common to both initialization methods."""
        self.n_materials = len(self.materials)
        self.n_material_options = self.n_materials
        
        # State and observation space setup
        self.state_space_size = self.max_layers * self.n_materials + self.max_layers
        self.state_space_shape = (self.max_layers, self.n_materials + 1)
        self.obs_space_size = self.max_layers * (2 + 1) # thickness, n, k for each layer
        self.obs_space_shape = (self.max_layers, (2 + 1))  # thickness, n, k for each layer
        
        # Environment state
        self.length = 0
        self.current_state = self.sample_state_space()
        self.current_index = 0
        self.previous_material = self.substrate_material_index
        
        # Initialize simple reward calculator
        reward_type = "default" if self.reward_function is None else str(self.reward_function)
        self.reward_calculator = RewardCalculator(
            reward_type=reward_type,
            optimise_parameters=self.get_parameter_names(),  # Use clean parameter names
            optimise_targets=self.optimise_targets,
            combine=self.combine, env=self
        )


    def reset(self):
        """Reset the state space and length."""
        self.length = 0
        self.current_state = self.sample_state_space()
        self.current_index = 0
        return self.current_state
    
    def print_state(self,):
        for i in range(len(self.current_state)):
            print(self.current_state[i])

    def check_numpy_cpu(self, x):
        if hasattr(x, 'numpy'):
            x = x.numpy()
        elif hasattr(x, 'detach'):
            x = x.detach().numpy()
        return x

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

    
    def compute_state_value(
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
        state_trim = state_utils.trim_state(state)
        # reverse state
        state_trim = state_trim[::-1]

        r, thermal_noise, e_integrated, total_thickness = coating_utils.merit_function(
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
        
    def compute_reward(self, new_state, max_value=0.0, target_reflectivity=1.0, objective_weights=None):
        """reward is the improvement of the state over the previous one

        Args:
            state (_type_): _description_
            action (_type_): _description_
        """

        new_reflectivity, new_thermal_noise, new_E_integrated, new_total_thickness = self.compute_state_value(new_state, return_separate=True)
        
        if objective_weights is not None:
            weights = {
                key:objective_weights[i] for i,key in enumerate(self.get_parameter_names())
            }
        else:
            weights=None

        total_reward, vals, rewards = self.reward_calculator.calculate(
                reflectivity=new_reflectivity,
                thermal_noise=new_thermal_noise,
                thickness=new_total_thickness,
                absorption=new_E_integrated,
                weights=weights,
                expert_constraints=self.current_expert_constraints,
                env=self
            )
        return total_reward, vals, rewards
    
    def set_expert_constraints(self, constraints: Dict[str, float]):
        """
        Set expert constraints for adaptive constraint specialization.
        
        Args:
            constraints: Dict mapping parameter names to target reward values
        """
        self.current_expert_constraints = constraints
    
    def clear_expert_constraints(self):
        """Clear expert constraints."""
        self.current_expert_constraints = None
    
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
        

        if self.min_thickness > thickness or thickness > self.max_thickness or not np.isfinite(thickness):
            self.current_state = new_state
            print("out of thickness bounds")
        elif self.current_index == self.max_layers-1 or material == self.air_material_index:
            finished = True
            self.current_state = new_state
            reward, vals, rewards = self.compute_reward(new_state, max_state, objective_weights=objective_weights)

        else:
            self.current_state = new_state
            if self.use_intermediate_reward:
                reward, vals, rewards = self.compute_reward(new_state, max_state, objective_weights=objective_weights)
        

        if np.any(np.isinf(new_state)) or np.any(np.isnan(new_state)) or np.isnan(reward) or np.isinf(reward):
            reward = neg_reward
            terminated = True
    
        self.previous_material = material
        #print(new_value)

        self.length += 1
        self.current_index += 1

        return new_state, rewards, terminated, finished, reward, full_action, vals

    # Convenience methods for working with the simple reward system
    def get_available_reward_functions(self):
        """Get list of available reward functions."""
        return self.reward_calculator.registry.list_functions()
    
    def change_reward_function(self, reward_type):
        """Change the reward function during runtime."""
        self.reward_function = reward_type
        self.reward_calculator = RewardCalculator(
            reward_type=reward_type,
            optimise_parameters=self.get_parameter_names(),  # Use clean parameter names
            optimise_targets=self.optimise_targets,
            combine=self.combine
        )
    
    def get_current_reward_function(self):
        """Get the currently configured reward function type."""
        return self.reward_calculator.reward_type

