import torch
import numpy as np
from typing import Tuple, Dict, Any, Optional, List, TYPE_CHECKING
from ..utils import coating_utils 
from ..utils import state_utils
from ..utils.EFI_tmm import CalculateEFI_tmm, optical_to_physical, physical_to_optical        # Observation space depends on whether electric field information is included

from ..reward_functions.reward_system import RewardCalculator
from .state import CoatingState
import time
import scipy
from tmm import coh_tmm
import matplotlib.pyplot as plt
import logging

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
        
        # Electric field configuration
        self.include_electric_field = getattr(config.data, 'include_electric_field', False)
        self.electric_field_points = getattr(config.data, 'electric_field_points', 50)
        
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
        
        # Objective bounds for reward normalisation (if provided in config)
        if hasattr(config.data, 'objective_bounds') and config.data.objective_bounds:
            self.objective_bounds = config.data.objective_bounds
        
        # Reward parameters from DataConfig
        self.reward_function = config.data.reward_function
        self.use_intermediate_reward = config.data.use_intermediate_reward
        self.combine = config.data.combine
        
        # Reward normalisation parameters from DataConfig
        self.use_reward_normalisation = getattr(config.data, 'use_reward_normalisation', False)
        self.reward_normalisation_mode = getattr(config.data, 'reward_normalisation_mode', 'fixed')
        self.reward_normalisation_ranges = getattr(config.data, 'reward_normalisation_ranges', {})
        self.reward_normalisation_alpha = getattr(config.data, 'reward_normalisation_alpha', 0.1)
        
        # Reward addon system configuration from DataConfig
        self.apply_normalisation = getattr(config.data, 'apply_normalisation', False)
        self.apply_boundary_penalties = getattr(config.data, 'apply_boundary_penalties', False)
        self.apply_divergence_penalty = getattr(config.data, 'apply_divergence_penalty', False)
        self.apply_air_penalty = getattr(config.data, 'apply_air_penalty', False)
        self.apply_pareto_improvement = getattr(config.data, 'apply_pareto_improvement', False)
        self.air_penalty_weight = getattr(config.data, 'air_penalty_weight', 1.0)
        self.divergence_penalty_weight = getattr(config.data, 'divergence_penalty_weight', 1.0)
        self.pareto_improvement_weight = getattr(config.data, 'pareto_improvement_weight', 1.0)
        
        # Training parameters from DataConfig
        self.ignore_air_option = config.data.ignore_air_option
        self.ignore_substrate_option = config.data.ignore_substrate_option
        
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
                                 objective_bounds=None,
                                 include_electric_field=False,
                                 electric_field_points=50,
                                 # Reward normalisation parameters
                                 use_reward_normalisation=False,
                                 reward_normalisation_mode="fixed",
                                 reward_normalisation_ranges=None,
                                 reward_normalisation_alpha=0.1,
                                 # Reward addon system configuration
                                 apply_normalisation=False,
                                 apply_boundary_penalties=False,
                                 apply_divergence_penalty=False,
                                 apply_air_penalty=False,
                                 apply_pareto_improvement=False,
                                 apply_preference_constraints=False,
                                 air_penalty_weight=1.0,
                                 divergence_penalty_weight=1.0,
                                 pareto_improvement_weight=1.0,
                                 preference_constraint_weight=1.0,
                                 multi_value_rewards=False,):
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
        self.multi_value_rewards = multi_value_rewards
        
        # Electric field configuration (legacy)
        self.include_electric_field = include_electric_field
        self.electric_field_points = electric_field_points
        
        # Objective bounds for reward normalisation (if provided)
        if objective_bounds is not None:
            self.objective_bounds = objective_bounds
        
        # Reward addon system configuration
        self.apply_normalisation = apply_normalisation
        self.apply_boundary_penalties = apply_boundary_penalties
        self.apply_divergence_penalty = apply_divergence_penalty
        self.apply_air_penalty = apply_air_penalty
        self.apply_pareto_improvement = apply_pareto_improvement
        self.apply_preference_constraints = apply_preference_constraints
        self.air_penalty_weight = air_penalty_weight
        self.divergence_penalty_weight = divergence_penalty_weight
        self.preference_constraint_weight = preference_constraint_weight
        self.pareto_improvement_weight = pareto_improvement_weight

    def _setup_common_attributes(self):
        """Setup attributes common to both initialization methods."""
        self.n_materials = len(self.materials)
        self.n_material_options = self.n_materials
        
        # State and observation space setup
        self.state_space_size = self.max_layers * self.n_materials + self.max_layers
        self.state_space_shape = (self.max_layers, self.n_materials + 1)
        
        # Environment state - always initialize with CoatingState
        self.length = 0
        self.current_state = self.sample_state_space()
        self.current_index = 0
        self.previous_material = self.substrate_material_index
        
        # Calculate observation space dynamically
        self.obs_space_shape = self._get_observation_shape()
        self.obs_space_size = self.obs_space_shape[0] * self.obs_space_shape[1]
        
        # Initialize reward calculator with addon configuration
        reward_type = "default" if self.reward_function is None else str(self.reward_function)
        
        # Gather reward calculator configuration parameters
        reward_calc_config = {
            'reward_type': reward_type,
            'optimise_parameters': self.get_parameter_names(),  # Use clean parameter names
            'optimise_targets': self.optimise_targets,
            'combine': self.combine,
            'use_reward_normalisation': getattr(self, 'use_reward_normalisation', False),
            'reward_normalisation_mode': getattr(self, 'reward_normalisation_mode', 'fixed'),
            'reward_normalisation_ranges': getattr(self, 'reward_normalisation_ranges', {}),
            'reward_normalisation_alpha': getattr(self, 'reward_normalisation_alpha', 0.1),
            # Addon configuration
            'apply_normalisation': getattr(self, 'apply_normalisation', False),
            'apply_boundary_penalties': getattr(self, 'apply_boundary_penalties', False),
            'apply_divergence_penalty': getattr(self, 'apply_divergence_penalty', False),
            'apply_air_penalty': getattr(self, 'apply_air_penalty', False),
            'apply_pareto_improvement': getattr(self, 'apply_pareto_improvement', False),
            'apply_preference_constraints': getattr(self, 'apply_preference_constraints', False),
            'air_penalty_weight': getattr(self, 'air_penalty_weight', 1.0),
            'divergence_penalty_weight': getattr(self, 'divergence_penalty_weight', 1.0),
            'pareto_improvement_weight': getattr(self, 'pareto_improvement_weight', 1.0),
            'preference_constraint_weight': getattr(self, 'preference_constraint_weight', 1.0),
            'multi_value_rewards': hasattr(self, 'multi_value_rewards')
        }
        
        self.reward_calculator = RewardCalculator(**reward_calc_config)

    def _get_state_shape(self):
        """
        Get shape of the underlying state tensor.
        
        Returns:
            Tuple of (max_layers, n_features) where n_features = 1 + n_materials
        """
        return (self.max_layers, 1 + self.n_materials)
    def _get_observation_shape(self):
        """
        Dynamically calculate observation shape by creating a sample CoatingState.
        This avoids hardcoding feature counts and automatically adapts to configuration.
        
        Returns:
            tuple: (max_layers, n_features) shape of observations
        """
        try:
            # Create a sample CoatingState with materials configured
            sample_state = CoatingState(
                max_layers=self.max_layers,
                n_materials=self.n_materials,
                air_material_index=self.air_material_index,
                substrate_material_index=self.substrate_material_index,
                include_electric_field=self.include_electric_field,
                materials=self.materials
            )
            
            # Get sample observation tensor
            sample_obs_tensor = sample_state.get_observation_tensor(
                include_field_data=self.include_electric_field,
                merit_function_callback=self._merit_function_wrapper if self.include_electric_field else None,
                light_wavelength=getattr(self, 'light_wavelength', 1064e-9),
                substrate_index=self.substrate_material_index,
                air_index=self.air_material_index,
                use_optical_thickness=getattr(self, 'use_optical_thickness', True)
            )

            # Return actual tensor shape
            return tuple(sample_obs_tensor.shape)
            
        except Exception as e:
            print(f"Warning: Could not calculate observation shape dynamically: {e}")
            # Fallback to basic calculation
            base_features = 4  # thickness, material_index, n, k
            if self.include_electric_field:
                base_features += 5  # + efield, grad, R, A, TN
            return (self.max_layers, base_features)  


    def reset(self):
        """Reset the environment state - always returns CoatingState object."""
        self.length = 0
        self.current_index = 0
        
        # Sample new state (always returns CoatingState)
        self.current_state = self.sample_state_space()
        
        # Ensure materials are set for enhanced observations
        if hasattr(self, 'materials'):
            self.current_state.materials = self.materials
        
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
        """
        Sample initial state space - always returns CoatingState object.

        Args:
            random_material: If True, assign random materials to layers
            
        Returns:
            CoatingState: Initial state as CoatingState object
        """
        if self.opt_init:
            layers_array = self.get_optimal_state()
            # Convert to CoatingState
            state = CoatingState.from_tensor(
                torch.from_numpy(layers_array).float(),
                air_material_index=self.air_material_index,
                substrate_material_index=self.substrate_material_index
            )
        else:
            # Create CoatingState directly
            state = CoatingState(
                max_layers=self.max_layers,
                n_materials=self.n_materials,
                air_material_index=self.air_material_index,
                substrate_material_index=self.substrate_material_index,
                include_electric_field=self.include_electric_field,
                materials=self.materials
            )
            
            # Set random thicknesses for all layers (initially as air)
            for layer_idx in range(self.max_layers):
                thickness = np.random.uniform(self.min_thickness, self.max_thickness)
                state.set_layer(layer_idx, thickness, self.air_material_index)

        if random_material:
            for layer_ind in range(self.max_layers):
                material = np.random.randint(1, self.n_materials)
                thickness = state.get_thickness_at_layer(layer_ind)
                state.set_layer(layer_ind, thickness, material)
        
        return state

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
            return_separate = False,
            return_field_data=False):
        """
        Compute state value with optional electric field data return.

        Args:
            state: CoatingState object or array (backward compatibility)
            material_sub (int, optional): Substrate material type 
            light_wavelength (int, optional): laser wavelength (m)
            frequency (int, optional): frequency of interest (Hz)
            wBeam (int, optional): laser beam radius on optic(m)
            Temp (int, optional): detector temperature (deg)
            return_separate (bool): If True, return individual metrics
            return_field_data (bool): If True, return electric field information

        Returns:
            _type_: Performance metrics and optionally field data
        """
        
        # Handle CoatingState input
        if isinstance(state, CoatingState):
            # Use state.get_tensor() for calculations
            state_array = state.get_array()
        else:
            # Backward compatibility with numpy arrays/tensors
            if isinstance(state, torch.Tensor):
                state_tensor = state
            else:
                state_tensor = torch.from_numpy(state).float()
            # Convert to numpy for existing trim_state function
            state_array = state_tensor.numpy()
        
        # trim out the duplicate air layers and inverse order
        state_trim = state_utils.trim_state(state_array)
        # reverse state
        state_trim = state_trim[::-1]

        # Call merit_function with field data option
        result = coating_utils.merit_function(
            np.array(state_trim),
            self.materials,
            light_wavelength=light_wavelength,
            frequency=frequency,
            wBeam=wBeam,
            Temp=Temp,
            substrate_index = self.substrate_material_index,
            air_index = self.air_material_index,
            use_optical_thickness=self.use_optical_thickness,
            return_field_data=return_field_data
            )
        
        if return_field_data:
            r, thermal_noise, e_integrated, total_thickness, field_data = result
            if return_separate:
                return r, thermal_noise, e_integrated, total_thickness, field_data
            else:
                return r, thermal_noise, field_data
        else:
            r, thermal_noise, e_integrated, total_thickness = result
            if return_separate:
                return r, thermal_noise, e_integrated, total_thickness
            else:
                return r, thermal_noise
        
    def compute_reward(self, new_state, max_value=0.0, target_reflectivity=1.0, objective_weights=None, 
                      pc_tracker=None, phase_info=None):
        """reward is the improvement of the state over the previous one

        Args:
            state (_type_): _description_
            action (_type_): _description_
            pc_tracker: PreferenceConstrainedTracker instance for preference-constrained optimization
            phase_info: Phase information from preference-constrained training
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
                env=self,
                pc_tracker=pc_tracker,
                phase_info=phase_info
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
        """
        Update state using CoatingState interface.
        
        Args:
            current_state: CoatingState object or array (for backward compatibility)
            thickness: New thickness value
            material: Material index

        Returns:
            tuple: (updated_state, new_layer_tensor)
        """
        # Convert to CoatingState if needed using universal loader
        if not isinstance(current_state, CoatingState):
            current_state = CoatingState.load_from_array(
                current_state,
                max_layers=self.max_layers,
                n_materials=self.n_materials,
                air_material_index=self.air_material_index,
                substrate_material_index=self.substrate_material_index
            )
            current_state.materials = self.materials
        
        # Use CoatingState.set_layer() for clean state manipulation
        current_state.set_layer(self.current_index, thickness, material)
        
        # For backward compatibility, create new_layer tensor
        material_onehot = current_state.material_index_to_onehot(material)
        thickness_tensor = torch.tensor([thickness])
        new_layer = torch.cat([thickness_tensor, material_onehot])

        return current_state, new_layer
    
    def get_observation_from_state(self, state=None):
        """
        Get observation from state - delegates to CoatingState for consistency.
        
        Args:
            state: CoatingState object or array (backward compatibility)
                   If None, uses self.current_state
            
        Returns:
            dict: Observation dictionary
        """
        # Use current state if no state provided
        if state is None:
            if not hasattr(self, 'current_state') or self.current_state is None:
                raise ValueError("No current state available. Call reset() first.")
            state = self.current_state
        
        # Handle backward compatibility - convert arrays to CoatingState
        if not isinstance(state, CoatingState):
            state = CoatingState.load_from_array(
                state,
                max_layers=self.max_layers,
                n_materials=self.n_materials,
                air_material_index=self.air_material_index,
                substrate_material_index=self.substrate_material_index
            )
            # Provide materials to CoatingState for enhanced observations
            state.materials = self.materials
        
        # Use CoatingState's observation generation with physics callback
        return state.get_enhanced_observation(
            include_field_data=self.include_electric_field,
            merit_function_callback=self._merit_function_wrapper if self.include_electric_field else None,
            light_wavelength=self.light_wavelength,
            substrate_index=self.substrate_material_index,
            air_index=self.air_material_index,
            use_optical_thickness=self.use_optical_thickness
        )

    def _merit_function_wrapper(self, state_trim, materials, **kwargs):
        """Wrapper for merit function to interface with CoatingState."""
        from ..utils import coating_utils
        return coating_utils.merit_function(
            np.array(state_trim),
            materials,
            **kwargs
        )
    
    def get_observation(self) -> torch.Tensor:
        """
        Get observation from current state in tensor format.
        Uses the existing CoatingState object for efficiency.
        
        Returns:
            torch.Tensor: Observation tensor ready for neural networks
        """
        if not hasattr(self, 'current_state') or self.current_state is None:
            raise ValueError("No current state available. Call reset() first.")
        
        # Since current_state is initialized as CoatingState in _setup_common_attributes,
        # we can use it directly without recreating
        return self.current_state.get_observation_tensor(
            include_field_data=self.include_electric_field,
            merit_function_callback=self._merit_function_wrapper if self.include_electric_field else None,
            light_wavelength=self.light_wavelength,
            substrate_index=self.substrate_material_index,
            air_index=self.air_material_index,
            use_optical_thickness=self.use_optical_thickness
        )
    
    def get_observation_array(self, state=None):
        """
        Get observation in legacy array format for backward compatibility.
        Converts state to simple [thickness, n, k] format per layer.
        
        Args:
            state: Current coating state (CoatingState object or array)
                   If None, uses self.current_state
            
        Returns:
            numpy.ndarray: Array with shape [n_layers, 3] containing [thickness, n, k]
        """
        # Use the centralized observation method and extract basic data
        obs_dict = self.get_observation_from_state(state)
        layer_stack = obs_dict['layer_stack']
        
        observation = []
        for layer in layer_stack:
            observation.append([
                layer['thickness'], 
                layer['n'], 
                layer['k']
            ])

        return np.array(observation)

    def step(self, action, max_state=0, verbose=False, state=None, layer_index=None, always_return_value=False, objective_weights=None, pc_tracker=None, phase_info=None):
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

        # Ensure state is CoatingState
        if not isinstance(state, CoatingState):
            state = CoatingState.load_from_array(
                state,
                max_layers=self.max_layers,
                n_materials=self.n_materials,
                air_material_index=self.air_material_index,
                substrate_material_index=self.substrate_material_index
            )
            self.current_state = state

        material = action[0]
        thickness = action[1] #* self.light_wavelength /(4*self.materials[material]["n"])
        
        # Create a copy for update_state
        if isinstance(state, CoatingState):
            state_copy = state.copy()
        else:
            state_copy = np.copy(state)
        
        new_state, full_action = self.update_state(state_copy, thickness, material)


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
            reward, vals, rewards = self.compute_reward(new_state, max_state, objective_weights=objective_weights, pc_tracker=pc_tracker, phase_info=phase_info)

        else:
            self.current_state = new_state
            if self.use_intermediate_reward:
                reward, vals, rewards = self.compute_reward(new_state, max_state, objective_weights=objective_weights, pc_tracker=pc_tracker, phase_info=phase_info)
        

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
        
        # Preserve addon configuration when changing reward function
        reward_calc_config = {
            'reward_type': reward_type,
            'optimise_parameters': self.get_parameter_names(),  # Use clean parameter names
            'optimise_targets': self.optimise_targets,
            'combine': self.combine,
            'use_reward_normalisation': getattr(self, 'use_reward_normalisation', False),
            'reward_normalisation_mode': getattr(self, 'reward_normalisation_mode', 'fixed'),
            'reward_normalisation_ranges': getattr(self, 'reward_normalisation_ranges', {}),
            'reward_normalisation_alpha': getattr(self, 'reward_normalisation_alpha', 0.1),
            # Addon configuration
            'apply_normalisation': getattr(self, 'apply_normalisation', False),
            'apply_boundary_penalties': getattr(self, 'apply_boundary_penalties', False),
            'apply_divergence_penalty': getattr(self, 'apply_divergence_penalty', False),
            'apply_air_penalty': getattr(self, 'apply_air_penalty', False),
            'apply_pareto_improvement': getattr(self, 'apply_pareto_improvement', False),
            'air_penalty_weight': getattr(self, 'air_penalty_weight', 1.0),
            'divergence_penalty_weight': getattr(self, 'divergence_penalty_weight', 1.0),
            'pareto_improvement_weight': getattr(self, 'pareto_improvement_weight', 1.0),
        }
        
        self.reward_calculator = RewardCalculator(**reward_calc_config)
    
    def get_current_reward_function(self):
        """Get the currently configured reward function type."""
        return self.reward_calculator.reward_type

    # State conversion methods for backward compatibility
    def tensor_to_coating_state(self, tensor: torch.Tensor) -> CoatingState:
        """Convert tensor to CoatingState object."""
        return CoatingState.load_from_array(
            tensor,
            air_material_index=self.air_material_index,
            substrate_material_index=self.substrate_material_index
        )
    
    def array_to_coating_state(self, array: np.ndarray) -> CoatingState:
        """Convert numpy array to CoatingState object."""
        return CoatingState.load_from_array(
            array,
            air_material_index=self.air_material_index,
            substrate_material_index=self.substrate_material_index
        )
    
    def coating_state_to_tensor(self, state: CoatingState) -> torch.Tensor:
        """Convert CoatingState to tensor."""
        return state.get_tensor()
    
    def coating_state_to_array(self, state: CoatingState) -> np.ndarray:
        """Convert CoatingState to numpy array."""
        return state.get_tensor().numpy()

