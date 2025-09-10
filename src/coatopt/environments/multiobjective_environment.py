"""
Pareto optimization coating environment.
Extends the base CoatingStack with multi-objective optimization capabilities.
"""
from typing import Optional, TYPE_CHECKING
from .hppo_environment import HPPOEnvironment
from .core.state import CoatingState
import numpy as np
import copy
import torch

from ..config.structured_config import CoatingOptimisationConfig
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from .utils.pareto_utils import incremental_pareto_update, EfficientParetoTracker
from .core.base_environment import BaseCoatingEnvironment
from .reward_functions.reward_system import RewardCalculator
from .utils import coating_utils, state_utils
from typing import Dict, List, Optional, Union, Tuple, Any
import random
import logging

class MultiObjectiveEnvironment(HPPOEnvironment):
    """
    Coating environment with Pareto multi-objective optimization.
    This extends the base functionality to support multi-objective optimization.
    """
    def __init__(self, config: Optional[CoatingOptimisationConfig] = None, 
                 use_reward_normalization=False, reward_normalization_mode="fixed",
                 reward_normalization_ranges=None, reward_normalization_alpha=0.1, **kwargs):
        """Initialize Pareto environment."""
        super().__init__(config, **kwargs)
        
        # Parse optimization directions from optimise_parameters
        self.optimization_directions = self._parse_optimization_parameters()
        
        # Enable multi-objective optimization
        self.multi_objective = True
        self.pareto_objectives = ["reflectivity", "thermal_noise", "absorption"]
        
        # Store normalization parameters
        self.use_reward_normalization = use_reward_normalization
        self.reward_normalization_mode = reward_normalization_mode
        self.reward_normalization_ranges = reward_normalization_ranges or {}
        self.reward_normalization_alpha = reward_normalization_alpha
        
        # Override reward calculator with normalization support
        reward_type = "default" if self.reward_function is None else str(self.reward_function)
        self.reward_calculator = RewardCalculator(
            reward_type=reward_type,
            optimise_parameters=self.get_parameter_names(),  # Use clean parameter names
            optimise_targets=self.optimise_targets,
            combine=self.combine, 
            env=self,
            use_reward_normalization=use_reward_normalization,
            reward_normalization_mode=reward_normalization_mode,
            reward_normalization_ranges=reward_normalization_ranges,
            reward_normalization_alpha=reward_normalization_alpha,
            apply_normalization=self.apply_normalization,
            apply_boundary_penalties=self.apply_boundary_penalties,
            apply_divergence_penalty=self.apply_divergence_penalty,
            apply_air_penalty=self.apply_air_penalty,
            apply_pareto_improvement=self.apply_pareto_improvement,
            air_penalty_weight=self.air_penalty_weight,
            divergence_penalty_weight=self.divergence_penalty_weight,
            pareto_improvement_weight=self.pareto_improvement_weight,
        )
        
        # Enhanced Pareto tracking with efficient algorithms
        self.pareto_update_interval = kwargs.get('pareto_update_interval', 10)  # Update every N steps
        self.use_efficient_pareto = kwargs.get('use_efficient_pareto', True)  # Use optimized algorithms
        
        self.pareto_tracker = EfficientParetoTracker(
                update_interval=self.pareto_update_interval,
                max_pending=50
            )


    def setup_multiobjective_specific_attributes(self, **kwargs):
        """Setup multi-objective specific attributes."""
        # Enable multi-objective optimization
        self.multi_objective = True
        self.pareto_objectives = ["reflectivity", "thermal_noise", "absorption"]
        self.pareto_front = []
        self.all_points = []
        self.all_vals = []
    
    def _parse_optimization_parameters(self):
        """
        Parse optimization parameters and directions from config.
        
        Supports formats like:
        - ["reflectivity:max", "absorption:min"] 
        - ["reflectivity", "absorption"] (fallback to defaults)
        
        Returns:
            dict: {parameter_name: direction} where direction is 'max' or 'min'
        """
        optimization_directions = {}
        
        for param in self.optimise_parameters:
            if isinstance(param, str) and ':' in param:
                # New format: "parameter:direction"
                param_name, direction = param.split(':', 1)
                param_name = param_name.strip()
                direction = direction.strip().lower()
                
                if direction in ['max', 'maximize', 'maximum']:
                    optimization_directions[param_name] = 'max'
                elif direction in ['min', 'minimize', 'minimum']:
                    optimization_directions[param_name] = 'min'
                else:
                    print(f"Warning: Unknown optimization direction '{direction}' for {param_name}, defaulting to 'min'")
                    optimization_directions[param_name] = 'min'
            else:
                # Legacy format: just parameter name, use defaults
                param_name = param if isinstance(param, str) else str(param)
                if param_name.lower() == 'reflectivity':
                    optimization_directions[param_name] = 'max'  # Default: maximize reflectivity
                else:
                    optimization_directions[param_name] = 'min'  # Default: minimize others
                    
        return optimization_directions
    
    
    def update_pareto_front(self, pareto_front, new_point, new_value, new_state):
        """Use efficient Pareto tracker for updates."""
        # Initialize tracker if needed
        if not hasattr(self, 'pareto_tracker') or self.pareto_tracker is None:
            self.pareto_tracker = EfficientParetoTracker(
                update_interval=self.pareto_update_interval,
                max_pending=50
            )
        # Add new point using efficient tracker (only rewards for environment)
        updated_front, was_updated = self.pareto_tracker.add_point(new_point, new_value, new_state)
        return updated_front, was_updated
    
    def _perform_batch_pareto_update(self, pareto_front):
        """Perform batch Pareto front update with all pending points."""
        if not self.pending_pareto_points:
            return pareto_front, False
        
        if isinstance(pareto_front, list):
            pareto_front = np.array(pareto_front)
        
        # Use efficient incremental update
        all_pending = np.vstack(self.pending_pareto_points)
        updated_pareto_front, pareto_updated = incremental_pareto_update(pareto_front, all_pending)
        
        # Reset tracking
        self._reset_pareto_update_tracking()
        return updated_pareto_front, pareto_updated
    
    def _reset_pareto_update_tracking(self):
        """Reset lazy Pareto update tracking variables."""
        if hasattr(self, 'pending_pareto_points'):
            self.pending_pareto_points = []
        if hasattr(self, 'steps_since_pareto_update'):
            self.steps_since_pareto_update = 0
        if hasattr(self, 'force_pareto_update'):
            self.force_pareto_update = False
    
    def force_pareto_front_update(self):
        """Force immediate Pareto front update on next call."""
        if self.use_efficient_pareto and hasattr(self, 'pareto_tracker'):
            self.pareto_tracker._perform_update()
        else:
            self.force_pareto_update = True
    
    def get_pareto_stats(self):
        """Get Pareto front management statistics."""
        if self.use_efficient_pareto and hasattr(self, 'pareto_tracker'):
            stats = self.pareto_tracker.get_stats()
            stats['algorithm'] = 'efficient'
            return stats
        else:
            raise Exception("Legacy Pareto tracking not supported in this version.")
            return {
                'algorithm': 'legacy',
                'front_size': len(self.pareto_front) if hasattr(self, 'pareto_front') else 0,
                'pending_points': len(getattr(self, 'pending_pareto_points', [])),
                'steps_since_update': getattr(self, 'steps_since_pareto_update', 0)
            }
    
    def compute_reward(self, new_state, max_value=0.0, target_reflectivity=1.0, objective_weights=None):
        """reward is the improvement of the state over the previous one

        Args:
            state (_type_): _description_
            action (_type_): _description_
        """
        if isinstance(new_state, CoatingState):
            # Use state.get_tensor() for calculations
            state_array = new_state.get_array()
        else:
            state_array = new_state  # Assume numpy array for backward compatibility

        new_reflectivity, new_thermal_noise, new_E_integrated, new_total_thickness = self.compute_state_value(new_state, return_separate=True)
        
        if objective_weights is not None:
            weights = {
                key: objective_weights[i] for i, key in enumerate(self.get_parameter_names())
            }
        else:
            weights = None

        # RewardCalculator now handles normalization internally
        total_reward, vals, rewards = self.reward_calculator.calculate(
                reflectivity=new_reflectivity,
                thermal_noise=new_thermal_noise,
                thickness=new_total_thickness,
                absorption=new_E_integrated,
                weights=weights,  # Pass weights directly to calculator
                expert_constraints=self.current_expert_constraints,
                env=self
            )
        
        new_point = np.zeros((len(self.optimise_parameters),))
        new_values = np.zeros((len(self.optimise_parameters),))

        i = 0
        for key in self.get_parameter_names():
            new_point[i] = rewards[key]
            new_values[i] = vals[key]
            i += 1

        updated_pareto_front, front_updated = self.update_pareto_front(copy.copy(self.pareto_front), copy.copy(new_point), copy.copy(new_values), state_array)

        rewards["updated_pareto_front"] = updated_pareto_front
        rewards["front_updated"] = front_updated

        return total_reward, vals, rewards
    
    def step(self, action, max_state=0, verbose=False, state=None, layer_index=None, always_return_value=False, objective_weights=None):
        """Step function simplified to work directly with CoatingState objects."""
        
        # Use current state if none provided
        if state is None:
            state = self.current_state
        else:
            self.current_state = state

        # Use current index if none provided
        if layer_index is None:
            layer_index = self.current_index
        else:
            self.current_index = layer_index

        # Extract action parameters (now action is a simple list)
        material = int(action[0])
        thickness = float(action[1])
        
        # Use the base class update_state method
        self.current_state, new_layer = self.update_state(self.current_state, thickness, material)
        
        # Initialize default values
        neg_reward = -1000
        reward = neg_reward
        terminated = False
        finished = False
        full_action = None
        
        rewards = {
            "reflectivity": 0, "thermal_noise": 0, "thickness": 0, 
            "absorption": 0, "total_reward": 0
        }
        vals = {
            "reflectivity": 0, "thermal_noise": 0, "thickness": 0, "absorption": 0
        }

        # Check termination conditions
        if self.min_thickness > thickness or thickness > self.max_thickness or not np.isfinite(thickness):
            print("out of thickness bounds")
        elif self.current_index == self.max_layers-1 or material == self.air_material_index:
            # Episode finished
            finished = True
            reward, vals, rewards = self.compute_reward(self.current_state, max_state, objective_weights=objective_weights)
        elif self.use_intermediate_reward:
            # Intermediate reward calculation
            reward, vals, rewards = self.compute_reward(self.current_state, max_state, objective_weights=objective_weights)

        # Check for invalid states using CoatingState validation
        if not self.current_state.is_valid() or np.isnan(reward) or np.isinf(reward):
            reward = neg_reward
            terminated = True

        # Update Pareto front if episode finished and front was updated
        if finished and rewards.get("front_updated", False):
            self.pareto_front = rewards["updated_pareto_front"]
            # Force final update to ensure we have the most current front
            self.force_pareto_front_update()

        # Update tracking variables
        self.previous_material = material
        self.length += 1
        self.current_index += 1

        return self.current_state, rewards, terminated, finished, reward, full_action, vals
