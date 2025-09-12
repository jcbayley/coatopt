"""
Refactored thermal noise environment - streamlined and focused on core functionality.
Weight cycling moved to trainer, reward calculations moved to reward_utils, 
utilities moved to coating_utils_env.
"""
import numpy as np
from typing import Optional, TYPE_CHECKING
from .core.base_environment import BaseCoatingEnvironment
from .core.state import CoatingState

from ..config.structured_config import CoatingOptimisationConfig


class HPPOEnvironment(BaseCoatingEnvironment):
    """
    Streamlined thermal noise coating environment.
    
    Core functionality only:
    - State/action sampling
    - Physics calculations (reflectivity, thermal noise, absorption)
    - Basic reward computation
    - RL environment interface (reset, step)
    
    Moved to other modules:
    - Weight cycling -> training utilities
    - Reward function selection -> reward_utils
    - Visualization -> coating_utils_env
    """

    def __init__(self, config: Optional[CoatingOptimisationConfig] = None, **kwargs):
        """
        Initialize HPPO environment with dual initialization support.
        
        Args:
            config: CoatingOptimisationConfig object (new approach)
            **kwargs: Individual parameters (legacy approach)
        """
        # Initialize base environment with all standard parameters
        super().__init__(config, **kwargs)
        
        self.setup_hppo_specific_attributes(**kwargs)

    def setup_hppo_specific_attributes(self, **kwargs):

        self.pareto_front = []
        self.all_points = []

    def sample_action_space(self):
        """
        Sample a random action from the action space.
        
        Returns:
            Random action array [thickness, material_one_hot]
        """
        action = np.zeros(self.n_materials + 1)
        
        # Sample thickness
        if self.use_optical_thickness:
            action[0] = np.random.uniform(0.01, 1.0)
        else:
            action[0] = np.random.uniform(self.min_thickness, self.max_thickness)
        
        # Sample material (one-hot encoded)
        material_idx = np.random.randint(0, self.n_materials)
        action[material_idx + 1] = 1
        
        return action

    def step(self, action, objective_weights=None, always_return_value=False):
        """
        Take a step in the environment.
        
        Args:
            action: Action array [thickness, material_one_hot]
            objective_weights: Optional weights for reward calculation
            always_return_value: Whether to always compute reward
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        # Extract action components
        thickness = action[0]
        material_one_hot = action[1:]
        material_idx = np.argmax(material_one_hot)
        
        # Ensure current_state is CoatingState
        if not isinstance(self.current_state, CoatingState):
            self.current_state = self.tensor_to_coating_state(self.current_state)
        
        # Update state using CoatingState.set_layer()
        self.current_state.set_layer(self.current_index, thickness, material_idx)
        
        # Add state validation call for debugging
        if hasattr(self, 'debug_enabled') and self.debug_enabled:
            issues = self.current_state.validate()
            if issues:
                print(f"DEBUG: State validation issues: {issues}")
        
        self.current_index += 1
        self.length += 1
        
        # Check if episode is done using state.get_num_active_layers()
        active_layers = self.current_state.get_num_active_layers()
        done = (self.current_index >= self.max_layers) or (material_idx == self.air_material_index and self.current_index > 1)
        
        # Calculate reward
        if done or always_return_value:
            total_reward, vals, rewards = self.compute_reward(self.current_state, objective_weights)
        else:
            total_reward = 0.0
            vals = {}
            rewards = {}
            
            # Intermediate reward if enabled
            if self.use_intermediate_reward:
                # Could implement intermediate reward logic here
                pass
        
        # Get observation
        observation = self.get_observation()
        
        # Info dictionary
        info = {
            'length': self.length,
            'values': vals,
            'individual_rewards': rewards,
            'done_reason': 'max_layers' if self.current_index >= self.max_layers else 'air_layer',
            'active_layers': active_layers
        }
        
        return observation, total_reward, done, info


