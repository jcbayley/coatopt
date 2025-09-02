"""
Layer Editing Environment for iterative coating stack modification.

This environment allows the agent to:
1. Insert new layers at any position in the stack
2. Replace existing layers 
3. Start from random initial coating configurations

Action space: [layer_thickness, material_one_hot, layer_index, insert_replace_token]
- layer_thickness: continuous value for layer thickness
- material_one_hot: discrete choice of material (same as before)
- layer_index: discrete choice of position in stack to edit
- insert_replace_token: discrete binary choice (0=replace, 1=insert)
"""
import numpy as np
from typing import Optional, Tuple, Dict, Any
from .multiobjective_environment import MultiObjectiveEnvironment
from ..config.structured_config import CoatingOptimisationConfig


class LayerEditingEnvironment(MultiObjectiveEnvironment):
    """
    Enhanced coating environment that supports layer insertion and replacement.
    
    Key differences from standard environment:
    - Starts with random initial coating stack instead of all-air
    - Actions can insert or replace layers at any position
    - Agent edits existing stacks rather than building from scratch
    - Supports variable-length episodes based on improvement potential
    """

    def __init__(self, config: Optional[CoatingOptimisationConfig] = None, **kwargs):
        """
        Initialize Layer Editing environment.
        
        Args:
            config: CoatingOptimisationConfig object (new approach)
            **kwargs: Individual parameters (legacy approach)
                Additional parameters:
                - initial_stack_size_range: tuple (min, max) for initial stack size
                - max_edits_per_episode: maximum number of edits allowed per episode
                - edit_success_threshold: minimum improvement required to continue
        """
        # Extract layer editing specific parameters before parent init
        self.initial_stack_size_range = kwargs.pop('initial_stack_size_range', (3, 8))
        self.max_edits_per_episode = kwargs.pop('max_edits_per_episode', 20)
        self.edit_success_threshold = kwargs.pop('edit_success_threshold', 0.01)
        
        # Initialize base environment with all standard parameters
        super().__init__(config, **kwargs)
        
        # Setup remaining layer editing attributes after parent init
        self.setup_layer_editing_attributes()

    def setup_layer_editing_attributes(self):
        """Setup attributes specific to layer editing environment that depend on parent attributes."""
        # Track editing history
        self.edit_count = 0
        self.edit_history = []
        self.improvement_history = []
        self.current_stack_length = 0
        
        # Override action space dimensions for new format
        # [thickness] + [material_one_hot] + [layer_index] + [insert_replace_token]
        self.continuous_action_dim = 1  # only thickness
        self.discrete_action_components = {
            'material': self.n_materials,      # one-hot material selection
            'layer_index': self.max_layers,    # position in stack (will be masked)
            'insert_replace': 2                # binary: 0=replace, 1=insert
        }
        
        # Total discrete action size for compatibility
        self.total_discrete_actions = sum(self.discrete_action_components.values())

    def generate_random_initial_stack(self) -> np.ndarray:
        """
        Generate a random initial coating stack.
        
        Returns:
            Random coating stack with varying materials and thicknesses
        """
        min_size, max_size = self.initial_stack_size_range
        stack_size = np.random.randint(min_size, max_size + 1)
        
        # Initialize empty state
        state = np.zeros((self.max_layers, self.n_materials + 1))
        
        # Fill in random layers
        for i in range(stack_size):
            # Random thickness
            if self.use_optical_thickness:
                state[i, 0] = np.random.uniform(0.01, 1.0)
            else:
                state[i, 0] = np.random.uniform(self.min_thickness, self.max_thickness)
            
            # Random material (excluding air for initial stack)
            available_materials = list(range(self.n_materials))
            if self.ignore_air_option and self.air_material_index in available_materials:
                available_materials.remove(self.air_material_index)
            if self.ignore_substrate_option and self.substrate_material_index in available_materials:
                available_materials.remove(self.substrate_material_index)
            
            material_idx = np.random.choice(available_materials)
            state[i, material_idx + 1] = 1
        
        # Fill remaining layers with air
        for i in range(stack_size, self.max_layers):
            state[i, 0] = np.random.uniform(self.min_thickness, self.max_thickness)
            state[i, self.air_material_index + 1] = 1
        
        return state, stack_size

    def sample_state_space(self, random_material=False):
        """
        Override to generate random initial coating stack instead of all-air.
        
        Returns:
            Random initial coating stack
        """
        if self.opt_init:
            return self.get_optimal_state()
        else:
            state, stack_length = self.generate_random_initial_stack()
            self.current_stack_length = stack_length
            return state

    def sample_action_space(self):
        """
        Sample a random action from the expanded action space.
        
        Returns:
            Dictionary with action components:
            {
                'thickness': float,
                'material': int,
                'layer_index': int, 
                'insert_replace': int
            }
        """
        # Sample thickness (continuous)
        if self.use_optical_thickness:
            thickness = np.random.uniform(0.01, 1.0)
        else:
            thickness = np.random.uniform(self.min_thickness, self.max_thickness)
        
        # Sample material (discrete)
        available_materials = list(range(self.n_materials))
        if self.ignore_air_option and self.air_material_index in available_materials:
            available_materials.remove(self.air_material_index)
        if self.ignore_substrate_option and self.substrate_material_index in available_materials:
            available_materials.remove(self.substrate_material_index)
        
        material = np.random.choice(available_materials)
        
        # Sample layer index (discrete, based on current stack length)
        max_index = min(self.current_stack_length + 1, self.max_layers - 1)
        layer_index = np.random.randint(0, max_index + 1)
        
        # Sample insert/replace token (discrete binary)
        # If layer_index == current_stack_length, force insert
        if layer_index == self.current_stack_length:
            insert_replace = 1  # must insert at end
        else:
            insert_replace = np.random.randint(0, 2)  # 0=replace, 1=insert
        
        return {
            'thickness': thickness,
            'material': material,
            'layer_index': layer_index,
            'insert_replace': insert_replace
        }

    def validate_action(self, action_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and clamp action components to valid ranges.
        
        Args:
            action_dict: Dictionary containing action components
            
        Returns:
            Validated action dictionary
        """
        validated = action_dict.copy()
        
        # Validate thickness
        if self.use_optical_thickness:
            validated['thickness'] = np.clip(validated['thickness'], 0.01, 1.0)
        else:
            validated['thickness'] = np.clip(validated['thickness'], 
                                           self.min_thickness, self.max_thickness)
        
        # Validate material index
        validated['material'] = np.clip(validated['material'], 0, self.n_materials - 1)
        
        # Validate layer index (must be within current stack bounds)
        max_valid_index = min(self.current_stack_length, self.max_layers - 1)
        validated['layer_index'] = np.clip(validated['layer_index'], 0, max_valid_index)
        
        # Validate insert/replace token
        validated['insert_replace'] = np.clip(validated['insert_replace'], 0, 1)
        
        # Force insert if at stack end and trying to replace
        if (validated['layer_index'] == self.current_stack_length and 
            validated['insert_replace'] == 0):
            validated['insert_replace'] = 1
            
        return validated

    def insert_layer(self, state: np.ndarray, position: int, thickness: float, material_idx: int) -> np.ndarray:
        """
        Insert a new layer at the specified position.
        
        Args:
            state: Current state array
            position: Position to insert at (0-indexed)
            thickness: Thickness of new layer
            material_idx: Material index for new layer
            
        Returns:
            Updated state array with layer inserted
        """
        new_state = state.copy()
        
        # Check if we can insert (not at max capacity)
        if self.current_stack_length >= self.max_layers:
            return new_state  # Cannot insert, return unchanged
        
        # Shift layers after insertion point down by one position
        for i in range(self.max_layers - 2, position - 1, -1):
            if i >= 0:
                new_state[i + 1] = new_state[i].copy()
        
        # Insert new layer at position
        new_state[position, :] = 0  # Clear the position
        new_state[position, 0] = thickness
        new_state[position, material_idx + 1] = 1
        
        # Update stack length
        self.current_stack_length = min(self.current_stack_length + 1, self.max_layers)
        
        return new_state

    def replace_layer(self, state: np.ndarray, position: int, thickness: float, material_idx: int) -> np.ndarray:
        """
        Replace an existing layer at the specified position.
        
        Args:
            state: Current state array
            position: Position to replace (0-indexed)
            thickness: Thickness of replacement layer
            material_idx: Material index for replacement layer
            
        Returns:
            Updated state array with layer replaced
        """
        new_state = state.copy()
        
        # Validate position
        if position >= self.current_stack_length:
            return new_state  # Invalid position, return unchanged
        
        # Replace layer at position
        new_state[position, :] = 0  # Clear the position
        new_state[position, 0] = thickness
        new_state[position, material_idx + 1] = 1
        
        return new_state

    def step(self, action, objective_weights=None, always_return_value=False):
        """
        Take a step in the environment using layer editing actions.
        
        Args:
            action: Either dict format {'thickness': float, 'material': int, ...} or 
                   legacy tensor format [thickness, material_one_hot] for compatibility
            objective_weights: Optional weights for reward calculation
            always_return_value: Whether to always compute reward
            
        Returns:
            Tuple of (next_state, rewards, done, finished, _, full_action, vals)
            - Compatible with existing trainer interface
        """
        # Handle both action formats for compatibility
        if isinstance(action, dict):
            action_dict = action
        else:
            # Convert legacy tensor format to dict format
            action_dict = self._convert_legacy_action_to_dict(action)
        
        # Validate action
        action_dict = self.validate_action(action_dict)
        
        # Extract action components
        thickness = action_dict['thickness']
        material_idx = action_dict['material']
        layer_index = action_dict['layer_index']
        insert_replace = action_dict['insert_replace']
        
        # Store previous state for comparison
        previous_state = self.current_state.copy()
        
        # Apply action based on insert/replace token
        if insert_replace == 1:  # Insert
            self.current_state = self.insert_layer(self.current_state, layer_index, thickness, material_idx)
            action_type = "insert"
        else:  # Replace
            self.current_state = self.replace_layer(self.current_state, layer_index, thickness, material_idx)
            action_type = "replace"
        
        # Increment edit count
        self.edit_count += 1
        
        # Record edit in history
        self.edit_history.append({
            'edit_count': self.edit_count,
            'action_type': action_type,
            'position': layer_index,
            'thickness': thickness,
            'material': material_idx,
            'stack_length_before': self.current_stack_length if action_type == "replace" else self.current_stack_length - 1,
            'stack_length_after': self.current_stack_length
        })
        
        # Check if episode is done
        done = self._check_episode_termination(action_dict)
        finished = done  # For compatibility with trainer interface
        
        # Calculate reward
        if done or always_return_value:
            total_reward, vals, rewards = self.compute_reward(self.current_state, objective_weights)
            
            # Track improvement for adaptive termination
            if len(self.improvement_history) > 0:
                improvement = total_reward - self.improvement_history[-1]
                self.improvement_history.append(total_reward)
            else:
                improvement = 0.0
                self.improvement_history.append(total_reward)
        else:
            # Intermediate reward calculation
            if self.use_intermediate_reward:
                total_reward, vals, rewards = self.compute_reward(self.current_state, objective_weights)
                if len(self.improvement_history) > 0:
                    improvement = total_reward - self.improvement_history[-1]
                self.improvement_history.append(total_reward)
            else:
                total_reward = 0.0
                vals = {}
                rewards = {"total_reward": 0.0}
                improvement = 0.0
        
        # Create full_action for compatibility (convert back to tensor format)
        full_action = self._convert_dict_action_to_legacy(action_dict)
        
        # Update base class tracking variables for compatibility
        self.current_index = self.current_stack_length
        self.length = self.current_stack_length
        
        # Add layer editing specific info to vals
        vals.update({
            'edit_count': self.edit_count,
            'stack_length': self.current_stack_length,
            'action_type': action_type,
            'improvement': improvement,
        })
        
        # Return in trainer-compatible format
        return (
            self.current_state,           # next_state
            rewards,                      # rewards dict with 'total_reward' key
            done,                         # done
            finished,                     # finished (same as done for this env)
            total_reward,                # reward value (for backward compatibility)
            full_action,                 # full_action (tensor format)
            vals                         # vals dict
        )

    def _convert_legacy_action_to_dict(self, action) -> Dict[str, Any]:
        """
        Convert legacy tensor action [thickness, material_one_hot] to dict format.
        
        Args:
            action: Legacy action tensor or array
            
        Returns:
            Action dictionary
        """
        if hasattr(action, 'numpy'):
            action = action.numpy()
        elif hasattr(action, 'detach'):
            action = action.detach().numpy()
        
        thickness = float(action[0])
        material_one_hot = action[1:]
        material_idx = int(np.argmax(material_one_hot))
        
        # Default behavior: append to end of stack (insert)
        return {
            'thickness': thickness,
            'material': material_idx,
            'layer_index': self.current_stack_length,  # Insert at end
            'insert_replace': 1  # Insert
        }

    def _convert_dict_action_to_legacy(self, action_dict: Dict[str, Any]):
        """
        Convert dict action back to legacy tensor format for compatibility.
        
        Args:
            action_dict: Action dictionary
            
        Returns:
            Legacy action tensor [thickness, material_one_hot]
        """
        legacy_action = np.zeros(1 + self.n_materials)
        legacy_action[0] = action_dict['thickness']
        legacy_action[1 + action_dict['material']] = 1.0
        
        return legacy_action

    def _check_episode_termination(self, action_dict: Dict[str, Any]) -> bool:
        """
        Check if episode should terminate based on various criteria.
        
        Args:
            action_dict: Current action dictionary
            
        Returns:
            Whether episode should terminate
        """
        # Maximum edits reached
        if self.edit_count >= self.max_edits_per_episode:
            return True
            
        # Stack at maximum capacity
        if self.current_stack_length >= self.max_layers:
            return True
            
        # Air layer selected (traditional termination condition)
        if action_dict['material'] == self.air_material_index and self.edit_count > 1:
            return True
            
        # Optionally: check for lack of improvement over recent edits
        if (len(self.improvement_history) >= 5 and 
            all(imp < self.edit_success_threshold for imp in self.improvement_history[-3:])):
            return True
            
        return False

    def _get_done_reason(self) -> str:
        """Get reason for episode termination."""
        if self.edit_count >= self.max_edits_per_episode:
            return 'max_edits'
        elif self.current_stack_length >= self.max_layers:
            return 'max_layers'
        elif len(self.improvement_history) >= 5 and all(imp < self.edit_success_threshold for imp in self.improvement_history[-3:]):
            return 'no_improvement'
        else:
            return 'air_layer'

    def reset(self):
        """Reset environment with new random initial stack."""
        # Reset counters and history
        self.edit_count = 0
        self.edit_history = []
        self.improvement_history = []
        
        # Generate new random initial stack
        self.current_state, self.current_stack_length = self.generate_random_initial_stack()
        
        # Reset base environment tracking
        self.current_index = self.current_stack_length  # Start at end of initial stack
        self.length = self.current_stack_length
        
        # Return the actual state, not observation
        return self.current_state

    def get_valid_layer_indices(self) -> np.ndarray:
        """
        Get array of valid layer indices for current stack.
        
        Returns:
            Array of valid indices where actions can be applied
        """
        # Can edit any existing layer (replace) or insert at any position up to stack end
        return np.arange(0, min(self.current_stack_length + 1, self.max_layers))

    def get_action_mask(self) -> Dict[str, np.ndarray]:
        """
        Get masks for valid actions in current state.
        
        Returns:
            Dictionary of masks for each action component
        """
        masks = {}
        
        # Material mask (same as before)
        material_mask = np.ones(self.n_materials, dtype=bool)
        if self.ignore_air_option:
            material_mask[self.air_material_index] = False
        if self.ignore_substrate_option:
            material_mask[self.substrate_material_index] = False
        masks['material'] = material_mask
        
        # Layer index mask (only valid positions)
        layer_index_mask = np.zeros(self.max_layers, dtype=bool)
        valid_indices = self.get_valid_layer_indices()
        layer_index_mask[valid_indices] = True
        masks['layer_index'] = layer_index_mask
        
        # Insert/replace mask
        insert_replace_mask = np.ones(2, dtype=bool)
        # If at stack end, can only insert
        if self.current_stack_length == 0:
            insert_replace_mask[0] = False  # Cannot replace empty stack
        masks['insert_replace'] = insert_replace_mask
        
        return masks

    def get_observation(self):
        """
        Get observation including current stack state and metadata.
        
        Returns:
            Observation array including stack information
        """
        # Get base observation (thickness, n, k for each layer)
        base_obs = super().get_observation() if hasattr(super(), 'get_observation') else self._compute_optical_properties()
        
        # Add metadata about current editing state
        # This could include: current_stack_length, edit_count, etc.
        # For now, just return base observation
        return base_obs

    def _compute_optical_properties(self):
        """
        Compute optical properties for observation.
        
        Returns:
            Array with thickness, n, k for each layer
        """
        obs = np.zeros(self.obs_space_shape)
        
        for i in range(self.max_layers):
            # Thickness
            obs[i, 0] = self.current_state[i, 0]
            
            # Get material index
            material_idx = np.argmax(self.current_state[i, 1:])
            
            # Get refractive index (n, k) for material
            if material_idx < len(self.materials):
                material = self.materials[material_idx]
                if hasattr(material, 'n') and hasattr(material, 'k'):
                    obs[i, 1] = material.n
                    obs[i, 2] = material.k
                else:
                    # Default values if material properties not available
                    obs[i, 1] = 1.0  # n
                    obs[i, 2] = 0.0  # k
        
        return obs

    def print_current_stack(self):
        """Print current coating stack for debugging."""
        print(f"\nCurrent stack (length: {self.current_stack_length}, edits: {self.edit_count}):")
        for i in range(self.current_stack_length):
            thickness = self.current_state[i, 0]
            material_idx = np.argmax(self.current_state[i, 1:])
            material_name = f"Material_{material_idx}" if material_idx < len(self.materials) else "Unknown"
            print(f"  Layer {i}: {thickness:.2e}m, {material_name}")
        print()

    def get_stack_summary(self) -> Dict[str, Any]:
        """
        Get summary of current stack state.
        
        Returns:
            Dictionary with stack information
        """
        return {
            'stack_length': self.current_stack_length,
            'edit_count': self.edit_count,
            'total_thickness': np.sum(self.current_state[:self.current_stack_length, 0]),
            'material_distribution': self._get_material_distribution(),
            'improvement_trend': self.improvement_history[-5:] if len(self.improvement_history) >= 5 else self.improvement_history
        }

    def _get_material_distribution(self) -> Dict[int, int]:
        """Get distribution of materials in current stack."""
        distribution = {}
        for i in range(self.current_stack_length):
            material_idx = np.argmax(self.current_state[i, 1:])
            distribution[material_idx] = distribution.get(material_idx, 0) + 1
        return distribution
