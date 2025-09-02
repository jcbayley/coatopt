"""
Layer Editing Agent - Enhanced PC-HPPO agent for iterative coating modification.

Extends the existing PCHPPO agent to handle multi-component discrete actions:
- Material selection (same as before)
- Layer index selection (new)
- Insert/replace token (new)

Continuous action remains the same (thickness only).
"""
from typing import Union, Optional, Tuple, List, Dict, Any
import numpy as np
import torch
import torch.nn.functional as F
from collections import deque

from .agent import PCHPPO
from .networks.layer_editing_networks import MultiComponentDiscretePolicy
from ...action_utils import (
    prepare_state_input, prepare_layer_number, 
    pack_state_sequence, validate_probabilities
)


class LayerEditingPCHPPO(PCHPPO):
    """
    Enhanced PC-HPPO agent for layer editing tasks.
    
    Extends the base agent to handle multi-component discrete actions while
    maintaining compatibility with the existing training infrastructure.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize with enhanced action space handling."""
        # Extract layer editing specific parameters
        self.max_stack_length = kwargs.pop('max_stack_length', 20)
        
        # Initialize base agent
        super().__init__(*args, **kwargs)
        
        # Flag to indicate this is a layer editing agent
        self.is_layer_editing = True
        
        # Override discrete action dimensions for multi-component output
        self.setup_multi_component_discrete_policy()
    
    def setup_multi_component_discrete_policy(self):
        """
        Modify the discrete policy to output multi-component actions.
        This replaces the single material selection with three separate outputs.
        """
        # Store original discrete output dimension for reference
        self.original_discrete_dim = self.num_discrete
        
        # Define multi-component discrete action structure
        self.discrete_action_components = {
            'material': self.num_discrete,           # Original material selection
            'layer_index': self.max_stack_length,    # Layer position selection
            'insert_replace': 2                      # Binary insert/replace choice
        }
        
        # Total discrete actions (for network output sizing)
        self.total_discrete_outputs = sum(self.discrete_action_components.values())
        
        # Create multi-component discrete policy network
        # Get the original discrete policy architecture parameters
        original_policy = self.policy_discrete
        
        self.multi_discrete_policy = MultiComponentDiscretePolicy(
            input_dim=original_policy.input_dim,
            action_components=self.discrete_action_components,
            hidden_dim=getattr(original_policy, 'hidden_dim', 64),
            n_layers=getattr(original_policy, 'n_layers', 2),
            include_layer_number=getattr(original_policy, 'include_layer_number', True),
            activation=getattr(original_policy, 'activation', 'relu'),
            n_objectives=getattr(original_policy, 'n_objectives', 3),
            use_hyper_networks=getattr(original_policy, 'use_hyper_networks', False),
            hyper_hidden_dim=getattr(original_policy, 'hyper_hidden_dim', 128),
            hyper_n_layers=getattr(original_policy, 'hyper_n_layers', 2)
        )
        
    def select_action(
        self, 
        state: Union[np.ndarray, torch.Tensor], 
        layer_number: Optional[Union[np.ndarray, torch.Tensor]] = None,
        actionc: Optional[torch.Tensor] = None,
        actiond_components: Optional[Dict[str, torch.Tensor]] = None,
        packed: bool = False,
        objective_weights: Optional[torch.Tensor] = None,
        action_masks: Optional[Dict[str, np.ndarray]] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, 
               Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, 
               torch.Tensor, torch.Tensor, torch.Tensor, 
               Dict[str, torch.Tensor], Dict[str, torch.Tensor], dict]:
        """
        Enhanced action selection for layer editing.
        
        Args:
            state: Current environment state
            layer_number: Current layer number
            actionc: Continuous action to evaluate (if None, sample new)
            actiond_components: Dict of discrete actions to evaluate (if None, sample new)
            packed: Whether state is already packed
            objective_weights: Multi-objective optimization weights
            action_masks: Masks for invalid actions
            
        Returns:
            Enhanced tuple with multi-component discrete actions
        """
        # Prepare inputs (same as base implementation)
        state_tensor = prepare_state_input(state, self.pre_type)
        layer_number, layer_numbers_validated = prepare_layer_number(layer_number)
        
        # Ensure objective_weights is properly formatted
        if objective_weights is not None:
            if not isinstance(objective_weights, torch.Tensor):
                objective_weights = torch.FloatTensor(objective_weights)
            else:
                objective_weights = objective_weights.float()
            
            if objective_weights.dim() == 1:
                objective_weights = objective_weights.unsqueeze(0)
        
        # Pack state sequence for LSTM processing
        if self.pre_type == "lstm":
            state_input = pack_state_sequence(state_tensor, layer_numbers_validated)
            use_packed = True
        else:
            state_input = state_tensor
            use_packed = False
        
        # Get pre-network outputs
        is_training = actionc is not None or actiond_components is not None
        pre_output_d, pre_output_c, pre_output_v = self._get_pre_network_outputs(
            state_input, layer_number, objective_weights, needs_gradients=is_training, packed=use_packed
        )
        
        # Get discrete probabilities using the multi-component network
        discrete_probs_dict = self.multi_discrete_policy.forward(
            pre_output_d, layer_number, objective_weights
        )
        
        if actiond_components is not None:
            # Convert tensor actions to dictionary format for evaluation
            if isinstance(actiond_components, torch.Tensor):
                actiond_components = self._convert_tensor_to_dict_actions(actiond_components)
            
            # Evaluate provided actions
            log_probs, entropies = self.multi_discrete_policy.evaluate_actions(
                pre_output_d, actiond_components, layer_number, objective_weights, action_masks
            )
        else:
            # Sample new actions
            actiond_components, log_probs = self.multi_discrete_policy.sample_action(
                pre_output_d, layer_number, objective_weights, action_masks
            )
            _, entropies = self.multi_discrete_policy.evaluate_actions(
                pre_output_d, actiond_components, layer_number, objective_weights, action_masks
            )
        
        # Get continuous action (same as before, but only thickness)
        continuous_output = self._handle_continuous_action(
            pre_output_c, layer_number, actionc, actiond_components, objective_weights
        )
        actionc, log_prob_continuous, c_means, c_std = continuous_output
        
        # Get state value
        state_value = self.value(pre_output_v, layer_number, objective_weights=objective_weights)
        
        # Calculate combined discrete log probability and entropy
        log_prob_discrete = sum(log_probs.values())
        entropy_discrete = sum(entropies.values())
        
        # Calculate continuous entropy (same as before)
        continuous_std = torch.exp(c_std)
        entropy_continuous = torch.sum(0.5 * torch.log(2 * np.pi * np.e * continuous_std**2), dim=-1)
        
        # Format action for environment (convert to expected format)
        formatted_action = self._format_layer_editing_action(actiond_components, actionc)
        
        # Return enhanced output (maintaining compatibility with trainer)
        return (
            formatted_action,                    # Combined action for environment
            actiond_components,                  # Multi-component discrete actions
            actionc,                            # Continuous action
            log_prob_discrete,                  # Combined discrete log prob
            log_prob_continuous,                # Continuous log prob
            discrete_probs_dict,                # Discrete probabilities dict
            c_means,                           # Continuous means
            c_std,                             # Continuous std
            state_value,                       # State value
            entropy_discrete,                  # Combined discrete entropy
            entropy_continuous,                # Continuous entropy
            {}                                 # MoE aux losses (empty for now)
        )
    
    def _convert_tensor_to_dict_actions(self, tensor_actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Convert concatenated discrete action indices back to dictionary format.
        
        The tensor contains concatenated discrete action indices:
        [material_idx, layer_idx, insert_replace_idx]
        
        Args:
            tensor_actions: Concatenated discrete action indices
            
        Returns:
            Dictionary with component actions as indices
        """
        if tensor_actions.dim() == 1 and tensor_actions.numel() == 3:
            # Single action: tensor has 3 elements [material, layer_index, insert_replace]
            return {
                'material': tensor_actions[0].long(),
                'layer_index': tensor_actions[1].long(), 
                'insert_replace': tensor_actions[2].long()
            }
        elif tensor_actions.dim() == 2 and tensor_actions.shape[1] == 3:
            # Batch of actions: shape is (batch_size, 3)
            return {
                'material': tensor_actions[:, 0].long(),
                'layer_index': tensor_actions[:, 1].long(),
                'insert_replace': tensor_actions[:, 2].long()
            }
        else:
            # Handle unexpected tensor shapes - try to extract 3 components
            if tensor_actions.numel() >= 3:
                flat_tensor = tensor_actions.flatten()
                return {
                    'material': flat_tensor[0].long(),
                    'layer_index': flat_tensor[1].long(),
                    'insert_replace': flat_tensor[2].long()
                }
            else:
                # Fallback for unexpected tensor shapes
                batch_size = tensor_actions.shape[0] if tensor_actions.dim() > 0 else 1
                print(f"Warning: unexpected tensor shape {tensor_actions.shape} for discrete actions")
                return {
                    'material': torch.zeros(batch_size, dtype=torch.long),
                    'layer_index': torch.zeros(batch_size, dtype=torch.long),
                    'insert_replace': torch.zeros(batch_size, dtype=torch.long)
                }
    
    def _get_multi_component_discrete_probs(
        self, 
        pre_output: torch.Tensor, 
        layer_number: Optional[torch.Tensor],
        objective_weights: Optional[torch.Tensor],
        action_masks: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Get probability distributions for all discrete action components.
        
        Args:
            pre_output: Pre-processed network input
            layer_number: Current layer number
            objective_weights: Multi-objective weights
            action_masks: Masks for invalid actions
            
        Returns:
            Dictionary of probability distributions for each component
        """
        # For now, use the existing discrete policy and split outputs
        # This is a temporary implementation - ideally we'd modify the network architecture
        
        if self.use_mixture_of_experts:
            all_discrete_probs, discrete_aux = self.policy_discrete(pre_output, objective_weights, layer_number)
        else:
            all_discrete_probs = self.policy_discrete(pre_output, layer_number, objective_weights=objective_weights)
        
        validate_probabilities(all_discrete_probs, "all_discrete_probs")
        
        # Split the output into components (temporary approach)
        # This assumes the discrete policy outputs have been expanded appropriately
        start_idx = 0
        probs_dict = {}
        
        for component, dim in self.discrete_action_components.items():
            end_idx = start_idx + dim
            component_logits = all_discrete_probs[:, start_idx:end_idx]
            
            # Apply component-specific masks if provided
            if action_masks and component in action_masks:
                mask = torch.tensor(action_masks[component], dtype=torch.float32, device=component_logits.device)
                component_logits = component_logits + torch.log(mask + 1e-8)
            
            probs_dict[component] = torch.softmax(component_logits, dim=-1)
            start_idx = end_idx
        
        return probs_dict

    def _handle_continuous_action(
        self, 
        pre_output: torch.Tensor,
        layer_number: Optional[torch.Tensor],
        actionc: Optional[torch.Tensor],
        actiond_components: Dict[str, torch.Tensor],
        objective_weights: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Handle continuous action processing (thickness only).
        
        Args:
            pre_output: Pre-processed network input
            layer_number: Current layer number
            actionc: Continuous action to evaluate (if None, sample)
            actiond_components: Discrete action components (for material conditioning)
            objective_weights: Multi-objective weights
            
        Returns:
            Tuple of (action, log_prob, means, log_std)
        """
        # Use material selection for conditioning continuous policy (if supported)
        material_action = actiond_components['material'] if actiond_components else None
        
        # Ensure material_action has correct shape for network input
        material_input = None
        if material_action is not None:
            if material_action.dim() == 0:
                # Scalar tensor, add batch and feature dimensions
                material_input = material_action.unsqueeze(0).unsqueeze(1)
            elif material_action.dim() == 1:
                # 1D tensor, add feature dimension
                material_input = material_action.unsqueeze(1)
            else:
                # Already has correct dimensions
                material_input = material_action
        
        if self.use_mixture_of_experts:
            continuous_means, continuous_log_std, continuous_aux = self.policy_continuous(
                pre_output, objective_weights, layer_number, material_input
            )
        else:
            continuous_means, continuous_log_std = self.policy_continuous(
                pre_output, layer_number, material_input, 
                objective_weights=objective_weights
            )
        
        continuous_std = torch.exp(continuous_log_std)
        
        # Use truncated normal for thickness sampling
        from coatopt.utils.math_utils import TruncatedNormalDist
        continuous_dist = TruncatedNormalDist(
            continuous_means, continuous_std, self.lower_bound, self.upper_bound
        )
        
        if actionc is None:
            actionc = continuous_dist.sample()
        
        actionc = torch.clamp(actionc, self.lower_bound, self.upper_bound)
        log_prob_continuous = torch.sum(continuous_dist.log_prob(actionc), dim=-1)
        
        return actionc, log_prob_continuous, continuous_means, continuous_log_std
    
    def _format_layer_editing_action(
        self, 
        discrete_components: Dict[str, torch.Tensor], 
        continuous_action: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Format actions for layer editing environment.
        
        Args:
            discrete_components: Dictionary of discrete action components
            continuous_action: Continuous action tensor
            
        Returns:
            Formatted action dictionary for environment
        """
        # Handle both single actions and batch actions
        if isinstance(continuous_action, torch.Tensor):
            if continuous_action.numel() == 1:
                # Single action
                thickness = continuous_action.item()
            else:
                # Batch of actions - take the first one or mean
                thickness = continuous_action[0].item() if continuous_action.dim() > 0 else continuous_action.mean().item()
        else:
            thickness = float(continuous_action)
            
        # Handle discrete components similarly
        def extract_value(tensor_or_value):
            if isinstance(tensor_or_value, torch.Tensor):
                if tensor_or_value.numel() == 1:
                    return tensor_or_value.item()
                else:
                    return tensor_or_value[0].item() if tensor_or_value.dim() > 0 else tensor_or_value[0]
            else:
                return int(tensor_or_value)
        
        return {
            'thickness': thickness,
            'material': extract_value(discrete_components['material']),
            'layer_index': extract_value(discrete_components['layer_index']),
            'insert_replace': extract_value(discrete_components['insert_replace'])
        }
    
    def update_discrete_policy_architecture(self, action_components: Dict[str, int]):
        """
        Update the discrete policy network to handle multi-component outputs.
        
        Args:
            action_components: Dictionary mapping component names to dimensions
        """
        # This would ideally recreate the discrete policy network with expanded outputs
        # For now, we'll work with the existing architecture and post-process outputs
        
        # Store component information
        self.discrete_action_components = action_components
        self.total_discrete_outputs = sum(action_components.values())
        
        # Note: In a full implementation, you'd want to recreate the policy_discrete network
        # with output_dim = total_discrete_outputs, then split the outputs appropriately
        
        print(f"Updated discrete policy for multi-component actions: {action_components}")
        print(f"Total discrete outputs: {self.total_discrete_outputs}")

    def update(self, update_policy=True, update_value=True, weights=None):
        """
        Override base update to handle dictionary discrete actions.
        """
        if len(self.replay_buffer.states) == 0:
            return 0, 0, 0

        # Check if we have dictionary actions that need special handling
        if (len(self.replay_buffer.discrete_actions) > 0 and 
            isinstance(self.replay_buffer.discrete_actions[0], dict)):
            
            # Convert dictionary actions to tensors for base class compatibility
            # Store original actions
            original_discrete_actions = list(self.replay_buffer.discrete_actions)
            
            # Convert to legacy tensor format temporarily
            legacy_actions = []
            for action_dict in original_discrete_actions:
                legacy_action = convert_layer_editing_action_to_legacy(action_dict, self.num_discrete)
                legacy_actions.append(legacy_action.unsqueeze(0))
            
            # Replace buffer contents temporarily
            self.replay_buffer.discrete_actions = legacy_actions
            
            # Call parent update method
            result = super().update(update_policy=update_policy, update_value=update_value)
            
            # Restore original dictionary actions
            self.replay_buffer.discrete_actions = original_discrete_actions
            
            return result
        else:
            # Standard tensor actions, use parent method
            return super().update(update_policy, update_value)


def create_layer_editing_agent(
    state_dim: Union[int, Tuple[int, ...]], 
    num_materials: int,
    max_stack_length: int,
    num_objectives: int = 3,
    **kwargs
) -> LayerEditingPCHPPO:
    """
    Factory function to create a layer editing agent with proper action space configuration.
    
    Args:
        state_dim: Dimensions of state space
        num_materials: Number of available materials
        max_stack_length: Maximum number of layers in stack
        num_objectives: Number of optimization objectives
        **kwargs: Additional arguments for PCHPPO initialization
        
    Returns:
        Configured LayerEditingPCHPPO agent
    """
    # Calculate discrete action dimensions
    discrete_components = {
        'material': num_materials,
        'layer_index': max_stack_length,
        'insert_replace': 2
    }
    total_discrete_dim = sum(discrete_components.values())
    
    # Create agent with expanded discrete action space
    agent = LayerEditingPCHPPO(
        state_dim=state_dim,
        num_discrete=total_discrete_dim,  # Expanded for multi-component
        num_cont=1,                       # Only thickness
        num_objectives=num_objectives,
        max_stack_length=max_stack_length,
        **kwargs
    )
    
    # Configure the multi-component action space
    agent.update_discrete_policy_architecture(discrete_components)
    
    return agent


def convert_legacy_action_to_layer_editing(legacy_action: torch.Tensor, current_stack_length: int) -> Dict[str, Any]:
    """
    Convert legacy action format [thickness, material_one_hot] to layer editing format.
    Useful for compatibility testing.
    
    Args:
        legacy_action: Action in format [thickness, material_one_hot]
        current_stack_length: Current length of coating stack
        
    Returns:
        Action dictionary in layer editing format
    """
    thickness = legacy_action[0].item()
    material_idx = torch.argmax(legacy_action[1:]).item()
    
    # Default behavior: append layer (insert at end)
    return {
        'thickness': thickness,
        'material': material_idx,
        'layer_index': current_stack_length,
        'insert_replace': 1  # Insert
    }


def convert_layer_editing_action_to_legacy(action_dict: Dict[str, Any], num_materials: int) -> torch.Tensor:
    """
    Convert layer editing action back to legacy format for compatibility.
    
    Args:
        action_dict: Action in layer editing format
        num_materials: Number of available materials
        
    Returns:
        Action tensor in legacy format [thickness, material_one_hot]
    """
    # Create legacy action tensor
    legacy_action = torch.zeros(1 + num_materials)
    legacy_action[0] = action_dict['thickness']
    legacy_action[1 + action_dict['material']] = 1.0
    
    return legacy_action
