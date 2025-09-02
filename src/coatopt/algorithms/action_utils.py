"""
Utility functions for action selection in PC-HPPO-OML.
Extracted from pc_hppo_oml.py to improve readability.
"""
from typing import List, Tuple, Union, Optional
import numpy as np
import torch
from torch.nn.utils.rnn import pack_padded_sequence


def prepare_state_input(state: Union[np.ndarray, torch.Tensor, dict], pre_type: str) -> torch.Tensor:
    """
    Prepare state input for neural networks.
    
    Args:
        state: Input state as numpy array, tensor, or enhanced observation dictionary
        pre_type: Type of pre-network ('linear', 'lstm', 'attn')
        
    Returns:
        Prepared state tensor with concatenated information
    """
    if isinstance(state, dict):
        # Handle enhanced observation dictionary format
        state_tensor = _process_enhanced_observation(state)
    elif isinstance(state, (np.ndarray, )):
        state_tensor = torch.from_numpy(state).unsqueeze(0).to(torch.float32)
    else:
        state_tensor = state
    
    if pre_type == "linear":
        state_tensor = state_tensor.flatten(1)
    
    return state_tensor


def _process_enhanced_observation(obs_dict: dict) -> torch.Tensor:
    """
    Process enhanced observation dictionary into tensor format.
    
    Args:
        obs_dict: Dictionary containing layer_stack and optional electric field info
        
    Returns:
        Concatenated tensor with all observation information
    """
    # Extract layer stack information
    layer_stack = obs_dict['layer_stack']
    
    # Convert layer stack to tensor format [n_layers, 4] -> [thickness, material_index, n, k]
    layer_data = []
    for layer in layer_stack:
        layer_data.append([
            layer['thickness'],
            layer['material_index'], 
            layer['n'],
            layer['k']
        ])
    
    layer_tensor = torch.tensor(layer_data, dtype=torch.float32).unsqueeze(0)  # [1, n_layers, 4]
    
    # Check if electric field information is available
    if 'electric_field' in obs_dict and obs_dict['electric_field'] is not None:
        # Convert electric field data to tensors
        efield_tensor = torch.tensor(obs_dict['electric_field'], dtype=torch.float32).unsqueeze(0)  # [1, field_points]
        grad_tensor = torch.tensor(obs_dict['field_gradients'], dtype=torch.float32).unsqueeze(0)   # [1, field_points]
        metrics_tensor = torch.tensor(obs_dict['cumulative_metrics'], dtype=torch.float32).unsqueeze(0) # [1, 3]
        
        # Concatenate electric field information as additional "layers" or features
        # Option 1: Add as additional features to each layer (broadcast)
        n_layers = layer_tensor.shape[1]
        field_points = efield_tensor.shape[1]
        
        # Reshape field info to match layer dimensions by interpolation or padding
        if field_points != n_layers:
            # Simple approach: interpolate field to match number of layers
            efield_interp = torch.nn.functional.interpolate(
                efield_tensor.unsqueeze(1), size=n_layers, mode='linear', align_corners=False
            ).squeeze(1)  # [1, n_layers]
            grad_interp = torch.nn.functional.interpolate(
                grad_tensor.unsqueeze(1), size=n_layers, mode='linear', align_corners=False  
            ).squeeze(1)  # [1, n_layers]
        else:
            efield_interp = efield_tensor  # [1, n_layers]
            grad_interp = grad_tensor      # [1, n_layers]
        
        # Add field information as additional features to each layer
        efield_expanded = efield_interp.unsqueeze(2)  # [1, n_layers, 1]
        grad_expanded = grad_interp.unsqueeze(2)      # [1, n_layers, 1]
        
        # Broadcast metrics to all layers
        metrics_expanded = metrics_tensor.unsqueeze(1).expand(-1, n_layers, -1)  # [1, n_layers, 3]
        
        # Concatenate: [thickness, material_index, n, k, efield, grad, R, A, TN]
        enhanced_tensor = torch.cat([
            layer_tensor,        # [1, n_layers, 4] 
            efield_expanded,     # [1, n_layers, 1]
            grad_expanded,       # [1, n_layers, 1] 
            metrics_expanded     # [1, n_layers, 3]
        ], dim=2)  # Final: [1, n_layers, 9]
        
        return enhanced_tensor
    else:
        # No electric field info, return just the base 4 features: thickness, material_index, n, k
        # This maintains consistency with the enhanced format but without field data
        return layer_tensor


def prepare_layer_number(
    layer_number: Optional[Union[np.ndarray, torch.Tensor]]
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    """
    Prepare layer number input and create validation tensor.
    
    Args:
        layer_number: Layer number input
        
    Returns:
        Tuple of (processed_layer_number, validated_layer_numbers)
    """
    if layer_number is None:
        return None, torch.tensor([1])
    
    if isinstance(layer_number, (np.ndarray, )):
        layer_number = torch.from_numpy(layer_number).flatten().unsqueeze(0)
    
    layer_number = layer_number.to(torch.int)
    
    # Create validated layer numbers (ensure no zeros for packing)
    layer_numbers_validated = layer_number.clone().detach().view(-1)
    layer_numbers_validated[layer_numbers_validated == 0] = 1
    
    return layer_number, layer_numbers_validated


def create_material_mask(
    discrete_probs: torch.Tensor,
    state: torch.Tensor,
    layer_number: Optional[torch.Tensor],
    substrate_material_index: int,
    air_material_index: int,
    ignore_air_option: bool,
    ignore_substrate_option: bool
) -> torch.Tensor:
    """
    Create mask to prevent selecting invalid materials.
    
    Args:
        discrete_probs: Discrete action probabilities
        state: Current state
        layer_number: Current layer number
        substrate_material_index: Index of substrate material
        air_material_index: Index of air material
        ignore_air_option: Whether to ignore air material
        ignore_substrate_option: Whether to ignore substrate material
        
    Returns:
        Masked probabilities tensor
    """
    if layer_number is None:
        return discrete_probs
    
    # Find previous materials to avoid repetition
    action_to_index = []
    if isinstance(layer_number, torch.Tensor) and layer_number.ndim > 0:
        for i, layer_num in enumerate(layer_number.squeeze(0)):
            if layer_num > 0:
                # Get previous layer's material (skip thickness dimension at index 0)
                previous_material_onehot = state[i, layer_num - 1, 1:]
                previous_material = torch.argmax(previous_material_onehot)
                
                # Check if previous material is valid
                if torch.sum(previous_material_onehot) == 0:
                    # No material set in previous layer, use substrate
                    previous_material = substrate_material_index
                    print(f"Warning: No material set in previous layer {layer_num-1}, using substrate")
            else:
                previous_material = substrate_material_index
            
            valid_indices = _get_valid_material_indices(
                discrete_probs.size(1),
                previous_material.item() if isinstance(previous_material, torch.Tensor) else previous_material,
                air_material_index,
                substrate_material_index,
                ignore_air_option,
                ignore_substrate_option
            )
            action_to_index.append(valid_indices)
    else:
        if layer_number > 0:
            # Get previous layer's material (skip thickness dimension at index 0)
            previous_material_onehot = state[0, layer_number - 1, 1:]
            previous_material = torch.argmax(previous_material_onehot)
            
            # Check if previous material is valid
            if torch.sum(previous_material_onehot) == 0:
                # No material set in previous layer, use substrate
                previous_material = substrate_material_index
                print(f"Warning: No material set in previous layer {layer_number-1}, using substrate")
        else:
            previous_material = substrate_material_index
            
        valid_indices = _get_valid_material_indices(
            discrete_probs.size(1),
            previous_material.item() if isinstance(previous_material, torch.Tensor) else previous_material,
            air_material_index,
            substrate_material_index,
            ignore_air_option,
            ignore_substrate_option
        )
        action_to_index.append(valid_indices)

    # Create mask based on valid indices
    mask = torch.zeros_like(discrete_probs)
    for i, indices in enumerate(action_to_index):
        mask[i, indices] = 1

    # Apply mask to probabilities
    masked_probs = discrete_probs * mask
    
    # Check if mask resulted in all zeros (no valid materials)
    # This can happen if we only have 2-3 materials and exclude previous + air/substrate
    if torch.all(mask.sum(dim=-1) == 0):
        print("Warning: Material mask resulted in no valid options. Falling back to original probabilities.")
        masked_probs = discrete_probs
    elif torch.any(mask.sum(dim=-1) == 0):
        # Handle batch case where some samples have no valid options
        zero_mask_indices = mask.sum(dim=-1) == 0
        masked_probs[zero_mask_indices] = discrete_probs[zero_mask_indices]
        print(f"Warning: {zero_mask_indices.sum().item()} samples had no valid material options. Using original probabilities for those.")
    else:
        pass
    if len(masked_probs.size()) == 1:
        masked_probs = masked_probs.unsqueeze(0)
    
    return masked_probs


def _get_valid_material_indices(
    n_materials: int,
    previous_material: int,
    air_material_index: int,
    substrate_material_index: int,
    ignore_air_option: bool,
    ignore_substrate_option: bool
) -> torch.Tensor:
    """
    Get valid material indices based on constraints.
    
    Args:
        n_materials: Total number of materials
        previous_material: Previously used material to avoid
        air_material_index: Index of air material
        substrate_material_index: Index of substrate material
        ignore_air_option: Whether to ignore air
        ignore_substrate_option: Whether to ignore substrate
        
    Returns:
        Tensor of valid material indices
    """
    valid_indices = torch.arange(n_materials)
    
    # Remove air if ignored
    if ignore_air_option:
        valid_indices = valid_indices[valid_indices != air_material_index]
    
    # Remove substrate if ignored
    if ignore_substrate_option:
        valid_indices = valid_indices[valid_indices != substrate_material_index]
    
    # Remove previous material to avoid repetition
    temp_valid_indices = valid_indices[valid_indices != previous_material]
    
    # Safety check: ensure we have at least one valid material
    if len(temp_valid_indices) == 0:
        # If removing previous material leaves no options, allow previous material
        # This is better than having no valid actions
        print(f"Warning: Removing previous material {previous_material} would leave no valid options from {valid_indices.tolist()}.")
        print(f"  Air index: {air_material_index}, Substrate index: {substrate_material_index}")
        print(f"  ignore_air_option: {ignore_air_option}, ignore_substrate_option: {ignore_substrate_option}")
        print(f"Allowing repetition.")
        return valid_indices
    else:
        return temp_valid_indices


def pack_state_sequence(state: torch.Tensor, layer_numbers_validated: torch.Tensor):
    """
    Pack state sequence for RNN processing.
    
    Args:
        state: State tensor
        layer_numbers_validated: Validated layer numbers
        
    Returns:
        Packed sequence
    """
    return pack_padded_sequence(
        state, 
        lengths=layer_numbers_validated, 
        batch_first=True, 
        enforce_sorted=False
    )


def validate_probabilities(probs: torch.Tensor, name: str) -> None:
    """
    Validate that probabilities don't contain NaN values.
    
    Args:
        probs: Probability tensor to validate
        name: Name for error reporting
        
    Raises:
        Exception: If NaN values found
    """
    if torch.isnan(probs).any():
        raise Exception(f"NaN found in {name}")


def format_action_output(
    discrete_action: torch.Tensor,
    continuous_action: torch.Tensor
) -> torch.Tensor:
    """
    Format discrete and continuous actions into single output tensor.
    
    Args:
        discrete_action: Discrete action component
        continuous_action: Continuous action component
        
    Returns:
        Combined action tensor
    """
    return torch.cat([
        discrete_action.detach().unsqueeze(0).T, 
        continuous_action
    ], dim=-1)[0]