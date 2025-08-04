"""
Utility functions for action selection in PC-HPPO-OML.
Extracted from pc_hppo_oml.py to improve readability.
"""
from typing import List, Tuple, Union, Optional
import numpy as np
import torch
from torch.nn.utils.rnn import pack_padded_sequence


def prepare_state_input(state: Union[np.ndarray, torch.Tensor], pre_type: str) -> torch.Tensor:
    """
    Prepare state input for neural networks.
    
    Args:
        state: Input state as numpy array or tensor
        pre_type: Type of pre-network ('linear', 'lstm', 'attn')
        
    Returns:
        Prepared state tensor
    """
    if isinstance(state, (np.ndarray, )):
        state_tensor = torch.from_numpy(state).unsqueeze(0).to(torch.float32)
    else:
        state_tensor = state
    
    if pre_type == "linear":
        state_tensor = state_tensor.flatten(1)
    
    return state_tensor


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
            previous_material = (
                torch.argmax(state[i, layer_num - 1, 1:]) 
                if layer_num > 0 
                else substrate_material_index
            )
            valid_indices = _get_valid_material_indices(
                discrete_probs.size(1),
                previous_material,
                air_material_index,
                substrate_material_index,
                ignore_air_option,
                ignore_substrate_option
            )
            action_to_index.append(valid_indices)
    else:
        previous_material = (
            torch.argmax(state[0, layer_number - 1, 1:]) 
            if layer_number > 0 
            else substrate_material_index
        )
        valid_indices = _get_valid_material_indices(
            discrete_probs.size(1),
            previous_material,
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
    valid_indices = valid_indices[valid_indices != previous_material]
    
    return valid_indices


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