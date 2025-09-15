"""
Enhanced action_utils.py with CoatingState support.

CoatingState is now the primary state representation for better state 
management and debugging.
"""
from typing import List, Tuple, Union, Optional
import numpy as np
import torch
from torch.nn.utils.rnn import pack_padded_sequence
from coatopt.environments.core.state import CoatingState


def create_material_mask_from_coating_state(
    discrete_probs: torch.Tensor,
    coating_state: 'CoatingState',
    layer_number: Optional[torch.Tensor],
    ignore_air_option: bool = False,
    ignore_substrate_option: bool = False
) -> torch.Tensor:
    """
    Create material mask using CoatingState interface.
    
    This version eliminates indexing errors by using CoatingState methods.
    
    Args:
        discrete_probs: Discrete action probabilities
        coating_state: CoatingState object
        layer_number: Current layer number
        ignore_air_option: Whether to ignore air material
        ignore_substrate_option: Whether to ignore substrate material
        
    Returns:
        Masked probabilities tensor
    """
    if layer_number is None:
        return discrete_probs
    
    # Handle batch processing
    if isinstance(layer_number, torch.Tensor) and layer_number.ndim > 0:
        # Batch processing - create mask for each sample
        batch_size = layer_number.size(0)
        mask = torch.ones_like(discrete_probs)
        
        for i in range(batch_size):
            layer_num = layer_number[i].item()
            
            # Get previous material using CoatingState method
            previous_material = coating_state.get_previous_material(layer_num)
            
            # Get valid materials (exclude previous material)
            valid_indices = _get_valid_material_indices(
                discrete_probs.size(1),
                previous_material,
                coating_state.air_material_index,
                coating_state.substrate_material_index,
                ignore_air_option,
                ignore_substrate_option
            )
            
            # Set mask for this sample
            mask[i, :] = 0  # Start with all invalid
            mask[i, valid_indices] = 1  # Set valid materials
    else:
        # Single sample processing
        layer_num = layer_number.item() if isinstance(layer_number, torch.Tensor) else layer_number
        
        # Get previous material using CoatingState method
        previous_material = coating_state.get_previous_material(layer_num)
        
        # Get valid materials (exclude previous material)
        valid_indices = _get_valid_material_indices(
            discrete_probs.size(1),
            previous_material,
            coating_state.air_material_index,
            coating_state.substrate_material_index,
            ignore_air_option,
            ignore_substrate_option
        )
        
        # Create mask
        mask = torch.zeros_like(discrete_probs)
        mask[:, valid_indices] = 1

    # Apply mask to probabilities
    masked_probs = discrete_probs * mask
    
    # Check if mask resulted in all zeros (no valid materials)
    if torch.sum(mask) == 0:
        print(f"WARNING: Material mask resulted in no valid materials! Layer {layer_num}, Previous material: {previous_material}")
        # Fallback: allow all materials except previous
        mask = torch.ones_like(discrete_probs)
        if layer_num > 0:
            mask[:, previous_material] = 0
        masked_probs = discrete_probs * mask
    
    # Renormalise probabilities
    masked_probs = masked_probs / (torch.sum(masked_probs, dim=-1, keepdim=True) + 1e-10)
    
    return masked_probs


def _get_valid_material_indices(
    n_materials: int,
    previous_material: int,
    air_material_index: int,
    substrate_material_index: int,
    ignore_air_option: bool,
    ignore_substrate_option: bool
) -> List[int]:
    """
    Get list of valid material indices based on constraints.
    
    Args:
        n_materials: Total number of materials
        previous_material: Previous material index to exclude
        air_material_index: Air material index
        substrate_material_index: Substrate material index
        ignore_air_option: Whether to ignore air material
        ignore_substrate_option: Whether to ignore substrate material
        
    Returns:
        List of valid material indices
    """
    valid_indices = []
    for material_idx in range(n_materials):
        # Skip previous material to prevent consecutive identical materials
        if material_idx == previous_material:
            continue
            
        # Skip air if ignore_air_option is True
        if ignore_air_option and material_idx == air_material_index:
            continue
            
        # Skip substrate if ignore_substrate_option is True
        if ignore_substrate_option and material_idx == substrate_material_index:
            continue
            
        valid_indices.append(material_idx)
    
    return valid_indices


def state_to_coating_state(state: Union[torch.Tensor, np.ndarray, dict], 
                          max_layers: int = 20, n_materials: int = 4,
                          air_material_index: int = 0, 
                          substrate_material_index: int = 1) -> 'CoatingState':
    """
    Convert various state formats to CoatingState object.
    
    Args:
        state: State in various formats
        max_layers: Maximum number of layers
        n_materials: Number of materials
        air_material_index: Air material index
        substrate_material_index: Substrate material index
        
    Returns:
        CoatingState object
    """
    return CoatingState.load_from_array(
        state, max_layers, n_materials, air_material_index, substrate_material_index
    )


def debug_state_consistency(state: Union[torch.Tensor, 'CoatingState'], 
                           layer_number: Optional[int] = None) -> None:
    """
    Debug helper to check state consistency.
    
    Args:
        state: State to check
        layer_number: Current layer number for context
    """
    if isinstance(state, CoatingState):
        print(f"\n=== CoatingState Debug (Layer {layer_number}) ===")
        print(state)
        
        # Validate state
        issues = state.validate()
        if issues:
            print("❌ Validation Issues:")
            for issue in issues:
                print(f"  {issue}")
        else:
            print("✅ State validation passed")
            
        # Show material sequence
        materials = state.get_material_sequence()
        print(f"Material sequence: {materials}")
        
        if layer_number is not None and layer_number > 0:
            prev_mat = state.get_previous_material(layer_number)
            print(f"Previous material for layer {layer_number}: {prev_mat}")
        
        print("=" * 50)
    else:
        print(f"\n=== Tensor State Debug (Layer {layer_number}) ===")
        if isinstance(state, torch.Tensor):
            print(f"Shape: {state.shape}")
            # Extract materials from tensor for debugging
            if state.dim() >= 3 and layer_number is not None and layer_number > 0:
                prev_material_onehot = state[0, layer_number - 1, 1:]
                prev_material = torch.argmax(prev_material_onehot).item()
                print(f"Previous material: {prev_material}")
        print("=" * 50)


# Import remaining functions from original action_utils
def prepare_layer_number(layer_number: Optional[Union[np.ndarray, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare layer number for neural networks.
    
    Args:
        layer_number: Layer number as numpy array or tensor
        
    Returns:
        Tuple of (layer_number, layer_numbers_validated)
    """
    if layer_number is False or layer_number is None:
        return False, False
    
    if not isinstance(layer_number, torch.Tensor):
        layer_number = torch.from_numpy(layer_number.astype(np.float32))
    
    layer_number = layer_number.to(torch.float32)
    layer_numbers_validated = layer_number.clone()
    layer_numbers_validated[layer_numbers_validated == 0] = 1
    
    return layer_number, layer_numbers_validated


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
            layer.get('thickness', 0.0),
            layer.get('material_index', 0),
            layer.get('n', 1.0),  # Refractive index
            layer.get('k', 0.0)   # Extinction coefficient
        ])
    
    layer_tensor = torch.tensor(layer_data, dtype=torch.float32)
    
    # Add electric field information if present
    if 'electric_field_data' in obs_dict:
        ef_data = obs_dict['electric_field_data']
        # Process electric field data - implementation depends on specific format
        # For now, placeholder implementation
        ef_tensor = torch.tensor([ef_data.get('intensity', 0.0)], dtype=torch.float32)
        
        # Concatenate layer and field information
        # This may need adjustment based on exact format requirements
        combined_tensor = torch.cat([layer_tensor.flatten(), ef_tensor])
        return combined_tensor.unsqueeze(0)
    else:
        return layer_tensor.unsqueeze(0)


def pack_state_sequence(state_tensor: torch.Tensor, layer_numbers_validated: torch.Tensor) -> torch.Tensor:
    """
    Pack state sequence for LSTM processing.
    
    Args:
        state_tensor: State tensor
        layer_numbers_validated: Validated layer numbers
        
    Returns:
        Packed sequence tensor
    """
    if layer_numbers_validated is False:
        return state_tensor
    
    batch_size = state_tensor.size(0)
    sequence_lengths = layer_numbers_validated.squeeze().long()
    
    # Handle single batch case
    if sequence_lengths.dim() == 0:
        sequence_lengths = sequence_lengths.unsqueeze(0)
    
    # Ensure we don't exceed actual sequence length
    max_len = state_tensor.size(1)
    sequence_lengths = torch.clamp(sequence_lengths, 1, max_len)
    
    # Pack sequence
    packed_input = pack_padded_sequence(
        state_tensor, sequence_lengths.cpu(), 
        batch_first=True, enforce_sorted=False
    )
    
    return packed_input


def validate_probabilities(probabilities: torch.Tensor, name: str = "probabilities") -> None:
    """
    Validate probability tensor for NaN or invalid values.
    
    Args:
        probabilities: Probability tensor to validate
        name: Name for error messages
        
    Raises:
        ValueError: If probabilities are invalid
    """
    if torch.isnan(probabilities).any():
        raise ValueError(f"{name} contains NaN values")
    
    if torch.isinf(probabilities).any():
        raise ValueError(f"{name} contains infinite values")
    
    if (probabilities < 0).any():
        raise ValueError(f"{name} contains negative values")


def format_action_output(discrete_action: torch.Tensor, continuous_action: torch.Tensor) -> list:
    """
    Format action output for environment.
    
    Args:
        discrete_action: Discrete action tensor
        continuous_action: Continuous action tensor
        
    Returns:
        List with [discrete_action_value, continuous_action_values...]
    """
    # Convert to simple list format that the environment expects
    discrete_val = discrete_action.item() if discrete_action.numel() == 1 else discrete_action
    
    if continuous_action.numel() == 1:
        continuous_vals = [continuous_action.item()]
    else:
        continuous_vals = continuous_action.squeeze().tolist() if continuous_action.dim() > 1 else continuous_action.tolist()
    
    # Return as simple list: [discrete, continuous1, continuous2, ...]
    return [discrete_val] + continuous_vals
