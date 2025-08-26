"""
Utility functions for coating environments.
Contains helper functions for state manipulation, visualization, and coating generation.
"""
import numpy as np
import matplotlib.pyplot as plt


def print_state(state):
    """Print the current state for debugging."""
    for i in range(len(state)):
        print(state[i])


def get_optimal_state(max_layers, materials, use_optical_thickness=True, 
                     reverse=False, inds_alternate=[1, 2]):
    """
    Generate optimal alternating coating state.
    
    Args:
        max_layers: Maximum number of layers
        materials: List of material properties
        use_optical_thickness: Whether to use optical thickness
        reverse: Whether to reverse the alternating pattern
        inds_alternate: Indices of materials to alternate between
        
    Returns:
        Optimal state array
    """
    if use_optical_thickness:
        thickness1 = 1/4
        thickness2 = 1/4
    else:
        wavelength = 1064e-9  # LIGO wavelength
        thickness1 = wavelength / (4 * materials[inds_alternate[0]]["n"])
        thickness2 = wavelength / (4 * materials[inds_alternate[1]]["n"])
    
    opt_state = []
    material = inds_alternate[1] if reverse else inds_alternate[0]
    
    for i in range(max_layers):
        current_material = inds_alternate[1] if material == inds_alternate[0] else inds_alternate[0]
        thickness = thickness1 if current_material == inds_alternate[0] else thickness2
        
        layer_state = [0] * (len(materials) + 1)
        layer_state[0] = thickness
        layer_state[current_material + 1] = 1
        
        opt_state.append(layer_state)
        material = current_material
    
    return np.array(opt_state)


def get_optimal_state_2mat(max_layers, materials, use_optical_thickness=True,
                          reverse=False, inds_alternate=[1, 2, 3]):
    """
    Generate optimal state with 2 material types alternating.
    
    Args:
        max_layers: Maximum number of layers
        materials: List of material properties  
        use_optical_thickness: Whether to use optical thickness
        reverse: Whether to reverse pattern
        inds_alternate: Material indices to use
        
    Returns:
        Optimal state array with 2 material types
    """
    if use_optical_thickness:
        thickness1 = 1/4
        thickness2 = 1/4  
        thickness3 = 1/4
    else:
        wavelength = 1064e-9
        thickness1 = wavelength / (4 * materials[inds_alternate[0]]["n"])
        thickness2 = wavelength / (4 * materials[inds_alternate[1]]["n"])
        thickness3 = wavelength / (4 * materials[inds_alternate[2]]["n"])
    
    thicknesses = [thickness1, thickness2, thickness3]
    opt_state = []
    
    for i in range(max_layers):
        # Cycle through materials
        material_idx = inds_alternate[i % len(inds_alternate)]
        thickness = thicknesses[i % len(thicknesses)]
        
        if reverse:
            material_idx = inds_alternate[-(i % len(inds_alternate)) - 1]
        
        layer_state = [0] * (len(materials) + 1)
        layer_state[0] = thickness
        layer_state[material_idx + 1] = 1
        
        opt_state.append(layer_state)
    
    return np.array(opt_state)


def get_air_only_state(max_layers, n_materials, air_material_index=0, n_layers=None):
    """
    Generate state with only air layers.
    
    Args:
        max_layers: Maximum number of layers
        n_materials: Number of materials
        air_material_index: Index of air material
        n_layers: Number of air layers (default: all layers)
        
    Returns:
        Air-only state array
    """
    if n_layers is None:
        n_layers = max_layers
    
    state = np.zeros((max_layers, n_materials + 1))
    
    for i in range(n_layers):
        state[i, 0] = 1  # thickness
        state[i, air_material_index + 1] = 1  # air material
    
    return state


def trim_state(state, air_material_index=0):
    """
    Remove air layers from the end of state.
    
    Args:
        state: State array to trim
        air_material_index: Index of air material
        
    Returns:
        Trimmed state array
    """
    trimmed_state = []
    
    for i in range(len(state)):
        # Check if this layer is air
        materials = state[i, 1:]  # Skip thickness column
        material_idx = np.argmax(materials)
        
        # If it's air and not the first layer, stop here
        if material_idx == air_material_index and i > 0:
            break
            
        trimmed_state.append(state[i])
    
    return np.array(trimmed_state) if trimmed_state else state


def plot_stack(data, title="Coating Stack"):
    """
    Plot the coating stack visualization.
    
    Args:
        data: Coating data to plot
        title: Plot title
    """
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract layers and materials for plotting
        if len(data.shape) == 2:  # State format
            thicknesses = data[:, 0]
            materials = np.argmax(data[:, 1:], axis=1)
        else:
            # Assume it's already in a plottable format
            thicknesses = data
            materials = range(len(data))
        
        # Create bar plot
        colors = plt.cm.tab10(materials)
        ax.bar(range(len(thicknesses)), thicknesses, color=colors)
        
        ax.set_xlabel('Layer Index')
        ax.set_ylabel('Thickness')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error plotting stack: {e}")


def generate_valid_arrangements(materials, num_layers):
    """
    Generate all valid coating arrangements.
    
    Args:
        materials: List of available materials
        num_layers: Number of layers
        
    Returns:
        List of valid arrangements
    """
    def backtrack(arrangement):
        if len(arrangement) == num_layers:
            return [arrangement[:]]
        
        valid_arrangements = []
        for material in materials:
            # Add constraints here if needed (e.g., no adjacent identical materials)
            arrangement.append(material)
            valid_arrangements.extend(backtrack(arrangement))
            arrangement.pop()
        
        return valid_arrangements
    
    return backtrack([])


def state_to_material_list(state, air_material_index=0):
    """
    Convert state representation to list of materials and thicknesses.
    
    Args:
        state: State array
        air_material_index: Index of air material
        
    Returns:
        Tuple of (material_indices, thicknesses)
    """
    material_indices = []
    thicknesses = []
    
    for i in range(len(state)):
        thickness = state[i, 0]
        materials = state[i, 1:]
        
        # Find active material
        material_idx = np.argmax(materials)
        
        # Stop at air layer (unless it's the first layer)
        if material_idx == air_material_index and i > 0:
            break
        
        # Only include layers with positive thickness
        if thickness > 0:
            material_indices.append(material_idx)
            thicknesses.append(thickness)
    
    return material_indices, thicknesses


def material_list_to_state(material_indices, thicknesses, max_layers, n_materials, air_material_index=0):
    """
    Convert material list to state representation.
    
    Args:
        material_indices: List of material indices
        thicknesses: List of thicknesses
        max_layers: Maximum number of layers
        n_materials: Number of materials
        air_material_index: Index of air material
        
    Returns:
        State array
    """
    if len(material_indices) != len(thicknesses):
        raise ValueError("Material indices and thicknesses must have same length")
    
    state = np.zeros((max_layers, n_materials + 1))
    
    # Fill in the specified layers
    for i, (mat_idx, thickness) in enumerate(zip(material_indices, thicknesses)):
        if i >= max_layers:
            break
        state[i, 0] = thickness
        state[i, mat_idx + 1] = 1
    
    # Fill remaining layers with air
    for i in range(len(material_indices), max_layers):
        state[i, 0] = 0
        state[i, air_material_index + 1] = 1
    
    return state


def validate_state(state, min_thickness=1e-9, max_thickness=1e-6):
    """
    Validate that a state is physically reasonable.
    
    Args:
        state: State array to validate
        min_thickness: Minimum allowed thickness
        max_thickness: Maximum allowed thickness
        
    Returns:
        bool: True if state is valid
    """
    try:
        # Check dimensions
        if len(state.shape) != 2:
            return False
        
        # Check thickness values
        thicknesses = state[:, 0]
        if np.any(thicknesses < 0):
            return False
        if np.any(thicknesses > max_thickness):
            return False
        
        # Check that each layer has exactly one material selected
        materials = state[:, 1:]
        material_sums = np.sum(materials, axis=1)
        if not np.allclose(material_sums, 1.0):
            return False
        
        return True
        
    except Exception:
        return False
