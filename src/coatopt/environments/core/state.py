#!/usr/bin/env python3
"""
CoatingState - Unified state representation for coating optimization.

Provides consistent interface for state manipulation and eliminates indexing errors.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch


class MaterialIndex(Enum):
    """Enum for material indices to avoid magic numbers"""

    AIR = 0
    SILICA = 1  # SiO2 - substrate
    TITANIA_TANTALUM = 2  # Ti:Ta2O5
    AMORPHOUS_SILICON = 3  # aSi


@dataclass
class LayerData:
    """Represents a single layer in the coating stack"""

    thickness: float
    material_index: int

    def is_active(self, min_thickness: float = 1e-9) -> bool:
        """Check if layer has significant thickness"""
        return self.thickness > min_thickness

    def __repr__(self) -> str:
        return (
            f"LayerData(thickness={self.thickness:.6f}, material={self.material_index})"
        )


class CoatingState:
    """
    Unified state representation for coating optimization.

    Handles conversion between different representations and provides
    consistent interface for state manipulation. Eliminates tensor indexing
    errors and provides validation capabilities.
    """

    def __init__(
        self,
        max_layers: int,
        n_materials: int,
        air_material_index: int = 0,
        substrate_material_index: int = 1,
        include_electric_field: bool = False,
        materials: Optional[List[Dict]] = None,
    ):
        """
        Initialize CoatingState.

        Args:
            max_layers: Maximum number of layers in the coating stack
            n_materials: Total number of available materials
            air_material_index: Index of air material (default: 0)
            substrate_material_index: Index of substrate material (default: 1)
            include_electric_field: Whether to include electric field information
            materials: Material properties for enhanced observations (n, k values)
        """
        self.max_layers = max_layers
        self.n_materials = n_materials
        self.air_material_index = air_material_index
        self.substrate_material_index = substrate_material_index
        self.include_electric_field = include_electric_field
        self.materials = materials or []

        # Internal storage as tensor: [thickness, material_onehot...]
        # Shape: (max_layers, n_materials + 1)
        # Column 0: thickness, Columns 1+: one-hot material encoding
        self._state_tensor = torch.zeros(max_layers, n_materials + 1)

        # Enhanced observation data (computed on demand)
        self._enhanced_observation_cache = None
        self._enhanced_observation_dirty = True

        # Debug tracking
        self._debug_enabled = False

        # Initialize all layers as air with zero thickness
        for i in range(max_layers):
            self.set_layer(i, 0.0, air_material_index)

    @classmethod
    def from_tensor(
        cls,
        state_tensor: torch.Tensor,
        air_material_index: int = 0,
        substrate_material_index: int = 1,
    ) -> "CoatingState":
        """
        Create CoatingState from existing tensor.

        Args:
            state_tensor: Tensor with shape (max_layers, n_materials + 1)
            air_material_index: Index of air material
            substrate_material_index: Index of substrate material

        Returns:
            New CoatingState instance
        """
        max_layers, n_materials_plus_one = state_tensor.shape
        n_materials = n_materials_plus_one - 1

        state = cls(
            max_layers, n_materials, air_material_index, substrate_material_index
        )
        state._state_tensor = state_tensor.clone()
        return state

    @classmethod
    def from_layers(
        cls,
        layers: List[LayerData],
        max_layers: int,
        n_materials: int,
        air_material_index: int = 0,
        substrate_material_index: int = 1,
    ) -> "CoatingState":
        """
        Create CoatingState from list of layers.

        Args:
            layers: List of LayerData objects
            max_layers: Maximum number of layers
            n_materials: Total number of materials
            air_material_index: Index of air material
            substrate_material_index: Index of substrate material

        Returns:
            New CoatingState instance
        """
        state = cls(
            max_layers, n_materials, air_material_index, substrate_material_index
        )

        for i, layer in enumerate(layers[:max_layers]):
            state.set_layer(i, layer.thickness, layer.material_index)

        # Fill remaining layers with air
        for i in range(len(layers), max_layers):
            state.set_layer(i, 0.0, air_material_index)

        return state

    @classmethod
    def load_from_array(
        cls,
        data: Union[np.ndarray, torch.Tensor, dict, List],
        max_layers: Optional[int] = None,
        n_materials: Optional[int] = None,
        air_material_index: int = 0,
        substrate_material_index: int = 1,
    ) -> "CoatingState":
        """
        Universal loader for CoatingState from various data formats.

        Handles numpy arrays, tensors, dictionaries, and lists with automatic
        format detection and conversion.

        Args:
            data: Input data (array, tensor, dict, or list)
            max_layers: Maximum layers (auto-detected if None)
            n_materials: Number of materials (auto-detected if None)
            air_material_index: Air material index
            substrate_material_index: Substrate material index

        Returns:
            New CoatingState instance
        """
        if isinstance(data, dict):
            # Handle dict format (e.g., from serialization)
            if "tensor" in data:
                return cls.from_dict(data)
            elif "state" in data:
                return cls.load_from_array(
                    data["state"],
                    max_layers,
                    n_materials,
                    air_material_index,
                    substrate_material_index,
                )
            else:
                raise ValueError("Dictionary must contain 'tensor' or 'state' key")

        elif isinstance(data, list):
            # Handle list of LayerData or list of arrays
            if data and isinstance(data[0], LayerData):
                if max_layers is None or n_materials is None:
                    raise ValueError(
                        "Must specify max_layers and n_materials for LayerData list"
                    )
                return cls.from_layers(
                    data,
                    max_layers,
                    n_materials,
                    air_material_index,
                    substrate_material_index,
                )
            else:
                # Convert list to numpy array and process
                data = np.array(data)

        # Convert to tensor format
        if isinstance(data, np.ndarray):
            state_tensor = torch.from_numpy(data).float()
        elif isinstance(data, torch.Tensor):
            state_tensor = data.float()
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

        # Handle different tensor shapes
        if state_tensor.dim() == 1:
            # Flat tensor - need to reshape
            if max_layers is None or n_materials is None:
                raise ValueError(
                    "Must specify max_layers and n_materials for 1D tensor"
                )
            expected_size = max_layers * (n_materials + 1)
            if state_tensor.size(0) != expected_size:
                raise ValueError(
                    f"1D tensor size {state_tensor.size(0)} doesn't match expected {expected_size}"
                )
            state_tensor = state_tensor.view(max_layers, n_materials + 1)
        elif state_tensor.dim() == 2:
            # 2D tensor - extract dimensions
            max_layers, n_materials_plus_one = state_tensor.shape
            n_materials = n_materials_plus_one - 1
        else:
            raise ValueError(f"Unsupported tensor shape: {state_tensor.shape}")

        return cls.from_tensor(
            state_tensor, air_material_index, substrate_material_index
        )

    def onehot_to_material_index(self, layer_idx: int) -> int:
        """
        Convert one-hot encoding to material index for a specific layer.

        Args:
            layer_idx: Layer index

        Returns:
            Material index (numeric encoding)
        """
        if layer_idx >= self.max_layers or layer_idx < 0:
            raise IndexError(
                f"Layer index {layer_idx} out of bounds [0, {self.max_layers})"
            )

        material_onehot = self._state_tensor[layer_idx, 1:]
        return torch.argmax(material_onehot).item()

    def material_index_to_onehot(self, material_idx: int) -> torch.Tensor:
        """
        Convert material index to one-hot encoding.

        Args:
            material_idx: Material index (numeric)

        Returns:
            One-hot encoded tensor
        """
        if material_idx >= self.n_materials or material_idx < 0:
            raise IndexError(
                f"Material index {material_idx} out of bounds [0, {self.n_materials})"
            )

        onehot = torch.zeros(self.n_materials)
        onehot[material_idx] = 1.0
        return onehot

    def get_state_shape(self) -> Tuple[int, int]:
        """
        Get shape of the underlying state tensor.

        Returns:
            Tuple of (max_layers, n_features) where n_features = 1 + n_materials
        """
        return self._state_tensor.shape

    def get_material_indices_sequence(
        self, max_layers: Optional[int] = None, active_only: bool = True
    ) -> List[int]:
        """
        Get sequence of material indices (numeric encoding).

        Args:
            max_layers: Maximum number of layers to include (None for all)
            active_only: If True, only return active layers (thickness > 0)

        Returns:
            List of material indices (numeric encoding for material properties)
        """
        if max_layers is None:
            max_layers = (
                self.get_num_active_layers() if active_only else self.max_layers
            )
        else:
            max_layers = min(max_layers, self.max_layers)

        indices = []
        for i in range(max_layers):
            if active_only and self._state_tensor[i, 0] <= 0:
                continue

            material_idx = self.onehot_to_material_index(i)
            if not active_only or material_idx != self.air_material_index:
                indices.append(material_idx)

        return indices

    @classmethod
    def from_observation(
        cls,
        observation: Union[Dict, np.ndarray, torch.Tensor],
        max_layers: Optional[int] = None,
        n_materials: Optional[int] = None,
        air_material_index: int = 0,
        substrate_material_index: int = 1,
    ) -> "CoatingState":
        """
        Create CoatingState from environment observation.

        Handles various observation formats (dict, array, tensor).

        Args:
            observation: Environment observation
            max_layers: Maximum layers (inferred if None)
            n_materials: Number of materials (inferred if None)
            air_material_index: Air material index
            substrate_material_index: Substrate material index

        Returns:
            New CoatingState instance
        """
        if isinstance(observation, dict):
            if "state" in observation:
                state_data = observation["state"]
            else:
                raise ValueError("Dictionary observation must contain 'state' key")
        else:
            state_data = observation

        # Convert to tensor if needed
        if isinstance(state_data, np.ndarray):
            state_tensor = torch.from_numpy(state_data).float()
        elif isinstance(state_data, torch.Tensor):
            state_tensor = state_data.float()
        else:
            raise ValueError(f"Unsupported observation type: {type(state_data)}")

        # Handle different tensor shapes
        if state_tensor.dim() == 1:
            # Flatten tensor - need to infer structure
            if max_layers is None or n_materials is None:
                raise ValueError(
                    "Must specify max_layers and n_materials for 1D tensor"
                )
            expected_size = max_layers * (n_materials + 1)
            if state_tensor.size(0) != expected_size:
                raise ValueError(
                    f"1D tensor size {state_tensor.size(0)} doesn't match expected {expected_size}"
                )
            state_tensor = state_tensor.view(max_layers, n_materials + 1)
        elif state_tensor.dim() == 2:
            max_layers, n_materials_plus_one = state_tensor.shape
            n_materials = n_materials_plus_one - 1
        else:
            raise ValueError(f"Unsupported tensor shape: {state_tensor.shape}")

        return cls.from_tensor(
            state_tensor, air_material_index, substrate_material_index
        )

    def enable_debug(self, enabled: bool = True):
        """Enable/disable debug mode for detailed logging"""
        self._debug_enabled = enabled

    def get_tensor(self) -> torch.Tensor:
        """Get the underlying tensor representation (immutable copy)"""
        return self._state_tensor.clone()

    def get_array(self) -> np.ndarray:
        """Get the underlying state as numpy array (for plotting, saving, etc.)"""
        return self._state_tensor.clone().numpy()

    def get_tensor_view(self) -> torch.Tensor:
        """Get view of underlying tensor (mutable, use with caution)"""
        return self._state_tensor

    def is_valid(self) -> bool:
        """Check if state contains valid (finite) values"""
        return not (
            torch.any(torch.isinf(self._state_tensor))
            or torch.any(torch.isnan(self._state_tensor))
        )

    def get_layer(self, layer_idx: int) -> LayerData:
        """
        Get layer data for specific layer index.

        Args:
            layer_idx: Layer index (0-based)

        Returns:
            LayerData for the specified layer

        Raises:
            IndexError: If layer_idx is out of bounds
        """
        # Ensure layer_idx is an integer (handle tensor inputs)
        if isinstance(layer_idx, torch.Tensor):
            layer_idx = layer_idx.item()
        layer_idx = int(layer_idx)

        if layer_idx >= self.max_layers or layer_idx < 0:
            raise IndexError(
                f"Layer index {layer_idx} out of bounds [0, {self.max_layers})"
            )

        thickness = self._state_tensor[layer_idx, 0].item()
        material_onehot = self._state_tensor[layer_idx, 1:]
        material_index = torch.argmax(material_onehot).item()

        return LayerData(thickness, material_index)

    def get_num_active_layers(self) -> int:
        """
        Get number of layers with non-zero thickness.

        Returns:
            Number of active layers
        """
        thicknesses = self._state_tensor[:, 0]
        return int(torch.sum(thicknesses > 0).item())

    def get_previous_material(self, layer_idx: int) -> int:
        """
        Get the material index of the previous layer.

        For layer 0, returns substrate material index.
        For other layers, returns material of previous layer.

        Args:
            layer_idx: Current layer index

        Returns:
            Material index of previous layer or substrate
        """
        # Ensure layer_idx is an integer (handle tensor inputs)
        if isinstance(layer_idx, torch.Tensor):
            layer_idx = layer_idx.item()
        layer_idx = int(layer_idx)

        if layer_idx <= 0:
            return self.substrate_material_index

        # Direct tensor access for efficiency
        material_onehot = self._state_tensor[layer_idx - 1, 1:]
        return torch.argmax(material_onehot).item()

    def get_material_at_layer(self, layer_idx: int) -> int:
        """
        Get material index for specific layer.

        Args:
            layer_idx: Layer index

        Returns:
            Material index for the layer
        """
        # Ensure layer_idx is an integer (handle tensor inputs)
        if isinstance(layer_idx, torch.Tensor):
            layer_idx = layer_idx.item()
        layer_idx = int(layer_idx)

        if layer_idx >= self.max_layers or layer_idx < 0:
            raise IndexError(
                f"Layer index {layer_idx} out of bounds [0, {self.max_layers})"
            )

        material_onehot = self._state_tensor[layer_idx, 1:]
        return torch.argmax(material_onehot).item()

    def get_thickness_at_layer(self, layer_idx: int) -> float:
        """
        Get thickness for specific layer.

        Args:
            layer_idx: Layer index

        Returns:
            Thickness for the layer
        """
        # Ensure layer_idx is an integer (handle tensor inputs)
        if isinstance(layer_idx, torch.Tensor):
            layer_idx = layer_idx.item()
        layer_idx = int(layer_idx)

        if layer_idx >= self.max_layers or layer_idx < 0:
            raise IndexError(
                f"Layer index {layer_idx} out of bounds [0, {self.max_layers})"
            )

        return self._state_tensor[layer_idx, 0].item()

    def get_material_sequence(self, max_layers: Optional[int] = None) -> List[int]:
        """
        Get sequence of material indices up to specified layer or all active layers.

        Args:
            max_layers: Maximum number of layers to include (None for all active)

        Returns:
            List of material indices
        """
        if max_layers is None:
            max_layers = self.get_num_active_layers()
        else:
            max_layers = min(max_layers, self.max_layers)

        # Direct tensor access for efficiency
        materials = []
        for i in range(max_layers):
            if self._state_tensor[i, 0] > 0:  # Only active layers
                material_onehot = self._state_tensor[i, 1:]
                material_idx = torch.argmax(material_onehot).item()
                if material_idx != self.air_material_index:  # Exclude air by default
                    materials.append(material_idx)

        return materials

    def get_thickness_sequence(self, max_layers: Optional[int] = None) -> List[float]:
        """
        Get sequence of thicknesses up to specified layer or all active layers.

        Args:
            max_layers: Maximum number of layers to include (None for all active)

        Returns:
            List of thicknesses
        """
        if max_layers is None:
            max_layers = self.get_num_active_layers()
        else:
            max_layers = min(max_layers, self.max_layers)

        # Direct tensor access for efficiency
        thicknesses = self._state_tensor[:max_layers, 0]
        active_mask = thicknesses > 0
        return thicknesses[active_mask].tolist()

    def get_active_layers(self, max_layers: Optional[int] = None) -> List[LayerData]:
        """
        Get active layers up to specified layer count.

        Args:
            max_layers: Maximum number of layers to include (None for all active)

        Returns:
            List of active LayerData objects
        """
        if max_layers is None:
            max_layers = self.get_num_active_layers()
        else:
            max_layers = min(max_layers, self.max_layers)

        layers = []
        for i in range(max_layers):
            if self._state_tensor[i, 0] > 0:  # Only active layers
                thickness = self._state_tensor[i, 0].item()
                material_onehot = self._state_tensor[i, 1:]
                material_idx = torch.argmax(material_onehot).item()
                layers.append(LayerData(thickness, material_idx))

        return layers

    def invalidate_enhanced_observation_cache(self):
        """Mark enhanced observation cache as dirty."""
        self._enhanced_observation_dirty = True
        self._enhanced_observation_cache = None

    def set_layer(self, layer_idx: int, thickness: float, material_index: int):
        """
        Set layer data for specific layer index.

        Args:
            layer_idx: Layer index (0-based)
            thickness: Layer thickness
            material_index: Material index

        Raises:
            IndexError: If indices are out of bounds
        """
        # Ensure indices are integers (handle tensor inputs)
        if isinstance(layer_idx, torch.Tensor):
            layer_idx = layer_idx.item()
        if isinstance(material_index, torch.Tensor):
            material_index = material_index.item()
        layer_idx = int(layer_idx)
        material_index = int(material_index)

        if layer_idx >= self.max_layers or layer_idx < 0:
            raise IndexError(
                f"Layer index {layer_idx} out of bounds [0, {self.max_layers})"
            )
        if material_index >= self.n_materials or material_index < 0:
            raise IndexError(
                f"Material index {material_index} out of bounds [0, {self.n_materials})"
            )

        # Set thickness
        self._state_tensor[layer_idx, 0] = thickness

        # Set material (one-hot encoding)
        self._state_tensor[layer_idx, 1:] = 0  # Clear all materials
        self._state_tensor[layer_idx, material_index + 1] = 1  # Set selected material

        # Invalidate enhanced observation cache
        self._enhanced_observation_dirty = True

        if self._debug_enabled:
            print(
                f"DEBUG: Set layer {layer_idx} -> thickness={thickness:.6f}, material={material_index}"
            )

    def invalidate_enhanced_observation_cache(self):
        """Invalidate cached enhanced observations when state changes."""
        self._enhanced_observation_dirty = True

    def get_enhanced_observation(
        self,
        include_field_data: bool = None,
        merit_function_callback=None,
        **physics_params,
    ) -> Dict:
        """
        Get enhanced observation with optional electric field information.
        This is the main observation method that environments should use.

        Args:
            include_field_data: Override for including field data (uses self.include_electric_field if None)
            merit_function_callback: Physics calculation function from environment
            **physics_params: Additional physics parameters (wavelength, etc.)

        Returns:
            Dictionary observation with consistent format
        """
        if include_field_data is None:
            include_field_data = self.include_electric_field

        # Check cache validity
        cache_key = f"field_{include_field_data}"
        if (
            not self._enhanced_observation_dirty
            and hasattr(self, "_enhanced_observation_cache")
            and self._enhanced_observation_cache is not None
            and cache_key in self._enhanced_observation_cache
        ):
            return self._enhanced_observation_cache[cache_key].copy()

        # Build base layer stack observation
        observation = self._build_layer_stack_observation()

        # Add electric field information if requested and possible
        if include_field_data and merit_function_callback and self.materials:
            field_info = self._compute_electric_field_info(
                merit_function_callback=merit_function_callback,
                materials=self.materials,
                **physics_params,
            )
            if field_info:
                observation.update(
                    {
                        "electric_field": field_info["field_normalised"],
                        "field_gradients": field_info["field_gradients"],
                        "cumulative_metrics": field_info["cumulative_metrics"],
                        "field_positions": field_info["field_positions"],
                        "field_layer_indices": field_info["layer_indices"],
                    }
                )

        # Cache the result
        if (
            not hasattr(self, "_enhanced_observation_cache")
            or self._enhanced_observation_cache is None
        ):
            self._enhanced_observation_cache = {}
        self._enhanced_observation_cache[cache_key] = observation.copy()
        self._enhanced_observation_dirty = False

        return observation

    def get_observation_tensor(
        self,
        include_field_data: bool = None,
        merit_function_callback=None,
        pre_type: str = None,
        **physics_params,
    ) -> torch.Tensor:
        """
        Get observation in tensor format ready for neural networks.

        Args:
            include_field_data: Whether to include electric field calculations
            merit_function_callback: Physics calculation function from environment
            pre_type: Format for pre-processing networks ('linear' for flattening, None for default)
            **physics_params: Additional physics parameters

        Returns:
            Tensor with shape [max_layers, n_features] or [batch, flattened_features]
            where n_features depends on whether field data is included:
            - Base: [thickness, material_index, n, k] = 4 features
            - Enhanced: [thickness, material_index, n, k, efield, grad, R, A, TN] = 9 features
        """
        # Get dictionary observation
        obs_dict = self.get_enhanced_observation(
            include_field_data=include_field_data,
            merit_function_callback=merit_function_callback,
            **physics_params,
        )

        # Convert to tensor format
        tensor = self._enhanced_observation_to_tensor(obs_dict)
        # Apply pre_type formatting
        if pre_type == "linear":
            # For linear networks, flatten all dimensions except batch
            if tensor.dim() == 1:
                # Already 1D, just return as is
                pass
            elif tensor.dim() == 2:
                # [n_layers, n_features] -> [n_layers * n_features]
                tensor = tensor.flatten()
            else:
                # [batch, n_layers, n_features] -> [batch, n_layers * n_features]
                tensor = tensor.flatten(1)

        return tensor

    def _build_layer_stack_observation(self) -> Dict:
        """Build basic layer stack observation with material properties."""
        layer_stack = []
        active_layers = self.get_active_layers()

        for layer in active_layers:
            # Get material properties
            if layer.material_index < len(self.materials):
                material = self.materials[layer.material_index]
                material_onehot = (
                    self.material_index_to_onehot(layer.material_index).numpy().tolist()
                )
                n = material.get("n", 1.0)
                k = material.get("k", 0.0)
            else:
                # Default values if material not found
                n = 1.0
                k = 0.0

            layer_stack.append(
                {
                    "thickness": layer.thickness,
                    "material_index": material_onehot,
                    "n": n,
                    "k": k,
                }
            )

        return {"layer_stack": layer_stack}

    def _compute_electric_field_info(
        self,
        num_field_points: int = 50,
        merit_function_callback=None,
        materials=None,
        **physics_params,
    ) -> Optional[Dict]:
        """
        Compute electric field information for enhanced observations.

        Args:
            num_field_points: Number of field sampling points
            merit_function_callback: Function to compute physics (from environment)
            materials: Material properties list
            **physics_params: Additional physics parameters

        Returns:
            Dictionary with field information or None
        """
        if merit_function_callback is None or materials is None:
            return None

        try:
            # Convert state to format needed for merit function
            from ..utils import state_utils

            state_array = self.get_tensor().numpy()
            state_trim = state_utils.trim_state(state_array)
            state_trim = state_trim[::-1]  # reverse state

            # Call merit function with field data
            result = merit_function_callback(
                state_trim, materials, return_field_data=True, **physics_params
            )

            if len(result) >= 5:  # Has field data
                r, thermal_noise, e_integrated, total_thickness, field_data = result

                return {
                    "field_normalised": field_data.get("field_normalised", []),
                    "field_gradients": field_data.get("field_gradients", []),
                    "cumulative_metrics": [r, e_integrated, thermal_noise],
                    "field_positions": field_data.get("field_positions", []),
                    "layer_indices": field_data.get("layer_indices", []),
                }
        except Exception as e:
            if self._debug_enabled:
                print(f"Warning: Could not compute electric field info: {e}")

        return None

    def get_tensor_for_network(
        self,
        include_enhanced_features: bool = False,
        merit_function_callback=None,
        **physics_params,
    ) -> torch.Tensor:
        """
        Get tensor representation optimized for neural network input.

        Args:
            include_enhanced_features: Whether to include enhanced features beyond basic state
            merit_function_callback: Physics calculation function (required for enhanced features)
            **physics_params: Additional physics parameters

        Returns:
            Tensor ready for network processing
        """
        if not include_enhanced_features or not self.include_electric_field:
            # Return basic state tensor
            return self._state_tensor.clone()

        # Get enhanced observation and convert to tensor format
        obs = self.get_enhanced_observation(
            merit_function_callback=merit_function_callback, **physics_params
        )
        return self._enhanced_observation_to_tensor(obs)

    def _enhanced_observation_to_tensor(self, obs: Dict) -> torch.Tensor:
        """
        Convert enhanced observation dictionary to tensor format.
        Similar to the old _process_enhanced_observation function.

        Args:
            obs: Enhanced observation dictionary

        Returns:
            Tensor with concatenated layer and field information
        """
        layer_stack = obs["layer_stack"]

        # Convert layer stack to tensor [n_layers, 4] -> [thickness,
        # material_index, n, k]
        layer_data = []
        for layer in layer_stack:
            # Flatten material_index (which is a list) into the row
            row = (
                [layer["thickness"]]
                + layer["material_index"]
                + [layer["n"], layer["k"]]
            )
            layer_data.append(row)

        if not layer_data:
            # Empty state - return minimal tensor
            return torch.zeros(1, 3 + self.n_materials)

        layer_tensor = torch.tensor(
            layer_data, dtype=torch.float32
        )  # [n_layers, 3+n_materials]

        # Add electric field information if available
        if "electric_field" in obs and obs["electric_field"] is not None:
            n_layers = layer_tensor.shape[0]

            # Convert field data to tensors
            efield = torch.tensor(obs["electric_field"], dtype=torch.float32)
            gradients = torch.tensor(obs["field_gradients"], dtype=torch.float32)
            metrics = torch.tensor(obs["cumulative_metrics"], dtype=torch.float32)

            # Interpolate field data to match number of layers
            field_points = efield.shape[0]
            if field_points != n_layers and n_layers > 1:
                efield = torch.nn.functional.interpolate(
                    efield.unsqueeze(0).unsqueeze(0),
                    size=n_layers,
                    mode="linear",
                    align_corners=False,
                ).squeeze()
                gradients = torch.nn.functional.interpolate(
                    gradients.unsqueeze(0).unsqueeze(0),
                    size=n_layers,
                    mode="linear",
                    align_corners=False,
                ).squeeze()

            # Ensure correct dimensions
            if efield.dim() == 0:
                efield = efield.unsqueeze(0)
            if gradients.dim() == 0:
                gradients = gradients.unsqueeze(0)

            # Expand to match layer dimensions
            efield_expanded = efield.unsqueeze(1)  # [n_layers, 1]
            grad_expanded = gradients.unsqueeze(1)  # [n_layers, 1]
            metrics_expanded = metrics.unsqueeze(0).expand(
                n_layers, -1
            )  # [n_layers, 3]

            # Concatenate: [thickness, material_index, n, k, efield, grad, R, A, TN]
            enhanced_tensor = torch.cat(
                [
                    layer_tensor,  # [n_layers, 4]
                    efield_expanded,  # [n_layers, 1]
                    grad_expanded,  # [n_layers, 1]
                    metrics_expanded,  # [n_layers, 3]
                ],
                dim=1,
            )  # Final: [n_layers, 9]

            return enhanced_tensor
        else:
            # No field info, return just the base features
            return layer_tensor

    def has_consecutive_materials(self, max_layers: Optional[int] = None) -> bool:
        """
        Check if state has consecutive identical materials.

        Args:
            max_layers: Maximum layers to check (None for all active)

        Returns:
            True if consecutive materials found
        """
        materials = self.get_material_sequence(max_layers)

        for i in range(1, len(materials)):
            if materials[i] == materials[i - 1]:
                return True
        return False

    def get_consecutive_violations(
        self, max_layers: Optional[int] = None
    ) -> List[Dict]:
        """
        Get detailed information about consecutive material violations.

        Args:
            max_layers: Maximum layers to check (None for all active)

        Returns:
            List of violation dictionaries with details
        """
        materials = self.get_material_sequence(max_layers)
        violations = []

        i = 0
        while i < len(materials):
            if i + 1 < len(materials) and materials[i] == materials[i + 1]:
                # Found consecutive materials, count how many
                material = materials[i]
                start_idx = i
                count = 1

                while i + 1 < len(materials) and materials[i + 1] == material:
                    count += 1
                    i += 1

                violations.append(
                    {
                        "material_index": material,
                        "start_layer": start_idx,
                        "count": count,
                        "layers": list(range(start_idx, start_idx + count)),
                    }
                )
            i += 1

        return violations

    def get_total_thickness(
        self, material_index: Optional[int] = None, max_layers: Optional[int] = None
    ) -> float:
        """
        Get total thickness, optionally filtered by material.

        Args:
            material_index: Material to filter by (None for all)
            max_layers: Maximum layers to check (None for all active)

        Returns:
            Total thickness
        """
        if max_layers is None:
            max_layers = self.get_num_active_layers()
        else:
            max_layers = min(max_layers, self.max_layers)

        total = 0.0
        for i in range(max_layers):
            if self._state_tensor[i, 0] > 0:  # Only active layers
                if material_index is None:
                    total += self._state_tensor[i, 0].item()
                else:
                    material_onehot = self._state_tensor[i, 1:]
                    layer_material = torch.argmax(material_onehot).item()
                    if layer_material == material_index:
                        total += self._state_tensor[i, 0].item()
        return total

    def get_layer_count(
        self, material_index: Optional[int] = None, max_layers: Optional[int] = None
    ) -> int:
        """
        Get number of active layers, optionally filtered by material.

        Args:
            material_index: Material to filter by (None for all)
            max_layers: Maximum layers to check (None for all active)

        Returns:
            Number of layers
        """
        if max_layers is None:
            max_layers = self.get_num_active_layers()
        else:
            max_layers = min(max_layers, self.max_layers)

        count = 0
        for i in range(max_layers):
            if self._state_tensor[i, 0] > 0:  # Only active layers
                if material_index is None:
                    count += 1
                else:
                    material_onehot = self._state_tensor[i, 1:]
                    layer_material = torch.argmax(material_onehot).item()
                    if layer_material == material_index:
                        count += 1
        return count

    def validate(
        self, check_consecutive: bool = True, check_bounds: bool = True
    ) -> List[str]:
        """
        Validate state and return list of issues.

        Args:
            check_consecutive: Whether to check for consecutive materials
            check_bounds: Whether to check material indices and thicknesses

        Returns:
            List of validation issue descriptions
        """
        issues = []

        # Check for consecutive materials
        if check_consecutive and self.has_consecutive_materials():
            violations = self.get_consecutive_violations()
            issues.append(f"Found {len(violations)} consecutive material violations")
            for v in violations:
                issues.append(
                    f"  - {v['count']} consecutive material {v['material_index']} "
                    f"layers starting at layer {v['start_layer']}"
                )

        if check_bounds:
            # Check material indices are valid
            for i in range(self.max_layers):
                layer = self.get_layer(i)
                if layer.material_index >= self.n_materials:
                    issues.append(
                        f"Layer {i} has invalid material index {layer.material_index}"
                    )

            # Check thickness values for active layers
            for i, layer in enumerate(self.get_active_layers()):
                if layer.thickness <= 0:
                    issues.append(
                        f"Active layer {i} has non-positive thickness {layer.thickness}"
                    )

        return issues

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "max_layers": self.max_layers,
            "n_materials": self.n_materials,
            "air_material_index": self.air_material_index,
            "substrate_material_index": self.substrate_material_index,
            "tensor": self._state_tensor.tolist(),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "CoatingState":
        """Create from dictionary"""
        state = cls(
            data["max_layers"],
            data["n_materials"],
            data["air_material_index"],
            data["substrate_material_index"],
        )
        state._state_tensor = torch.tensor(data["tensor"])
        return state

    def copy(self) -> "CoatingState":
        """Create a deep copy of this state"""
        new_state = CoatingState(
            self.max_layers,
            self.n_materials,
            self.air_material_index,
            self.substrate_material_index,
        )
        new_state._state_tensor = self._state_tensor.clone()
        new_state._debug_enabled = self._debug_enabled
        return new_state

    def __repr__(self) -> str:
        active_count = self.get_num_active_layers()
        return f"CoatingState({active_count} active layers, {self.max_layers} total)"

    def __str__(self) -> str:
        active_count = self.get_num_active_layers()
        lines = [f"CoatingState ({active_count} active layers):"]

        # Show active layers
        for i in range(active_count):
            if self._state_tensor[i, 0] > 0:
                thickness = self._state_tensor[i, 0].item()
                material_onehot = self._state_tensor[i, 1:]
                material_idx = torch.argmax(material_onehot).item()

                material_name = (
                    MaterialIndex(material_idx).name
                    if material_idx < 4
                    else f"MAT_{material_idx}"
                )
                lines.append(
                    f"  Layer {i}: {material_name} ({material_idx}), Thickness {thickness:.6f}"
                )

        # Show consecutive violations
        violations = self.get_consecutive_violations()
        if violations:
            lines.append("⚠️  Consecutive material violations:")
            for v in violations:
                material_name = (
                    MaterialIndex(v["material_index"]).name
                    if v["material_index"] < 4
                    else f"MAT_{v['material_index']}"
                )
                lines.append(
                    f"  - {v['count']} consecutive {material_name} ({v['material_index']}) layers"
                )

        return "\n".join(lines)
