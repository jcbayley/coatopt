from typing import List, Optional, Tuple

import numpy as np
import torch


class CoatingState:
    """
    Simple state representation for coating stacks.

    State is stored as a 2D numpy array with shape (max_layers, 2):
        - Column 0: thickness (in meters)
        - Column 1: material_index (integer)
    """

    def __init__(
        self,
        max_layers: int,
        n_materials: int,
        air_material_index: int = 0,
        substrate_material_index: int = 1,
        materials: Optional[dict] = None,
    ):
        """
        Initialize coating state.

        Args:
            max_layers: Maximum number of layers
            n_materials: Total number of available materials
            air_material_index: Index representing air (default: 0)
            substrate_material_index: Index representing substrate (default: 1)
            materials: Material properties dict (for n, k values)
        """
        self.max_layers = max_layers
        self.n_materials = n_materials
        self.air_material_index = air_material_index
        self.substrate_material_index = substrate_material_index
        self.materials = materials or {}

        # Internal state: (max_layers, 2) array
        # Column 0: thickness, Column 1: material_index
        self._state = np.zeros((max_layers, 2), dtype=np.float32)
        self._state[:, 1] = air_material_index  # Initialize all to air

    @classmethod
    def from_array(
        cls,
        state_array: np.ndarray,
        n_materials: int,
        air_material_index: int = 0,
        substrate_material_index: int = 1,
        materials: Optional[dict] = None,
    ) -> "CoatingState":
        """Create CoatingState from numpy array."""
        max_layers = state_array.shape[0]
        obj = cls(
            max_layers,
            n_materials,
            air_material_index,
            substrate_material_index,
            materials,
        )
        obj._state = state_array.copy()
        return obj

    @classmethod
    def from_tensor(
        cls,
        state_tensor: torch.Tensor,
        n_materials: int,
        air_material_index: int = 0,
        substrate_material_index: int = 1,
        materials: Optional[dict] = None,
    ) -> "CoatingState":
        """Create CoatingState from PyTorch tensor (with one-hot materials)."""
        # Assume tensor shape is (max_layers, n_materials + 1)
        # Column 0: thickness, Columns 1+: one-hot encoded materials
        max_layers = state_tensor.shape[0]

        # Extract thickness and material indices
        thicknesses = state_tensor[:, 0].cpu().numpy()
        material_onehot = state_tensor[:, 1:].cpu().numpy()
        material_indices = np.argmax(material_onehot, axis=1)

        # Create state array
        state_array = np.stack([thicknesses, material_indices], axis=1).astype(
            np.float32
        )

        return cls.from_array(
            state_array,
            n_materials,
            air_material_index,
            substrate_material_index,
            materials,
        )

    def get_array(self) -> np.ndarray:
        """Get state as numpy array in one-hot encoded format (max_layers, n_materials + 1).

        This matches the format expected by merit_function and base_environment.
        Format: [thickness, material_0, material_1, ..., material_n]
        """
        return self.get_tensor().numpy()

    def get_tensor(self) -> torch.Tensor:
        """Get state as one-hot encoded PyTorch tensor (max_layers, n_materials + 1)."""
        # Create tensor with shape (max_layers, n_materials + 1)
        tensor = torch.zeros(self.max_layers, self.n_materials + 1, dtype=torch.float32)

        # Set thicknesses
        tensor[:, 0] = torch.from_numpy(self._state[:, 0])

        # Set one-hot encoded materials
        material_indices = self._state[:, 1].astype(int)
        for i, mat_idx in enumerate(material_indices):
            tensor[i, 1 + mat_idx] = 1.0

        return tensor

    def get_observation_tensor(
        self,
        pre_type: str = "linear",
        include_field_data: bool = False,
        merit_function_callback=None,
        constraints: Optional[dict] = None,
        objective_names: Optional[list] = None,
        **physics_params,
    ) -> torch.Tensor:
        """
        Get observation tensor formatted for neural networks.

        Returns only ACTIVE layers (thickness > 0), not padded to max_layers.

        Base format per layer: [thickness, material_0, ..., material_n, n, k]
        Enhanced format (with field): [thickness, materials, n, k, efield, grad, R, A, TN]
        Appends constraint thresholds at the end if provided.

        Args:
            pre_type: "linear" for flattening, else 2D
            include_field_data: Include electric field calculations (default: False)
            merit_function_callback: Physics function from environment
            constraints: Dict of constraint thresholds {objective_name: threshold}
            objective_names: Ordered list of objective names for consistent constraint ordering
            **physics_params: Additional physics parameters

        Returns:
            Tensor with active layers only (or all layers if pre_type=="lstm")
        """
        # Build layer stack observation
        # For LSTM: include all max_layers (with zero padding for inactive)
        # For others: only include active layers
        layer_data = []

        for i in range(self.max_layers):
            thickness, material_idx = self.get_layer(i)
            material_idx = int(material_idx)

            # For non-LSTM, skip inactive layers for variable-length sequences
            if thickness <= 0 and pre_type != "lstm":
                continue

            # Build row: [thickness, material_onehot, n, k]
            # Always encode material (air for inactive layers) to match original behavior
            material_onehot = [0.0] * self.n_materials
            material_onehot[material_idx] = 1.0

            # Get n and k from materials dict
            if self.materials and material_idx in self.materials:
                material = self.materials[material_idx]
                n_val = float(material.get("n", 1.0))
                k_val = float(material.get("k", 0.0))
            else:
                n_val = 1.0  # Default for air
                k_val = 0.0

            row = [float(thickness)] + material_onehot + [n_val, k_val]
            layer_data.append(row)

        # Handle empty state (shouldn't happen with LSTM since we pad)
        if not layer_data:
            base_features = 3 + self.n_materials
            features = base_features + 5 if include_field_data else base_features
            tensor = torch.zeros(1, features, dtype=torch.float32)
        else:
            tensor = torch.tensor(layer_data, dtype=torch.float32)

            # Add electric field info if requested
            if include_field_data and merit_function_callback and self.materials:
                field_results = merit_function_callback(self, **physics_params)
                if field_results and isinstance(field_results, dict):
                    tensor = self._add_field_info(tensor, field_results)

        # Append constraint thresholds (before pre_type formatting for 2D tensors)
        if constraints is not None and objective_names:
            constraint_values = [constraints.get(obj, 0.0) for obj in objective_names]
            constraint_tensor = torch.tensor(constraint_values, dtype=torch.float32)

            if tensor.dim() == 2:
                # 2D tensor: add constraints as extra features to each layer
                n_layers = tensor.shape[0]
                constraint_expanded = constraint_tensor.unsqueeze(0).expand(
                    n_layers, -1
                )
                tensor = torch.cat([tensor, constraint_expanded], dim=1)
            else:
                # 1D tensor: just append
                tensor = torch.cat([tensor, constraint_tensor])

        # Apply pre_type formatting
        if pre_type == "linear":
            if tensor.dim() == 2:
                tensor = tensor.flatten()
            elif tensor.dim() > 2:
                tensor = tensor.flatten(1)

        return tensor

    def _add_field_info(
        self, layer_tensor: torch.Tensor, field_results: dict
    ) -> torch.Tensor:
        """Add electric field information to layer tensor."""
        efield = field_results.get("electric_field")
        gradients = field_results.get("field_gradients")
        metrics = field_results.get("cumulative_metrics")

        if efield is None or gradients is None or metrics is None:
            return layer_tensor

        n_layers = layer_tensor.shape[0]

        # Convert to tensors
        efield = (
            torch.tensor(efield, dtype=torch.float32)
            if not isinstance(efield, torch.Tensor)
            else efield
        )
        gradients = (
            torch.tensor(gradients, dtype=torch.float32)
            if not isinstance(gradients, torch.Tensor)
            else gradients
        )
        metrics = (
            torch.tensor(metrics, dtype=torch.float32)
            if not isinstance(metrics, torch.Tensor)
            else metrics
        )

        # Interpolate field data to match number of layers if needed
        field_points = efield.shape[0] if efield.dim() > 0 else 1
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
        efield_expanded = efield.unsqueeze(1)
        grad_expanded = gradients.unsqueeze(1)
        metrics_expanded = metrics.unsqueeze(0).expand(n_layers, -1)

        # Concatenate: [base_features, efield, grad, R, A, TN]
        return torch.cat(
            [layer_tensor, efield_expanded, grad_expanded, metrics_expanded], dim=1
        )

    def set_layer(self, layer_idx: int, thickness: float, material_index: int):
        """Set thickness and material for a specific layer."""
        if layer_idx < 0 or layer_idx >= self.max_layers:
            raise IndexError(
                f"Layer index {layer_idx} out of range [0, {self.max_layers})"
            )

        self._state[layer_idx, 0] = thickness
        self._state[layer_idx, 1] = material_index

    def get_layer(self, layer_idx: int) -> Tuple[float, int]:
        """Get (thickness, material_index) for a specific layer."""
        # Handle tensor indices
        if hasattr(layer_idx, "item"):
            layer_idx = layer_idx.item()
        layer_idx = int(layer_idx)

        if layer_idx < 0 or layer_idx >= self.max_layers:
            raise IndexError(
                f"Layer index {layer_idx} out of range [0, {self.max_layers})"
            )

        return self._state[layer_idx, 0], int(self._state[layer_idx, 1])

    def get_thicknesses(self) -> np.ndarray:
        """Get all layer thicknesses."""
        return self._state[:, 0].copy()

    def get_materials(self) -> np.ndarray:
        """Get all material indices."""
        return self._state[:, 1].astype(int)

    def get_active_layers(self, min_thickness: float = 1e-9) -> np.ndarray:
        """Get indices of active layers (thickness > min_thickness)."""
        return np.where(self._state[:, 0] > min_thickness)[0]

    def get_num_active_layers(self, min_thickness: float = 1e-9) -> int:
        """Get count of active layers."""
        return len(self.get_active_layers(min_thickness))

    def get_total_thickness(
        self,
        max_layers: Optional[int] = None,
        ignore_air: bool = True,
        ignore_substrate: bool = True,
    ) -> float:
        """Get total thickness of the coating stack."""
        n = max_layers if max_layers is not None else self.max_layers

        total = 0.0
        for i in range(n):
            thickness, material = self.get_layer(i)

            # Skip air and substrate if requested
            if ignore_air and material == self.air_material_index:
                continue
            if ignore_substrate and material == self.substrate_material_index:
                continue

            total += thickness

        return total

    def get_previous_material(self, layer_idx: int) -> Optional[int]:
        """
        Get material of the previous layer (for constraint checking).

        Args:
            layer_idx: Current layer index (can be int or tensor)

        Returns:
            Material index of previous layer, or None if at first layer
        """
        # Handle tensor indices
        if hasattr(layer_idx, "item"):
            layer_idx = layer_idx.item()
        layer_idx = int(layer_idx)

        if layer_idx <= 0:
            return None

        _, material = self.get_layer(layer_idx - 1)
        return int(material)

    def copy(self) -> "CoatingState":
        """Create a deep copy of this state."""
        return CoatingState.from_array(
            self._state.copy(),
            self.n_materials,
            self.air_material_index,
            self.substrate_material_index,
            self.materials,
        )

    def __repr__(self) -> str:
        active = self.get_num_active_layers()
        total_thick = self.get_total_thickness()
        return f"CoatingState({active}/{self.max_layers} active layers, {total_thick*1e9:.1f}nm total)"

    def __str__(self) -> str:
        """Human-readable representation showing all layers."""
        lines = [f"CoatingState with {self.max_layers} layers:"]
        for i in range(self.max_layers):
            thickness, material = self.get_layer(i)
            if thickness > 1e-9:  # Only show active layers
                lines.append(f"  Layer {i}: {thickness*1e9:.2f}nm, material {material}")
        return "\n".join(lines)


def create_state_from_tensor(
    tensor: torch.Tensor, n_materials: int, air_idx: int = 0, substrate_idx: int = 1
) -> CoatingState:
    """Helper function for creating states from tensors."""
    return CoatingState.from_tensor(tensor, n_materials, air_idx, substrate_idx)


def create_state_from_array(
    array: np.ndarray, n_materials: int, air_idx: int = 0, substrate_idx: int = 1
) -> CoatingState:
    """Helper function for creating states from arrays."""
    return CoatingState.from_array(array, n_materials, air_idx, substrate_idx)


def convert_to_tensor_state(state: CoatingState) -> torch.Tensor:
    """Convert CoatingState to one-hot tensor representation."""
    return state.get_tensor()


def convert_to_array_state(state: CoatingState) -> np.ndarray:
    """Convert CoatingState to array representation."""
    return state.get_array()
