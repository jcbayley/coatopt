"""
Policy and value networks for PC-HPPO algorithm.
Refactored with shared base class to reduce code duplication.
"""
from typing import Optional, Tuple, List, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_activation_function(name: str) -> nn.Module:
    """Get activation function by name."""
    activations = {
        "relu": nn.ReLU(),
        "tanh": nn.Tanh(),
        "silu": nn.SiLU(),
        "leaky_relu": nn.LeakyReLU(),
    }
    if name not in activations:
        raise ValueError(f"Unsupported activation: {name}")
    return activations[name]


class BaseNetwork(nn.Module):
    """
    Base class for policy and value networks with shared functionality.
    Handles both standard networks and hyper-networks.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dims: Dict[str, int],  # e.g. {'logits': 4} or {'mean': 1, 'log_std': 1}
        hidden_dim: int,
        n_layers: int = 2,
        include_layer_number: bool = False,
        include_material: bool = False,
        activation: str = "relu",
        n_objectives: int = 0,
        use_hyper_networks: bool = False,
        hyper_hidden_dim: int = 128,
        hyper_n_layers: int = 2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dims = output_dims
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.include_layer_number = include_layer_number
        self.include_material = include_material
        self.use_hyper_networks = use_hyper_networks
        self.n_objectives = n_objectives
        self.activation_fn = get_activation_function(activation)
        
        # Calculate input dimension for target network
        objective_dim = 0 if use_hyper_networks else n_objectives
        layer_dim = 1 if include_layer_number else 0
        material_dim = 1 if include_material else 0
        self.total_input_dim = input_dim + layer_dim + material_dim + objective_dim
        
        if use_hyper_networks:
            self.target_input_dim = input_dim + layer_dim + material_dim  # Exclude objectives
            self._build_hyper_network(hyper_hidden_dim, hyper_n_layers, n_objectives)
        else:
            self._build_standard_network()
    
    def _build_standard_network(self):
        """Build standard feed-forward network."""
        self.input_layer = nn.Linear(self.total_input_dim, self.hidden_dim)
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList([
            nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(self.n_layers)
        ])
        
        # Output layers - one for each output type
        self.output_layers = nn.ModuleDict({
            name: nn.Linear(self.hidden_dim, dim) 
            for name, dim in self.output_dims.items()
        })
        
    def _build_hyper_network(self, hyper_hidden_dim: int, hyper_n_layers: int, n_objectives: int):
        """Build hyper-network that generates weights for target network."""
        self.target_params = self._compute_target_param_count()
        
        # Hyper-network takes only objective weights as input
        hyper_layers = []
        in_dim = n_objectives
        
        for _ in range(hyper_n_layers):
            hyper_layers.append(nn.Linear(in_dim, hyper_hidden_dim))
            hyper_layers.append(self.activation_fn)
            in_dim = hyper_hidden_dim
            
        hyper_layers.append(nn.Linear(hyper_hidden_dim, self.target_params))
        self.hyper_network = nn.Sequential(*hyper_layers)
    
    def _compute_target_param_count(self) -> int:
        """Compute total parameters needed for target network."""
        count = 0
        # Input layer
        count += self.target_input_dim * self.hidden_dim + self.hidden_dim
        # Hidden layers
        for _ in range(self.n_layers):
            count += self.hidden_dim * self.hidden_dim + self.hidden_dim
        # Output layers
        for dim in self.output_dims.values():
            count += self.hidden_dim * dim + dim
        return count
    
    def _prepare_input(self, state: torch.Tensor, layer_number: Optional[torch.Tensor] = None, 
                      material: Optional[torch.Tensor] = None, objective_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Prepare input tensor by concatenating all components."""
        inputs = [state]
        
        if self.include_layer_number and layer_number is not None:
            # Ensure layer_number has the right shape for concatenation with state
            if layer_number.dim() == 1 and state.dim() == 2:
                # If layer_number is 1D and state is 2D, expand layer_number
                layer_number = layer_number.unsqueeze(-1)  # Shape: (B, 1)
            elif layer_number.dim() == 0 and state.dim() == 2:
                # If layer_number is scalar and state is 2D, expand to match batch
                layer_number = layer_number.unsqueeze(0).unsqueeze(-1)  # Shape: (1, 1)
                layer_number = layer_number.expand(state.size(0), -1)  # Shape: (B, 1)
            elif layer_number.dim() == 1 and state.dim() == 1:
                # Both 1D, expand both to 2D
                layer_number = layer_number.unsqueeze(-1)  # Shape: (B, 1)
                
            # Ensure batch dimensions match
            if layer_number.size(0) != state.size(0):
                layer_number = layer_number.expand(state.size(0), -1)
                
            inputs.append(layer_number)
        elif self.include_layer_number:
            raise ValueError("layer_number must be provided when include_layer_number=True")
            
        if self.include_material and material is not None:
            inputs.append(material)
        elif self.include_material:
            raise ValueError("material must be provided when include_material=True")
            
        # For standard networks, concatenate objective weights
        if not self.use_hyper_networks and objective_weights is not None:
            inputs.append(objective_weights)
            
        return torch.cat(inputs, dim=-1)
    
    def _forward_standard_network(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through standard network. Returns dict of outputs."""
        x = self.activation_fn(self.input_layer(x))
        
        for layer in self.hidden_layers:
            x = self.activation_fn(layer(x))
        
        # Compute all outputs
        outputs = {}
        for name, layer in self.output_layers.items():
            outputs[name] = layer(x)
            
        return outputs
    
    def _forward_hyper_network(self, state_input: torch.Tensor, objective_weights: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through hyper-network."""
        batch_size = state_input.size(0)
        
        # Generate all target network parameters
        all_params = self.hyper_network(objective_weights)
        
        # Parse parameters and perform functional forward pass
        param_idx = 0
        
        # Input layer
        input_weight_size = self.target_input_dim * self.hidden_dim
        input_weight = all_params[:, param_idx:param_idx + input_weight_size].view(
            batch_size, self.target_input_dim, self.hidden_dim
        )
        param_idx += input_weight_size
        
        input_bias = all_params[:, param_idx:param_idx + self.hidden_dim].view(
            batch_size, 1, self.hidden_dim
        )
        param_idx += self.hidden_dim
        
        # Forward through input layer
        x = torch.bmm(state_input.unsqueeze(1), input_weight) + input_bias
        x = self.activation_fn(x)
        
        # Hidden layers
        for _ in range(self.n_layers):
            weight_size = self.hidden_dim * self.hidden_dim
            weight = all_params[:, param_idx:param_idx + weight_size].view(
                batch_size, self.hidden_dim, self.hidden_dim
            )
            param_idx += weight_size
            
            bias = all_params[:, param_idx:param_idx + self.hidden_dim].view(
                batch_size, 1, self.hidden_dim
            )
            param_idx += self.hidden_dim
            
            x = torch.bmm(x, weight) + bias
            x = self.activation_fn(x)
        
        # Output layers
        outputs = {}
        for name, dim in self.output_dims.items():
            weight_size = self.hidden_dim * dim
            weight = all_params[:, param_idx:param_idx + weight_size].view(
                batch_size, self.hidden_dim, dim
            )
            param_idx += weight_size
            
            bias = all_params[:, param_idx:param_idx + dim].view(
                batch_size, 1, dim
            )
            param_idx += dim
            
            output = torch.bmm(x, weight) + bias
            outputs[name] = output.squeeze(1)
            
        return outputs
    
    def forward(self, state: torch.Tensor, layer_number: Optional[torch.Tensor] = None, 
               material: Optional[torch.Tensor] = None, objective_weights: Optional[torch.Tensor] = None):
        """Forward pass. Returns processed outputs specific to each subclass."""
        if self.use_hyper_networks:
            if objective_weights is None:
                raise ValueError("objective_weights required for hyper-network variant")
            state_input = self._prepare_input(state, layer_number, material, None)
            raw_outputs = self._forward_hyper_network(state_input, objective_weights)
        else:
            x = self._prepare_input(state, layer_number, material, objective_weights)
            raw_outputs = self._forward_standard_network(x)
        
        return self._process_outputs(raw_outputs)
    
    def _process_outputs(self, raw_outputs: Dict[str, torch.Tensor]):
        """Process raw network outputs into final form. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement _process_outputs")


class DiscretePolicy(BaseNetwork):
    """
    Discrete policy network that outputs categorical distributions.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim_discrete: int,
        hidden_dim: int,
        n_layers: int = 2,
        lower_bound: float = 0,
        upper_bound: float = 1,
        include_layer_number: bool = False,
        activation: str = "relu",
        n_objectives: int = 0,
        use_hyper_networks: bool = False,
        hyper_hidden_dim: int = 128,
        hyper_n_layers: int = 2,
    ):
        self.output_dim_discrete = output_dim_discrete
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        
        # Define output structure
        output_dims = {'logits': output_dim_discrete}
        
        super().__init__(
            input_dim, output_dims, hidden_dim, n_layers, include_layer_number, False,  # No material
            activation, n_objectives, use_hyper_networks, hyper_hidden_dim, hyper_n_layers
        )
    
    def _process_outputs(self, raw_outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Process discrete policy output (softmax probabilities)."""
        return torch.softmax(raw_outputs['logits'], dim=-1)


class ContinuousPolicy(BaseNetwork):
    """
    Continuous policy network that outputs Gaussian distributions.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim_continuous: int,
        hidden_dim: int,
        n_layers: int = 2,
        lower_bound: float = 0.1,
        upper_bound: float = 1.0,
        include_layer_number: bool = False,
        include_material: bool = False,
        activation: str = "relu",
        n_objectives: int = 0,
        use_hyper_networks: bool = False,
        hyper_hidden_dim: int = 128,
        hyper_n_layers: int = 2,
    ):
        self.output_dim_continuous = output_dim_continuous
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        
        # Define output structure - separate heads for mean and log_std
        output_dims = {
            'mean': output_dim_continuous,
            'log_std': output_dim_continuous
        }
        
        super().__init__(
            input_dim, output_dims, hidden_dim, n_layers, include_layer_number, include_material,
            activation, n_objectives, use_hyper_networks, hyper_hidden_dim, hyper_n_layers
        )
    
    def _process_outputs(self, raw_outputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process continuous policy output (bounded mean and clamped log_std)."""
        mean = torch.sigmoid(raw_outputs['mean'])
        mean = self.lower_bound + (self.upper_bound - self.lower_bound) * mean
        
        log_std = torch.clamp(raw_outputs['log_std'], min=-10, max=2)
        
        return mean, log_std


class ValueNetwork(BaseNetwork):
    """
    Value function network that outputs state values.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_layers: int = 2,
        include_layer_number: bool = False,
        activation: str = "relu",
        n_objectives: int = 0,
        use_hyper_networks: bool = False,
        hyper_hidden_dim: int = 128,
        hyper_n_layers: int = 2,
    ):
        # Define output structure

        output_dim = n_objectives if n_objectives > 1 else 1
        output_dims = {'value': output_dim}
        
        n_objectives=0 # Temporary fix for new model
        
        super().__init__(
            input_dim, output_dims, hidden_dim, n_layers, include_layer_number, False,  # No material
            activation, n_objectives, use_hyper_networks, hyper_hidden_dim, hyper_n_layers
        )
    
    def _process_outputs(self, raw_outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Process value function output (direct output)."""
        return raw_outputs['value']
    
    def forward(self, state: torch.Tensor, layer_number: Optional[torch.Tensor] = None, 
               objective_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Override to match expected signature and return raw multi-objective values."""
        # Get the raw multi-objective values from parent
        values = super().forward(state, layer_number, None, None)
        
        # Always return the raw N-dimensional values
        # Weighting will be handled in the agent during advantage computation
        return values
