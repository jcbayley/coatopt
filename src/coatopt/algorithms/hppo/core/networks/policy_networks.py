"""
Policy networks for HPPO agent.
Consolidates standard and hyper-network variants.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def get_activation_function(activation: str) -> nn.Module:
    """Get activation function by name."""
    activations = {
        "relu": nn.ReLU(),
        "tanh": nn.Tanh(),
        "leaky_relu": nn.LeakyReLU(),
        "gelu": nn.GELU(),
        "swish": nn.SiLU(),
    }
    return activations.get(activation.lower(), nn.ReLU())


class DiscretePolicy(nn.Module):
    """
    Discrete policy network for material selection.
    Supports both standard and hyper-network variants.
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
        super().__init__()
        self.output_dim_discrete = output_dim_discrete
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.n_layers = n_layers
        self.include_layer_number = include_layer_number
        self.use_hyper_networks = use_hyper_networks
        self.activation_fn = get_activation_function(activation)
        
        # Calculate input dimension
        objective_dim = n_objectives if use_hyper_networks else 0
        layer_dim = 1 if include_layer_number else 0
        total_input_dim = input_dim + layer_dim + objective_dim
        
        if use_hyper_networks:
            self._build_hyper_network(total_input_dim, hidden_dim, hyper_hidden_dim, hyper_n_layers, n_objectives)
        else:
            self._build_standard_network(total_input_dim, hidden_dim)
            
    def _build_standard_network(self, input_dim: int, hidden_dim: int):
        """Build standard discrete policy network."""
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(self.n_layers)
        ])
        
        self.output_layer = nn.Linear(hidden_dim, self.output_dim_discrete)
        
    def _build_hyper_network(self, input_dim: int, hidden_dim: int, hyper_hidden_dim: int, hyper_n_layers: int, n_objectives: int):
        """Build hyper-network variant."""
        self.target_input_dim = input_dim - n_objectives  # Exclude objectives from target input
        self.target_params = self._compute_target_param_count(hidden_dim)
        
        # Hyper-network that generates weights for target network
        hyper_input_dim = n_objectives
        self.hyper_layers = nn.ModuleList()
        
        # First hyper layer
        self.hyper_layers.append(nn.Linear(hyper_input_dim, hyper_hidden_dim))
        
        # Hidden hyper layers
        for _ in range(hyper_n_layers - 1):
            self.hyper_layers.append(nn.Linear(hyper_hidden_dim, hyper_hidden_dim))
            
        # Output layer generates all target network parameters
        self.hyper_output = nn.Linear(hyper_hidden_dim, self.target_params)
        
    def _compute_target_param_count(self, hidden_dim: int) -> int:
        """Compute total parameters needed for target network."""
        param_count = 0
        
        # Input layer parameters
        param_count += self.target_input_dim * hidden_dim + hidden_dim  # weights + biases
        
        # Hidden layer parameters
        for _ in range(self.n_layers):
            param_count += hidden_dim * hidden_dim + hidden_dim  # weights + biases
            
        # Output layer parameters
        param_count += hidden_dim * self.output_dim_discrete + self.output_dim_discrete  # weights + biases
        
        return param_count
        
    def forward(self, state: torch.Tensor, layer_number: Optional[torch.Tensor] = None, objectives: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through discrete policy."""
        
        # Prepare input
        inputs = [state]
        if self.include_layer_number and layer_number is not None:
            inputs.append(layer_number)
            
        if self.use_hyper_networks:
            if objectives is not None:
                return self._forward_hyper_network(torch.cat(inputs, dim=-1), objectives)
            else:
                raise ValueError("Objectives required for hyper-network variant")
        else:
            if objectives is not None:
                inputs.append(objectives)
            x = torch.cat(inputs, dim=-1)
            return self._forward_standard_network(x)
            
    def _forward_standard_network(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through standard network."""
        x = self.activation_fn(self.input_layer(x))
        
        for layer in self.hidden_layers:
            x = self.activation_fn(layer(x))
            
        return torch.softmax(self.output_layer(x), dim=-1)
        
    def _forward_hyper_network(self, state_input: torch.Tensor, objectives: torch.Tensor) -> torch.Tensor:
        """Forward pass through hyper-network."""
        batch_size = state_input.size(0)
        
        # Generate target network parameters using hyper-network
        hyper_input = objectives
        for layer in self.hyper_layers:
            hyper_input = self.activation_fn(layer(hyper_input))
        
        # Generate all target network parameters
        all_params = self.hyper_output(hyper_input)
        
        # Parse parameters for target network layers
        param_idx = 0
        hidden_dim = (all_params.size(-1) - self.target_input_dim - self.output_dim_discrete * 2) // (self.n_layers + 1)
        
        # Input layer parameters
        input_weight_size = self.target_input_dim * hidden_dim
        input_weight = all_params[:, param_idx:param_idx + input_weight_size].view(batch_size, self.target_input_dim, hidden_dim)
        param_idx += input_weight_size
        
        input_bias = all_params[:, param_idx:param_idx + hidden_dim].view(batch_size, 1, hidden_dim)
        param_idx += hidden_dim
        
        # Forward through input layer
        x = torch.bmm(state_input.unsqueeze(1), input_weight) + input_bias
        x = self.activation_fn(x)
        
        # Hidden layers
        for _ in range(self.n_layers):
            hidden_weight_size = hidden_dim * hidden_dim
            hidden_weight = all_params[:, param_idx:param_idx + hidden_weight_size].view(batch_size, hidden_dim, hidden_dim)
            param_idx += hidden_weight_size
            
            hidden_bias = all_params[:, param_idx:param_idx + hidden_dim].view(batch_size, 1, hidden_dim)
            param_idx += hidden_dim
            
            x = torch.bmm(x, hidden_weight) + hidden_bias
            x = self.activation_fn(x)
            
        # Output layer
        output_weight_size = hidden_dim * self.output_dim_discrete
        output_weight = all_params[:, param_idx:param_idx + output_weight_size].view(batch_size, hidden_dim, self.output_dim_discrete)
        param_idx += output_weight_size
        
        output_bias = all_params[:, param_idx:param_idx + self.output_dim_discrete].view(batch_size, 1, self.output_dim_discrete)
        
        x = torch.bmm(x, output_weight) + output_bias
        
        return torch.softmax(x.squeeze(1), dim=-1)


class ContinuousPolicy(nn.Module):
    """
    Continuous policy network for thickness selection.
    Outputs mean and log_std for thickness distribution.
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
        include_material: bool = False,  # Added to match original interface
        activation: str = "relu",
        n_objectives: int = 0,
        use_hyper_networks: bool = False,
        hyper_hidden_dim: int = 128,
        hyper_n_layers: int = 2,
    ):
        super().__init__()
        self.output_dim_continuous = output_dim_continuous
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.n_layers = n_layers
        self.include_layer_number = include_layer_number
        self.include_material = include_material
        self.use_hyper_networks = use_hyper_networks
        self.activation_fn = get_activation_function(activation)
        
        # Calculate input dimension
        objective_dim = n_objectives if use_hyper_networks else 0
        layer_dim = 1 if include_layer_number else 0
        material_dim = 1 if include_material else 0
        total_input_dim = input_dim + layer_dim + material_dim + objective_dim
        
        if use_hyper_networks:
            self._build_hyper_network(total_input_dim, hidden_dim, hyper_hidden_dim, hyper_n_layers, n_objectives)
        else:
            self._build_standard_network(total_input_dim, hidden_dim)
            
        # Initialize saved data (to match original interface)
        self.saved_log_probs = []
        self.rewards = []
        
        # Calculate input dimension
        objective_dim = n_objectives if use_hyper_networks else 0
        layer_dim = 1 if include_layer_number else 0
        total_input_dim = input_dim + layer_dim + objective_dim
        
        if use_hyper_networks:
            self._build_hyper_network(total_input_dim, hidden_dim, hyper_hidden_dim, hyper_n_layers, n_objectives)
        else:
            self._build_standard_network(total_input_dim, hidden_dim)
            
    def _build_standard_network(self, input_dim: int, hidden_dim: int):
        """Build standard continuous policy network."""
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(self.n_layers)
        ])
        
        # Output layers for mean and log_std
        self.mean_layer = nn.Linear(hidden_dim, self.output_dim_continuous)
        self.log_std_layer = nn.Linear(hidden_dim, self.output_dim_continuous)
        
    def _build_hyper_network(self, input_dim: int, hidden_dim: int, hyper_hidden_dim: int, hyper_n_layers: int, n_objectives: int):
        """Build hyper-network variant."""
        self.target_input_dim = input_dim - n_objectives
        self.target_params = self._compute_target_param_count(hidden_dim)
        
        # Hyper-network
        hyper_input_dim = n_objectives
        self.hyper_layers = nn.ModuleList()
        
        self.hyper_layers.append(nn.Linear(hyper_input_dim, hyper_hidden_dim))
        
        for _ in range(hyper_n_layers - 1):
            self.hyper_layers.append(nn.Linear(hyper_hidden_dim, hyper_hidden_dim))
            
        self.hyper_output = nn.Linear(hyper_hidden_dim, self.target_params)
        
    def _compute_target_param_count(self, hidden_dim: int) -> int:
        """Compute total parameters for target network."""
        param_count = 0
        
        # Input layer
        param_count += self.target_input_dim * hidden_dim + hidden_dim
        
        # Hidden layers
        for _ in range(self.n_layers):
            param_count += hidden_dim * hidden_dim + hidden_dim
            
        # Output layers (mean and log_std)
        param_count += 2 * (hidden_dim * self.output_dim_continuous + self.output_dim_continuous)
        
        return param_count
        
    def forward(self, state: torch.Tensor, layer_number: Optional[torch.Tensor] = None, objectives: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning mean and log_std."""
        
        # Prepare input
        inputs = [state]
        if self.include_layer_number and layer_number is not None:
            inputs.append(layer_number)
            
        if self.use_hyper_networks:
            if objectives is not None:
                return self._forward_hyper_network(torch.cat(inputs, dim=-1), objectives)
            else:
                raise ValueError("Objectives required for hyper-network variant")
        else:
            if objectives is not None:
                inputs.append(objectives)
            x = torch.cat(inputs, dim=-1)
            return self._forward_standard_network(x)
            
    def _forward_standard_network(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through standard network."""
        x = self.activation_fn(self.input_layer(x))
        
        for layer in self.hidden_layers:
            x = self.activation_fn(layer(x))
        
        mean = torch.sigmoid(self.mean_layer(x))
        mean = self.lower_bound + (self.upper_bound - self.lower_bound) * mean
        
        log_std = torch.clamp(self.log_std_layer(x), min=-10, max=2)
        
        return mean, log_std
        
    def _forward_hyper_network(self, state_input: torch.Tensor, objectives: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through hyper-network."""
        batch_size = state_input.size(0)
        
        # Generate target network parameters
        hyper_input = objectives
        for layer in self.hyper_layers:
            hyper_input = self.activation_fn(layer(hyper_input))
        
        all_params = self.hyper_output(hyper_input)
        
        # Similar parameter parsing as discrete policy but for continuous output
        param_idx = 0
        hidden_dim = (all_params.size(-1) - self.target_input_dim - 2 * self.output_dim_continuous * 2) // (self.n_layers + 1)
        
        # Forward through network (similar structure to discrete)
        # ... (implementation details omitted for brevity, follows same pattern as discrete)
        
        # For now, return placeholder - full implementation would follow discrete pattern
        # This is where the actual hyper-network forward pass would be implemented
        raise NotImplementedError("Hyper-network continuous policy not fully implemented")


class ValueNetwork(nn.Module):
    """
    Value function network for state evaluation.
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
    ):
        super().__init__()
        self.n_layers = n_layers
        self.include_layer_number = include_layer_number
        self.use_hyper_networks = use_hyper_networks
        self.activation_fn = get_activation_function(activation)
        
        # Calculate input dimension
        objective_dim = n_objectives if use_hyper_networks else 0
        layer_dim = 1 if include_layer_number else 0
        total_input_dim = input_dim + layer_dim + objective_dim
        
        # Build network
        self.input_layer = nn.Linear(total_input_dim, hidden_dim)
        
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers)
        ])
        
        # Output single value
        self.output_layer = nn.Linear(hidden_dim, n_objectives if n_objectives > 1 else 1)
        
    def forward(self, state: torch.Tensor, layer_number: Optional[torch.Tensor] = None, objectives: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through value network."""
        
        inputs = [state]
        if self.include_layer_number and layer_number is not None:
            inputs.append(layer_number)
        if self.use_hyper_networks and objectives is not None:
            inputs.append(objectives)
            
        x = torch.cat(inputs, dim=-1)
        
        x = self.activation_fn(self.input_layer(x))
        
        for layer in self.hidden_layers:
            x = self.activation_fn(layer(x))
            
        return self.output_layer(x)
