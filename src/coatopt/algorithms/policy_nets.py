import torch
import torch.nn as nn
import torch.nn.functional as F

class DiscretePolicy(torch.nn.Module):
    def __init__(
            self, 
            input_dim, 
            output_dim_discrete, 
            hidden_dim, 
            n_layers=2,
            lower_bound=0, 
            upper_bound=1,
            include_layer_number=False,
            activation="relu",
            n_objectives=0):
        super(DiscretePolicy, self).__init__()
        self.output_dim_discrete = output_dim_discrete
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.n_layers = n_layers
        self.include_layer_number = include_layer_number

        if include_layer_number:
            self.input = torch.nn.Linear(input_dim + 1 + n_objectives, hidden_dim)
        else:
            self.input = torch.nn.Linear(input_dim + n_objectives, hidden_dim)

        for i in range(self.n_layers):
            setattr(self, f"affine{i}", torch.nn.Linear(hidden_dim, hidden_dim))

        if activation == "relu":
            self.activation = torch.nn.ReLU()
        elif activation == "tanh":
            self.activation = torch.nn.Tanh()
        elif activation == "silu":
            self.activation = torch.nn.SiLU()
        else:
            raise Exception(f"Activation not supported: {activation}")
        #self.affine2 = torch.nn.Linear(hidden_dim, hidden_dim)
        #self.affine3 = torch.nn.Linear(hidden_dim, hidden_dim)
        #self.affine4 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.dropout = torch.nn.Dropout(p=0.0)
        self.output_discrete = torch.nn.Linear(hidden_dim, output_dim_discrete)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x, layer_number=None, zero_idx=None, objective_weights=None):
        if layer_number is not None and self.include_layer_number:
            x = torch.cat([x, layer_number], dim=1)
        if objective_weights is not None:
            x = torch.cat([x, objective_weights], dim=1)
        x = self.input(x)
        x = self.activation(x)
        for i in range(self.n_layers):
            x = getattr(self, f"affine{i}")(x)
            x = self.activation(x)

        action_discrete = self.output_discrete(x)

        #astd = torch.nn.functional.softplus(action_std) + 1e-5
        adisc = torch.nn.functional.softmax(action_discrete, dim=-1)
        return adisc
    
    
class ContinuousPolicy(torch.nn.Module):
    def __init__(
            self, 
            input_dim, 
            output_dim_continuous, 
            hidden_dim, 
            n_layers=2,
            lower_bound=0, 
            upper_bound=1, 
            include_layer_number=False,
            include_material=False,
            activation="relu",
            n_objectives=0):
        super(ContinuousPolicy, self).__init__()
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.n_layers = n_layers
        self.include_layer_number = include_layer_number
        self.include_material = include_material

        indim = input_dim
        if include_layer_number:
            indim += 1
        if include_material:
            indim += 1
        if n_objectives > 0:
            indim += n_objectives
            
        self.input = torch.nn.Linear(indim, hidden_dim)
        
        for i in range(self.n_layers):
            setattr(self, f"affine{i}", torch.nn.Linear(hidden_dim, hidden_dim))

        if activation == "relu":
            self.activation = torch.nn.ReLU()
        elif activation == "tanh":
            self.activation = torch.nn.Tanh()
        elif activation == "silu":
            self.activation = torch.nn.SiLU()
        else:
            raise Exception(f"Activation not supported: {activation}")
        
        self.dropout = torch.nn.Dropout(p=0.0)
        self.output_continuous_mean = torch.nn.Linear(hidden_dim, output_dim_continuous)
        self.output_continuous_std = torch.nn.Linear(hidden_dim, output_dim_continuous)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x, layer_number=None, material=None, objective_weights=None):

        if layer_number is not None and self.include_layer_number:
            x = torch.cat([x, layer_number], dim=1)
        if layer_number is not None and self.include_material:
            x = torch.cat([x, material], dim=1)
        if objective_weights is not None:
            x = torch.cat([x, objective_weights], dim=1)

        x = self.input(x)
        x = self.activation(x)  
        #x = self.dropout(x)
        for i in range(self.n_layers):
            x = getattr(self, f"affine{i}")(x)
            x = self.activation(x)
        action_mean = self.output_continuous_mean(x)
        action_std = self.output_continuous_std(x)
        amean = torch.sigmoid(action_mean)*(self.upper_bound - self.lower_bound) + self.lower_bound
        astd = torch.sigmoid(action_std)*(self.upper_bound - self.lower_bound)*0.1 + 1e-8
        return amean, astd
    
class Value(torch.nn.Module):
    def __init__(
            self, 
            input_dim, 
            hidden_dim, 
            n_layers=2,
            output_dim=1,
            lower_bound=0, 
            upper_bound=1,
            include_layer_number=False,
            activation="relu",
            n_objectives=0):
        super(Value, self).__init__()
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.n_layers = n_layers
        self.include_layer_number = include_layer_number

        if include_layer_number:
            self.input = torch.nn.Linear(input_dim + 1 + n_objectives, hidden_dim)
        else:
            self.input = torch.nn.Linear(input_dim + n_objectives, hidden_dim)
        for i in range(self.n_layers):
            setattr(self, f"affine{i}", torch.nn.Linear(hidden_dim, hidden_dim))

        if activation == "relu":
            self.activation = torch.nn.ReLU()
        elif activation == "tanh":
            self.activation = torch.nn.Tanh()
        elif activation == "silu":
            self.activation = torch.nn.SiLU()
        else:
            raise Exception(f"Activation not supported: {activation}")
        
        self.output = torch.nn.Linear(hidden_dim, output_dim)
        self.dropout = torch.nn.Dropout(p=0.0)


    def forward(self, x, layer_number=None, objective_weights=None):
        if layer_number is not None and self.include_layer_number:
            x = torch.cat([x, layer_number], dim=1)
        if objective_weights is not None:
            x = torch.cat([x, objective_weights], dim=1)
        x = self.input(x)
        x = self.activation(x)

        for i in range(self.n_layers):
            x = getattr(self, f"affine{i}")(x)
            x = self.activation(x)

        output = self.output(x)
        #x = self.dropout(x)
        return output
    


class HyperNetwork(torch.nn.Module):
    def __init__(self, latent_dim, lstm_input_size, lstm_hidden_size, fc_in_size, fc_out_size):
        super(HyperNetwork, self).__init__()
        self.latent_dim = latent_dim
        self.lstm_input_size = lstm_input_size
        self.lstm_hidden_size = lstm_hidden_size
        self.fc_in_size = fc_in_size
        self.fc_out_size = fc_out_size

        # Size of weights for LSTM: (input_size + hidden_size) * hidden_size * 4 (for input, forget, cell, output gates)
        self.lstm_weight_size = (lstm_input_size + lstm_hidden_size) * lstm_hidden_size * 4

        # Size of weights for Linear layer: fc_in_size * fc_out_size
        self.linear_weight_size = fc_in_size * fc_out_size

        # Total size of generated weights
        total_weights = self.lstm_weight_size + self.linear_weight_size

        # Simple MLP as the hypernetwork
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, total_weights)
        )

    def forward(self, z):
        batch_size = z.size(0)
        weights = self.mlp(z)  # Shape: (batch_size, total_weights)

        # Split weights for LSTM and Linear
        lstm_weights = weights[:, :self.lstm_weight_size]  # Shape: (batch_size, lstm_weight_size)
        linear_weights = weights[:, self.lstm_weight_size:]  # Shape: (batch_size, linear_weight_size)

        # Reshape into weight matrices
        # LSTM weights: (batch_size, 4 * hidden_size, input_size + hidden_size)
        lstm_weights = lstm_weights.view(batch_size, 4 * self.lstm_hidden_size, self.lstm_input_size + self.lstm_hidden_size)

        # Linear weights: (batch_size, out_features, in_features)
        linear_weights = linear_weights.view(batch_size, self.fc_out_size, self.fc_in_size)

        return lstm_weights, linear_weights


class HyperDiscretePolicy(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim_discrete,
        hidden_dim,
        n_layers=2,
        hyper_hidden_dim=128,
        hyper_n_layers=2,
        activation="relu",
        n_objectives=0
    ):
        super(HyperDiscretePolicy, self).__init__()
        self.input_dim = input_dim
        self.output_dim_discrete = output_dim_discrete
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_objectives = n_objectives

        self.total_params = self._compute_target_params()

        # Hypernetwork takes input_dim + n_objectives as input
        self.hyper_input_dim = input_dim + n_objectives

        # Simple MLP hypernetwork
        layers = []
        in_dim = self.hyper_input_dim
        for _ in range(hyper_n_layers):
            layers.append(nn.Linear(in_dim, hyper_hidden_dim))
            layers.append(self._get_activation(activation))
            in_dim = hyper_hidden_dim
        layers.append(nn.Linear(hyper_hidden_dim, self.total_params))  # Output all weights
        self.hyper_mlp = nn.Sequential(*layers)

    def _get_activation(self, name):
        if name == "relu":
            return nn.ReLU()
        elif name == "tanh":
            return nn.Tanh()
        elif name == "silu":
            return nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {name}")

    def _compute_target_params(self):
        """Compute total number of parameters for target DiscretePolicy-like network."""
        # Input layer (no n_objectives in target network input)
        params = (self.input_dim * self.hidden_dim) + self.hidden_dim

        # Hidden layers
        for _ in range(self.n_layers):
            params += (self.hidden_dim * self.hidden_dim) + self.hidden_dim

        # Output layer
        params += (self.hidden_dim * self.output_dim_discrete) + self.output_dim_discrete
        return params

    def forward(self, x, objective_weights):
        """
        Inputs:
            x: [batch_size, input_dim]
            objective_weights: [batch_size, n_objectives]
        Returns:
            weight_dict: dictionary of parameter tensors for target network
        """
        hyper_input = torch.cat([x, objective_weights], dim=-1)
        flat_params = self.hyper_mlp(hyper_input)  # [batch_size, total_params]

        return self._unpack_weights(flat_params)

    def _unpack_weights(self, flat_params):
        """
        Converts flat parameter vector into a dictionary of weights/biases for the target network.
        Assumes batch size of 1 or processes batch-wise weights.
        """
        batch_size = flat_params.size(0)
        idx = 0
        weights = []

        def slice_weights(shape):
            nonlocal idx
            numel = shape[0] * shape[1]
            w = flat_params[:, idx:idx + numel].reshape(batch_size, *shape)
            idx += numel
            return w

        def slice_bias(shape):
            nonlocal idx
            b = flat_params[:, idx:idx + shape[0]].reshape(batch_size, shape[0])
            idx += shape[0]
            return b

        param_dict = {}
        # Input layer
        param_dict['input.weight'] = slice_weights((self.hidden_dim, self.input_dim))
        param_dict['input.bias'] = slice_bias((self.hidden_dim,))

        # Hidden layers
        for i in range(self.n_layers):
            param_dict[f"affine{i}.weight"] = slice_weights((self.hidden_dim, self.hidden_dim))
            param_dict[f"affine{i}.bias"] = slice_bias((self.hidden_dim,))

        # Output layer
        param_dict['output.weight'] = slice_weights((self.output_dim_discrete, self.hidden_dim))
        param_dict['output.bias'] = slice_bias((self.output_dim_discrete,))

        return param_dict