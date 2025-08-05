import torch
import torch.nn as nn
import torch.nn.functional as F

class DiscretePolicy(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim_discrete,
        hidden_dim,
        n_layers=2,
        include_layer_number=False,
        hyper_hidden_dim=128,
        hyper_n_layers=2,
        activation="relu",
        n_objectives=0,
        lower_bound=0,
        upper_bound=1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim_discrete = output_dim_discrete
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.include_layer_number = include_layer_number
        self.activation_fn = self._get_activation(activation)
        self.n_objectives = n_objectives

        self.target_input_dim = input_dim + 1 if include_layer_number else input_dim
        self.total_params = self._compute_target_param_count()

        # Hypernetwork: takes in only objective_weights
        layers = []
        in_dim = n_objectives
        for _ in range(hyper_n_layers):
            layers.append(nn.Linear(in_dim, hyper_hidden_dim))
            layers.append(self._get_activation(activation))
            in_dim = hyper_hidden_dim
        layers.append(nn.Linear(hyper_hidden_dim, self.total_params))
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

    def _compute_target_param_count(self):
        count = 0
        # Input layer
        count += self.target_input_dim * self.hidden_dim + self.hidden_dim
        # Hidden layers
        for _ in range(self.n_layers):
            count += self.hidden_dim * self.hidden_dim + self.hidden_dim
        # Output layer
        count += self.hidden_dim * self.output_dim_discrete + self.output_dim_discrete
        return count

    def forward(self, x, layer_number=None, objective_weights=None):
        """
        x: [B, input_dim]
        layer_number: [B, 1] or None
        objective_weights: [B, n_objectives]
        """
        if objective_weights is None:
            raise ValueError("objective_weights must be provided")

        batch_size = x.size(0)

        if self.include_layer_number and layer_number is not None:
            x = torch.cat([x, layer_number], dim=1)
        elif self.include_layer_number and layer_number is None:
            raise ValueError("layer_number must be provided if include_layer_number is True")

        # Generate weights from hypernetwork
        flat_params = self.hyper_mlp(objective_weights)
        param_dict = self._unpack_weights(flat_params)

        # Functional forward pass with generated weights
        x = self._functional_forward(x, param_dict)
        return x

    def _unpack_weights(self, flat_params):
        batch_size = flat_params.size(0)
        idx = 0
        params = {}

        def slice_weight(out_dim, in_dim, name):
            nonlocal idx
            w = flat_params[:, idx:idx + out_dim * in_dim].reshape(batch_size, out_dim, in_dim)
            idx += out_dim * in_dim
            params[f"{name}.weight"] = w

        def slice_bias(out_dim, name):
            nonlocal idx
            b = flat_params[:, idx:idx + out_dim].reshape(batch_size, out_dim)
            idx += out_dim
            params[f"{name}.bias"] = b

        # Input layer
        slice_weight(self.hidden_dim, self.target_input_dim, "input")
        slice_bias(self.hidden_dim, "input")

        # Hidden layers
        for i in range(self.n_layers):
            slice_weight(self.hidden_dim, self.hidden_dim, f"affine{i}")
            slice_bias(self.hidden_dim, f"affine{i}")

        # Output layer
        slice_weight(self.output_dim_discrete, self.hidden_dim, "output_discrete")
        slice_bias(self.output_dim_discrete, "output_discrete")

        return params

    def _functional_forward(self, x, param_dict):
        """
        Functional forward pass using generated weights.
        """
        batch_size = x.size(0)

        def linear(batch_w, batch_b, input_x):
            out = torch.bmm(batch_w, input_x.unsqueeze(-1)).squeeze(-1) + batch_b
            return out

        # Input layer
        x = linear(param_dict["input.weight"], param_dict["input.bias"], x)
        x = self.activation_fn(x)

        # Hidden layers
        for i in range(self.n_layers):
            x = linear(param_dict[f"affine{i}.weight"], param_dict[f"affine{i}.bias"], x)
            x = self.activation_fn(x)

        # Output layer
        x = linear(param_dict["output_discrete.weight"], param_dict["output_discrete.bias"], x)
        return F.softmax(x, dim=-1)
    

class ContinuousPolicy(nn.Module):
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
        hyper_hidden_dim=128,
        hyper_n_layers=2,
        activation="relu",
        n_objectives=0,
    ):
        super().__init__()
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.include_layer_number = include_layer_number
        self.include_material = include_material
        self.activation_fn = self._get_activation(activation)
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.output_dim_continuous = output_dim_continuous

        self.input_dim = input_dim
        self.n_objectives = n_objectives

        # Input to target network
        self.target_input_dim = input_dim
        if include_layer_number:
            self.target_input_dim += 1
        if include_material:
            self.target_input_dim += 1

        self.total_params = self._compute_target_param_count()

        # Hypernetwork: generates parameters from objective_weights
        layers = []
        in_dim = n_objectives
        for _ in range(hyper_n_layers):
            layers.append(nn.Linear(in_dim, hyper_hidden_dim))
            layers.append(self._get_activation(activation))
            in_dim = hyper_hidden_dim
        layers.append(nn.Linear(hyper_hidden_dim, self.total_params))
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

    def _compute_target_param_count(self):
        count = 0
        # Input layer
        count += self.target_input_dim * self.hidden_dim + self.hidden_dim
        # Hidden layers
        for _ in range(self.n_layers):
            count += self.hidden_dim * self.hidden_dim + self.hidden_dim
        # Output layers: mean and std
        count += 2 * (self.hidden_dim * self.output_dim_continuous + self.output_dim_continuous)
        return count

    def forward(self, x, layer_number=None, material=None, objective_weights=None):
        if objective_weights is None:
            raise ValueError("objective_weights must be provided")

        batch_size = x.size(0)

        # Construct input to target net
        if self.include_layer_number:
            if layer_number is None:
                raise ValueError("layer_number must be provided if include_layer_number=True")
            x = torch.cat([x, layer_number], dim=1)
        if self.include_material:
            if material is None:
                raise ValueError("material must be provided if include_material=True")
            x = torch.cat([x, material], dim=1)

        # Generate parameters
        flat_params = self.hyper_mlp(objective_weights)
        param_dict = self._unpack_weights(flat_params)

        # Functional forward
        mean, std = self._functional_forward(x, param_dict)
        return mean, std

    def _unpack_weights(self, flat_params):
        batch_size = flat_params.size(0)
        idx = 0
        params = {}

        def slice_weight(out_dim, in_dim, name):
            nonlocal idx
            w = flat_params[:, idx:idx + out_dim * in_dim].reshape(batch_size, out_dim, in_dim)
            idx += out_dim * in_dim
            params[f"{name}.weight"] = w

        def slice_bias(out_dim, name):
            nonlocal idx
            b = flat_params[:, idx:idx + out_dim].reshape(batch_size, out_dim)
            idx += out_dim
            params[f"{name}.bias"] = b

        # Input layer
        slice_weight(self.hidden_dim, self.target_input_dim, "input")
        slice_bias(self.hidden_dim, "input")

        # Hidden layers
        for i in range(self.n_layers):
            slice_weight(self.hidden_dim, self.hidden_dim, f"affine{i}")
            slice_bias(self.hidden_dim, f"affine{i}")

        # Output: mean
        slice_weight(self.output_dim_continuous, self.hidden_dim, "output_mean")
        slice_bias(self.output_dim_continuous, "output_mean")

        # Output: std
        slice_weight(self.output_dim_continuous, self.hidden_dim, "output_std")
        slice_bias(self.output_dim_continuous, "output_std")

        return params

    def _functional_forward(self, x, param_dict):
        batch_size = x.size(0)

        def linear(batch_w, batch_b, input_x):
            return torch.bmm(batch_w, input_x.unsqueeze(-1)).squeeze(-1) + batch_b

        x = linear(param_dict["input.weight"], param_dict["input.bias"], x)
        x = self.activation_fn(x)

        for i in range(self.n_layers):
            x = linear(param_dict[f"affine{i}.weight"], param_dict[f"affine{i}.bias"], x)
            x = self.activation_fn(x)

        mean = linear(param_dict["output_mean.weight"], param_dict["output_mean.bias"], x)
        std = linear(param_dict["output_std.weight"], param_dict["output_std.bias"], x)

        # Apply bounds
        mean = torch.sigmoid(mean) * (self.upper_bound - self.lower_bound) + self.lower_bound
        std = torch.sigmoid(std) * (self.upper_bound - self.lower_bound) * 0.1 + 1e-8

        return mean, std
    

class Value(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        n_layers=2,
        output_dim=1,
        lower_bound=0,
        upper_bound=1,
        include_layer_number=False,
        hyper_hidden_dim=128,
        hyper_n_layers=2,
        activation="relu",
        n_objectives=0,
    ):
        super().__init__()
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.include_layer_number = include_layer_number
        self.activation_fn = self._get_activation(activation)
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.n_objectives = n_objectives

        # Input to target network
        self.target_input_dim = input_dim + int(include_layer_number)

        self.total_params = self._compute_target_param_count()

        # Hypernetwork
        layers = []
        in_dim = n_objectives
        for _ in range(hyper_n_layers):
            layers.append(nn.Linear(in_dim, hyper_hidden_dim))
            layers.append(self._get_activation(activation))
            in_dim = hyper_hidden_dim
        layers.append(nn.Linear(hyper_hidden_dim, self.total_params))
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

    def _compute_target_param_count(self):
        count = 0
        # Input layer
        count += self.target_input_dim * self.hidden_dim + self.hidden_dim
        # Hidden layers
        for _ in range(self.n_layers):
            count += self.hidden_dim * self.hidden_dim + self.hidden_dim
        # Output layer
        count += self.hidden_dim * self.output_dim + self.output_dim
        return count

    def forward(self, x, layer_number=None, objective_weights=None):
        if objective_weights is None:
            raise ValueError("objective_weights must be provided")

        batch_size = x.size(0)

        # Add layer number if included
        if self.include_layer_number:
            if layer_number is None:
                raise ValueError("layer_number must be provided if include_layer_number=True")
            x = torch.cat([x, layer_number], dim=1)

        flat_params = self.hyper_mlp(objective_weights)
        param_dict = self._unpack_weights(flat_params)
        output = self._functional_forward(x, param_dict)
        return output

    def _unpack_weights(self, flat_params):
        batch_size = flat_params.size(0)
        idx = 0
        params = {}

        def slice_weight(out_dim, in_dim, name):
            nonlocal idx
            w = flat_params[:, idx:idx + out_dim * in_dim].reshape(batch_size, out_dim, in_dim)
            idx += out_dim * in_dim
            params[f"{name}.weight"] = w

        def slice_bias(out_dim, name):
            nonlocal idx
            b = flat_params[:, idx:idx + out_dim].reshape(batch_size, out_dim)
            idx += out_dim
            params[f"{name}.bias"] = b

        # Input layer
        slice_weight(self.hidden_dim, self.target_input_dim, "input")
        slice_bias(self.hidden_dim, "input")

        # Hidden layers
        for i in range(self.n_layers):
            slice_weight(self.hidden_dim, self.hidden_dim, f"affine{i}")
            slice_bias(self.hidden_dim, f"affine{i}")

        # Output layer
        slice_weight(self.output_dim, self.hidden_dim, "output")
        slice_bias(self.output_dim, "output")

        return params

    def _functional_forward(self, x, param_dict):
        batch_size = x.size(0)

        def linear(batch_w, batch_b, input_x):
            return torch.bmm(batch_w, input_x.unsqueeze(-1)).squeeze(-1) + batch_b

        x = linear(param_dict["input.weight"], param_dict["input.bias"], x)
        x = self.activation_fn(x)

        for i in range(self.n_layers):
            x = linear(param_dict[f"affine{i}.weight"], param_dict[f"affine{i}.bias"], x)
            x = self.activation_fn(x)

        output = linear(param_dict["output.weight"], param_dict["output.bias"], x)
        return output