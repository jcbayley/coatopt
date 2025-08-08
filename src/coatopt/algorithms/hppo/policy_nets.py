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
    
