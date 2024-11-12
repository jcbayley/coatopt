import torch
import numpy as np
from torch.nn.utils.rnn import pad_packed_sequence


class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        #print(x.size())
        #print(self.pe[:x.size(1)].size())
        x = x + torch.swapaxes(self.pe[:x.size(1)], 0,1)
        return self.dropout(x)


class PreNetworkAttention(torch.nn.Module):
    def __init__(self, input_dim, output_dim, embed_dim, num_heads=2, num_layers=2):
        super(PreNetworkAttention, self).__init__()
        self.embedding = torch.nn.Linear(input_dim, embed_dim)
        encoder_layers = torch.nn.TransformerEncoderLayer(embed_dim, num_heads)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = torch.nn.Linear(embed_dim, output_dim)
        
    def forward(self, x, layer_number=None):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)  # Change to shape (seq_len, batch_size, embed_dim) for Transformer encoder
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)  # Change back to shape (batch_size, seq_len, embed_dim)
        x = self.fc(x)
        if layer_number != None:
            #indices = layer_number.flatten().view(x.size(0), 1, 1).to(torch.int64)
            #indices = indices.expand(x.size(0), 1, x.size(2))
            #x = torch.gather(x, 1, indices)
            x = torch.mean(x, dim=1)
            #x[:, layer_number.flatten()]
        else:
            x = torch.mean(x, dim=1)  # Global average pooling
        return x.flatten(start_dim=1)
    
class PreNetworkLinear(torch.nn.Module):
    def __init__(
            self, 
            input_dim, 
            output_dim, 
            hidden_dim, 
            n_layers=3,
            lower_bound=0, 
            upper_bound=1,
            include_layer_number=False,
            activation="relu"):
        super(PreNetworkLinear, self).__init__()
        self.output_dim = output_dim
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        if include_layer_number:
            input_dim = input_dim + 1
        self.input = torch.nn.Linear(input_dim, hidden_dim)
        self.n_layers = n_layers

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

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x, layer_number=None):
        x = x.flatten(1)
        if layer_number is not None:
            x = torch.cat([x, layer_number], dim=1)
        x = self.input(x)
        x = self.activation(x)

        for i in range(self.n_layers):
            x = getattr(self, f"affine{i}")(x)
            x = self.activation(x)

        out= self.output(x)
        return out
    
class PreNetworkLSTM(torch.nn.Module):
    def __init__(
            self, 
            input_dim, 
            output_dim, 
            hidden_dim, 
            lower_bound=0, 
            upper_bound=1,
            include_layer_number=False,
            n_layers=2):
        super(PreNetworkLSTM, self).__init__()
        self.output_dim = output_dim
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.include_layer_number = include_layer_number
        
        # LSTM layer
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers = n_layers)
        
        # Fully connected layers after LSTM
        self.affine1 = torch.nn.Linear(hidden_dim+ (1 if include_layer_number else 0), hidden_dim)
        self.affine2 = torch.nn.Linear(hidden_dim, hidden_dim)

        self.output = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x, layer_number=None, packed=False):
        # x shape: (B, N, I)
        
        # LSTM expects input shape (B, N, I) and we use batch_first=True
        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out shape: (B, N, hidden_dim)

        if packed: 
            lstm_out, output_lengths = pad_packed_sequence(lstm_out, batch_first=True)
            # Use the last output of LSTM for further processing
            out = lstm_out[torch.arange(lstm_out.size(0)), output_lengths - 1, :]
            #out = lstm_out[:, output_lengths.view(-1) - 1, :]        # Use the last output of LSTM for further processing
        else:
            out = lstm_out[:, -1, :]  # Take the output from the last layer (N)
        
        if self.include_layer_number and layer_number is not None:
            # Expand layer number to match the (B, N) shape
            layer_number = layer_number#.unsqueeze(-1)  # Shape: (B, N, 1)
            #print(out.size(), layer_number.size())
            out = torch.cat([out, layer_number], dim=-1)  # Concatenate along the last dimension

        # Fully connected layers
        out = self.affine1(out)
        out = torch.nn.functional.relu(out)
        out = self.affine2(out)
        out = torch.nn.functional.relu(out)
        
        # Final output layer
        out = self.output(out)  # Shape: (B, O)
        
        return out
    