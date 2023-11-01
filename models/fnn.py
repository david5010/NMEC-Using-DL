# Feed-Forward Architecture
import torch
import torch.nn as nn

class FNN(nn.Module):
    def __init__(self, input_dims, output_dim, hidden_dims, activation=nn.ReLU(), batch_norm = False):
        super(FNN, self).__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim

        layers = []
        prev_dim = input_dims
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(activation)
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    