import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, output_sizes, activation='GELU', dropout=0.0):
        super().__init__()
        
        if isinstance(activation, str):
            activation = getattr(nn, activation)()
        
        layers = []
        for idx, output_size in enumerate(output_sizes[:-1]):
            if idx == 0: continue
            
            layers.append(nn.Linear(output_sizes[idx - 1], output_size))
            layers.append(activation)
        layers.append(nn.Linear(output_sizes[-2], output_sizes[-1]))
        layers.append(nn.Dropout(dropout))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class BiAffine(nn.Module):
    """Biaffine attention layer."""

    def __init__(self, input_dim, output_dim, hidden_dim=128, dropout=0.0):
        super(BiAffine, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Mapping to intermediate dimension
        self.lab_head = MLP([input_dim, hidden_dim, input_dim], nn.ReLU(), dropout=dropout)
        self.lab_dep = MLP([input_dim, hidden_dim, input_dim], nn.ReLU(), dropout=dropout)

        self.U = nn.Parameter(torch.FloatTensor(output_dim, input_dim, input_dim))
        nn.init.xavier_uniform(self.U)

    def forward(self, Rh, Rd):
        Rh = self.lab_head(Rh)
        Rd = self.lab_dep(Rd)

        Rh = Rh.unsqueeze(1)
        Rd = Rd.unsqueeze(1)

        # TODO: Elaborate the score by taking into account head (dep) score and bias
        S = Rh @ self.U @ Rd.transpose(-1, -2)

        return S.squeeze(1)
