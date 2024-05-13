import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers, dropout_rate, activation):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        layers = [nn.Linear(input_dim, hidden_layers[0]), activation()]
        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            layers.append(activation())
            layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(hidden_layers[-1], output_dim))
        self.layers = nn.Sequential(*self.layers)
        
    def forward(self, x):
        return self.layers(x)


class LinearTransformer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearTransformer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.linear(x)
    