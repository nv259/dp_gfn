from torch import nn
from dp_gfn.nets.attention import LinearMultiHeadAttention


class MLP(nn.Module):
    def __init__(self, output_sizes, activation, dropout=0.0):
        super().__init__()

        layers = []
        for idx, output_size in enumerate(output_sizes[:-1]):
            layers.append(nn.Linear(output_sizes[idx - 1], output_size)) 
            layers.append(activation)
        layers.append(nn.Linear(output_sizes[-2], output_sizes[-1]))
        layers.append(nn.Dropout(dropout))
        
        self.layers = nn.Sequential(*layers)
         
    def forward(self, x):
        return self.layers(x)