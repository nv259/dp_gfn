import torch
import torch.nn as nn
import torch.nn.functional as F

from dp_gfn.nets.initial_encoders import PrefEncoder, StateEncoder
from dp_gfn.nets.encoders import LinearTransformer, MLP 


class DPGFlowNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DPGFlowNet, self).__init__()

        self.pref_embedding = PrefEncoder()
        self.state_embedding = StateEncoder()
        self.encoder = LinearTransformer(input_dim, hidden_dim, output_dim)
        self.output = MLP(hidden_dim, output_dim)
        self.Z_mod = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.initial_encoder(x)
        x = self.state_encoder(x)
        x = self.main_encoder(x)
        return x