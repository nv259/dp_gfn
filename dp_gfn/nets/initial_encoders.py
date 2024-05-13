import torch
import torch.nn as nn
import torch.nn.functional as F


class StateEncoder(nn.Module):
    def __init__(self):
        super(StateEncoder, self).__init__()
        
    def forward(self, x):
        return x
    

class PrefEncoder(nn.Module):
    def __init__(self):
        super(PrefEncoder, self).__init__()
        
    def forward(self, x):
        return x
    