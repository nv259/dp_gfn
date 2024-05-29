import numpy as np
import torch


def encode(decoded):
    encoded = decoded.reshape(decoded.shape[0], -1).detach().cpu().numpy()
    return torch.from_numpy(np.packbits(encoded, axis=1)) 


def decode(encoded, num_variables, dtype=torch.float32):
    encoded = encoded.detach().cpu().numpy()
    decoded = np.unpackbits(encoded, axis=-1, count=num_variables ** 2) 
    decoded = decoded.reshape(*encoded.shape[:-1], num_variables, num_variables)
    
    return torch.from_numpy(decoded).to(dtype)


def base_mask(num_variables, root_first=True, device=torch.device('cpu')) -> torch.Tensor:
    pass