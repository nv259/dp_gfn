import numpy as np
import torch


def encode(decoded):
    encoded = decoded.reshape(decoded.shape[0], -1).detach().cpu().numpy()
    return torch.from_numpy(np.packbits(encoded, axis=1)) 


def decode(encoded, dtype=np.float32):
    pass


def base_mask(num_variables, root_first=True, device=torch.device('cpu')) -> torch.Tensor:
    pass