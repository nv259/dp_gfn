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


def batched_base_mask(num_words: torch.Tensor|list[int], num_variables: int, root_first=True, device=torch.device('cpu')) -> torch.Tensor:
    """
    Generate the batched base mask for the given number of words and number of variables. 
    """
    mask = torch.zeros(num_variables, num_variables, dtype=torch.bool, device=device) 
    mask = mask.repeat(len(num_words), 1, 1)
    
    for batch_idx, num_word in enumerate(num_words):
        if root_first:
            mask[batch_idx, 0, 1 : num_word + 1] = True
        else:
            mask[batch_idx, num_variables - 1, 1 : num_word + 1] = True
    
    return mask


def base_mask(num_words: int, num_variables: int, root_first=True, device=torch.device('cpu')) -> torch.Tensor:
    mask = torch.zeros(num_variables, num_variables, dtype=torch.bool, device=device)
    
    if root_first:
        mask[0, 1 : num_words + 1] = True
    else:
        mask[num_variables - 1, 1 : num_words + 1] = True
    
    return mask


def check_done(adjacency):
    """
    Checks if the graph is fully connected, indicating that the parsing is done.

    Args:
        adjacency (torch.Tensor): Adjacency matrix of the graph.

    Returns:
        torch.Tensor: Boolean tensor indicating whether the graph is fully connected.
    """
    return torch.all(torch.sum(adjacency, dim=1) == 1, dim=1)