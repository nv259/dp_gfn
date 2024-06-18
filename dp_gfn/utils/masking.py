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

    Only edges from ROOT to 1-num_words nodes are left available (True). Otherwise, set to False.
    """
    print(num_words, len(num_words))
    input()
    mask = torch.zeros(num_variables, num_variables, dtype=torch.bool, device=device) 
    mask = mask.repeat(len(num_words), 1, 1)
    
    for batch_idx, num_word in enumerate(num_words):
        num_word = int(num_word)
        
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


def mask_logits(logits, masks, MASKED_VALUE=1e-5):
    masks = masks.reshape(logits.shape)
    
    return masks * logits + (1 - masks) * MASKED_VALUE


def mask_uniform_logits(logits, masks):
    masks = masks.reshape(logits.shape)
    uniform_logits = torch.ones_like(logits) * 0.5
    
    return masks * uniform_logits + (1 - masks) * logits