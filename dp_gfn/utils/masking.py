import numpy as np
import jax.numpy as jnp 
from jax import random


MASKED_VALUE = -1e5


def encode(decoded):
    encoded = decoded.reshape(decoded.shape[0], -1)
    return np.packbits(encoded.astype(bool), axis=1)


def decode(encoded, num_variables):
    decoded = np.unpackbits(encoded, axis=-1, count=num_variables**2)
    decoded = decoded.reshape(*encoded.shape[:-1], num_variables, num_variables)

    return decoded


def batched_base_masks(
    batch_size: int,
    num_variables: int,
):
    # Mask for sampling edge with shape [len(num_words), num_variables]
    # 1. mask[visitted nodes] == True
    # 2. mask[un-visitted nodes] == False
    # To this end, only first elements, i.e. the `ROOT`, of the initial masks are set to True
    edge_mask = np.zeros((batch_size, num_variables), dtype=np.bool_)
    edge_mask[:, 0] = True
   
    # Mask for sampling next node (reverse of the previous edge mask)
    node_mask = np.ones((batch_size, num_variables), dtype=np.bool_)
    node_mask[:, 0] = False
    
    return edge_mask, node_mask


def base_masks(
    num_variables: int
):
    edge_mask = np.zeros(num_variables, dtype=np.bool_)
    edge_mask[0] = True
    
    node_mask = np.ones(num_variables, dtype=np.bool_)
    node_mask[0] = False 
    
    return edge_mask, node_mask
    

def mask_logits(logits, masks):
    return masks * logits + (1 - masks) * MASKED_VALUE


def check_done(masks, num_words):
    return masks[0].sum(axis=1) == num_words


def batch_random_choice(key, probas, masks):
    # Sample from the distribution
    uniform = random.uniform(key, shape=(probas.shape[0], 1))
    cum_probas = jnp.cumsum(probas, axis=1)
    samples = jnp.sum(cum_probas < uniform, axis=1, keepdims=True)

    # masks = masks.reshape(masks.shape[0], -1)
    # is_valid = jnp.take_along_axis(masks, samples, axis=1)    # TODO: Potential risk

    return samples


def uniform_log_policy(masks, is_forward=True):
    masks = masks.reshape(masks.shape[0], -1)
    num_valid_actions = jnp.sum(masks, axis=-1, keepdims=True)
    log_pi = -jnp.log(num_valid_actions)
   
    if is_forward:  
        log_pi = mask_logits(log_pi, masks)    
    else: 
        log_pi = log_pi.squeeze(-1)
        
    return log_pi


class StateBatch:
    def __init__(
        self,
        batch_size,
        num_variables,
        num_words,
    ):
        self.batch_size = batch_size
        self.num_variables = num_variables 
        
        labels = np.zeros(
            (
                batch_size, 
                num_variables
            ),
            dtype=np.int_,
        )

        self._data = {
            "labels": labels,
            "masks": batched_base_masks(    # (edge_mask, node_mask)
                    batch_size=self.batch_size,
                    num_variables=self.num_variables,
            ),
            "num_words": num_words,
        }

    def __len__(self):
        return len(self._data["labels"], )

    def __getitem__(self, key: str):
        return self._data[key]

    def get_full_data(self, index: int):
        labels = self._data["labels"][index]
        masks = self._data["masks"][index]
        num_words = self._data["num_words"][index]

        return {
            "labels": labels,
            "masks": masks,
            "num_words": num_words,
        }

    def step(self, actions, t):
        masks = self.__getitem__("masks")
        num_words = self.__getitem__('num_words')
        dones = check_done(masks, num_words)
        
        # Only pick the desired node at step 0 
        if t == 0:
            masks[1][:, actions] = False
            return 0
         
        masks[0] = np.logical_not(masks[1].copy())
        masks[0][:, 0] = False
        masks[1] = masks[1][:, actions] = False
        
        return 1
    
    def reset(
        self,
    ):
        pass
