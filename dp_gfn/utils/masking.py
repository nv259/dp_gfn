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


def batched_base_mask(
    batch_size: int,
    num_variables: int,
    num_words_list: list[int],
) -> np.ndarray:
    """Generate batch of base masks for initial graph with shape [batch_size, num_variables, num_variables], i.e., batchified version of base_mask

    Args:
        batch_size (int): Batch size
        num_variables (int): Total number of nodes in the graphs (including [ROOT] and [PAD])
        num_words_list (list[int]): List of actual number of nodes w.r.t each graph (excluding [PAD])
        
    Returns: 
        np.ndarray: base mask with shape [batch_size, num_variables, num_variables]
    """
    mask = base_mask(num_variables, num_variables)
    mask = np.repeat(mask[np.newaxis, ...], batch_size, axis=0) 
    
    for idx, num_words in enumerate(num_words_list):
        mask[idx, num_words + 1:, :] = False
        mask[idx, :, num_words + 1:] = False
    
    return mask


def base_mask(
    num_variables: int,
    num_words: int,
) -> np.ndarray:
    """Generate base mask for initial graph with shape [num_variables, num_variables]

    Args:
        num_variables (int): Total number of nodes in the graph (including [ROOT] and [PAD])
        num_words (int): Actual number of nodes in the graph (excluding [PAD])

    Returns:
        np.ndarray: base mask with shape [num_variables, num_variables]
    """
    mask = np.ones((num_variables, num_variables), dtype=np.bool_)
    mask[:, 0] = False                              # No incoming edges into ROOT
    np.fill_diagonal(mask, False)                   # No self-loop edges
    mask[num_words + 1:, :] = False                 # No [PAD] (head)
    mask[:, num_words + 1:] = False                 # No [PAD] (dependent)
    return mask


def mask_logits(logits, mask):
    return mask * logits + (1 - mask) * MASKED_VALUE


def batch_random_choice(key, probas, mask, delta):
    # Sample from the distribution
    uniform = random.uniform(key, shape=(delta.shape))
    cum_probas = jnp.cumsum(probas, axis=-1)
    samples = jnp.sum(cum_probas < uniform, axis=-1)

    # mask = mask.reshape(mask.shape[0], -1)
    # is_valid = jnp.take_along_axis(mask, samples, axis=1)    # TODO: Implement precautionary measure for potential failure in sampling actions

    return samples


def sample_action(key, log_pi, mask, delta, ret_backward=False):
    key, subkey1, subkey2 = random.split(key, 3)
    
    log_uniform = uniform_log_policy(mask)
    is_exploration = random.bernoulli(
        subkey1, p=delta, shape=(delta.shape)
    )

    log_pi = jnp.where(is_exploration, log_uniform, log_pi)

    actions = batch_random_choice(
        subkey2, jnp.exp(log_pi), mask, delta
    )

    log_pF = log_pi[actions]

    if ret_backward:
        log_pB = uniform_log_policy(mask, is_forward=False)
        return key, actions, log_pF, log_pB
        
    return key, actions, log_pF
        

def uniform_log_policy(mask, is_forward=True):
    num_valid_actions = jnp.sum(mask, axis=-1, keepdims=True)
    log_pi = -jnp.log(num_valid_actions)

    if is_forward:
        log_pi = mask_logits(log_pi, mask)
    else:
        log_pi = log_pi.squeeze(-1)

    return log_pi


class StateBatch:
    def __init__(
        self,
        num_variables,
        num_words_list,
    ):
        self.batch_size = len(num_words_list)
        self.num_variables = num_variables
        
        self._data = {
            "labels": np.zeros((self.batch_size, self.num_variables), dtype=np.int32),
            "mask": batched_base_mask( 
                batch_size=self.batch_size,
                num_variables=self.num_variables,
                num_words_list=num_words_list,
            ),
            "num_words": num_words_list,
            "adjacency": np.zeros(
                (self.batch_size, self.num_variables, self.num_variables),
                dtype=np.int32,
            ),
        }
        
        self._closure_T = np.repeat(
            np.eye(self.num_variables, dtype=np.bool_)[np.newaxis],
            self.batch_size,
            axis=0,
        )
        
        # Who let a sentence with only one word here???
        for i in range(self.batch_size):
            if self['num_words'][i] == 1:
                self['adjacency'][i, 0, 1] = 1
                self['labels'][i, 1] = 1

    def __len__(self):
        return len(self["labels"])

    def __getitem__(self, key: str):
        return self._data[key]

    def get_full_data(self, index: int):
        labels = self["labels"][index]
        mask = self["mask"][index]
        num_words = self["num_words"][index]
        adjacency = self["adjacency"][index]

        return {
            "labels": labels,
            "mask": mask,
            "num_words": num_words,
            "adjacency": adjacency,
        }

    ## The base code of this function is from: https://github.com/tristandeleu/jax-dag-gflownet/blob/master/dag_gflownet/env.py#L96
    def step(self, actions):
        targets, sources = actions
        dones = self.check_done()
        sources, targets = sources[~dones], targets[~dones]
 
        assert np.all(sources <= self["num_words"][~dones]), "Invalid head node(s): Node(s) out of range"
        assert np.all(targets <= self["num_words"][~dones]), "Invalid dependent node(s): Node(s) out of range"
        
        if not np.all(self["mask"][~dones, sources, targets]):
            raise ValueError("Invalid action(s): Already existed edge(s), Self-loop, or Causing-cycle edge(s).")

        # Update adjacency matrices
        self._data["adjacency"][~dones, sources, targets] = 1
        self._data["labels"][~dones, targets] = 1
        # self["adjacency"][dones] = 0

        # Update transitive closure of transpose
        source_rows = np.expand_dims(self._closure_T[~dones, sources, :], axis=1)
        target_cols = np.expand_dims(self._closure_T[~dones, :, targets], axis=2)
        self._closure_T[~dones] |= np.logical_and(source_rows, target_cols)  # Outer product
        self._closure_T[dones] = np.eye(self.num_variables, dtype=np.bool_)

        # Update the mask
        self._data["mask"] = 1 - (self._data["adjacency"] + self._closure_T)
        num_parents = np.sum(self["adjacency"], axis=1, keepdims=True)
        self._data["mask"] *= (num_parents < 1)
        
        # Exclude all undue edges
        for batch_idx, num_word in enumerate(self["num_words"]):
            self._data['mask'][batch_idx, num_word + 1:, :] = False
            self._data['mask'][batch_idx, :, num_word + 1:] = False
        
        self._data["mask"][:, :, 0] = False 
    
    def check_done(self):
        return self['adjacency'].sum(axis=-1).sum(axis=-1) == self['num_words']