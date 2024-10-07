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
        np.bool_: base mask with shape [num_variables, num_variables]
    """
    mask = np.ones((num_variables, num_variables), dtype=np.bool_)
    mask[:, 0] = False                              # No incoming edges into ROOT
    np.fill_diagonal(mask, False)                   # No self-loop edges
    mask[num_words + 1:, :] = False                 # No [PAD] (head)
    mask[:, num_words + 1:] = False                 # No [PAD] (dependent)
    return mask


def mask_logits(logits, mask):
    return mask * logits + (1 - mask) * MASKED_VALUE


def check_done(mask, num_words):
    return mask[0].sum(axis=1) == num_words, mask[1].sum(axis=1) == 0


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
        
        # Who let a sentence with only one word here???
        for i in range(self.batch_size):
            if self._data['num_words'][i] == 1:
                self._data['adjacency'][i, 0, 1] = 1
                self._data['labels'][i, 1] = 1

    def __len__(self):
        return len(
            self._data["labels"],
        )

    def __getitem__(self, key: str):
        return self._data[key]

    def get_full_data(self, index: int):
        labels = self._data["labels"][index]
        mask = self._data["mask"][index]
        num_words = self._data["num_words"][index]
        adjacency = self._data["adjacency"][index]

        return {
            "labels": labels,
            "mask": mask,
            "num_words": num_words,
            "adjacency": adjacency,
        }

    def step(self, node_ids, prev_node_ids, actions=None):
        mask = self.__getitem__("mask")
        num_words = self.__getitem__("num_words")
        edge_dones, node_dones = check_done(mask, num_words)
        batch_ids = np.arange(self.batch_size)

        mask[1][batch_ids[~node_dones], node_ids[~node_dones]] = False
        self._data["mask"][1] = mask[1]

        if actions is None:
            return 0

        mask[0][batch_ids[~edge_dones], prev_node_ids[~edge_dones]] = True
        mask[0][batch_ids[~edge_dones], 0] = False  # Ensure no outcoming edge from ROOT
        self._data["mask"][0] = mask[0]
        actions = actions.squeeze(-1)
        self._data["adjacency"][batch_ids[~edge_dones], actions[~edge_dones], prev_node_ids[~edge_dones]] = 1
        self._data["labels"][batch_ids[~edge_dones], prev_node_ids[~edge_dones]] = 1

        return 1

    def reset(
        self,
    ):
        pass
