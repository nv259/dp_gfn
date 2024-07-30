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
    num_words: list[int],
    num_variables: int,
):
    mask = np.zeros((num_variables, num_variables), dtype=bool)
    mask = np.repeat(mask[np.newaxis, ...], len(num_words), axis=0)

    for batch_idx, num_word in enumerate(num_words):
        num_word = int(num_word)
        mask[batch_idx, 0, 1 : num_word + 1] = True

    return mask


def base_mask(
    num_words: int, num_variables: int
):
    mask = np.zeros((num_variables, num_variables), dtype=np.bool_)
    mask[0, 1 : num_words + 1] = True

    return mask


def mask_logits(logits, masks):
    return masks * logits + (1 - masks) * MASKED_VALUE


def check_done(adjacency_matrices, num_words):
    num_edges = adjacency_matrices.sum((1, 2)) 

    return num_edges == num_words


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
                num_variables**2 
            ),
            dtype=np.int_,
        )

        self._data = {
            "labels": labels,
            "mask": batched_base_mask(
                    num_words=num_words,
                    num_variables=self.num_variables,
            ),
            "adjacency": 
                np.zeros(
                    (self.batch_size, self.num_variables, self.num_variables), dtype=np.int_
            ),
            "num_words": num_words,
            "done": np.zeros(batch_size, dtype=np.bool_),
        }
        self._closure_T = np.eye(self.num_variables, dtype=np.bool_)
        self._closure_T = np.repeat(self._closure_T[np.newaxis, ...], batch_size, axis=0)

    def __len__(self):
        return len(self._data["labels"], )

    def __getitem__(self, key: str):
        return self._data[key]

    def get_full_data(self, index: int):
        labels = self._data["labels"][index]
        mask = self._data["mask"][index]
        adjacency = self._data["adjacency"][index]
        num_words = self._data["num_words"][index]
        done = self._data["done"][index]

        return {
            "labels": labels,
            "mask": mask,
            "adjacency": adjacency,
            "num_words": num_words,
            "done": done,
        }

    def step(self, actions):
        sources, targets = divmod(actions, self.num_variables)
        dones = check_done(self._data["adjacency"], self._data["num_words"])
        sources, targets = sources[~dones], targets[~dones]
        masks = self.__getitem__("mask")
        adjacencies = self.__getitem__("adjacency")

        if not np.all(masks[~dones, sources, targets]):
            raise ValueError("Invalid action")

        # Update adjacency matrices
        adjacencies[~dones, sources, targets] = 1
        # adjacencies[dones] = 0

        # Update transitive closure of transpose
        source_rows = np.expand_dims(self._closure_T[~dones, sources, :], axis=1)
        target_cols = np.expand_dims(self._closure_T[~dones, :, targets], axis=2)
        self._closure_T[~dones] |= np.logical_and(
            source_rows, target_cols
        )  # Outer product
        self._closure_T[dones] = np.eye(self.num_variables, dtype=np.bool_)

        # Update dones
        self._data["done"][~dones] = check_done(
            adjacencies[~dones], self._data["num_words"][~dones]
        )

        # Update the mask
        masks = 1 - (adjacencies + self._closure_T)
        num_parents = np.sum(adjacencies, axis=1, keepdims=True)
        masks *= num_parents < 1  # each node has only one parent node
        # Exclude all undue edges
        for batch_idx, num_words in enumerate(self._data["num_words"]):
            masks[batch_idx, num_words + 1: ] = False
            masks[batch_idx, :, num_words + 1:] = False
            
        masks[:, :, 0] = False

        self._data["mask"] = masks
        self._data["adjacency"] = adjacencies
        self._data["labels"] = self._data["adjacency"].copy().reshape(
            self.batch_size, -1
        )

    def reset(
        self,
    ):
        pass
