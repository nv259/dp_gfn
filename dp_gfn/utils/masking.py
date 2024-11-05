import torch.nn.functional as F
import jax.numpy as jnp
import numpy as np
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
        mask[idx, num_words + 1 :, :] = False
        mask[idx, :, num_words + 1 :] = False

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
    mask[:, 0] = False  # No incoming edges into ROOT
    np.fill_diagonal(mask, False)  # No self-loop edges
    mask[num_words + 1 :, :] = False  # No [PAD] (head)
    mask[:, num_words + 1 :] = False  # No [PAD] (dependent)
    return mask


def mask_logits(logits, mask):
    return mask * logits + (1 - mask) * MASKED_VALUE


def sample_action(logits, mask, exp_temp=1.0, rand_coef=0.0):
    logits = mask_logits(logits, mask)
    probs = F.softmax(logits, dim=1)
    
    # Manipulate the original distribution 
    probs = probs ** (1 / exp_temp) 
    probs = probs / (1e-9 + probs.sum(1, keepdim=True))
    probs = (1 - rand_coef) * probs + rand_coef * uniform_mask_logit(mask)
    
    # Sample from the distribution 
    sample = probs.multinomial(1)
    log_p = logits.log_softmax(1).gather(1, sample).squeeze(1)
    
    return sample, log_p


def uniform_mask_logit(mask):
    logits = mask / (1e-9 + mask.sum(-1, keepdim=True))
    return logits 
     
    
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
            if self["num_words"][i] == 1:
                self["adjacency"][i, 0, 1] = 1
                self["labels"][i, 1] = 1

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

        assert np.all(
            sources <= self["num_words"][~dones]
        ), "Invalid head node(s): Node(s) out of range"
        assert np.all(
            targets <= self["num_words"][~dones]
        ), "Invalid dependent node(s): Node(s) out of range"

        if not np.all(self["mask"][~dones, sources, targets]):
            np.save("./output/log_errors/targets.npy", targets)
            np.save("./output/log_errors/sources.npy", sources)
            np.save("./output/log_errors/mask.npy", self["mask"])
            np.save("./output/log_errors/labels.npy", self["labels"])
            np.save("./output/log_errors/adjacency.npy", self["adjacency"])
            np.save("./output/log_errors/num_words.npy", self["num_words"])

            raise ValueError(
                "Invalid action(s): Already existed edge(s), Self-loop, or Causing-cycle edge(s)."
            )

        # Update adjacency matrices
        self._data["adjacency"][~dones, sources, targets] = 1
        self._data["labels"][~dones, targets] = 1
        # self["adjacency"][dones] = 0

        # Update transitive closure of transpose
        source_rows = np.expand_dims(self._closure_T[~dones, sources, :], axis=1)
        target_cols = np.expand_dims(self._closure_T[~dones, :, targets], axis=2)
        self._closure_T[~dones] |= np.logical_and(
            source_rows, target_cols
        )  # Outer product
        self._closure_T[dones] = np.eye(self.num_variables, dtype=np.bool_)

        # Update the mask
        self._data["mask"] = 1 - (self._data["adjacency"] + self._closure_T)
        num_parents = np.sum(self["adjacency"], axis=1, keepdims=True)
        self._data["mask"] *= num_parents < 1

        # Exclude all undue edges
        for batch_idx, num_word in enumerate(self["num_words"]):
            self._data["mask"][batch_idx, num_word + 1 :, :] = False
            self._data["mask"][batch_idx, :, num_word + 1 :] = False

        # Filter already linked ROOT
        self._data["mask"][:, :, 0] = False
        self._data["mask"][:, 0] *= np.logical_not(
            np.any(self["adjacency"][:, 0], axis=1, keepdims=True)
        )

    def check_done(self):
        return self["adjacency"].sum(axis=-1).sum(axis=-1) == self["num_words"]
