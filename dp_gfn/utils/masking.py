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
    mask = jnp.zeros((num_variables, num_variables), dtype=bool)
    mask = mask.repeat(len(num_words), 1, 1)

    for batch_idx, num_word in enumerate(num_words):
        num_word = int(num_word)
        mask[batch_idx, 0, 1 : num_word + 1] = True

    return mask


def base_mask(
    num_words: int, num_variables: int
):
    mask = jnp.zeros((num_variables, num_variables), dtype=bool)
    mask[0, 1 : num_words + 1] = True

    return mask


def mask_logits(logits, masks):
    return masks * logits + (1 - masks) * MASKED_VALUE


def check_done(adjacency_matrices, num_words):
    num_edges = adjacency_matrices.sum((1, 2)) - 1

    return num_edges == num_words


def batch_random_choice(key, probas, masks):
    # Sample from the distribution
    uniform = random.uniform(key, shape=(probas.shape[0], 1))
    cum_probas = jnp.cumsum(probas, axis=1)
    samples = jnp.sum(cum_probas < uniform, axis=1, keepdims=True)

    masks = masks.reshape(masks.shape[0], -1)
    # is_valid = jnp.take_along_axis(masks, samples, axis=1)    # TODO: Potential risk

    return jnp.squeeze(samples, axis=1)


def uniform_log_policy(masks):
    masks = masks.reshape(masks.shape[0], -1)
    num_valid_actions = jnp.sum(masks, axis=-1, keepdims=True)
    log_pi = -jnp.log(num_valid_actions)
    
    log_pi = mask_logits(log_pi, masks)

    return log_pi