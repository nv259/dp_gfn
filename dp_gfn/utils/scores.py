from functools import partial

import jax.numpy as jnp
from jax import jit, vmap


def reward(predict, gold, graph_distance_fn):
    return (1.0 - graph_distance_fn(predict, gold)) > 0.99
    return jnp.exp(1.0 - graph_distance_fn(predict, gold))


def unlabeled_graph_edit_distance(predict, gold):
    # Retain edges only
    predict = predict.astype(bool)
    gold = gold.astype(bool)

    ged = (predict != gold).sum(-1).sum(-1) / (2 * gold.sum(-1).sum(-1))

    reward = jnp.exp(1.0 - ged)

    return reward


# TODO: Implement bayesian graph edit distance
def bayesian_graph_edit_distance(predict, gold):
    pass


# TL, DR: the smaller, the better
@partial(vmap, in_axes=(0, 0))
def frobenius_norm_distance(A, B):
    numerator = jnp.linalg.norm(A - B, "fro")
    denominator = jnp.sqrt(
        jnp.linalg.norm(A, "fro") ** 2 + jnp.linalg.norm(B, "fro") ** 2
    )

    return numerator / denominator


# TL, DR: the larger, the better
@partial(vmap, in_axes=(0, 0))
def cosine_similarity(A, B):
    A_flat = A.flatten()
    B_flat = B.flatten()
    cos_sim = jnp.dot(A_flat, B_flat) / (
        jnp.linalg.norm(A_flat) * jnp.linalg.norm(B_flat)
    )

    return (1 + cos_sim) / 2  # Normalize to [0, 1]


# TL,DR: the larger, the better
@partial(vmap, in_axes=(0, 0))
def jaccard_index(A, B):
    A_set = set(zip(*jnp.where(A != 0)))
    B_set = set(zip(*jnp.where(B != 0)))
    intersection = len(A_set & B_set)
    union = len(A_set | B_set)

    return intersection / union if union != 0 else 1


# TL,DR: the smaller, the better
@partial(vmap, in_axes=(0, 0))
def hamming_distance(A, B):
    return jnp.sum(A != B) / A.size


# TL,DR: the smaller, the better
@partial(vmap, in_axes=(0, 0))
def spectral_distance(A, B):
    eigenvalues_A = jnp.linalg.eigvals(A)
    eigenvalues_B = jnp.linalg.eigvals(B)
    numerator = jnp.linalg.norm(eigenvalues_A - eigenvalues_B)
    denominator = jnp.sqrt(
        jnp.linalg.norm(eigenvalues_A) ** 2 + jnp.linalg.norm(eigenvalues_B) ** 2
    )

    return numerator / denominator


@partial(
    jit,
)
def scale_between(inputs, original_min, original_max, scaled_min, scaled_max):
    return (scaled_max - scaled_min) * (inputs - original_min) / (
        original_max - original_min
    ) + scaled_min
