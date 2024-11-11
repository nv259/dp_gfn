import torch
from collections.abc import MutableMapping


def flatten_config(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_config(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def remove_at_indices(x, indices):
    """Creates new embeddings by excluding elements at specified indices.

    Args:
        x: A tensor of shape (B, N, D) representing the embeddings.
        indices: A tensor of shape (B, 1) representing the indices to exclude.

    Returns:
        A tensor of shape (B, N-1, D) containing the new embeddings.
    """

    B, N, D = x.shape

    all_indices = torch.arange(N).unsqueeze(0).expand(B, -1).to(indices.device)
    mask = all_indices != indices

    return x[mask].reshape(B, -1, D)
