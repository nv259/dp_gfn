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


def get_parent_indices(adjacency_matrix, child_indices):
    """
    Retrieves parent indices for given child indices in a batch of adjacency matrices.

    Args:
        adjacency_matrix (torch.Tensor): Batch of adjacency matrices (B, N, N).
        child_indices (torch.Tensor): Batch of child node indices (B, 1).

    Returns:
        torch.Tensor: Batch of parent node indices (B, 1).
    """
    B, N, _ = adjacency_matrix.shape
    parent_indices = torch.zeros_like(child_indices)

    for b in range(B):
        # Select the row corresponding to the child index for the current batch item
        row = adjacency_matrix[b, :, child_indices[b].item()]
        # Use argmax to find the index of the parent (non-zero value in the row)
        parent_indices[b] = torch.argmax(row)

    return parent_indices
    

def create_graph_relations(graphs, num_tags, device):
    terminal_states = graphs.clone().to(int)
    
    for b in range(terminal_states.shape[0]): 
        for i in range(terminal_states.shape[1]):
            for j in range(terminal_states.shape[2]):
                if terminal_states[b, i, j] != 0 and terminal_states[b, j, i] == 0:
                    terminal_states[b, j, i] = terminal_states[b, i, j] + num_tags
                
    return terminal_states.to(device)