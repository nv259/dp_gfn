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


def to_undirected(adjacency_matrices, device=None):
    undirected_adj = adjacency_matrices | adjacency_matrices.transpose(0, 2, 1)
    return undirected_adj if device is None else torch.tensor(undirected_adj, device=device)


def post_processing(predicted_adjacency, traj_log_pF, device, method="best", threshold=0.5, num_edges=None, x=None, y=None):
    """
    Processes a batch of predicted adjacency matrices to produce a single final prediction with `num_edges`.

    Args:
        predicted_adjacency (torch.Tensor): Batch of predicted adjacency matrices (B, N, N).
        traj_log_pF (torch.Tensor): Trajectory log probabilities for each sample in the batch (B,).
        device (torch.device): Device where tensors reside.
        method (str): Post-processing method. Options: "best", "hard_voting", "probability_voting".
            Defaults to "best".
        threshold (float): Threshold for probability voting. Defaults to 0.5.
        num_edges (int): The exact number of edges required in the final output. If None, uses the number of edges in the gold standard (N).
                           If provided, and the chosen method doesn't result in exactly num_edges, the function will modify the output to ensure this constraint.

    Returns:
        torch.Tensor: Final predicted adjacency matrix (1, N, N) with exactly `num_edges` edges.
    """
    predicted_adjacency = torch.tensor(predicted_adjacency)
     
    predicted_adjacency = torch.cat([torch.tensor(predicted_adjacency, device=device), x.to(device)], dim=0)
    traj_log_pF = torch.cat([traj_log_pF, y.to(device)], dim=0)
    
    if method == "best":
        best_index = traj_log_pF.argmax().item()
        final_predicted_adjacency = predicted_adjacency[best_index].unsqueeze(0)  # Add batch dimension
    elif method == "hard_voting":
        vote_counts = predicted_adjacency.sum(dim=0)
        final_predicted_adjacency = (vote_counts > (len(predicted_adjacency) / 2)).float().unsqueeze(0)
    elif method == "probability_voting":  # Assumes predicted_adjacency are edge probabilities
        avg_probabilities = predicted_adjacency.mean(dim=0)
        final_predicted_adjacency = (avg_probabilities > threshold).float().unsqueeze(0)
    else:
        raise ValueError(f"Invalid post-processing method: {method}")

    final_predicted_adjacency = final_predicted_adjacency.to(device)

    if num_edges is not None:
        # Ensure exactly num_edges are present 
        current_edges = final_predicted_adjacency.to(torch.bool).sum().item()

        if current_edges != num_edges:
            diff = num_edges - current_edges

            if diff != 0:
                if method == "hard_voting":
                    sorted_votes, indices = torch.sort(vote_counts.view(-1), descending=True)
                
                    for i in range(int(abs(diff))):
                        idx = indices[i] // vote_counts.size()[0]
                        idy = indices[i] % vote_counts.size()[1]
                        
                        if diff > 0:    # Add edges 
                            final_predicted_adjacency[0][idx, idy] = 1
                        else:           # Remove edges
                            final_predicted_adjacency[0][idx, idy] = 0
                else:
                    print("Not yet implemented") 
            #     # Find top probabilities among unconnected edges and connect them
            #     sorted_probs = torch.sort(avg_probabilities.view(-1), descending=True)
            #     for i in range(int(diff)):
            #         idx = sorted_probs.indices[i] // avg_probabilities.size()[0]
            #         idy = sorted_probs.indices[i] % avg_probabilities.size()[0]
                    
            #         final_predicted_adjacency[0][idx, idy] = 1
                    
            # elif diff < 0:  # Remove edges

            #     sorted_probs = torch.sort(avg_probabilities.view(-1))
            #     for i in range(int(abs(diff))):
            #         idx = sorted_probs.indices[i] // avg_probabilities.size()[0]
            #         idy = sorted_probs.indices[i] % avg_probabilities.size()[0]
            #         final_predicted_adjacency[0][idx, idy] = 0
                    
    return final_predicted_adjacency


def align_shape(A, B):
    min_dim = min(A.shape[-1], B.shape[-1])
    A = A[:, :min_dim, :min_dim]
    B = B[:, :min_dim, :min_dim]
    
    return A, B