import matplotlib.pyplot as plt
import networkx as nx


def visualize_graphs(adjacency_matrices, save_path_prefix="output/graph"):
    """
    Visualizes a batch of graphs from adjacency matrices and saves them as JPG images.

    Args:
        adjacency_matrices (np.ndarray): A batch of adjacency matrices with
                                          shape (batch_size, num_nodes, num_nodes).
        save_path_prefix (str, optional): The prefix for the save file names.
                                          Defaults to "graph".
    """

    for i, adj_matrix in enumerate(adjacency_matrices):
        graph = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
        plt.figure()
        nx.draw(graph, with_labels=True)
        plt.savefig(f"{save_path_prefix}_{i}.jpg")
        plt.close()


# Example usage:
# Assuming `batch_of_adjacency_matrices` is your numpy array of shape (batch_size, num_nodes, num_nodes)
# visualize_graphs(batch_of_adjacency_matrices)
