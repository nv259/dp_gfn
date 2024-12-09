import matplotlib.pyplot as plt
import networkx as nx

def visualize_graphs(adjacency_matrices, save_path_prefix="output/graph", batch_plot=False, graphs_per_row=5):
    """
    Visualizes a batch of graphs from adjacency matrices and saves them as JPG images.

    Args:
        adjacency_matrices (np.ndarray): A batch of adjacency matrices with
                                          shape (batch_size, num_nodes, num_nodes).
        save_path_prefix (str, optional): The prefix for the save file names.
                                          Defaults to "graph".
        batch_plot (bool, optional): Whether to plot all graphs in a single image.
                                       Defaults to False.
        graphs_per_row (int, optional): Number of graphs per row in batch plot.
                                         Defaults to 5.
    """

    if batch_plot:
        num_graphs = len(adjacency_matrices)
        num_rows = (num_graphs + graphs_per_row - 1) // graphs_per_row  # Calculate rows needed
        fig, axes = plt.subplots(num_rows, graphs_per_row, figsize=(20, 4 * num_rows))  # Adjust figsize as needed

        for i, adj_matrix in enumerate(adjacency_matrices):
            row = i // graphs_per_row
            col = i % graphs_per_row
            ax = axes[row, col] if num_rows > 1 else axes[col]  # Handle single row case
            graph = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
            nx.draw(graph, with_labels=True, ax=ax)
            ax.set_title(f"Graph {i}")  # Add title to each subplot
            
            # Add a frame (border) to the subplot
            for spine in ax.spines.values():
                spine.set_visible(True)  # Ensure spines are visible
                spine.set_linewidth(2)  # Adjust line width as desired
                spine.set_edgecolor('black') # Set border color

        # Turn off remaining axes if num_graphs < num_rows * graphs_per_row
        for i in range(num_graphs, num_rows * graphs_per_row):
             row = i // graphs_per_row
             col = i % graphs_per_row
             ax = axes[row, col] if num_rows > 1 else axes[col]
             ax.axis('off')


        plt.tight_layout() # Adjust subplot parameters for a tight layout
        plt.savefig(f"{save_path_prefix}_batch.jpg")
        plt.close()


    else:  # Original functionality of saving each graph separately
        for i, adj_matrix in enumerate(adjacency_matrices):
            graph = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
            plt.figure()
            nx.draw(graph, with_labels=True)
            plt.savefig(f"{save_path_prefix}_{i}.jpg")
            plt.close()



# Example usage:
# Assuming `batch_of_adjacency_matrices` is your numpy array of shape (batch_size, num_nodes, num_nodes)
# visualize_graphs(batch_of_adjacency_matrices, batch_plot=True)  # To save as a single image
# visualize_graphs(batch_of_adjacency_matrices) # To save each graph in separate image.
