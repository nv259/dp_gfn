import os
import xml.etree.ElementTree as ET

import conllu
import networkx as nx
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def parse_conllu_for_edges(current_node: conllu.models.TokenTree, parent_index: int):
    edges = []

    current_index = current_node.token["id"]
    edges.append((parent_index, current_index, current_node.token["deprel"]))

    if current_node.children == 0:
        return edges

    for child_node in current_node.children:
        edges += parse_conllu_for_edges(child_node, parent_index=current_index)

    return edges


def adjacency_matrix_from_edges_list(edges_list, num_variables: int) -> torch.Tensor:
    G = torch.zeros(num_variables + 1, num_variables + 1, dtype=torch.int)
    for source, target, tag in edges_list:
        G[source, target] = tag

    return G


def get_dependency_relation_dict(path_to_stats_file: str) -> dict:
    tree = ET.parse(path_to_stats_file)
    root = tree.getroot()
    deps = root.find("deps")

    rel_id = {}
    for idx, dep in enumerate(deps.findall("dep")):
        rel = dep.get("name")
        rel_id[rel] = idx

    return rel_id


def collate_nx_graphs(batch):
    """
    Custom collate function for batching NetworkX graphs.

    Args:
        batch: A list of dictionaries, where each dictionary contains a 'graph' key with a NetworkX graph.

    Returns:
        A dictionary containing a batched graph and text.
    """

    graphs = [item['graph'] for item in batch]
    text = [item['text'] for item in batch]

    # Find the maximum number of nodes in the batch
    max_num_nodes = max(len(graph.nodes) for graph in graphs)

    # Create a batched adjacency matrix
    batched_graph = torch.zeros((len(batch), max_num_nodes, max_num_nodes), dtype=torch.int)
    batched_labels = torch.zeros((len(batch), max_num_nodes, max_num_nodes), dtype=torch.int)

    for i, graph in enumerate(graphs):
        for u, v, data in graph.edges(data=True):
            batched_graph[i, u, v] = 1
            batched_labels[i, u, v] = data['tag']

    return {'graph': batched_graph, 'labels': batched_labels, 'text': text}


class BaseDataset(Dataset):
    def __init__(
        self,
        path_to_conllu_file: str,
        max_num_nodes: int = 160,
        store_nx_graph: bool = False,
        return_edges: bool = False,
    ):
        super(BaseDataset, self).__init__()

        self.path = path_to_conllu_file
        self.store_nx_graph = store_nx_graph
        self.max_num_nodes = max_num_nodes
        self.return_edges = return_edges

        self.rel_id = get_dependency_relation_dict(
            os.path.join(os.path.dirname(path_to_conllu_file), "stats.xml")
        )
        self.id_rel = {v: k for k, v in self.rel_id.items()}

        self.data = []
        with open(self.path, "r", encoding="utf-8") as data_file:
            for tokenlist in conllu.parse_tree_incr(data_file):
                self.data.append(tokenlist)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]

        edges_list = parse_conllu_for_edges(item, parent_index=0)
        edges_list = [(source, target, self.rel_id[tag]) for source, target, tag in edges_list]

        if self.store_nx_graph:
            G = nx.DiGraph()
            for source, target, tag in edges_list:
                G.add_edge(source, target, tag=tag)
        else:
            G = adjacency_matrix_from_edges_list(
                edges_list, num_variables=self.max_num_nodes
            )
        
        if self.return_edges:
            return {"text": item.metadata['text'], "graph": G, "edges": edges_list}
        else:  
            return {"text": item.metadata['text'], "graph": G} 


def get_dataloader(path_to_conllu_file: str, max_num_nodes: int = 160, return_edges: bool = True, store_nx_graph: bool = False, batch_size: int = 1, shuffle: bool = True, num_workers: int = 0):
    dataset = BaseDataset(
        path_to_conllu_file=path_to_conllu_file,
        max_num_nodes=max_num_nodes, 
        return_edges=return_edges,
        store_nx_graph=store_nx_graph,
    )
    
    if store_nx_graph:  
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_nx_graphs, num_workers=num_workers)
    else: 
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    return dataloader
    
    