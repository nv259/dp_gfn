import os
import xml.etree.ElementTree as ET

import networkx as nx
import numpy as np

import conllu
from torch.utils.data import DataLoader, Dataset


def parse_token_tree(
    current_node: conllu.models.TokenTree, tokens: dict[int:str], parent_index: int = 0
):
    edges = []

    current_index = current_node.token["id"]
    edges.append((parent_index, current_index, current_node.token["deprel"]))
    tokens[current_index] = current_node.token["form"]

    if current_node.children == 0:
        return edges

    for child_node in current_node.children:
        edges += parse_token_tree(child_node, parent_index=current_index, tokens=tokens)

    return edges


def adjacency_matrix_from_edges_list(edges_list, num_variables: int):
    G = np.zeros((num_variables + 1, num_variables + 1), dtype=int)
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

    graphs = [item["graph"] for item in batch]
    text = [item["text"] for item in batch]
    num_words = [item["num_words"] for item in batch]

    # Find the maximum number of nodes in the batch
    max_num_nodes = max(len(graph.nodes) for graph in graphs)

    # Create a batched adjacency matrix
    batched_graph = np.zeros(
        (len(batch), max_num_nodes, max_num_nodes), dtype=np.int_
    )
    batched_labels = np.zeros(
        (len(batch), max_num_nodes, max_num_nodes), dtype=np.int_
    )

    for i, graph in enumerate(graphs):
        for u, v, data in graph.edges(data=True):
            batched_graph[i, u, v] = 1
            batched_labels[i, u, v] = data["tag"]

    return {
        "graph": batched_graph,
        "labels": batched_labels,
        "text": text,
        "num_words": num_words,
    }


def np_collate_fn(batch):
    graphs = [item['graph'] for item in batch]
    text = [item['text'] for item in batch]
    num_words = [item['num_words'] for item in batch] 
    
    graphs = np.array(graphs, dtype=np.int32)
    text = np.array(text)
    num_words = np.array(num_words, dtype=np.int32)
     
    return {
        "graph": graphs,
        "text": text,
        "num_words": num_words
    }


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
        self.num_tags = len(self.rel_id)

        self.data = []
        with open(self.path, "r", encoding="utf-8") as data_file:
            for tokenlist in conllu.parse_tree_incr(data_file):
                self.data.append(tokenlist)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]

        word_dict = {}
        edges_list = parse_token_tree(item, tokens=word_dict)
        edges_list = np.array(
            [(source, target, self.rel_id[tag]) for source, target, tag in edges_list]
        )
        words_list = [word_dict[idx] for idx in range(1, len(word_dict) + 1)]
        joined_words = " ".join(words_list)

        if self.store_nx_graph:
            G = nx.DiGraph()
            for source, target, tag in edges_list:
                G.add_edge(source, target, tag=tag)
        else:
            G = adjacency_matrix_from_edges_list(
                edges_list, num_variables=self.max_num_nodes
            )

        if (
            self.return_edges
        ):  # cannot batch because of the discrepancy of sizes between edges_lists
            return {
                "text": joined_words,
                "graph": G,
                "edges": edges_list,
                "num_words": len(words_list),
            }
        else:
            return {"text": joined_words, "graph": G, "num_words": len(words_list)}


def get_dataloader(
    path_to_conllu_file: str,
    max_num_nodes: int = 160,
    return_edges: bool = True,
    store_nx_graph: bool = False,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 0,
    get_num_tags: bool = False,
):
    dataset = BaseDataset(
        path_to_conllu_file=path_to_conllu_file,
        max_num_nodes=max_num_nodes,
        return_edges=return_edges,
        store_nx_graph=store_nx_graph,
    )

    if store_nx_graph:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_nx_graphs,
            num_workers=num_workers,
        )
    else:
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
            collate_fn=np_collate_fn,
        )

    if get_num_tags:
        return dataloader, dataset.num_tags
    else:
        return dataloader
