import os
import xml.etree.ElementTree as ET

import conllu
import networkx as nx
import numpy as np
from torch.utils.data import DataLoader, Dataset


def parse_token_tree(
    current_node: conllu.models.TokenTree, tokens: dict[int:str], parent_index: int = 0
):
    edges = []

    current_index = current_node.token["id"]

    if type(current_index) == int:
        edges.append((parent_index, current_index, current_node.token["deprel"]))
        tokens[current_index] = current_node.token["form"]
    else:
        raise Exception("Token id is not an integer")

    if current_node.children == 0:
        return edges

    for child_node in current_node.children:
        edges += parse_token_tree(child_node, parent_index=current_index, tokens=tokens)

    return edges


def adjacency_matrix_from_edges_list(edges_list, num_variables: int):
    G = np.zeros((num_variables, num_variables), dtype=int)
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
    batched_graph = np.zeros((len(batch), max_num_nodes, max_num_nodes), dtype=np.int_)
    batched_labels = np.zeros((len(batch), max_num_nodes, max_num_nodes), dtype=np.int_)

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
    graphs = [item["graph"] for item in batch]
    text = [item["text"] for item in batch]
    num_words = [item["num_words"] for item in batch]

    graphs = np.array(graphs, dtype=np.int32)
    text = text
    num_words = np.array(num_words, dtype=np.int32)

    return {"graph": graphs, "text": text, "num_words": num_words}


class BaseDataset(Dataset):
    def __init__(
        self,
        path_to_conllu_file: str,
        max_number_of_words: int = 100,
        store_nx_graph: bool = False,
        return_edges: bool = False,
        debug: bool = False,
    ):
        super(BaseDataset, self).__init__()

        self.path = path_to_conllu_file
        self.store_nx_graph = store_nx_graph
        self.max_num_nodes = 0
        self.return_edges = return_edges

        self.rel_id = get_dependency_relation_dict(
            os.path.join(os.path.dirname(path_to_conllu_file), "stats.xml")
        )
        self.id_rel = {v: k for k, v in self.rel_id.items()}
        self.num_tags = len(self.rel_id)

        self.data = []
        with open(self.path, "r", encoding="utf-8") as data_file:
            for tokenlist in conllu.parse_tree_incr(data_file):
                word_dict = {}
                edges_list = parse_token_tree(tokenlist, tokens=word_dict)
                edges_list = np.array(
                    [
                        (source, target, self.rel_id[tag])
                        for source, target, tag in edges_list
                    ]
                )
                words_list = [word_dict[idx] for idx in range(1, len(word_dict) + 1)]
                joined_words = " ".join(words_list)

                if (
                    len(words_list) > max_number_of_words - 1
                    and "train" in path_to_conllu_file
                ):
                    continue
                elif len(words_list) > max_number_of_words:
                    continue

                self.data.append(
                    {
                        "text": "<s> " + joined_words if not debug else joined_words,
                        "edges": edges_list,
                        "num_words": len(words_list),
                    }
                )

                self.max_num_nodes = max(self.max_num_nodes, len(words_list))

        if "train" in path_to_conllu_file:
            self.max_num_nodes = (
                min(self.max_num_nodes, max_number_of_words) + 1
            )  # ROOT node
        else:
            self.max_num_nodes = max_number_of_words

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]

        if self.store_nx_graph:
            G = nx.DiGraph()
            for source, target, tag in item["edges"]:
                G.add_edge(source, target, tag=tag)
        else:
            G = adjacency_matrix_from_edges_list(
                item["edges"], num_variables=self.max_num_nodes
            )

        if self.return_edges:
            return {
                "text": item["text"],
                "graph": G,
                "edges": item["edges"],
                "num_words": item["num_words"],
            }
        else:
            return {"text": item["text"], "graph": G, "num_words": item["num_words"]}


def get_dataloader(
    path_to_conllu_file: str,
    max_number_of_words: int = 100,
    return_edges: bool = False,
    store_nx_graph: bool = False,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 0,
    is_torch: bool = True,
):
    dataset = BaseDataset(
        path_to_conllu_file=path_to_conllu_file,
        max_number_of_words=max_number_of_words,
        store_nx_graph=store_nx_graph,
        return_edges=return_edges,
    )
    
    collate_fn = collate_nx_graphs if store_nx_graph else np_collate_fn
    if is_torch: collate_fn = None
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        drop_last=True,
    )

    return dataloader, (dataset.id_rel, dataset.num_tags, dataset.max_num_nodes)
