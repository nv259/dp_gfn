import torch
import conllu
import networkx as nx
from torch.utils.data import DataLoader, Dataset
import xml.etree.ElementTree as ET


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
    G = torch.zeros(num_variables, num_variables, dtype=torch.int)
    for source, target, tag in edges_list: 
        G[source, target] = tag
     
    return G


def get_dependency_relation_dict(path_to_stats_file: str) -> dict:
    tree = ET.parse(path_to_stats_file)
    root = tree.getroot()
    deps = root.find('deps')
    
    rel_id = {}
    for id, dep in enumerate(deps.findall('dep')):
        rel = dep.get('name')
        rel_id[rel] = id
        
    return rel_id 


class BaseDataset(Dataset):
    def __init__(self, path_to_conllu_file: str, max_num_nodes: int = 160, store_nx_graph: bool = False):
        super(BaseDataset, self).__init__()

        self.path = path_to_conllu_file
        self.store_nx_graph = store_nx_graph
        self.max_num_nodes = max_num_nodes
        self.data = []

        with open(self.path, "r", encoding="utf-8") as data_file:
            for tokenlist in conllu.parse_tree_incr(data_file):
                self.data.append(tokenlist)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        
        edges_list = parse_conllu_for_edges(item, parent_index=0)
        
        if self.store_nx_graph:
            G = nx.from_edgelist(edges_list)
        else:
            G = adjacency_matrix_from_edges_list(edges_list, num_variables=self.max_num_nodes)
            
        return {"metadata": item.metadata, "graph": G}