import conllu
import networkx as nx
from torch.utils.data import DataLoader, Dataset


def parse_conllu_for_edges(current_node: conllu.models.TokenTree, parent_index: int):
    edges = []

    current_index = current_node.token["id"]
    edges.append((parent_index, current_node.token["id"]))

    if current_node.children == 0:
        return edges

    for child_node in current_node.children:
        edges += parse_conllu_for_edges(child_node, parent_index=current_index)

    return edges
