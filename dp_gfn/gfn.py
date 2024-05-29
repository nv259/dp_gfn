import torch
from dp_gfn.utils.masking import base_mask, decode, encode


def get_reward():
    pass


def evaluation():
    pass


def sample():
    pass


def step():
    pass


def setup():
    pass


class StateBatch:
    def __init__(self, edges, labels, num_variables, device, root_first=True):
        self.num_variables = num_variables
        self.device = device

        self._data = {
            "edges": edges.to(device),
            "labels": labels.to(device),
            "mask": encode(
                base_mask(
                    num_variables=num_variables, root_first=root_first, device=device
                ).repeat(edges.shape[0], 1, 1)
            ),
            "adjacency": encode(
                torch.zeros(edges.shape[0], num_variables, num_variables, device=device)
            ),
            "isolated_root": torch.ones(
                edges.shape[0], dtype=torch.bool, device=device
            ),
            # "num_edges": torch.zeros(edges.shape[0], dtype=torch.int).to(device),
            "num_variables": num_variables.to(device),
            "done": torch.zeros(edges.shape[0], dtype=torch.bool, device=device),
        }
        self._closure_T = torch.eye(num_variables, dtype=torch.bool, device=device)
        self._closure_T = self._closure_T.repeat(edges.shape[0], 1, 1)

    def __getitem__(
        self,
    ):
        pass

    def step(
        self,
    ):
        pass

    def reset(
        self,
    ):
        pass
