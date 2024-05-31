import torch
from dp_gfn.utils import masking


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
    def __init__(self, edges: torch.Tensor, labels: torch.Tensor, num_words: torch.Tensor, num_variables: torch.Tensor, device: torch.Tensor, root_first=True):
        self.num_variables = num_variables
        self.device = device

        self._data = {
            "edges": edges.to(device),
            "labels": labels.to(device),
            "mask": masking.encode(
                masking.base_mask(
                    num_words=num_words, root_first=root_first, device=device
                ).repeat(edges.shape[0], 1, 1)
            ),
            "adjacency": masking.encode(
                torch.zeros(edges.shape[0], num_variables, num_variables, device=device)
            ),
            "isolated_root": torch.ones(
                edges.shape[0], dtype=torch.bool, device=device
            ),
            # "num_edges": torch.zeros(edges.shape[0], dtype=torch.int).to(device),
            "num_words": num_words.to(device),
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
        actions
    ):
        sources = actions // self.num_variables
        targets = actions % self.num_variables
        dones = self._data["done"]
         
        if not torch.all(self._data['mask'][dones, sources, targets]):
            raise ValueError("Invalid action")
        
        # Update adjacency matrices
        self._data["adjacency"][~dones, sources, targets] = 1
        self._data["adjacency"][dones] = 0
        
        # Update transitive closure of transpose
        source_rows = torch.unsqueeze(self._closure_T[~dones, sources, :], axis=1)
        target_cols = torch.unsqueeze(self._closure_T[~dones, :, targets], axis=2)
        self._closure_T[~dones] |= torch.logical_and(source_rows, target_cols)  # Outer product
        self._closure_T[dones] = torch.eye(self.num_variables, dtype=torch.bool)

        # Update dones
        self._data["done"][dones] = masking.check_done(self._data["adjacency"])
        
        # Update the mask
        self._data["mask"] = 1 - (self._data['adjacency'] + self._closure_T)
        num_parents = torch.sum(self._data['adjacency'], axis=1, keepdim=True)
        self._data['mask'] *= (num_parents < 1).to(self.device)     # each node has only one parent node

    def to(self, device):
        self.device = device
        
        self._data["edges"] = self._data["edges"].to(device)
        self._data["labels"] = self._data["labels"].to(device)
        self._data["mask"] = self._data["mask"].to(device)
        self._data["adjacency"] = self._data["adjacency"].to(device)
        self._data["isolated_root"] = self._data["isolated_root"].to(device)
        self._data["num_words"] = self._data["num_words"].to(device)
        self._data["done"] = self._data["done"].to(device)
        self._closure_T.to(device)
        
    def reset(
        self,
    ):
        pass
