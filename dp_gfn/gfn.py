import torch
from dp_gfn.utils import masking
from dp_gfn.nets.gflownet import DPGFlowNet

class DPGFN():
    def __init__(self, cfg, model: DPGFlowNet, device=None):
        pass 
    
    def train(self, train_loader, val_loader):
        pass
    
    def step(self, ): 
        pass
    
    def evaluation(self, ):
        pass
    
    def val_step(self, ):
        pass
    

class StateBatch:
    def __init__(
        self,
        edges: torch.Tensor,
        labels: torch.Tensor,
        num_words: torch.Tensor,
        num_variables: torch.Tensor,
        device,
        root_first=True,
    ):
        self.num_variables = num_variables
        self.device = device
        self.batch_size = edges.shape[0]

        self._data = {
            "edges": edges.to(device),
            "labels": labels.to(device),
            "mask": masking.encode(
                masking.batched_base_mask(
                    num_words=num_words,
                    num_variables=num_variables,
                    root_first=root_first,
                    device=device,
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

    def __getitem__(self, index: int, return_dict: bool = True):
        edges = self._data["edges"][index]
        labels = self._data["labels"][index]
        mask = self._data["mask"][index]
        adjacency = self._data["adjacency"][index]
        isolated_root = self._data["isolated_root"][index]
        num_words = self._data["num_words"][index]
        done = self._data["done"][index]

        if return_dict:
            return {
                "edges": edges,
                "labels": labels,
                "mask": mask,
                "adjacency": adjacency,
                "isolated_root": isolated_root,
                "num_words": num_words,
                "done": done,
            }

        return edges, labels, mask, adjacency, isolated_root, num_words, done

    def step(self, actions):
        sources = actions // self.num_variables
        targets = actions % self.num_variables
        dones = self._data["done"]

        if not torch.all(self._data["mask"][dones, sources, targets]):
            raise ValueError("Invalid action")

        # Update adjacency matrices
        self._data["adjacency"][~dones, sources, targets] = 1
        self._data["adjacency"][dones] = 0

        # Update transitive closure of transpose
        source_rows = torch.unsqueeze(self._closure_T[~dones, sources, :], axis=1)
        target_cols = torch.unsqueeze(self._closure_T[~dones, :, targets], axis=2)
        self._closure_T[~dones] |= torch.logical_and(
            source_rows, target_cols
        )  # Outer product
        self._closure_T[dones] = torch.eye(self.num_variables, dtype=torch.bool)

        # Update dones
        self._data["done"][dones] = masking.check_done(self._data["adjacency"])

        # Update the mask
        self._data["mask"] = 1 - (self._data["adjacency"] + self._closure_T)
        num_parents = torch.sum(self._data["adjacency"], axis=1, keepdim=True)
        self._data["mask"] *= (num_parents < 1).to(
            self.device
        )  # each node has only one parent node
        # Exclude all undue edges
        for batch_idx, num_word in enumerate(self._data["num_words"]):
            self._data["mask"][
                batch_idx,
                num_word + 1 : self.num_variables,
                num_word + 1 : self.num_variables,
            ] = False
        self._data["mask"][:, :, 0] = False

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
