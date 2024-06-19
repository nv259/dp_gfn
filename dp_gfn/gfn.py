import math
import os

from tqdm import tqdm

import hydra
import torch
from dp_gfn.nets.gflownet import DPGFlowNet
from dp_gfn.utils import masking
from torch.distributions import Categorical
from torch.utils.data import DataLoader

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class DPGFN:
    def __init__(
        self,
        config,
        model: DPGFlowNet,
    ):
        super().__init__()
        self.config = config
        self.model = model
        # self.score_fn = hydra.utils.instantiate(config.algorithm.score_fn)
        # self.loss_fn = hydra.utils.instantiate(config.algorithm.train.loss_fn)

        self.initialize_vars()
        self.init_policy()

    def initialize_vars(self):
        self.max_number_of_words = self.config.max_number_of_words
        self.batch_size = self.config.batch_size
        self.device = self.config.device

        config = self.config.algorithm
        self.num_tags = self.model.num_tags
        self.backward_policy = config.backward_policy
        self.score_fn = config.score_fn

        # train hyperparameters
        config = config.train
        self.n_grad_accumulation_steps = config.n_grad_accumulation_steps
        self.max_steps = config.max_steps
        self.eval_on_train = config.eval_on_train
        self.exploration_rate = config.exploration_rate
        # self.stimulated_annealing = self.stimulated_annealing  # TODO: Future work
        self.clip_grad = config.clip_grad

    def init_policy(self):
        self.model.to(self.config.device)

        policy_lr = self.config.algorithm.train.optimizer.policy_lr
        Z_lr = self.config.algorithm.train.optimizer.Z_lr
        bert_factor = self.config.algorithm.train.optimizer.bert_factor
        weight_decay = self.config.algorithm.train.optimizer.weight_decay

        # TODO: Implement lr_scheduler
        self.opt_bert = torch.optim.Adam(
            self.model.bert_params(),
            lr=policy_lr * bert_factor,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
        )
        self.opt_model = torch.optim.Adam(
            self.model.flow_params(),
            lr=policy_lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
        )
        self.opt_Z = torch.optim.Adam(
            self.model.Z_params(),
            lr=Z_lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
        )

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        losses, rewards = [], []
        for step in tqdm(range(self.max_steps)):
            assert self.model.training == True

            batch = next(iter(train_loader))
            initial_states, pref_embeddings = self.model.create_initial_state(
                batch["text"]
            )
            batch = StateBatch(
                initial_states=initial_states,
                gold_tree=batch["graph"],
                num_words=batch["num_words"],
                node_embedding_dim=self.model.state_encoder.node_embedding_dim,
            )

            self.step(batch, pref_embeddings)

            # evaluation

    def step(
        self,
        batch,
        pref_embeddings: torch.Tensor,
    ):
        log_Z = self.model.Z(pref_embeddings)
        states, log_probs = self.sample_trajectory(batch)

        # log_r = scores.calculate_reward()
        # optimizer.zero_grad()
        # compute tb loss
        # optimizer.step()

        return loss, reward

    def sample_trajectory(self, batch, is_train: bool = True):
        uniform_pol = torch.empty(self.batch_size, device=self.device).fill_(
            self.exploration_rate
        )
        traj_log_prob = torch.zeros(self.batch_size, device=self.device)

        for t in range(self.max_number_of_words):
            logits = self.model(batch["edges"], batch["labels"], batch["mask"])
            logits = masking.mask_logits(logits, batch["mask"])
            uniform_logits = masking.mask_uniform_logits(logits, batch["mask"]).to(
                self.device
            )

            exploitation_dist = Categorical(logits=logits)
            policy_dist = Categorical(logits=logits)
            actions = exploitation_dist.sample()

            if is_train:
                uniform_mix = torch.bernoulli(uniform_pol).bool()
                exploration_dist = Categorical(logits=uniform_logits)
                explore_actions = exploration_dist.sample()

                actions = torch.where(uniform_mix, explore_actions, actions)

            log_prob = policy_dist.log_prob(actions) * torch.logical_not(batch["done"])
            traj_log_prob += log_prob

            batch.step(actions)

            if batch["done"].all() == True:
                break

        return batch, traj_log_prob

    def evaluation(
        self,
    ):
        pass

    def val_step(
        self,
    ):
        pass


class StateBatch:
    def __init__(
        self,
        initial_states: torch.Tensor,
        gold_tree: torch.Tensor,
        num_words: torch.Tensor,
        node_embedding_dim: int = 128,
        root_first: bool = True,
    ):
        edges = initial_states[:, :, : node_embedding_dim * 2]
        labels = initial_states[:, :, node_embedding_dim * 2 :]

        self.num_variables = int(math.sqrt(initial_states.shape[1]))
        self.device = initial_states.device
        self.batch_size = edges.shape[0]
        self.encoded_key = ["mask", "adjacency"]

        self._data = {
            "edges": edges,
            "labels": labels,
            "gold_tree": gold_tree,
            "mask": masking.encode(
                masking.batched_base_mask(
                    num_words=num_words,
                    num_variables=self.num_variables,
                    root_first=root_first,
                    device=self.device,
                )
            ),
            "adjacency": masking.encode(
                torch.zeros(
                    edges.shape[0],
                    self.num_variables,
                    self.num_variables,
                    device=self.device,
                    dtype=torch.bool,
                )
            ),
            "num_words": num_words,
            "done": torch.zeros(edges.shape[0], dtype=torch.bool, device=self.device),
        }
        self._closure_T = torch.eye(
            self.num_variables, dtype=torch.bool, device=self.device
        )
        self._closure_T = self._closure_T.repeat(edges.shape[0], 1, 1)

    def __len__(self):
        return self.batch_size

    def __getitem__(self, key: str):
        if key in self.encoded_key:
            return masking.decode(
                self._data[key], self.num_variables, device=self.device
            )

        return self._data[key]

    def get_full_data(self, index: int):
        edges = self._data["edges"][index]
        labels = self._data["labels"][index]
        mask = self._data["mask"][index]
        adjacency = self._data["adjacency"][index]
        num_words = self._data["num_words"][index]
        done = self._data["done"][index]

        return {
            "edges": edges,
            "labels": labels,
            "mask": mask,
            "adjacency": adjacency,
            "num_words": num_words,
            "done": done,
        }

    def step(self, actions):
        sources = actions // self.num_variables
        targets = actions % self.num_variables
        dones = self._data["done"]
        sources, targets = sources[~dones], targets[~dones]
        masks = self.__getitem__("mask")
        adjacencies = self.__getitem__("adjacency")

        print(dones, dones.shape)
        print(sources, sources.shape)
        print(targets, targets.shape)

        if not torch.all(masks[~dones, sources, targets]):
            raise ValueError("Invalid action")

        # Update adjacency matrices
        adjacencies[~dones, sources, targets] = 1
        adjacencies[dones] = 0

        # Update transitive closure of transpose
        source_rows = torch.unsqueeze(self._closure_T[~dones, sources, :], axis=1)
        target_cols = torch.unsqueeze(self._closure_T[~dones, :, targets], axis=2)
        self._closure_T[~dones] |= torch.logical_and(
            source_rows, target_cols
        )  # Outer product
        self._closure_T[dones] = torch.eye(
            self.num_variables, dtype=torch.bool, device=self.device
        )

        # Update dones
        self._data["done"][~dones] = masking.check_done(
            adjacencies[~dones], self._data["num_words"][~dones]
        )

        # Update the mask
        masks = 1 - (adjacencies + self._closure_T)
        num_parents = torch.sum(adjacencies, axis=1, keepdim=True)
        masks *= (num_parents < 1).to(self.device)  # each node has only one parent node
        # Exclude all undue edges
        for batch_idx, num_word in enumerate(self._data["num_words"]):
            masks[
                batch_idx,
                num_word + 1 : self.num_variables,
                num_word + 1 : self.num_variables,
            ] = False
        masks[:, :, 0] = False

        self._data["mask"] = masking.encode(masks)
        self._data["adjacency"] = masking.encode(adjacencies)

    def to(self, device):
        self.device = device

        self._data["edges"] = self._data["edges"].to(device)
        self._data["labels"] = self._data["labels"].to(device)
        self._data["mask"] = self._data["mask"].to(device)
        self._data["adjacency"] = self._data["adjacency"].to(device)
        self._data["num_words"] = self._data["num_words"].to(device)
        self._data["done"] = self._data["done"].to(device)
        self._closure_T.to(device)

    def reset(
        self,
    ):
        pass
