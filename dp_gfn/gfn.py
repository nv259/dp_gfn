import logging
import os
import pickle
import subprocess

import dgl
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import trange

from dp_gfn.nets.gflownet import DPGFlowNet
from dp_gfn.utils import masking, scores

try:
    from evaluation import save_predictions
except:
    pass


class DPGFN:
    def __init__(self, config, num_tags, id2rel, pretrained_path=None, debug=False):
        super().__init__()
        self.config = config
        self.num_tags = num_tags
        self.id2rel = id2rel
        self.debug = debug

        self.initialize_vars(config)
        self.model = DPGFlowNet(config.model)
        self.initialize_policy(config.algorithm)

        if pretrained_path is not None:
            self.load_weights(pretrained_path)

    def initialize_vars(self, config):
        self.batch_size = config.batch_size
        self.dump_foldername = config.dump_foldername
        self.max_number_of_words = config.max_number_of_words

        nodes_list = [i for i in range(self.max_number_of_words + 1)]
        self.graph = dgl.batch(
            [dgl.graph((nodes_list, nodes_list)) for _ in range(self.batch_size)]
        )

        config = config.algorithm
        self.max_steps = config.train.max_steps
        self.eval_every_n = config.train.eval_every_n
        self.save_every_n = config.train.save_every_n
        self.reward_scale_factor = config.reward_scale_factor

    def sample(self, states):
        pass

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        save_folder = os.path.join(
            os.path.dirname(logging.getLogger().handlers[1].baseFilename),
            self.dump_foldername,
        )
        os.makedirs(save_folder, exist_ok=True)
        os.makedirs(os.path.join(save_folder, "model"))
        os.makedirs(os.path.join(save_folder, "predicts"))
        if self.debug:
            os.makedirs(os.path.join(save_folder, "debug"))
        logging.info(f"Save folder: {save_folder}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        train_loader = cycle(train_loader)
        train_losses, val_losses = [], []
        train_loss, val_loss = 0, 0
        log_Zs = []
        rewards = []

        with trange(self.max_step, desc="Training") as pbar:
            for iteration in pbar:
                batch = next(train_loader)

                # Initialize s0 and its tracer
                state = masking.StateBatch(
                    self.batch_size,
                    self.max_number_of_words + 1,
                    batch["num_words"][0].item(),
                )
                word_embeddings = self.model.init_state(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    word_ids=batch["word_ids"].to(device),
                )
                self.graph.ndata["s0"] = word_embeddings.repeat(
                    self.batch_size, 1, 1
                ).reshape(-1, self.model.bert_model.config.hidden_size)

                log_Z = self.model.logZ(batch["num_words"])
                complete_states, traj_log_pF, traj_log_pB = self.sample(state)
                log_R = torch.log(
                    scores.reward(
                        complete_states, batch["graph"], scores.frobenius_norm_distance
                    )
                    + 1e-9  # offset to prevent -inf from occurring
                )

                loss = trajectory_balance_loss(log_Z, traj_log_pF, log_R, traj_log_pB)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                pbar.set_postfix()

        return train_losses, val_losses

    def val_step(self, val_loader, *args, **kwargs):
        losses = []

        return np.mean(losses)

    def inference(
        self,
    ):
        pass

    def load_weights(self, filename):
        pass


def trajectory_balance_loss(log_Z, traj_log_pF, log_R, traj_log_pB, delta=1.0):
    assert log_Z.shape == traj_log_pF.shape == traj_log_pB.shape == log_R.shape
    loss = (log_Z + traj_log_pF - log_R - traj_log_pB) ** 2
    logs = {"loss": loss, "log_R": log_R, "log_Z": log_Z}

    return (loss, logs)


def cycle(dataloader):
    while True:
        for batch in dataloader:
            yield batch
