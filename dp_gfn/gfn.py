import logging
import os
import pickle
import subprocess
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

try:
    from evaluation import save_predictions
except:
    pass



os.environ["TOKENIZERS_PARALLELISM"] = "false"
GFlowNetState = namedtuple("GFlowNetState", ["optimizer", "step"])
GFlowNetParams = namedtuple("GFlowNetParams", ["bert", "gfn", "logZ"])


class DPGFN:
    def __init__(self, config, num_tags, id2rel, pretrained_path=None, debug=False):
        super().__init__()
        self.config = config
        self.num_tags = num_tags
        self.id2rel = id2rel
        self.debug = debug

        self.initialize_vars()
        self.init_policy()

        if pretrained_path is not None:
            self.load_weights(pretrained_path)

    def initialize_vars(self):
        pass
    
    def init_policy(self):
        pass

    def sample(
        self,
    ):
        pass

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        save_folder = os.path.join(
            "/".join(logging.getLogger().handlers[1].baseFilename.split("/")[:-1]),
            self.dump_foldername,
        )
        os.makedirs(save_folder, exist_ok=True)
        os.makedirs(os.path.join(save_folder, "model"))
        os.makedirs(os.path.join(save_folder, "predicts"))
        if self.debug:
            os.makedirs(os.path.join(save_folder, "debug"))
        logging.info(f"Save folder: {save_folder}")

        train_loader = cycle(train_loader)
        train_losses, val_losses = [], []
        train_loss, val_loss = 0, 0
        log_Zs = []
        rewards = []

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


def trajectory_balance_loss(log_Z, traj_log_pF, log_R, traj_log_pB, delta=1):
    assert log_Z.shape == traj_log_pF.shape == traj_log_pB.shape == log_R.shape
    loss = (log_Z + traj_log_pF - log_R - traj_log_pB) ** 2
    logs = {"loss": loss, "log_R": log_R, "log_Z": log_Z}

    return (loss, logs)


def cycle(dataloader):
    while True:
        for batch in dataloader:
            yield batch
