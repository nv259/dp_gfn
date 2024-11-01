from tqdm import trange
import jax
import logging
import os
import random
import warnings

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from dp_gfn.gfn import DPGFN
from dp_gfn.utils.data import get_dataloader
from dp_gfn.utils.misc import flatten_config
from dp_gfn.utils import io, masking, scores
from dp_gfn.utils.pretrains import (
    batch_token_embeddings_to_batch_word_embeddings,
    create_position_ids_from_input_ids,
)


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


@hydra.main(config_path="./configs", config_name="main")
def main(config):
    os.chdir(hydra.utils.get_original_cwd())
    config.seed = random.randint(1, 100000) if config.seed is None else config.seed
    set_seed(config.seed)

    log_config = flatten_config(OmegaConf.to_container(config, resolve=True), sep="/")
    log_config = {"/".join(("config", key)): val for key, val in log_config.items()}

    config = OmegaConf.to_container(config, resolve=True)
    config = DictConfig(config)

    logging.info("Loading Data")
    train_loader, num_tags, max_num_nodes, id2rel = get_dataloader(
        path_to_conllu_file=config.train_path,
        max_number_of_words=config.max_number_of_words,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=True,
        is_train=True,
    )
    config.model.num_variables = max_num_nodes
    config.max_number_of_words = max_num_nodes - 1
    print(OmegaConf.to_yaml(config))
    with open("hydra_config.txt", "w") as f:
        f.write(OmegaConf.to_yaml(config))

    try:
        val_loader, _ = get_dataloader(
            path_to_conllu_file=config.train_path.replace("train", "dev"),
            max_number_of_words=max_num_nodes,
            batch_size=config.algorithm.eval.batch_size,
            num_workers=config.num_workers,
            shuffle=False,
        )
    except:
        val_loader = None
        logging.warning("No validation data found")

    logging.info("Initializing Algorithm")
    algorithm = UniformForwardPolicy(
        config=config, 
        num_tags=num_tags, 
        id2rel=id2rel,
        # pretrained_path="/mnt/yth/dp_gfn/output/logging_rewards_/run_bs=12_epsilon=0.005_dim=256_nlayers=3_nheads=4/model_2000.npz"
        debug=True,
    )
    
    algorithm.train(train_loader=train_loader, val_loader=val_loader)


class UniformForwardPolicy(DPGFN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def train(self, train_loader, val_loader):
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
        rewards = []
        delta = 1.
        
        try:
            with trange(self.max_steps, desc="Training") as pbar:
                for iteration in pbar:
                    batch = next(train_loader)

                    tokens = self.tokenizer(
                        batch["text"],
                        return_tensors="jax",
                        padding="max_length",
                        truncation=True,
                        add_special_tokens=False,
                    )

                    position_ids = create_position_ids_from_input_ids(
                        tokens["input_ids"]
                    )

                    loss, logs = self.loss(
                        self.bert_params,
                        self.gfn_params,
                        self.logZ_params,
                        tokens,
                        position_ids,
                        batch["num_words"],
                        batch["graph"],
                        delta,
                    )

                    rewards.append(np.exp(logs["log_R"]))

                    if iteration % self.eval_every_n == 0:
                        np.save(os.path.join(save_folder, f'rewards_{iteration}.npy'), np.concat(rewards))
                        rewards = []
                        print("-" * 50)

                    if self.eval_on_train:
                        pbar.set_postfix(
                            epsilon=f"{delta:.4f}",
                            loss=f"{logs['loss']:.5f}",
                            reward=f"{np.exp(logs['log_R']).mean():.6f}",
                            train_loss=f"{train_loss:.5f}",
                            val_loss=f"{val_loss:.5f}",
                        )
                    else:
                        pbar.set_postfix(
                            epsilon=f"{delta:.4f}",
                            loss=f"{logs['loss']:.5f}",
                            reward=f"{np.exp(logs['log_R']).mean():.6f}",
                            val_loss=f"{val_loss:.5f}",
                        )

                    train_losses.append(logs["loss"])

        except Exception as e:  # Save current training information
            io.save(
                os.path.join(save_folder, "except.npz"),
                bert=self.bert_params,
                gfn=self.gfn_params,
                logZ=self.logZ_params,
                states=self.states,
                step=iteration
            )

            raise e

        return train_losses, val_losses


def cycle(dataloader):
    while True:
        for batch in dataloader:
            yield batch
            

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
