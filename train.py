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
    train_loader, (id2rel, num_tags, max_num_nodes) = get_dataloader(
        path_to_conllu_file=config.train_path,
        max_number_of_words=config.max_number_of_words,
        batch_size=1,
        num_workers=config.num_workers,
        shuffle=True,
        pre_tokenize=config.pre_tokenize
    )
    torch.save(train_loader, os.path.join(os.path.dirname(config.train_path), "train_loader.pt"))
    
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
            pre_tokenize=config.pre_tokenize
        )
        torch.save(val_loader, os.path.join(os.path.dirname(config.train_path), "val_loader.pt"))
    except:
        val_loader = None
        logging.warning("No validation data found")

    logging.info("Initializing Algorithm")
    algorithm = DPGFN(
        config=config, 
        # num_tags=num_tags, 
        num_tags=config.num_tags,
        id2rel=id2rel,
        # pretrained_path="/mnt/yth/dp_gfn/output/logging_rewards_/run_bs=12_epsilon=0.005_dim=256_nlayers=3_nheads=4/model_2000.npz"
    )

    algorithm.train(train_loader=train_loader, val_loader=val_loader)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
