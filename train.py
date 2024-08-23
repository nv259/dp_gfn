import logging
import os
import random
import warnings

import hydra
from dp_gfn.gfn import DPGFN
from dp_gfn.utils.data import get_dataloader
from dp_gfn.utils.misc import flatten_config
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="./configs", config_name="main")
def main(config):
    os.chdir(hydra.utils.get_original_cwd())
    config.seed = random.randint(1, 100000) if config.seed is None else config.seed
    config.model.num_variables = config.max_number_of_words + 1

    log_config = flatten_config(OmegaConf.to_container(config, resolve=True), sep="/")
    log_config = {"/".join(("config", key)): val for key, val in log_config.items()}

    config = OmegaConf.to_container(config, resolve=True)
    config = DictConfig(config)

    print(OmegaConf.to_yaml(config))
    with open("hydra_config.txt", "w") as f:
        f.write(OmegaConf.to_yaml(config))

    logging.info("Loading Data")
    train_loader, num_tags, max_num_nodes = get_dataloader(
        path_to_conllu_file=config.train_path,
        max_number_of_words=config.max_number_of_words,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        is_train=True,
    )

    try:
        val_loader = get_dataloader(
            path_to_conllu_file=config.train_path.replace("train", "dev"),
            max_number_of_words=max_num_nodes,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            shuffle=False,
        )
    except:
        val_loader = None
        logging.warning("No validation data found")

    logging.info("Initializing Model")

    logging.info("Initializing Algorithm")
    algorithm = DPGFN(config=config, num_tags=num_tags)

    algorithm.train(train_loader=train_loader, val_loader=val_loader)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
