import logging
import os
import random
import warnings
import torch

import hydra
from dp_gfn.gfn import DPGFN
from dp_gfn.utils.data import get_dataloader
from dp_gfn.utils.misc import flatten_config
from dp_gfn.utils.pretrains import create_position_ids_from_input_ids
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="./configs", config_name="atis")
def main(config):
    os.chdir(hydra.utils.get_original_cwd())
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    config.seed = random.randint(1, 100000) if config.seed is None else config.seed

    log_config = flatten_config(OmegaConf.to_container(config, resolve=True), sep="/")
    log_config = {"/".join(("config", key)): val for key, val in log_config.items()}

    config = OmegaConf.to_container(config, resolve=True)
    config = DictConfig(config)

    print(OmegaConf.to_yaml(config))
    with open("hydra_config.txt", "w") as f:
        f.write(OmegaConf.to_yaml(config))

    logging.info("Loading Data")
    val_loader, id2rel = get_dataloader(
        path_to_conllu_file=config.train_path.replace("train", "dev"),
        max_number_of_words=config.max_number_of_words,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
    )

    logging.info("Initializing Model")

    logging.info("Initializing Algorithm")
    algorithm = DPGFN(config=config, num_tags=config.num_tags, id2rel=id2rel, pretrained_path="/media/doublemint/SharedDisk/repo/GFlowNets-DP/dp_gfn/output/model_3000.npz")

    for batch in val_loader:
        tokens = algorithm.tokenizer(
            batch["text"],
            return_tensors="jax", 
            padding="max_length",
            truncation=True,
            add_special_tokens=False,
        )
        
        position_ids = create_position_ids_from_input_ids(tokens['input_ids'])
        
        complete_states = algorithm.inference(tokens, position_ids, batch["num_words"], delta=0.)
        
        print("abcdef")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
