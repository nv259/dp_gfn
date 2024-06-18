import logging
import random
import warnings

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf, open_dict

from dp_gfn.gfn import DPGFN
from dp_gfn.utils.misc import flatten_config
from dp_gfn.utils.data import get_dataloader


def set_seed(seed=42):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def check_model_num_tags(model, orig_num_tags):
    assert (
        model.num_tags
        == model.state_encoder.num_tags
        == model.backbone.layers[0].num_tags
        == model.label_scorer.num_tags
        == orig_num_tags + 2
    )


@hydra.main(config_path="./configs", config_name="main")
def main(config):
    config.seed = random.randint(1, 100000) if config.seed is None else config.seed
    config.model.num_variables = config.max_number_of_words + 1 
    config.max_number_of_words += 1 # ROOT token
    # config.model.backbone.encoder_block.label_embedded = not config.model.state_encoder.encode_label
 
    log_config = flatten_config(OmegaConf.to_container(config, resolve=True), sep="/")
    log_config = {"/".join(("config", key)): val for key, val in log_config.items()}

    config = OmegaConf.to_container(config, resolve=True)
    config = DictConfig(config)

    print(OmegaConf.to_yaml(config))
    with open("hydra_config.txt", "w") as f:
        f.write(OmegaConf.to_yaml(config))

    logging.info("Loading Data")
    train_loader, num_tags = get_dataloader(
        path_to_conllu_file=config.train_path,
        max_num_nodes=config.model.num_variables,
        return_edges=False,
        batch_size=config.batch_size,
        num_workers=config.num_workers, 
        get_num_tags=True
    )
    
    try:
        val_loader = get_dataloader(
            path_to_conllu_file=config.train_path.replace('train', 'dev'),
            max_num_nodes=config.model.num_variables,
            return_edges=False,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            get_num_tags=False
        )
    except:
        val_loader = None
        logging.warning("No validation data found")
    
    logging.info("Initializing Model")
    model = hydra.utils.instantiate(config.model, num_tags=num_tags)
    check_model_num_tags(model, num_tags)
    
    logging.info("Initializing Algorithm")
    algorithm = DPGFN(config=config, model=model)
    
    algorithm.train(train_loader=train_loader, val_loader=val_loader)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
