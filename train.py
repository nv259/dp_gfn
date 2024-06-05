import hydra
import warnings
import random
import numpy as np
import torch
import logging

from dp_gfn.utils.misc import flatten_config
from omegaconf import OmegaConf, DictConfig


def set_seed(seed=42):
    torch.backends.cudnn.benchmark = False 
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


@hydra.main(config_path='./configs', config_name='main')
def main(config):
    config.seed = random.randint(1, 100000) if config.seed is None else config.seed
    
    log_config = flatten_config(OmegaConf.to_container(config, resolve=True), sep='/')
    log_config = {'/'.join(('config', key)): val for key, val in log_config.items()}
    
    config = OmegaConf.to_container(config, resolve=True)
    config = DictConfig(config) 
    
    print(OmegaConf.to_yaml(config))
    with open("hydra_config.txt", 'w') as f:
        f.write(OmegaConf.to_yaml(config))
    
    config.num_tags += 2    # ROOT-edge & NO-edge
    logging.info("Initializing Model") 
    model = hydra.utils.instantiate(config.model)
    logging.info("Model Initialized")
    

if __name__=='__main__':
    warnings.filterwarnings("ignore")
    main()