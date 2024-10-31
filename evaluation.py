import logging
import os
import warnings
from argparse import ArgumentParser

import conllu
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

try:
    from dp_gfn.gfn import DPGFN
except:
    pass
from dp_gfn.utils.data import get_dataloader
from dp_gfn.utils.misc import flatten_config
from dp_gfn.utils.pretrains import create_position_ids_from_input_ids


def save_predictions(algorithm, loader, config, id2rel, original, gold, system, verbose=False):
    save_folder = '/'.join(gold.split('/')[:-1])
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    token_lists = []
    output_file = open(system, "w")
    truncated_old_file = open(gold, "w")
    if verbose:
        pbar = tqdm(
            total=len(loader) * config.algorithm.eval.batch_size, desc="Inference"
        )  # TODO: this code drop the last batch if number of sentences remains does not suffice
    loader = iter(loader)

    with open(original, "r") as dev_file:
        for token_list_ in conllu.parse_incr(dev_file):
            if len(token_lists) == config.algorithm.eval.batch_size:
                # Predict parse tree
                batch = next(loader)

                tokens = algorithm.tokenizer(
                    batch["text"],
                    return_tensors="jax",
                    padding="max_length",
                    truncation=True,
                    add_special_tokens=False,
                )

                position_ids = create_position_ids_from_input_ids(tokens["input_ids"])

                complete_states, log = algorithm.inference(
                    tokens, position_ids, batch["num_words"], delta=0.0
                )

                # Update token list content
                predict_graphs = complete_states["adjacency"]

                for idx, token_list in enumerate(token_lists):
                    # Write old data
                    serialized = token_list.serialize()
                    truncated_old_file.write(serialized)

                    predict = predict_graphs[idx]
                    # Update token list's content
                    for token in token_list:
                        if type(token["id"]) != int:
                            continue

                        token["head"] = np.argmax(predict[:, token["id"]]).item()
                        if token["head"] > batch["num_words"][idx]:
                            print("debug")
                        token["deprel"] = id2rel[
                            predict[token["head"], token["id"]].item()
                        ]
                        token["deps"] = [(token["deprel"], token["head"])]

                    # Write updated token list to file
                    serialized = token_list.serialize()
                    output_file.write(serialized)
                    
                    if verbose: 
                        pbar.update(1)

                # Reset
                token_lists = []
            token_lists.append(token_list_)

    output_file.close()
    truncated_old_file.close()


@hydra.main(config_path="./configs", config_name="atis")
def main(config):
    os.chdir(hydra.utils.get_original_cwd())
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
        # batch_size=config.algorithm.eval.batch_size,
        batch_size=8,
        num_workers=config.num_workers,
        shuffle=False,
    )

    logging.info("Initializing Algorithm")
    algorithm = DPGFN(
        config=config,
        num_tags=config.num_tags,
        id2rel=id2rel,
        pretrained_path="/mnt/yth/dp_gfn/output/run_bs=16_epsilon=0.005_dim=256_nlayers=3_nheads=4/model_55000.npz",
    )

    save_predictions(
        algorithm=algorithm,
        loader=val_loader,
        config=config,
        id2rel=id2rel,
        original=config.train_path.replace("train", "dev"),
        gold="./output/gold.conllu",
        system="./output/system.conllu",
    )


if __name__ == "__main__":
    # warnings.filterwarnings("ignore")
    main()
