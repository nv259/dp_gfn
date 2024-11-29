import logging
import os

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
        self.model = self.model.to(self.device)
        self.initialize_policy(config.algorithm)

        if pretrained_path is not None:
            self.load_weights(pretrained_path)

    def initialize_vars(self, config):
        self.batch_size = config.batch_size
        self.dump_foldername = config.dump_foldername
        self.max_number_of_words = config.max_number_of_words
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        config = config.algorithm
        self.max_steps = config.train.max_steps
        self.eval_every_n = config.train.eval_every_n
        self.save_every_n = config.train.save_every_n
        self.reward_scale_factor = config.reward_scale_factor
        
    def initialize_policy(self, config):
        config = config.train
        
        bert_params = [params for name, params in self.model.named_parameters() if 'bert' in name]
        logZ_params = [params for name, params in self.model.named_parameters() if 'logZ' in name]
        gfn_params = [params for name, params in self.model.named_parameters() if ('bert' not in name) and ('logZ' not in name)]
        
        self.optimizer = torch.optim.Adam([
            {'params': bert_params, 'lr': config.optimizer.gfn_lr * config.optimizer.bert_factor},
            {'params': logZ_params, 'lr': config.optimizer.Z_lr},
            {'params': gfn_params, 'lr': config.optimizer.gfn_lr},
        ])

    def sample(self, node_embeddings, states, exp_temp=1.0, rand_coef=0.0):
        traj_log_pF = torch.zeros((self.batch_size,), dtype=torch.float32, device=self.device)
        traj_log_pB = torch.zeros((self.batch_size,), dtype=torch.float32, device=self.device)
        
        actions, log_pF, backward_logits = self.model(
            node_embeddings=node_embeddings, 
            graph_relations=torch.tensor(states["relations"], device=self.device), 
            mask=torch.tensor(states["mask"]).to(self.device), 
            exp_temp=exp_temp, 
            rand_coef=rand_coef
        )

        for step in range(states.num_words):
            traj_log_pF += log_pF[1] + log_pF[0]

            np_actions = actions.cpu().numpy()
            # print(np_actions)
            states.step(np_actions)
            prev_actions = actions.clone()

            actions, log_pF, backward_logits = self.model(
                node_embeddings=node_embeddings, 
                graph_relations=torch.tensor(states["relations"], device=self.device), 
                mask=torch.tensor(states["mask"]).to(self.device), 
                exp_temp=exp_temp, 
                rand_coef=rand_coef
            )
            
            traj_log_pB += (
                backward_logits.log_softmax(1).gather(1, prev_actions[:, 1].unsqueeze(-1)).squeeze(-1)
            )

        return states["adjacency"], traj_log_pF, traj_log_pB

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

        train_loader = cycle(train_loader)
        train_losses, val_losses = [], []
        log_Zs = []
        rewards = []

        with trange(self.max_steps, desc="Training") as pbar:
            for iteration in pbar:
                batch = next(train_loader)

                # Initialize s0 and its tracer
                state = masking.StateBatch(
                    self.batch_size,
                    batch["num_words"][0].item() + 1, # TODO: Generalize to num_variables
                    batch["num_words"][0].item(),
                )
                word_embeddings = self.model.init_state(
                    input_ids=batch["input_ids"].to(self.device),
                    attention_mask=batch["attention_mask"].to(self.device),
                    word_ids=batch["word_ids"].to(self.device),
                )

                log_Z = self.model.logZ(batch["num_words"].to(torch.float32).to(self.device))
                # log_Z = torch.tensor([4.6], device=self.device)
                complete_states, traj_log_pF, traj_log_pB = self.sample(word_embeddings, state)
                log_R = torch.log(
                    scores.reward(
                        torch.tensor(complete_states, dtype=torch.float32, device=self.device), 
                        batch["graph"].to(torch.bool).to(torch.float32).to(self.device), 
                        scores.frobenius_norm_distance
                    )
                    + 1e-9  # offset to prevent -inf from occurring
                )

                loss, logs = trajectory_balance_loss(log_Z, traj_log_pF, log_R, traj_log_pB)
                # loss = loss.mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                rewards.append(logs['log_R'])
                train_losses.append(logs['loss'])
                log_Zs.append(logs['log_Z'])
                pbar.set_postfix(
                    loss=f"{train_losses[-1]:.5f}",
                    reward=f"{np.exp(logs['log_R']).mean():.6f}",
                    Z=f"{np.exp(logs['log_Z']):.5f}",
                )
                
                if iteration % self.eval_every_n == 0:
                    np.save(os.path.join(save_folder, f'rewards_{iteration}.npy'), np.array(rewards))
                    np.save(os.path.join(save_folder, f'logZ_{iteration}.npy'), np.array(log_Zs))
                    np.save(os.path.join(save_folder, f'train_losses_{iteration}.npy'), np.array(train_losses))
                    # rewards = []
                    # log_Zs = [] 
        
        np.save(os.path.join(save_folder, "train_losses.npy"), np.array(train_losses))
        np.save(os.path.join(save_folder, "logR.npy"), np.array(rewards))
        np.save(os.path.join(save_folder, "logZ.npy"), np.array(log_Zs)) 
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
    # loss = (log_Z + traj_log_pF - log_R - traj_log_pB) ** 2
    error = (log_Z + traj_log_pF - log_R - traj_log_pB)
    loss = torch.nn.HuberLoss(delta=delta)(error, torch.zeros_like(error))
    logs = {"loss": loss.tolist(), "log_R": log_R.tolist(), "log_Z": log_Z.item()}

    return loss, logs


def cycle(dataloader):
    while True:
        for batch in dataloader:
            yield batch
