import logging
import os
import traceback

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import trange

from dp_gfn.nets.gflownet import DPGFlowNet
from dp_gfn.utils import masking, scores, io
# from dp_gfn.utils.replay_buffer import ReplayBuffer

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
        self.syn_batch_size = config.train.syn_batch_size
        # self.buffer_capacity = config.train.buffer_capacity
        self.exp_temp = config.train.exp_temp
        self.rand_coef = config.train.rand_coef
        self.p_init = config.train.p_init
        self.reward_scale_factor = config.reward_scale_factor

    def initialize_policy(self, config):
        config = config.train

        bert_params = [params for name, params in self.model.named_parameters() 
                       if "bert" in name]
        logZ_params = [params for name, params in self.model.named_parameters() 
                       if "logZ" in name]
        gfn_params = [params for name, params in self.model.named_parameters() 
                      if ("bert" not in name) and ("logZ" not in name)]

        self.optimizer = torch.optim.Adam(
            [
                {
                    "params": bert_params,
                    "lr": config.optimizer.gfn_lr * config.optimizer.bert_factor,
                },
                {"params": logZ_params, "lr": config.optimizer.Z_lr},
                {"params": gfn_params, "lr": config.optimizer.gfn_lr},
            ]
        )

    def sample(self, node_embeddings, states, exp_temp=1.0, rand_coef=0.0):
        traj_log_pF = torch.zeros((self.batch_size,), dtype=torch.float32, device=self.device)
        traj_log_pB = torch.zeros((self.batch_size,), dtype=torch.float32, device=self.device)

        actions, log_pF, backward_logits = self.model(
            node_embeddings=node_embeddings,
            graph_relations=torch.tensor(states["relations"], device=self.device),
            mask=torch.tensor(states["mask"]).to(self.device),
            exp_temp=exp_temp,
            rand_coef=rand_coef,
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
                rand_coef=rand_coef,
            )

            traj_log_pB += (
                backward_logits.log_softmax(1)
                .gather(1, prev_actions[:, 1].unsqueeze(-1))
                .squeeze(-1)
            )

        return states["adjacency"], traj_log_pF, traj_log_pB

    def synthesize_trajectory(self, node_embeddings, states, graph_relations, exp_temp=1.0, rand_coef=0.0):
        traj_log_pF = torch.zeros((self.syn_batch_size,), dtype=torch.float32, device=self.device)
        traj_log_pB = torch.zeros((self.syn_batch_size,), dtype=torch.float32, device=self.device)
       
        action_list = self.model.trace_backward(
            node_embeddings=node_embeddings,
            graph_relations=graph_relations,
            exp_temp=exp_temp,
            rand_coef=rand_coef
        )
        
        log_pF, backward_logits = self.model(
            node_embeddings=node_embeddings,
            graph_relations=graph_relations,
            mask=torch.tensor(states['mask'].to(self.device)),
            actions=action_list[0],
            exp_temp=exp_temp,
            rand_coef=rand_coef
        ) 
        
        for step in range(states.num_words):
            traj_log_pF += log_pF[1] + log_pF[0]

            np_actions = actions.cpu().numpy()
            states.step(np_actions)

            actions, log_pF, backward_logits = self.model(
                node_embeddings=node_embeddings,
                graph_relations=torch.tensor(states["relations"], device=self.device),
                mask=torch.tensor(states["mask"]).to(self.device),
                actions=action_list[step + 1],
                exp_temp=exp_temp,
                rand_coef=rand_coef,
            )

            traj_log_pB += (
                backward_logits.log_softmax(1)
                .gather(1, action_list[step][:, 1].unsqueeze(-1))
                .squeeze(-1)
            )
            
        return traj_log_pF, traj_log_pB

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

        # replay = ReplayBuffer(self.buffer_capacity, self.max_number_of_words + 1, self.p_init)
        train_loader = cycle(train_loader)
        train_losses, val_losses = [], []
        log_Zs = []
        rewards = []

        with trange(self.max_steps, desc="Training") as pbar:
            for iteration in pbar:
                batch = next(train_loader)

                try:
                    word_embeddings = self.model.init_state(
                        input_ids=batch["input_ids"].to(self.device),
                        attention_mask=batch["attention_mask"].to(self.device),
                        word_ids=batch["word_ids"].to(self.device),
                    )
                    log_Z = self.model.logZ(
                        batch["num_words"].to(torch.float32).to(self.device)
                    )
                    # log_Z = torch.tensor([4.6], device=self.device)
                    
                    state = masking.StateBatch(
                        self.batch_size,
                        batch["num_words"][0].item()
                        + 1,  # TODO: Generalize to num_variables
                        batch["num_words"][0].item(),
                    )
                    
                    # Collect trajectories using pF
                    complete_states, traj_log_pF, traj_log_pB = self.sample(
                        word_embeddings, state, self.exp_temp, self.rand_coef
                    )

                    # Synthesize trajectories using pB
                    state.reset(self.syn_batch_size)
                    syn_traj_log_pF, syn_traj_log_pB = self.synthesize_trajectory(
                        word_embeddings, state, batch["graph"], self.exp_temp, self.rand_coef
                    )
                    
                    # Gather both forward trajectories and backward trajectories
                    complete_states = torch.cat([complete_states, batch["graph"].repeat(self.syn_batch_size, 1, 1)], dim=0)
                    traj_log_pB = torch.cat([traj_log_pB, syn_traj_log_pB], dim=0)
                    traj_log_pF = torch.cat([traj_log_pF, syn_traj_log_pF], dim=0)

                    # Calculate reward                
                    log_R = torch.log(
                        scores.reward(
                            torch.tensor(
                                complete_states, dtype=torch.float32, device=self.device
                            ),
                            batch["graph"].to(torch.bool).to(torch.float32).to(self.device),
                            scores.frobenius_norm_distance,
                        )
                        + 1e-9  # offset to prevent -inf from occurring
                    )

                    loss, logs = trajectory_balance_loss(log_Z, traj_log_pF, log_R, traj_log_pB)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    rewards.append(logs["log_R"])
                    train_losses.append(logs["loss"])
                    log_Zs.append(logs["log_Z"])
                    pbar.set_postfix(
                        loss=f"{train_losses[-1]:.5f}",
                        reward=f"{np.exp(logs['log_R']).mean():.6f}",
                        Z=f"{np.exp(logs['log_Z']):.5f}",
                    )

                    if iteration % self.eval_every_n == 0:
                        rewards, log_Zs, train_losses = io.save_train_aux(
                            save_folder, iteration, rewards, log_Zs, train_losses
                        )
                
                
                except Exception as e:
                    logging.error(f"Error encountered at iteration {iteration}: {e}") 
                    traceback.print_exc()   # Print detailed error traceback
                   
                    error_save_path = os.path.join(save_folder, f"error_iteration_{iteration}")
                    os.makedirs(error_save_path, exist_ok=True)

                    # Save model parameters, optimizer state
                    torch.save(self.model.state_dict(), os.path.join(error_save_path, "model_state_dict.pt"))
                    torch.save(self.optimizer.state_dict(), os.path.join(error_save_path, "optimizer_state_dict.pt"))
                    # Save current batch (optional, but might be helpful for debugging)
                    torch.save(batch, os.path.join(error_save_path, "batch.pt"))

                    # Save dataloader state (if possible)
                    try:
                        torch.save(train_loader.state_dict(), os.path.join(error_save_path, "dataloader_state.pt"))
                    except Exception as e_dataloader:  # Not all dataloaders are easily saveable
                         logging.warning(f"Could not save dataloader state: {e_dataloader}")
                         
                    break
                
        np.save(os.path.join(save_folder, "train_losses.npy"), np.array(train_losses))
        np.save(os.path.join(save_folder, "logR.npy"), np.array(rewards))
        np.save(os.path.join(save_folder, "logZ.npy"), np.array(log_Zs))
        return train_losses, val_losses

    @torch.no_grad()  # This decorator prevents gradient updates
    def val_step(self, val_loader: DataLoader):
        """Performs a validation step.

        Args:
            val_loader (DataLoader): The validation data loader.

        Returns:
            float: The average loss over the validation set.
        """
        self.model.eval()  # Set the model to evaluation mode
        total_loss = 0
        total_samples = 0

        for batch in val_loader:
            word_embeddings = self.model.init_state(
                input_ids=batch["input_ids"].to(self.device),
                attention_mask=batch["attention_mask"].to(self.device),
                word_ids=batch["word_ids"].to(self.device),
            )
            log_Z = self.model.logZ(
                batch["num_words"].to(torch.float32).to(self.device)
            )

            state = masking.StateBatch(
                self.batch_size,
                batch["num_words"][0].item() + 1,
                batch["num_words"][0].item(),  # Assuming batch size > 1 and consistent number of words
            )
            complete_states, traj_log_pF, traj_log_pB = self.sample(
                word_embeddings, state, self.exp_temp, self.rand_coef
            )

            log_R = torch.log(
                scores.reward(
                    torch.tensor(
                        complete_states, dtype=torch.float32, device=self.device
                    ),
                    batch["graph"].to(torch.bool).to(torch.float32).to(self.device),
                    scores.frobenius_norm_distance,
                )
                + 1e-9
            )
            loss, _ = trajectory_balance_loss(log_Z, traj_log_pF, log_R, traj_log_pB)

            total_loss += loss.item() * len(batch)  # Weighted average for different batch sizes
            total_samples += len(batch)

        self.model.train()  # Set the model back to training mode
        avg_loss = total_loss / total_samples if total_samples > 0 else 0  # Avoid division by zero

        return avg_loss

    def inference(
        self,
    ):
        pass

    def load_weights(self, filename):
        pass


def trajectory_balance_loss(log_Z, traj_log_pF, log_R, traj_log_pB, delta=1.0):
    error = log_Z + traj_log_pF - log_R - traj_log_pB
    loss = torch.nn.HuberLoss(delta=delta)(error, torch.zeros_like(error))
    logs = {"loss": loss.tolist(), "log_R": log_R.tolist(), "log_Z": log_Z.item()}

    return loss, logs


def cycle(dataloader):
    while True:
        for batch in dataloader:
            yield batch
