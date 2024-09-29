import os
from collections import namedtuple

import numpy as np
from tqdm import trange
try:
    from evaluation import save_predictions
except:
    pass

import haiku as hk
import jax
import jax.numpy as jnp
import optax
from dp_gfn.nets import bert
from dp_gfn.nets.gflownet import gflownet_forward_fn, output_total_flow_fn
from dp_gfn.nets.initial_encoders import label_score_fn
from dp_gfn.utils import masking, scores, io
from dp_gfn.utils.pretrains import \
    batch_token_embeddings_to_batch_word_embeddings, create_position_ids_from_input_ids
from jax import grad, jit, vmap
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer
import subprocess


os.environ["TOKENIZERS_PARALLELISM"] = "false"
GFlowNetState = namedtuple("GFlowNetState", ["optimizer", "step"])
GFlowNetParams = namedtuple("GFlowNetParams", ["bert", "gflownet", "Z"])


class DPGFN:
    def __init__(self, config, num_tags, id2rel, pretrained_path=None):
        super().__init__()
        self.config = config
        self.num_tags = num_tags
        self.id2rel = id2rel

        self.initialize_vars()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.pref_encoder.pretrained_path
        )

        # Initalizer
        dummy_input_ids = jnp.ones((self.batch_size, self.tokenizer.model_max_length), dtype=jnp.int32)

        bert.init(self.config.model.pref_encoder.pretrained_path)
        self.bert_model = hk.transform(bert.get_bert_token_embeddings_fn)
        self.bert_params = self.bert_model.init(
            self.key,
            dummy_input_ids,
            jnp.ones_like(dummy_input_ids),
            jnp.zeros_like(dummy_input_ids),
            jnp.ones_like(dummy_input_ids),
            True
        )

        # Backbone
        self.gflownet = hk.without_apply_rng(hk.transform(gflownet_forward_fn))
        base_masks = masking.base_masks(self.num_variables, self.num_variables)
        self.gflownet_params = self.gflownet.init(
            self.key,
            jax.random.normal(self.key, (self.num_variables, self.bert_config['hidden_size'])),
            jnp.array(1, dtype=jnp.int32),
            np.zeros((self.num_variables,), dtype=np.int32),
            base_masks,
            self.num_tags,
            self.num_layers,
            self.num_heads,
            self.key_size,
        )
        self.gflownet = vmap(
            self.gflownet.apply, in_axes=(None, 0, 0, 0, 0, None, None, None, None)
        )

        self.Z = hk.without_apply_rng(hk.transform(output_total_flow_fn))
        self.Z_params = self.Z.init(self.key, jnp.ones((self.bert_config['hidden_size'],)))
        self.Z = vmap(self.Z.apply, in_axes=(None, 0))

        # TODO: Tags predictor
        # self.label_scorer = hk.without_apply_rng(hk.transform(label_scorer_fn))
        # self.label_scorer_params = self.label_scorer.init(
        #     self.key,
        #     jnp.ones((self.node_embedding_dim, )),
        #     jnp.ones((self.node_embedding_dim, )),
        #     self.num_tags
        # )
        # self.label_scorer = vmap(self.label_scorer.apply, in_axes=(None, 0, 0, None))

        self.init_policy()
        
        if pretrained_path is not None:
            self.load_weights(pretrained_path)

    def initialize_vars(self):
        self.key = jax.random.PRNGKey(self.config.seed)
        self.bert_config = AutoConfig.from_pretrained(
            self.config.model.pref_encoder.pretrained_path
        ).to_dict()

        self.num_variables = self.config.model.num_variables
        self.batch_size = self.config.batch_size
        self.save_path = self.config.save_path
        
        self.num_layers = self.config.model.backbone.num_layers
        self.num_heads = self.config.model.backbone.encoder_block.num_heads
        self.key_size = self.config.model.backbone.encoder_block.d_k
        self.node_embedding_dim = self.config.model.common.node_embedding_dim
        self.model_size = self.num_heads * self.key_size
        self.init_scale = 2.0 / self.config.model.backbone.num_layers
        self.agg_func = self.config.model.pref_encoder.agg_func

        config = self.config.algorithm
        self.reward_scale_factor = config.reward_scale_factor

        # train hyperparameters
        config = config.train
        self.n_grad_accumulation_steps = config.n_grad_accumulation_steps
        self.max_steps = config.max_steps
        self.eval_on_train = config.eval_on_train
        self.exploration_rate = config.exploration_rate
        self.clip_grad = config.clip_grad
        self.eval_every_n = config.eval_every_n
        self.save_every_n = config.save_every_n

    def init_policy(self):
        self.model = hk.without_apply_rng(hk.transform(GFlowNetState))

        gflownet_lr = self.config.algorithm.train.optimizer.gflownet_lr
        Z_lr = self.config.algorithm.train.optimizer.Z_lr
        bert_factor = self.config.algorithm.train.optimizer.bert_factor
        # weight_decay = self.config.algorithm.train.optimizer.weight_decay

        self.Z_optimizer = optax.adam(Z_lr)
        self.Z_state = self.Z_optimizer.init(self.Z_params)
        self.bert_optimizer = optax.adam(bert_factor * Z_lr)
        self.bert_state = self.bert_optimizer.init(self.bert_params)
        self.gflownet_optimizer = optax.adam(gflownet_lr)
        self.gflownet_state = self.gflownet_optimizer.init(self.gflownet_params)

    def init_states(self, bert_params, tokens, position_ids, training=False):
        self.key, bert_key = jax.random.split(self.key)
        # Present initial state (s0) as a set of node_embeddings
        token_embeddings = jit(self.bert_model.apply, static_argnums=(6, ))(
            bert_params, bert_key, tokens['input_ids'], position_ids, jnp.zeros_like(tokens['input_ids']), tokens['attention_mask'], training=training
        )  
        
        node_embeddings = batch_token_embeddings_to_batch_word_embeddings(  # TODO: Inspect the others (first, last)
            tokens=tokens, 
            token_embeddings=token_embeddings,
            agg_func=self.agg_func,
            max_word_length=self.num_variables
        )   # TODO: Find another way that allow parallelization -> JIT
        
        # Embeddings for computing intitial flow 
        sentence_embeddings = token_embeddings.mean(1)  # TODO: Using sentence embeddings might be plain 
        
        return node_embeddings, sentence_embeddings

    def loss(
        self,
        bert_params,
        gflownet_params,
        Z_params,
        tokens,
        position_ids,
        num_words_list,
        golds,
        delta
    ):
        node_embeddings, sentence_embeddings = self.init_states(bert_params, tokens, position_ids, training=True)
        
        log_Z = jit(self.Z)(Z_params, sentence_embeddings).squeeze(axis=-1)
        
        # Sample trajectory $\tau = (s_0 -> s_1 -> ... -> s_n)$
        traj_log_pF, traj_log_pB, complete_states = self.sample(
            gflownet_params, node_embeddings, num_words_list, delta
        )
       
        # Compute reward: # TODO: inspect other metrics?
        golds = (golds != 0).astype(bool)
        log_R = jnp.log(
            scores.scale_between(
                inputs=scores.reward(complete_states["adjacency"], golds, scores.frobenius_norm_distance),
                original_min=1, 
                original_max=jnp.exp(1),
                scaled_min=3,
                scaled_max=10
            )
        )

        return trajectory_balance_loss(log_Z, traj_log_pF, log_R, traj_log_pB)
    
    def sample(self, gflownet_params, node_embeddings, num_words_list, delta=0.001):
        states = masking.StateBatch(self.num_variables, num_words_list)
        node_ids = jnp.zeros((len(num_words_list),), dtype=jnp.int32)
        actions = None

        traj_log_pF = jnp.zeros((len(num_words_list),), dtype=jnp.float32)
        traj_log_pB = jnp.zeros((len(num_words_list),), dtype=jnp.float32)

        for t in range(self.num_variables):
            edge_dones, node_dones = masking.check_done(states["masks"], states["num_words"])
            if np.all(edge_dones):
                break

            # Exploitation: Sample action based on GFlowNet policy
            log_pi_t, log_node_tp1 = jit(self.gflownet, static_argnums=(5, 6, 7, 8))(
                gflownet_params,
                node_embeddings,
                node_ids,
                states["labels"],
                states["masks"],
                self.num_tags,
                self.num_layers,
                self.num_heads,
                self.key_size,
            )

            next_node_ids, log_pF_node, _ = jit(self.sample_action)(log_node_tp1, states['masks'][1], delta)
            log_pF_node = jnp.where(node_dones, jnp.zeros_like(log_pF_node), log_pF_node)
            traj_log_pF += log_pF_node * (1 - node_dones)
            # log_pB_node = jnp.where(node_dones, jnp.zeros_like(log_pB_node), log_pB_node)
            # traj_log_pB += log_pB_node * (1 - node_dones) # log_pB_node = inf if node_done 

            # Only sample next node at step 0
            # TODO: JIT here?
            if t != 0:
                actions, log_pF, _ = self.sample_action(log_pi_t, states['masks'][0], delta)
                traj_log_pF += log_pF * (1 - edge_dones)
                # log_pB = 
                # traj_log_pB += log_pB * (1 - edge_dones)  
            
            # Move to the next state
            next_node_ids = next_node_ids.squeeze(-1)
            states.step(node_ids=next_node_ids, prev_node_ids=node_ids, actions=actions)
            node_ids = next_node_ids    # num_word = 1 => edge_dones at step 0

        return traj_log_pF, traj_log_pB, states

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        save_folder = os.path.join(
            self.save_path, 
            f"run_bs={self.batch_size}_epsilon={self.exploration_rate}_dim={self.model_size}_nlayers={self.num_layers}_nheads={self.num_heads}")
        os.makedirs(save_folder, exist_ok=True)
        
        # train_loader = iter(train_loader)
        train_loader = cycle(train_loader)
        train_losses, val_losses = [], []
        train_loss, val_loss = 0, 0
         
        exploration_schedule = jax.jit(optax.linear_schedule(
            init_value=jnp.array(self.exploration_rate),
            end_value=jnp.array(self.exploration_rate / 10.),
            transition_steps=self.max_steps // 2,
            transition_begin=self.max_steps // 1000
        ))

        with trange(self.max_steps, desc="Training") as pbar:
            for iteration in pbar:
                delta = exploration_schedule(iteration)
                batch = next(train_loader)
                 
                tokens = self.tokenizer(
                    batch["text"],
                    return_tensors="jax",
                    padding="max_length",
                    truncation=True,
                    add_special_tokens=False,
                )
                
                position_ids = create_position_ids_from_input_ids(tokens["input_ids"])

                (bert_grads, gflownet_grads, Z_grads), logs = grad(
                    self.loss, argnums=(0, 1, 2), has_aux=True)(
                    self.bert_params,
                    self.gflownet_params,
                    self.Z_params,
                    tokens,
                    position_ids,
                    batch["num_words"],
                    batch["graph"],
                    delta
                )

                bert_updates, self.bert_state = self.bert_optimizer.update(
                    bert_grads, self.bert_state
                )
                gflownet_updates, self.gflownet_state = self.gflownet_optimizer.update(
                    gflownet_grads, self.gflownet_state
                )
                Z_updates, self.Z_state = self.Z_optimizer.update(Z_grads, self.Z_state)
                
                self.bert_params = optax.apply_updates(self.bert_params, bert_updates)
                self.gflownet_params = optax.apply_updates(self.gflownet_params, gflownet_updates) 
                self.Z_params = optax.apply_updates(self.Z_params, Z_updates)

                if iteration % self.eval_every_n == 0:
                    gold = os.path.join(save_folder, f"gold_{iteration}.conllu")
                    system = os.path.join(save_folder, f"system_{iteration}.conllu")
                    save_predictions(algorithm=self, loader=val_loader, config=self.config, id2rel=self.id2rel, original=self.config.train_path.replace("train", "dev"), gold=gold, system=system)
                    subprocess.run(['./ud_eval.py', gold, system, '-v'])
                    
                    val_loss = self.val_step(val_loader)
                    print('loss on val:', val_loss)
                    if len(train_losses):
                        print('loss on train:', sum(train_losses) / len(train_losses))
                    train_losses = []
                    val_losses.append(val_loss)

                    if self.eval_on_train:
                        train_loss = self.val_step(train_loader) 
                        train_losses.append(train_loss)
                    
                    print("-"*50)
                
                if iteration % self.save_every_n == 0:
                    io.save(os.path.join(save_folder, f"model_{iteration}.npz"),
                            bert=self.bert_params,
                            gflownet=self.gflownet_params,
                            Z=self.Z_params) 
                 
                if self.eval_on_train: 
                    pbar.set_postfix(
                        epsilon=f"{self.exploration_rate:.6f}",
                        loss=f"{logs['loss']:.6f}", 
                        reward=f"{np.exp(logs['log_R']):.6f}",
                        train_loss=f"{train_loss:.6f}",
                        val_loss=f"{val_loss:.6f}"
                    )
                else:
                    pbar.set_postfix(
                        epsilon=f"{self.exploration_rate:.6f}",
                        loss=f"{logs['loss']:.6f}", 
                        reward=f"{np.exp(logs['log_R']):.6f}",
                        val_loss=f"{val_loss:.6f}"
                    )
                train_losses.append(logs['loss'])
            
        # Save model parameters
        io.save(os.path.join(save_folder, "model.npz"), 
                bert=self.bert_params, 
                gflownet=self.gflownet_params, 
                Z=self.Z_params)
        
        return train_losses, val_losses

    def val_step(
        self,
        val_loader,
        delta=0.
    ):
        losses = []
        
        for batch in val_loader:
            tokens = self.tokenizer(
                batch["text"],
                return_tensors="jax",
                padding="max_length",
                truncation=True,
                add_special_tokens=False,
            )
            
            position_ids = create_position_ids_from_input_ids(tokens['input_ids'])
            
            (loss, _) = self.loss(
                self.bert_params, 
                self.gflownet_params, 
                self.Z_params, 
                tokens, 
                position_ids,
                batch["num_words"], 
                batch["graph"], 
                delta=delta
            )
            losses.append(loss)
        
        return np.mean(losses)

    def inference(self, tokens, position_ids, num_words_list, delta=0.):
        node_embeddings, sentence_embeddings = self.init_states(self.bert_params, tokens, position_ids, training=False)
        traj_log_pF, traj_log_pB, complete_states = self.sample(
            self.gflownet_params, node_embeddings, num_words_list, delta=delta
        )
        log_Z = jit(self.Z)(self.Z_params, sentence_embeddings).squeeze(axis=-1)
        log = (log_Z, traj_log_pF, traj_log_pB) 
        
        return complete_states, log
    
    def load_weights(self, filename):
        params = io.load(filename)
        
        self.bert_params = params['bert']
        self.gflownet_params = params['gflownet']
        self.Z_params = params['Z']
     

def trajectory_balance_loss(log_Z, traj_log_pF, log_R, traj_log_pB, delta=1): 
    assert log_Z.shape == traj_log_pF.shape == traj_log_pB.shape == log_R.shape
     
    error = log_Z + traj_log_pF - log_R - traj_log_pB
    loss = jnp.mean(optax.huber_loss(error, delta=delta))

    logs = {
        "error": error,
        "loss": loss,
        "log_R": log_R.mean().item()
    }

    return (loss, logs)


def cycle(dataloader):
    while True:
        for batch in dataloader: 
            yield batch
            
            