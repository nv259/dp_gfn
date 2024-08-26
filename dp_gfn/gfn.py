import os
from collections import namedtuple

import numpy as np
from tqdm import trange

import haiku as hk
import jax
import jax.numpy as jnp
import optax
from dp_gfn.nets import bert
from dp_gfn.nets.gflownet import gflownet_forward_fn, output_total_flow_fn
from dp_gfn.nets.initial_encoders import label_score_fn
from dp_gfn.utils import masking, scores, io
from dp_gfn.utils.pretrains import \
    batch_token_embeddings_to_batch_word_embeddings
from jax import grad, jit, vmap
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"
GFlowNetState = namedtuple("GFlowNetState", ["optimizer", "step"])
GFlowNetParams = namedtuple("GFlowNetParams", ["bert", "gflownet", "Z"])


class DPGFN:
    def __init__(self, config, num_tags):
        super().__init__()
        self.config = config
        self.num_tags = num_tags

        self.initialize_vars()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.pref_encoder.pretrained_path
        )

        # Initalizer
        bert.init(self.config.model.pref_encoder.pretrained_path)
        self.bert_model = hk.without_apply_rng(
            hk.transform(bert.get_bert_token_embeddings_fn)
        )
        self.bert_params = self.bert_model.init(
            self.key,
            self.model_size,
            jnp.ones(
                (
                    self.batch_size,
                    self.bert_config["max_position_embeddings"],
                ),
                dtype=jnp.int32,
            ),
            jnp.ones(
                (
                    self.batch_size,
                    self.bert_config["max_position_embeddings"],
                ),
                dtype=jnp.int32,
            ),
            jnp.ones(
                (
                    self.batch_size,
                    self.bert_config["max_position_embeddings"],
                ),
                dtype=jnp.int32,
            ),
        )

        # Backbone
        self.gflownet = hk.without_apply_rng(hk.transform(gflownet_forward_fn))
        base_masks = masking.base_masks(self.num_variables, self.num_variables)
        self.gflownet_params = self.gflownet.init(
            self.key,
            jnp.ones((self.num_variables, self.model_size)),
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
        self.Z_params = self.Z.init(self.key, jnp.ones((self.model_size,)))
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

    def init_states(self, bert_params, tokens):
        # Present initial state (s0) as a set of node_embeddings
        token_embeddings = jit(self.bert_model.apply, static_argnums=(1, ))(
        # token_embeddings = self.bert_model.apply(
            bert_params, self.model_size, **tokens
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
        num_words_list,
        golds,
        delta
    ):
        node_embeddings, sentence_embeddings = self.init_states(bert_params, tokens)
        
        log_Z = jit(self.Z)(Z_params, sentence_embeddings).squeeze(axis=-1)
        
        # Sample trajectory $\tau = (s_0 -> s_1 -> ... -> s_n)$
        traj_log_pF, traj_log_pB, complete_states = self.sample(
            gflownet_params, node_embeddings, num_words_list, delta
        )
       
        # Compute reward: # TODO: inspect other metrics?
        log_R = jnp.log(jit(
            scores.unlabeled_graph_edit_distance)(complete_states["adjacency"], golds))

        return trajectory_balance_loss(log_Z, traj_log_pF, log_R, traj_log_pB)

    def sample(self, gflownet_params, node_embeddings, num_words_list, delta=0.001):
        states = masking.StateBatch(self.num_variables, num_words_list)
        node_ids = jnp.zeros((self.batch_size,), dtype=jnp.int32)
        actions = None

        traj_log_pF = jnp.zeros((self.batch_size,), dtype=jnp.float32)
        traj_log_pB = jnp.zeros((self.batch_size,), dtype=jnp.float32)

        for t in range(self.num_variables):
            self.key, subkey1, subkey2 = jax.random.split(self.key, 3)

            dones = masking.check_done(states["masks"], states["num_words"])
            if np.all(dones):
                break

            # Exploitation: Sample action based on GFlowNet policy
            log_pi, next_node_logits = jit(self.gflownet, static_argnums=(5, 6, 7, 8))(
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

            next_node_ids = next_node_logits.argmax(axis=-1)

            # Only sample next node at step 0
            # TODO: JIT here?
            if t != 0:
                # Exploration: Sample action uniformly at random
                log_uniform = masking.uniform_log_policy(masks=states["masks"][0])
                is_exploration = jax.random.bernoulli(
                    subkey1, p=delta, shape=(self.batch_size, 1)
                )  # TODO: stimulated annealing

                # Mixing GFlowNet policy and uniform policy:
                # \pi = (1 - delta) * Policy + delta * Uniform
                log_pi = jnp.where(is_exploration, log_uniform, log_pi)

                # Sample actions
                actions = masking.batch_random_choice(
                    subkey2, jnp.exp(log_pi), states["masks"]
                )

                log_probs = jnp.take_along_axis(log_pi, actions, axis=1).squeeze(-1)
                traj_log_pF += log_probs * (1 - dones)
                 
                traj_log_pB += masking.uniform_log_policy(
                    states["masks"][0],
                    is_forward=False,
                ) * (1 - dones)  # Uniform backward policy

            # Move to the next state
            states.step(node_ids=next_node_ids, prev_node_ids=node_ids, actions=actions)
            node_ids = next_node_ids

        return traj_log_pF, traj_log_pB, states

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        save_folder = os.path.join(
            self.save_path, 
            f"run_bs={self.batch_size}_epsilon={self.exploration_rate}_dim={self.model_size}_nlayers={self.num_layers}_nheads={self.num_heads}")
        os.makedirs(save_folder, exist_ok=True)
        
        train_losses, val_losses = [], []
        
        exploration_schedule = jax.jit(optax.linear_schedule(
            init_value=jnp.array(self.exploration_rate),
            end_value=jnp.array(self.exploration_rate / 10.),
            transition_steps=self.max_steps // 2,
            transition_begin=self.max_steps // 1000
        ))

        with trange(self.max_steps, desc="Training") as pbar:
            for iteration in pbar:
                delta = exploration_schedule(iteration)
                batch = next(iter(train_loader))

                tokens = self.tokenizer(
                    batch["text"],
                    return_tensors="jax",
                    padding="max_length",
                    truncation=True,
                    add_special_tokens=False,
                )

                (bert_grads, gflownet_grads, Z_grads), logs = grad(
                    self.loss, argnums=(0, 1, 2), has_aux=True)(
                    self.bert_params,
                    self.gflownet_params,
                    self.Z_params,
                    tokens,
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
                    val_loss = self.val_step(val_loader)
                    val_losses.append(val_loss)
                    
                    if self.eval_on_train:
                        train_loss = self.val_step(train_loader) 
                        train_losses.append(train_loss)
                
                if iteration % self.save_every_n == 0:
                    io.save(os.path.join(save_folder, f"model_{iteration}.npz"),
                            bert=self.bert_params,
                            gflownet=self.gflownet_params,
                            Z=self.Z_params) 
                 
                if self.eval_on_train: 
                    pbar.set_postfix(
                        epsilon=f"{self.exploration_rate:.2f}",
                        loss=f"{logs['loss']:.2f}", 
                        train_loss=f"{train_loss:.2f}",
                        val_loss=f"{val_loss:.2f}"
                    )
                else:
                    pbar.set_postfix(
                        epsilon=f"{self.exploration_rate:.2f}",
                        loss=f"{logs['loss']:.2f}", 
                        val_loss=f"{val_loss:.2f}"
                    )
            
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
            
            (loss, _) = self.loss(
                self.bert_params, 
                self.gflownet_params, 
                self.Z_params, 
                tokens, 
                batch["num_words"], 
                batch["graph"], 
                delta=delta
            )
            losses.append(loss)
        
        return np.mean(losses)


def trajectory_balance_loss(log_Z, traj_log_pF, log_R, traj_log_pB, delta=1): 
    
    assert log_Z.shape == traj_log_pF.shape == traj_log_pB.shape == log_R.shape
     
    error = log_Z + traj_log_pF - log_R - traj_log_pB
    loss = jnp.mean(optax.huber_loss(error, delta=delta))

    logs = {
        "error": error,
        "loss": loss,
    }

    return (loss, logs)
