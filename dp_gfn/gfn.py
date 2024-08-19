import os
from collections import namedtuple

import haiku as hk
import numpy as np
import jax
import jax.numpy as jnp
import optax
from jax import grad, jit, vmap
from torch.utils.data import DataLoader
from tqdm import trange
from transformers import AutoTokenizer, AutoConfig

from dp_gfn.nets import bert
from dp_gfn.nets.gflownet import gflownet_forward_fn, output_total_flow_fn
from dp_gfn.nets.initial_encoders import label_score_fn
from dp_gfn.utils import masking, scores
from dp_gfn.utils.pretrains import \
    batch_token_embeddings_to_batch_word_embeddings

os.environ["TOKENIZERS_PARALLELISM"] = "false"
GFlowNetState = namedtuple("GFlowNetState", ["optimizer", "step"])
GFlowNetParams = namedtuple("GFlowNetParams", ["bert", "policy", "Z"])


class DPGFN:
    def __init__(self, config, num_tags):
        super().__init__()
        self.config = config
        self.num_tags = num_tags

        self.initialize_vars()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.pref_encoder.pretrained_path
        )
        
        bert.init(self.config.model.pref_encoder.pretrained_path) 
        self.bert_model = hk.without_apply_rng(hk.transform(bert.get_bert_token_embeddings_fn))
        self.bert_params = self.bert_model.init(
            self.key,
            self.model_size,
            jnp.ones(
                (
                    self.batch_size,
                    self.bert_config["max_position_embeddings"],
                ),
                dtype=jnp.int32
            ),
            jnp.ones((self.batch_size, self.bert_config["max_position_embeddings"],), dtype=jnp.int32),
            jnp.ones((self.batch_size, self.bert_config["max_position_embeddings"],), dtype=jnp.int32),
        )

        self.gflownet = hk.without_apply_rng(hk.transform(gflownet_forward_fn))
        base_masks = masking.base_masks(self.num_variables, self.num_variables) 
        self.gflownet_params = self.gflownet.init(
            self.key,
            jnp.ones((self.num_variables, self.model_size)),
            jnp.array(1, dtype=jnp.int32),
            np.zeros((self.num_variables, ), dtype=np.int32),
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
        self.Z_params = self.Z.init(
            self.key, jnp.ones((self.model_size,))
        )
        self.Z = vmap(self.Z.apply, in_axes=(None, 0))

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

        self.num_variables = self.config.max_number_of_words
        self.batch_size = self.config.batch_size
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
        
        # pretrained config

    def init_policy(self):
        self.model = hk.without_apply_rng(hk.transform(GFlowNetState))

        policy_lr = self.config.algorithm.train.optimizer.policy_lr
        Z_lr = self.config.algorithm.train.optimizer.Z_lr
        bert_factor = self.config.algorithm.train.optimizer.bert_factor
        # weight_decay = self.config.algorithm.train.optimizer.weight_decay

        # TODO: Implement lr_scheduler
        self.Z_optimizer = optax.adam(Z_lr)
        self.bert_optimizer = optax.adam(bert_factor * Z_lr)
        self.policy_optimizer = optax.adam(policy_lr)

    def loss(
        self,
        bert_params,
        gflownet_params,
        Z_params,
        tokens,
        num_words_list,
        golds,
    ):
        # Initialize state embeddings
        token_embeddings = jit(self.bert_model.apply, static_argnums=(1, ))(bert_params, self.model_size, **tokens)
        sentence_embeddings = token_embeddings.mean(1)
        node_embeddings = batch_token_embeddings_to_batch_word_embeddings( # TODO: choose first token?
            tokens=tokens,
            token_embeddings=token_embeddings,
            agg_func=self.agg_func,
            max_word_length=self.num_variables,
        )

        log_Z = jit(self.Z)(Z_params, sentence_embeddings).squeeze(axis=-1)
        traj_log_pF, traj_log_pB, complete_states = self.sample(
            gflownet_params, node_embeddings, num_words_list
        )
        log_R = jnp.log(scores.unlabeled_graph_edit_distance(complete_states['adjacency'], golds))

        return trajectory_balance_loss(log_Z, traj_log_pF, log_R, traj_log_pB)

    def sample(self, gflownet_params, node_embeddings, num_words_list):
        states = masking.StateBatch(self.batch_size, self.num_variables, num_words_list)
        node_ids = jnp.zeros((self.batch_size, ), dtype=jnp.int32)
        
        traj_log_pF = jnp.zeros((self.batch_size,), dtype=jnp.float32)
        traj_log_pB = jnp.zeros((self.batch_size,), dtype=jnp.float32)
        
        actions = None
        
        for t in range(self.num_variables):
            print(t)
            self.key, subkey1, subkey2 = jax.random.split(self.key, 3)

            dones = masking.check_done(states['masks'], states['num_words']) 
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
            if t != 0:
                # Exploration: Sample action uniformly at random
                log_uniform = masking.uniform_log_policy(masks=states["masks"][0])
                is_exploration = jax.random.bernoulli(
                    subkey1, p=self.exploration_rate, shape=(self.batch_size, 1)
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
                )  * (1 - dones) # Uniform backward policy 

            # Move to the next state
            states.step(node_ids=next_node_ids, prev_node_ids=node_ids, actions=actions)
            node_ids = next_node_ids

        return traj_log_pF, traj_log_pB, states

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        losses, rewards = [], []

        with trange(self.max_steps, desc="Training") as pbar:
            for iteration in pbar:
                batch = next(iter(train_loader))
                
                tokens = self.tokenizer(
                    batch["text"],
                    return_tensors="jax",
                    padding="max_length",
                    truncation=True,
                    add_special_tokens=False,
                )

                grads, logs = grad(self.loss, argnums=(0, 1, 2, ), has_aux=True)(
                    self.bert_params,
                    self.gflownet_params,
                    self.Z_params,
                    tokens,
                    batch["num_words"],
                    batch["graph"]
                ) 
                
                print(grads)
                
                # updates, opt_state = self.bert_optimizer.update(
                #     grads,
                # )
                
                pbar.set_postfix(loss=f"{logs['loss']:.2f}")

        # TODO: Continue here
        pass

    def evaluation(
        self,
    ):
        pass

    def val_step(
        self,
    ):
        pass


def trajectory_balance_loss(log_Z, traj_log_pF, log_R, traj_log_pB, delta=1):
    error = jnp.squeeze(log_Z + traj_log_pF - log_R - traj_log_pB, axis=-1)
    loss = jnp.mean(optax.huber_loss(error, delta=delta))

    logs = {
        "error": error,
        "loss": loss,
    }

    return (loss, logs)
