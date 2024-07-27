import os
from collections import namedtuple

import haiku as hk
import jax
import jax.numpy as jnp
import optax
from jax import grad, jit, vmap
from torch.utils.data import DataLoader
from tqdm import trange
from transformers import AutoTokenizer, AutoConfig

from dp_gfn.nets.bert import get_bert_token_embeddings_fn
from dp_gfn.nets.gflownet import output_logits_fn, output_total_flow_fn
from dp_gfn.nets.initial_encoders import label_score_fn, state_featurizer_fn
from dp_gfn.utils import masking, scores
from dp_gfn.utils.pretrains import \
    batch_token_embeddings_to_batch_word_embeddings

os.environ["TOKENIZERS_PARALLELISM"] = "false"
GFlowNetState = namedtuple("GFlowNetState", ["optimizer", "step"])
GFlowNetParams = namedtuple("GFlowNetParams", ["bert", "state_encoder", "policy", "Z"])


class DPGFN:
    def __init__(self, config, num_tags):
        super().__init__()
        self.config = config
        self.num_tags = num_tags

        self.initialize_vars()

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model.pref_encoder.pretrained_path
        )
        
        self.bert = hk.without_apply_rng(hk.transform(get_bert_token_embeddings_fn))
        self.bert_params = self.bert.init(
            self.key,
            jnp.ones(
                (
                    self.batch_size,
                    self.bert_config["max_position_embeddings"],
                    self.bert_config["hidden_size"],
                )
            ),
        )

        self.state_encoder = hk.without_apply_rng(hk.transform(state_featurizer_fn))
        self.state_encoder_params = self.state_encoder.init(
            self.key,
            jnp.ones(
                (
                    self.batch_size,
                    self.max_number_of_words,
                    self.bert_config["hidden_size"],
                )
            ),
            self.max_number_of_words,
            self.node_embedding_dim,
            self.init_scale,
        )
        self.state_encoder = vmap(self.state_encoder.apply, in_axes=(None, 0, None))

        self.gflownet = hk.without_apply_rng(hk.transform(output_logits_fn))
        self.gflownet_params = self.gflownet.init(
            self.key,
            jnp.ones((self.max_number_of_words**2, self.node_embedding_dim * 2)),
            jnp.ones(
                (self.max_number_of_words**2,), dtype=int
            ),  # TODO: use int label here, what about labels' embeddings?
            masking.base_mask(7, self.max_number_of_words),
            self.num_tags,
            self.num_layers,
            self.num_heads,
            self.key_size,
        )
        self.gflownet = vmap(
            self.gflownet.apply, in_axes=(None, 0, 0, 0, None, None, None, None)
        )

        self.Z = hk.without_apply_rng(hk.transform(output_total_flow_fn))
        self.Z_params = self.Z.init(
            self.key, jnp.ones((self.bert_config["hidden_size"],))
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
        self.init_scale = 2.0 / self.config.model.backbone.num_layers
        self.key = jax.random.PRNGKey(self.config.seed)
        self.bert_config = AutoConfig.from_pretrained(
            self.config.model.pref_encoder.pretrained_path
        ).to_dict()

        self.max_number_of_words = self.config.max_number_of_words
        self.batch_size = self.config.batch_size
        # self.device = self.config.device
        self.node_embedding_dim = self.config.model.common.node_embedding_dim

        config = self.config.algorithm
        # self.backward_policy = config.backward_policy
        # self.score_fn = config.score_fn

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
        weight_decay = self.config.algorithm.train.optimizer.weight_decay

        # TODO: Implement lr_scheduler
        self.Z_optimizer = optax.adam(Z_lr, weight_decay=weight_decay)
        self.bert_optimizer = optax.adam(bert_factor * Z_lr, weight_decay=weight_decay)
        self.policy_optimizer = optax.adam(policy_lr, weight_decay=weight_decay)

    def loss(
        self,
        bert_params,
        state_encoder_params,
        gflownet_params,
        Z_params,
        tokens,
        num_words_list,
        golds,
    ):
        # Initialize state embeddings
        token_embeddings = jit(self.bert.apply)(bert_params, **tokens)
        sentence_embeddings = token_embeddings.mean(1)
        word_embeddings = batch_token_embeddings_to_batch_word_embeddings(
            tokens=tokens,
            token_embeddings=token_embeddings,
            agg_func=self.agg_func,
            max_word_length=self.max_word_length,
        )
        state_embeddings = jit(self.state_encoder, static_argnums=(2,))(
            state_encoder_params, word_embeddings, self.node_embedding_dim
        )

        log_Z = jit(self.Z)(Z_params, sentence_embeddings)
        traj_log_pF, traj_log_pB, complete_states = self.sample(
            gflownet_params, state_embeddings, num_words_list
        )
        log_R = jnp.log(scores.unlabeled_graph_edit_distance(complete_states, golds))

        return trajectory_balance_loss(log_Z, traj_log_pF, log_R, traj_log_pB)

    def sample(self, gflownet_params, state_embeddings, num_words_list):
        states = masking.StateBatch(self.batch_size, self.num_variables, num_words_list)

        traj_log_pF = jnp.zeros((self.batch_size,), dtype=jnp.float32)
        traj_log_pB = jnp.zeros((self.batch_size,), dtype=jnp.float32)

        for t in range(self.max_number_of_words):
            self.key, subkey1, subkey2 = jax.random.split(self.key, 3)

            # Exploitation: Sample action based on GFlowNet policy
            log_pi = jit(self.gflownet, static_argnums=(4, 5, 6, 7))(
                gflownet_params,
                state_embeddings,
                states["labels"],
                states["masks"],
                self.num_tags,
                self.num_layers,
                self.num_heads,
                self.key_size,
            )

            # Exploration: Sample action uniformly at random
            log_uniform = masking.uniform_log_policy(states["masks"])
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

            log_probs = jnp.take_along_axis(log_pi, actions, axis=1)
            traj_log_pF += log_probs
            traj_log_pB += masking.uniform_log_policy(
                states["masks"]
            )  # Uniform backward policy

            # Move to the next state
            states.step(actions)

        return traj_log_pF, traj_log_pB, states

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        losses, rewards = [], []

        with trange(self.max_steps, desc="Training") as pbar:
            for iteration in pbar:
                print(self.bert_params) 
                print()
                print(self.state_encoder_params)
                print()
                print(self.gflownet_params)
                print()
                print(self.Z_params)
                print()
                
                batch = next(iter(train_loader))

                tokens = self.tokenizer(
                    batch["text"],
                    return_tensors="jax",
                    padding="max_length",
                    truncation=True,
                    add_special_tokens=False,
                )

                grads, logs = grad(self.loss, argnums=(0, 1, 2, 3, ),has_aux=True)(
                    self.bert_params,
                    self.state_encoder_params,
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
