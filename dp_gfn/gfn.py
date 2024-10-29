import pickle
import logging
import os
from collections import namedtuple

import numpy as np
from tqdm import trange

try:
    from evaluation import save_predictions
except:
    pass

import subprocess

import haiku as hk
import jax
import jax.numpy as jnp
import optax
from jax import grad, jit, vmap
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer

from dp_gfn.nets import bert
from dp_gfn.nets.gflownet import gflownet_fn, output_total_flow_fn
from dp_gfn.nets.initial_encoders import label_score_fn
from dp_gfn.utils import io, masking, scores
from dp_gfn.utils.pretrains import (
    batch_token_embeddings_to_batch_word_embeddings,
    create_position_ids_from_input_ids,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
GFlowNetState = namedtuple("GFlowNetState", ["optimizer", "step"])
GFlowNetParams = namedtuple("GFlowNetParams", ["bert", "gfn", "logZ"])


class DPGFN:
    def __init__(self, config, num_tags, id2rel, pretrained_path=None, debug=False):
        super().__init__()
        self.config = config
        self.num_tags = num_tags
        self.id2rel = id2rel
        self.debug = debug

        self.initialize_vars()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.pref_encoder.pretrained_path
        )

        dummy_input_ids = jnp.ones(
            (self.batch_size, self.tokenizer.model_max_length), dtype=jnp.int32
        )

        bert.init(self.config.model.pref_encoder.pretrained_path)
        self.bert_model = hk.transform(bert.get_bert_token_embeddings_fn)
        self.bert_params = self.bert_model.init(
            self.key,
            dummy_input_ids,
            jnp.ones_like(dummy_input_ids),
            jnp.zeros_like(dummy_input_ids),
            jnp.ones_like(dummy_input_ids),
            True,
        )

        self.gflownet = hk.without_apply_rng(hk.transform(gflownet_fn))
        base_masks = masking.base_mask(self.num_variables, self.num_variables)
        self.gfn_params = self.gflownet.init(
            self.key,
            self.key,
            jax.random.normal(
                self.key, (self.num_variables, self.bert_config["hidden_size"])
            ),
            jnp.zeros((self.num_variables,), dtype=jnp.int32),
            base_masks,
            self.num_tags,
            self.num_layers,
            self.num_heads,
            self.key_size,
            jnp.array(0.0),
        )
        self.gflownet = vmap(
            self.gflownet.apply, in_axes=(None, 0, 0, 0, 0, None, None, None, None, 0)
        )

        self.logZ = hk.without_apply_rng(hk.transform(output_total_flow_fn))
        self.logZ_params = self.logZ.init(
            self.key, jnp.ones((self.bert_config["hidden_size"],))
        )
        self.logZ = vmap(self.logZ.apply, in_axes=(None, 0))

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
        self.dump_foldername = self.config.dump_foldername

        self.num_layers = self.config.model.backbone.num_layers
        self.num_heads = self.config.model.backbone.encoder_block.num_heads
        self.key_size = self.config.model.backbone.encoder_block.d_k
        self.node_embedding_dim = self.config.model.common.node_embedding_dim
        self.model_size = self.num_heads * self.key_size
        self.init_scale = 2.0 / self.config.model.backbone.num_layers
        self.agg_func = self.config.model.pref_encoder.agg_func

        config = self.config.algorithm
        self.reward_scale_factor = config.reward_scale_factor

        config = config.train
        self.n_grad_accumulation_steps = config.n_grad_accumulation_steps
        self.max_steps = config.max_steps
        self.eval_on_train = config.eval_on_train
        self.exploration_scheduler = config.exploration_scheduler
        self.clip_grad = config.clip_grad
        self.eval_every_n = config.eval_every_n
        self.save_every_n = config.save_every_n

    def init_policy(self):
        gflownet_lr = self.config.algorithm.train.optimizer.gflownet_lr
        Z_lr = self.config.algorithm.train.optimizer.Z_lr
        bert_factor = self.config.algorithm.train.optimizer.bert_factor

        self.optimizer = optax.multi_transform(
            {
                "bert": optax.adam(gflownet_lr * bert_factor),
                "gfn": optax.adam(gflownet_lr),
                "logZ": optax.adam(Z_lr),
            },
            ("bert", "gfn", "logZ"),
        )
        self.states = self.optimizer.init(
            (self.bert_params, self.gfn_params, self.logZ_params)
        )

    def init_states(self, bert_params, tokens, position_ids, training=False):
        self.key, bert_key = jax.random.split(self.key)

        token_embeddings = jit(self.bert_model.apply, static_argnums=(6,))(
            bert_params,
            bert_key,
            tokens["input_ids"],
            position_ids,
            jnp.zeros_like(tokens["input_ids"]),
            tokens["attention_mask"],
            training=training,
        )

        node_embeddings = batch_token_embeddings_to_batch_word_embeddings(
            tokens=tokens,
            token_embeddings=token_embeddings,
            agg_func=self.agg_func,
            max_word_length=self.num_variables,
        )  # TODO: Find another way that allows parallelization -> JIT 

        # Embeddings for computing intitial flow
        sentence_embeddings = token_embeddings.mean(1)

        return node_embeddings, sentence_embeddings

    def loss(
        self,
        bert_params,
        gfn_params,
        Z_params,
        tokens,
        position_ids,
        num_words_list,
        golds,
        delta,
    ):
        node_embeddings, sentence_embeddings = self.init_states(
            bert_params, tokens, position_ids, training=False
        )
        log_Z = jit(self.logZ)(Z_params, sentence_embeddings).squeeze(axis=-1)

        traj_log_pF, traj_log_pB, complete_states = self.sample(
            gfn_params, node_embeddings, num_words_list, delta
        )

        golds = (golds != 0).astype(bool)
        log_R = jnp.log(scores.reward(complete_states["adjacency"], golds, scores.frobenius_norm_distance)).clip(-100)
        # log_R = jnp.log(
        #     scores.reward(
        #         complete_states["adjacency"], golds, scores.frobenius_norm_distance
        #     ) ** self.reward_scale_factor
        # ).clip(-100)

        return trajectory_balance_loss(log_Z, traj_log_pF, log_R, traj_log_pB)

    def sample(self, gfn_params, node_embeddings, num_words_list, delta=0.001):
        key = jax.random.split(self.key, len(num_words_list))
        delta = jnp.array([delta] * len(num_words_list))
        states = masking.StateBatch(self.num_variables, num_words_list)

        traj_log_pF = jnp.zeros((len(num_words_list),), dtype=jnp.float32)
        traj_log_pB = jnp.zeros((len(num_words_list),), dtype=jnp.float32)

        for step in range(self.num_variables):
            dones = states.check_done()

            key, actions, (log_pF_dep, log_pF_head), log_pBs = jit(
                self.gflownet, static_argnums=(5, 6, 7, 8)
            )(
                gfn_params,
                key,
                node_embeddings,
                states["labels"],
                states["mask"],
                self.num_tags,
                self.num_layers,
                self.num_heads,
                self.key_size,
                delta,
            )

            log_pF_dep = jnp.where(dones, jnp.zeros_like(log_pF_dep), log_pF_dep)
            log_pF_head = jnp.where(dones, jnp.zeros_like(log_pF_head), log_pF_head)
            traj_log_pF += log_pF_dep + log_pF_head

            if step > 0:
                log_pB = jnp.take_along_axis(
                    log_pBs, (prev_actions - 1)[..., jnp.newaxis], axis=1
                ).squeeze(-1)
                log_pB = jnp.where(prev_dones, jnp.zeros_like(log_pB), log_pB)

                traj_log_pB += log_pB
            
            states.step(np.array(actions))
            prev_dones = dones.copy()
            prev_actions = actions[0].copy()

            if prev_dones.all():
                break
            
        self.key = key[0]

        return traj_log_pF, traj_log_pB, states

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        save_folder = os.path.join(
            "/".join(logging.getLogger().handlers[1].baseFilename.split("/")[:-1]),
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
        train_loss, val_loss = 0, 0
        rewards = []

        exploration_schedule = jax.jit(
            optax.linear_schedule(
                init_value=self.exploration_scheduler['init_value'],
                end_value=self.exploration_scheduler['end_value'],
                transition_steps=self.exploration_scheduler['transition_steps'],
                transition_begin=self.exploration_scheduler['transition_begin'],
            )
        )

        try:
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

                    position_ids = create_position_ids_from_input_ids(
                        tokens["input_ids"]
                    )

                    grads, logs = grad(self.loss, argnums=(0, 1, 2), has_aux=True)(
                        self.bert_params,
                        self.gfn_params,
                        self.logZ_params,
                        tokens,
                        position_ids,
                        batch["num_words"],
                        batch["graph"],
                        delta,
                    )

                    updates, self.states = self.optimizer.update(
                        grads,
                        self.states,
                        (self.bert_params, self.gfn_params, self.logZ_params),
                    )
                    self.bert_params, self.gfn_params, self.logZ_params = (
                        optax.apply_updates(
                            (self.bert_params, self.gfn_params, self.logZ_params),
                            updates,
                        )
                    )
                    rewards.append(np.exp(logs["log_R"]))

                    if iteration % self.eval_every_n == 0:
                        gold = os.path.join(save_folder, "predicts", f"gold.conllu")
                        system = os.path.join(
                            save_folder, "predicts", f"system_{iteration}.conllu"
                        )
                        save_predictions(
                            algorithm=self,
                            loader=val_loader,
                            config=self.config,
                            id2rel=self.id2rel,
                            original=self.config.train_path.replace("train", "dev"),
                            gold=gold,
                            system=system,
                        )
                        subprocess.run(["./ud_eval.py", gold, system])

                        val_loss = self.val_step(val_loader)
                        train_losses = []
                        val_losses.append(val_loss)

                        if self.eval_on_train:
                            train_loss = self.val_step(train_loader)
                            train_losses.append(train_loss)

                        logging.info(
                            f"Iteration {iteration}: loss = {logs['loss']:.5f} "
                            f"--- log_Z = {logs['log_Z'].mean():.5f} "
                            f"--- train_loss = {train_loss:.5f} "
                            f"--- val_loss = {val_loss:.5f} --- epsilon = {delta:.5f}"
                        )
                        # str_rewards = str(rewards).replace('\n', '\t')
                        # logging.info("Rewards: " + str_rewards)
                        logging.info(f"Mean reward: {np.concat(rewards).mean():.6f}")
                        rewards = []

                        if self.debug is True:
                            np.save(
                                os.path.join(
                                    save_folder, "debug", f"logR_{iteration}.npy"
                                ),
                                logs["log_R"],
                            )

                        print("-" * 50)

                    if iteration % self.save_every_n == 0:
                        io.save(
                            os.path.join(
                                save_folder, "model", f"model_{iteration}.npz"
                            ),
                            bert=self.bert_params,
                            gfn=self.gfn_params,
                            logZ=self.logZ_params,
                        )

                    if self.eval_on_train:
                        pbar.set_postfix(
                            epsilon=f"{delta:.4f}",
                            loss=f"{logs['loss']:.5f}",
                            reward=f"{np.exp(logs['log_R']).mean():.6f}",
                            train_loss=f"{train_loss:.5f}",
                            val_loss=f"{val_loss:.5f}",
                            log_Z=f"{logs['log_Z'].mean():.5f}"
                        )
                    else:
                        pbar.set_postfix(
                            epsilon=f"{delta:.4f}",
                            loss=f"{logs['loss']:.5f}",
                            reward=f"{np.exp(logs['log_R']).mean():.6f}",
                            val_loss=f"{val_loss:.5f}",
                            log_Z=f"{logs['log_Z'].mean():.5f}"
                        )

                    train_losses.append(logs["loss"])

        except Exception as e:  # Save current training information
            io.save(
                os.path.join(save_folder, "last.npz"),
                bert=self.bert_params,
                gfn=self.gfn_params,
                logZ=self.logZ_params,
            )
            
            with open(os.path.join(save_folder, "opt.pkl") , 'wb') as f:
                pickle.dump({'states': self.states, 'step': iteration}, f)

            raise e

        return train_losses, val_losses

    def val_step(self, val_loader, delta=0.0):
        losses = []

        for batch in val_loader:
            tokens = self.tokenizer(
                batch["text"],
                return_tensors="jax",
                padding="max_length",
                truncation=True,
                add_special_tokens=False,
            )

            position_ids = create_position_ids_from_input_ids(tokens["input_ids"])

            (loss, _) = self.loss(
                self.bert_params,
                self.gfn_params,
                self.logZ_params,
                tokens,
                position_ids,
                batch["num_words"],
                batch["graph"],
                delta=delta,
            )
            losses.append(loss)

        return np.mean(losses)

    def inference(self, tokens, position_ids, num_words_list, delta=0.0):
        node_embeddings, sentence_embeddings = self.init_states(
            self.bert_params, tokens, position_ids, training=False
        )
        traj_log_pF, traj_log_pB, complete_states = self.sample(
            self.gfn_params, node_embeddings, num_words_list, delta=delta
        )
        log_Z = jit(self.logZ)(self.logZ_params, sentence_embeddings).squeeze(axis=-1)
        log = (log_Z, traj_log_pF, traj_log_pB)

        return complete_states, log

    def load_weights(self, filename):
        params = io.load(filename)

        self.bert_params = params["bert"]
        self.gfn_params = params["gfn"]
        self.logZ_params = params["logZ"]

        self.states = params["states"] if "states" in params else self.states


def trajectory_balance_loss(log_Z, traj_log_pF, log_R, traj_log_pB, delta=1):
    assert log_Z.shape == traj_log_pF.shape == traj_log_pB.shape == log_R.shape

    error = log_Z + traj_log_pF - log_R - traj_log_pB
    loss = jnp.power(error, 2).mean()

    logs = {"error": error, "loss": loss, "log_R": log_R, "log_Z": log_Z}

    return (loss, logs)


def cycle(dataloader):
    while True:
        for batch in dataloader:
            yield batch
