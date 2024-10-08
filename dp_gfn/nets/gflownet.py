import haiku as hk
import jax
import jax.numpy as jnp

from dp_gfn.nets.encoders import DenseBlock, LinearTransformerBlock
from dp_gfn.nets.initial_encoders import Biaffine
from dp_gfn.utils import masking


class DPGFlowNet(hk.Module):
    def __init__(
        self,
        num_variables=100,
        num_tags=41,
        num_layers=5,
        num_heads=4,
        key_size=64,
        model_size=None,
    ):
        super().__init__()

        self.num_variables = num_variables
        self.num_tags = num_tags + 1  # including no-edge type

        self.num_layers = num_layers
        self.key_size = key_size
        self.num_heads = num_heads
        self.model_size = (
            model_size if model_size is not None else self.num_heads * self.key_size
        )

        self.init_scale = 2.0 / self.num_layers

    def __call__(self, key, x, labels, mask, delta):
        for _ in range(self.num_layers):
            x = LinearTransformerBlock(
                num_heads=self.num_heads,
                key_size=self.key_size,
                model_size=self.model_size,
                init_scale=self.init_scale,
                num_tags=self.num_tags,
            )(x, labels)

        dep_mask = jnp.any(mask, axis=0)
        log_pi_dep = self.dep_policy(x.mean(axis=-2), x[1:], dep_mask[1:])
        key, dep_id, log_pF_dep = masking.sample_action(
            key, log_pi_dep, dep_mask[1:], delta
        )
        dep_id = (
            dep_id + 1
        )  # Offset dep_id by 1 since ROOT cannot be chosen as a dependant

        head_mask = mask[:, dep_id]
        log_pi_head = self.head_policy(x, dep_id, head_mask)
        key, head_id, log_pF_head = masking.sample_action(
            key, log_pi_head, head_mask, delta
        )

        log_pBs = self.backward_policy(x[1:], labels[1:].astype(jnp.bool_))

        return key, (dep_id, head_id), (log_pF_dep, log_pF_head), log_pBs

    def dep_policy(self, graph_emb, node_embeddings, mask):
        graph_embs = jnp.repeat(
            graph_emb[jnp.newaxis, :], node_embeddings.shape[0], axis=0
        )

        logits = jax.vmap(
            Biaffine(num_tags=1, init_scale=self.init_scale), in_axes=(0, 0)
        )(graph_embs, node_embeddings)
        logits = logits.squeeze(-1)
        log_pi = log_policy(logits, mask)

        return log_pi

    def head_policy(self, node_embeddings, node_id, mask):
        deps = jnp.repeat(
            node_embeddings[jnp.newaxis, node_id], node_embeddings.shape[-2], axis=-2
        )

        # IDEA: What if we use edge policy at step 0?
        logits = jax.vmap(
            Biaffine(num_tags=1, init_scale=self.init_scale), in_axes=(0, 0)
        )(node_embeddings, deps)
        logits = logits.squeeze(-1)
        log_pi = log_policy(logits, mask)

        return log_pi

    def backward_policy(self, node_embeddings, mask):
        logits = DenseBlock(1, init_scale=self.init_scale)(node_embeddings)
        logits = logits.squeeze(-1)
        log_pB = log_policy(logits, mask)

        return log_pB


def log_policy(logits, mask):
    mask = mask.reshape(logits.shape)
    logits = masking.mask_logits(logits, mask)
    log_pi = jax.nn.log_softmax(logits, axis=-1)

    return log_pi


def gflownet_forward_fn(
    key, x, labels, mask, num_tags, num_layers, num_heads, key_size, delta=0.0
):
    num_variables = int(x.shape[0])

    key, actions, (log_pF_dep, log_pF_head), log_pBs = DPGFlowNet(
        num_variables=num_variables,
        num_tags=num_tags,
        num_layers=num_layers,
        num_heads=num_heads,
        key_size=key_size,
    )(key, x, labels, mask, delta)

    return key, actions, (log_pF_dep, log_pF_head), log_pBs


def output_total_flow_fn(pref):
    return DenseBlock(output_size=1, name="Z_flow", init_scale=2.0 / 12.0)(pref)
