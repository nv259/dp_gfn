import haiku as hk
import jax
import jax.numpy as jnp
from dp_gfn.nets.encoders import DenseBlock, LinearTransformerBlock
from dp_gfn.nets.initial_encoders import Biaffine
from dp_gfn.utils import masking


class DPGFlowNet(hk.Module):
    def __init__(
        self,
        num_variables=160,
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

    def __call__(self, x, node_id, labels, masks):
        for _ in range(self.num_layers):
            x = LinearTransformerBlock(
                num_heads=self.num_heads,
                key_size=self.key_size,
                model_size=self.model_size,
                init_scale=self.init_scale,
                num_tags=self.num_tags,
            )(x, labels)

        log_pi = self.edge_policy(x, node_id, masks[0])
        
        # TODO: Should we use only visitted nodes to predict next node to visit?
        log_node_pi = self.next_node_policy(x.mean(axis=-2), x, masks[1])  

        return log_pi, log_node_pi

    def Z(self, pref): # TODO: Redesign Z to use input the encoded state
        return DenseBlock(
            output_size=self.model_size, init_scale=self.init_scale, name="Z_flow"
        )(pref).mean(-1)

    def edge_policy(self, x, node_id, masks):
        deps = jnp.repeat(x[jnp.newaxis, node_id], x.shape[-2], axis=-2)
        
        # TODO: What if we use edge policy at step 0?
        logits = jax.vmap(
            Biaffine(num_tags=1, init_scale=self.init_scale), in_axes=(0, 0)
        )(x, deps)
        logits = logits.squeeze(-1)
        log_pi = log_policy(logits, masks)

        return log_pi

    def next_node_policy(self, graph_emb, node_embeddings, masks):
        graph_embs = jnp.repeat(
            graph_emb[jnp.newaxis, :], node_embeddings.shape[0], axis=0
        )
        hidden = jnp.concatenate([graph_embs, node_embeddings], axis=-1)

        # logits = DenseBlock(1, init_scale=self.init_scale)(hidden)
        logits = jax.vmap(
            Biaffine(num_tags=1, init_scale=self.init_scale), in_axes=(0, 0)
        )(graph_embs, hidden)
        
        logits = logits.squeeze(-1)
        log_pi = log_policy(logits, masks)

        return log_pi


def log_policy(logits, masks):
    masks = masks.reshape(logits.shape)
    logits = masking.mask_logits(logits, masks)
    log_pi = jax.nn.log_softmax(logits, axis=-1)

    return log_pi


def gflownet_forward_fn(
    x, node_id, labels, masks, num_tags, num_layers, num_heads, key_size, model_size=None
):
    # model_size = x.shape[-1]
    num_variables = int(x.shape[0])

    log_pi, next_node_logits = DPGFlowNet(
        num_variables=num_variables,
        num_tags=num_tags,
        num_layers=num_layers,
        num_heads=num_heads,
        key_size=key_size,
        model_size=model_size,
    )(x, node_id, labels, masks)

    return log_pi, next_node_logits


def output_total_flow_fn(pref):
    return DenseBlock(output_size=1, name="Z_flow", init_scale=2. / 12.)(pref)
