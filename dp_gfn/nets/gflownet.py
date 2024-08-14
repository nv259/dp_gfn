import logging

from dp_gfn.nets.encoders import (
    DenseBlock,
    LinearTransformerBlock,
)
from dp_gfn.nets.initial_encoders import Biaffine

import haiku as hk
import jax.numpy as jnp
import numpy as np
import math
from dp_gfn.utils import masking
import jax


class DPGFlowNet(hk.Module):
    def __init__(
        self,
        num_variables=160,
        num_tags=41,
        node_embedding_dim=128,
        num_layers=5,
        num_heads=4,
        key_size=64,
    ):
        assert (
            node_embedding_dim * 2 == key_size * num_heads
        ), "transformer's model_size must be equal to state_embedding_dim"

        super().__init__()

        self.num_variables = num_variables
        self.num_tags = num_tags + 1  # including no-edge type
        self.node_embdding_dim = node_embedding_dim

        self.num_layers = num_layers
        self.key_size = key_size
        self.num_heads = num_heads
        self.model_size = self.key_size * self.num_heads

        self.init_scale = 2.0 / self.num_layers

    def __call__(self, x, node_id, labels, masks):
        log_pi, node_embeddings = self.edge_policy(x, node_id, labels, masks, output_nodes=True)
        next_node_logits = self.next_node(node_embeddings.mean(axis=-2), node_embeddings)   # TODO: Should we use only visitted nodes to predict next node to visit?
        
        return log_pi, next_node_logits

    def Z(self, pref):
        return DenseBlock(
            output_size=self.model_size, init_scale=self.init_scale, name="Z_flow"
        )(pref).mean(-1)
    
    def edge_policy(self, x, node_id, labels, masks, output_nodes=False):
        for _ in range(self.num_layers):
            x = LinearTransformerBlock(
                num_heads=self.num_heads,
                key_size=self.key_size,
                embedding_size=self.node_embdding_dim,
                init_scale=self.init_scale,
                num_tags=self.num_tags,
            )(x, labels)

        deps = jnp.repeat(x[jnp.newaxis, node_id], x.shape[-2], axis=-2)
        assert x.shape == deps.shape
        logits = jax.vmap(Biaffine(num_tags=1), in_axes=(0, 0))(x, deps)
          
        # logits = DenseBlock(1, init_scale=self.init_scale)(x)
        logits = logits.squeeze(-1)
        log_pi = log_policy(logits, masks)  # TODO: edit mask
        
        if output_nodes is True:
            graph_emb = x
        else:
            graph_emb = x.mean(axis=-2) # TODO: Inspect other ways to create graph embeddings from nodes' embeddings
             
        return log_pi, graph_emb
    
    def next_node(self, graph_emb, node_embeddings, masks):
        graph_embs = jnp.repeat(graph_emb, node_embeddings.shape[-2], axis=-2)
        hidden = jnp.concatenate([graph_embs, node_embeddings], axis=-1)
        
        logits = DenseBlock(1, init_scale=self.init_scale)(hidden)
        logits = (logits, masks)
                
        return logits


def log_policy(logits, masks):
    masks = masks.reshape(logits.shape)
    logits = masking.mask_logits(logits, masks)
    log_pi = jax.nn.log_softmax(logits, axis=-1)
    
    return log_pi


def output_logits_fn(
    edges_embedding,
    labels,
    masks, 
    num_tags,
    num_layers,
    num_heads,
    key_size,
):
    num_variables = int(math.sqrt(edges_embedding.shape[-2]))
    node_embedding_dim = edges_embedding.shape[-1] // 2
    
    logits = DPGFlowNet(
        num_variables=num_variables,
        num_tags=num_tags, 
        node_embedding_dim=node_embedding_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        key_size=key_size
    )(edges_embedding, labels, masks)
    
    return logits


def output_total_flow_fn(pref):
    # TODO: Compare output_size = 1 vs output_size = model_size -> mean
    return DenseBlock(output_size=1, name="Z_flow")(pref)
