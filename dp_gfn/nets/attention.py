## the base code is from https://github.com/tristandeleu/jax-dag-gflownet
import haiku as hk
import jax.nn as nn
import jax.numpy as jnp


class LinearMultiHeadAttention(hk.MultiHeadAttention):
    def __call__(self, query, key, value, arc_keys, arc_values, mask=None):
        feature_map = lambda x: nn.elu(x) + 1.0
        eps = 1e-6

        query_heads = self._linear_projection(query, self.key_size, "query")
        key_heads = self._linear_projection(key, self.key_size, "key")
        value_heads = self._linear_projection(value, self.value_size, "value")
        arc_keys = arc_keys.reshape(key_heads.shape)
        arc_values = arc_values.reshape(value_heads.shape)
        key_heads = key_heads + arc_keys
        value_heads = value_heads + arc_values

        # Map the query & key with a feature map
        query_heads = feature_map(query_heads)
        key_heads = feature_map(key_heads)

        key_values = jnp.einsum("...thd,...thk->...hkd", key_heads, value_heads)
        normalizer = 1.0 / (
            jnp.einsum("...thd,...hd->...th", query_heads, jnp.sum(key_heads, axis=-3))
            + eps
        )
        attn = jnp.einsum(
            "...thd,...hkd,...th->...thk", query_heads, key_values, normalizer
        )

        # Concatenate attention matrix of all heads into a single vector.
        attn_vec = jnp.reshape(attn, (*query.shape[:-1], -1))
        return hk.Linear(self.model_size, w_init=self.w_init)(attn_vec)
