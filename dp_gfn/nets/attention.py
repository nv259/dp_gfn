"""
MIT License

Copyright (c) 2022 Tristan Deleu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import jax.numpy as jnp
import haiku as hk

import jax.nn as nn


class LinearMultiHeadAttention(hk.MultiHeadAttention):
    def __call__(self, query, key, value, mask=None):
        feature_map = lambda x: nn.elu(x) + 1.
        eps = 1e-6
        
        query_heads = self._linear_projection(query, self.key_size, 'query') 
        key_heads = self._linear_projection(key, self.key_size, 'key')
        value_heads = self._linear_projection(value, self.value_size, 'value')
        
        # Map the query & key with a feature map
        query_heads = feature_map(query_heads)
        key_heads = feature_map(key_heads)

        key_values = jnp.einsum('...thd,...thk->...hkd', key_heads, value_heads)
        normalizer = 1. / (jnp.einsum('...thd,...hd->...th',
            query_heads, jnp.sum(key_heads, axis=-3)) + eps)
        attn = jnp.einsum('...thd,...hkd,...th->...thk',
            query_heads, key_values, normalizer)
        
        # Concatenate attention matrix of all heads into a single vector.
        attn_vec = jnp.reshape(attn, (*query.shape[:-1], -1))
        return hk.Linear(self.model_size, w_init=self.w_init)(attn_vec)