"""
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

import haiku as hk
import jax.nn as nn
import jax.numpy as jnp
from dp_gfn.nets.attention import LinearMultiHeadAttention


class DenseBlock(hk.Module):
    def __init__(self, output_size, init_scale=None, activation="gelu", name=None):
        super().__init__(name=name)
        self.output_size = output_size
        self.init_scale = init_scale
        self.activation = activation

    def __call__(self, inputs):
        input_size = inputs.shape[-1]

        w_init = (
            hk.initializers.RandomNormal()
            if self.init_scale is None
            else hk.initializers.VarianceScaling(self.init_scale)
        )

        hiddens = hk.Linear((input_size + self.output_size) // 2, w_init=w_init)(inputs)

        activation = getattr(nn, self.activation)
        hiddens = activation(hiddens)

        return hk.Linear(self.output_size, w_init=w_init)(hiddens)


class LinearTransformerBlock(hk.Module):
    def __init__(
        self, num_heads, key_size, embedding_size, init_scale, num_tags, name=None
    ):
        super().__init__(name=name)
        self.num_heads = num_heads
        self.key_size = key_size
        self.embedding_size = embedding_size
        self.init_scale = init_scale
        self.num_tags = num_tags

    def __call__(self, edges_embedding, labels):
        # w_init = hk.initializers.VarianceScaling(self.init_scale)

        # Attention layer
        preattn_labels_embedding = hk.Embed(
            self.num_tags, self.embedding_size, name="preattn_embedding"
        )(labels)
        hiddens = hk.LayerNorm(
            axis=-1, create_scale=True, create_offset=True, name="preattn_layernorm"
        )(jnp.concatenate([preattn_labels_embedding, edges_embedding], axis=-1))
        attn = LinearMultiHeadAttention(
            num_heads=self.num_heads,
            key_size=self.key_size,
            w_init_scale=self.init_scale,
        )(hiddens, hiddens, hiddens)

        edges_embedding = edges_embedding + attn

        # FFN layer
        preffn_labels_embedding = hk.Embed(
            self.num_tags, self.embedding_size, name="preffn_embedding"
        )(labels)
        hiddens = hk.LayerNorm(
            axis=-1, create_scale=True, create_offset=True, name="preffn_layernorm"
        )(jnp.concatenate([preffn_labels_embedding, edges_embedding], axis=-1))
        hiddens = DenseBlock(
            output_size=self.num_heads * self.key_size,
            init_scale=self.init_scale,
        )(hiddens)

        hiddens = hiddens + edges_embedding

        return hiddens
