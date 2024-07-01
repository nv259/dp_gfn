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

from typing import Optional

import haiku as hk
import jax.nn as nn
from dp_gfn.nets.attention import LinearMultiHeadAttention
import jax.numpy as jnp


class DenseBlock(hk.Module):
    def __init__(self, output_size, init_scale, activation='gelu', name=None):
        super().__init__(name=name)
        self.output_size = output_size
        self.init_scale = init_scale
        self.activation = activation
        
    def __call__(self, inputs):
        input_size = inputs.shape[-1]
         
        w_init = hk.initializers.VarianceScaling(self.init_scale)
        hiddens = hk.Linear(
            self.widening_factor * self.output_size, 
            w_init = (input_size + self.output_size) // 2,
            w_init=w_init
        )(inputs)
        
        activation = getattr(nn, self.activation)
        hiddens = activation(hiddens)
        
        return hk.Linear(self.output_size, w_init=w_init)(hiddens)

    

# class Backbone(nn.Module):
#     def __init__(self, encoder_block, num_layers, input_dim, output_dim, num_tags):
#         super(Backbone, self).__init__()
#         self.num_layers = num_layers
#         self.input_dim = input_dim
#         self.output_dim = output_dim
        
#         self.layers = nn.ModuleList([encoder_block(num_tags=num_tags) for _ in range(num_layers)])
    
#     def forward(self, x, aux=None):
#         for layer in self.layers:
#             if aux is not None:
#                 x = layer(x, aux)
#             else: 
#                 x = layer(x)
        
#         return x  
    
#     def __getitem__(self, index):
#         return self.layers[index]
    
class TransformerBlock(hk.Module):
    def __init__(
        self,
        num_heads,
        key_size,
        embedding_size,
        init_scale, 
        widening_factor=4,
        name=None
    ):
        super().__init__(name=name)
        self.num_heads = num_heads
        self.key_size = key_size
        self.embedding_size = embedding_size
        self.init_scale = init_scale
        self.widening_factor = widening_factor

    def __call__(self, edges_embedding, labels):
        w_init = hk.initializers.VarianceScaling(self.init_scale)
        
        # Attention layer
        preattn_labels_embedding = hk.Linear(
            self.embedding_size,
            w_init=w_init,
            name='preattn_linear'
        )(labels)
        hiddens = hk.LayerNorm(
            axis=-1,
            create_scale=True, 
            create_offset=True,
            name='preattn_layernorm'
        )(jnp.concatenate([preattn_labels_embedding, edges_embedding], axis=-1))
        attn = LinearMultiHeadAttention(
            num_heads=self.num_heads,
            key_size=self.key_size,
            w_init_scale=self.init_scale
        )(hiddens, hiddens, hiddens)
        
        edges_embedding = edges_embedding + attn
        
        # FFN layer 
        preffn_labels_embedding = hk.Linear(
            self.embedding_size,
            w_init=w_init,
            name='preffn_linear'
        )(labels)
        hiddens = hk.LayerNorm(
            axis=-1,
            create_scale=True,
            create_offset=True,
            name='preffn_layernorm'
        )(jnp.concatenate([preffn_labels_embedding, edges_embedding], axis=-1))
        hiddens = DenseBlock(
            output_size=self.num_heads * self.key_size,
            init_scale=self.init_scale,
            # TODO: widening factor
        )
        
        hiddens = hiddens + edges_embedding
        
        return hiddens
    