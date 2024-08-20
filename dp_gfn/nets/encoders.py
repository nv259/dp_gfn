## the base code is from https://github.com/tristandeleu/jax-dag-gflownet

import haiku as hk
import jax.nn as nn
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
        self, num_heads, key_size, model_size, init_scale, num_tags, name=None
    ):
        super().__init__(name=name)
        self.num_heads = num_heads
        self.key_size = key_size
        self.model_size = model_size
        self.init_scale = init_scale
        self.num_tags = num_tags

    def __call__(self, x, labels):
        # w_init = hk.initializers.VarianceScaling(self.init_scale)

        arc_keys = hk.Embed(self.num_tags, self.model_size)(labels)
        arc_values = hk.Embed(self.num_tags, self.model_size)(labels)

        # Attention layer
        attn = LinearMultiHeadAttention(
            num_heads=self.num_heads,
            key_size=self.key_size,
            w_init_scale=self.init_scale,
        )(x, x, x, arc_keys, arc_values)

        # Add & Norm
        hiddens = x + attn
        hiddens = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(hiddens)

        # FFN layer
        mlp_output = DenseBlock(
            output_size=self.model_size,
            init_scale=self.init_scale,
        )(hiddens)

        # Add & Norm
        output = mlp_output + hiddens
        output = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(output)

        return hiddens
