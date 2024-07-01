from typing import Optional

import haiku as hk
import jax.nn as nn
from dp_gfn.nets.attention import LinearMultiHeadAttention


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

class Backbone(nn.Module):
    
    def __init__(self, encoder_block, num_layers, input_dim, output_dim, num_tags):
        super(Backbone, self).__init__()
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.layers = nn.ModuleList([encoder_block(num_tags=num_tags) for _ in range(num_layers)])
    
    def forward(self, x, aux=None):
        for layer in self.layers:
            if aux is not None:
                x = layer(x, aux)
            else: 
                x = layer(x)
        
        return x  
    
    def __getitem__(self, index):
        return self.layers[index]
    
    
class LinearTransformerBlock(nn.Module):
    def __init__(
        self,
        input_dim:int,
        num_tags: Optional[int],
        num_heads: int,
        d_k: int,
        d_v: Optional[int] = None,
        d_model: Optional[int] = None,
        activation: str = "GELU",
        attn_dropout: float = 0.1,
        mlp_dropout: float = 0.1,
        dropout_rate: float = 0.1,
        eps: float = 1e-6,
        label_embedded: bool = False,
        label_embedding_dim: Optional[int] = 0,
    ):
        assert input_dim == d_model + label_embedding_dim, "input_dim must be equal to d_model + label_embedding_dim"
         
        super(LinearTransformerBlock, self).__init__()

        self.input_dim = input_dim
        self.d_model = d_model or d_k * num_heads
        self.label_embedding_dim = label_embedding_dim
        # TODO: Implement dropout properly
        self.dropout_rate = dropout_rate
        self.attn_dropout = attn_dropout
        self.mlp_dropout = mlp_dropout
        self.label_embedded = label_embedded
        self.categorical_label = not label_embedded
        self.num_tags = num_tags
       
        if self.categorical_label:  
            self.label_embeddings = nn.ModuleList([
                nn.Embedding(self.num_tags, label_embedding_dim) for _ in range(2)
            ])
        else:
            self.label_embeddings = nn.ModuleList([
                nn.Linear(self.label_embedding_dim, self.label_embedding_dim) for _ in range(2)
            ])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(self.input_dim) for _ in range(2)])

        self.attention = LinearMultiHeadAttention(
            input_dim=input_dim,
            num_heads=num_heads,
            d_k=d_k,
            d_v=d_v,
            d_model=self.d_model,
            eps=eps,
        )

        # TODO: Implement widening factor & adjust layers (maybe)
        self.dense = MLP(
            input_dim=self.input_dim,
            output_dim=self.d_model,
            hidden_layers=[(self.input_dim + self.d_model) // 2],
            activation=activation,
            # dropout_rate=self.mlp_dropout
        )

    def forward(self, x, labels=None):
        # Embed labels in case labels is retained
        labels_attn = self.label_embeddings[0](labels)
        labels_dense = self.label_embeddings[1](labels)

        # Layer normalization
        hiddens = self.layer_norms[0](torch.cat([x, labels_attn], dim=-1))
        # Multi-head attention
        attn = self.attention(hiddens, hiddens, hiddens)
        # Residual connection
        x = x + attn
        # Layer normalization
        hiddens = self.layer_norms[1](torch.cat([x, labels_dense], dim=-1))
        # Project to final dimension
        hiddens = self.dense(hiddens)
        # Residual connection
        hiddens = x + hiddens

        return hiddens
