from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from dp_gfn.nets.attention import LinearMultiHeadAttention


class MLP(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_layers=None,
        dropout_rate=0.1,
        activation="ReLU",
    ):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        activation = getattr(nn, activation)

        if not hidden_layers:
            hidden_layers = [(input_dim + output_dim) // 2] 
            
        layers = [nn.Linear(input_dim, hidden_layers[0]), activation()]
        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            layers.append(activation())
            layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(hidden_layers[-1], output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Backbone(nn.Module):
    
    def __init__(self, encoder_block, num_layers, input_dim, output_dim):
        super(Backbone, self).__init__()
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.layers = nn.ModuleList([encoder_block for _ in range(num_layers)])
    
    def forward(self, x, aux=None):
        for layer in self.layers:
            if aux is not None:
                x = layer(x, aux)
            else: 
                x = layer(x)
            
        return self.layers(x)    
    
    
class LinearTransformerBlock(nn.Module):
    def __init__(
        self,
        input_dim:int,
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
        num_tags: Optional[int] = 0,
        d_label: Optional[int] = 0,
    ):
        assert ((num_tags != 0) ^ label_embedded), "Either label_embedded or d_label must be True (d_label != 0)"
        assert input_dim == d_model + d_label, "input_dim must be equal to d_model + d_label"
        
        if label_embedded and (input_dim != d_model):
            print("warning: label_embedded is True but d_model is not equal to input_dim, using input_dim instead")
            d_model = input_dim
            
        if not label_embedded and (input_dim != d_model + d_label):
            print("warning: label_embedded is False but input_dim is not equal to d_model + d_label, using d_label = input_dim - d_model instead")
            d_label = input_dim - d_model 
            
            if d_label <= 0: 
                raise NotImplementedError("d_label must be greater than 0")
             
        super(LinearTransformerBlock, self).__init__()

        self.input_dim = input_dim
        self.d_model = d_model or d_k * num_heads
        self.d_label = d_label
        # TODO: Implement dropout properly
        self.dropout_rate = dropout_rate
        self.attn_dropout = attn_dropout
        self.mlp_dropout = mlp_dropout
        self.label_embedded = label_embedded
        self.num_tags = num_tags + 2
        
        if (not label_embedded) or (d_label != 0):
            self.label_embeddings = [
                nn.Embedding(self.num_tags, d_label) for _ in range(2)
            ]
        self.layer_norms = [nn.LayerNorm(self.input_dim) for _ in range(2)]

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
            hidden_layers=[self.d_model],
            activation=activation,
            # dropout_rate=self.mlp_dropout
        )

    def forward(self, x, labels=None):
        # Embed labels in case labels is retained
        labels_attn = (
            self.label_embeddings[0](labels)
            if not self.label_embedded 
            else torch.zeros(x.shape[0], x.shape[1], 0)
        )
        labels_dense = (
            self.label_embeddings[1](labels) 
            if not self.label_embedded
            else labels_attn
        )

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
