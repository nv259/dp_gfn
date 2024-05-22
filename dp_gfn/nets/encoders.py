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
        hidden_layers=[],
        dropout_rate=0.1,
        activation="ReLU",
    ):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        activation = getattr(nn, activation)

        layers = [nn.Linear(input_dim, hidden_layers[0]), activation()]
        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            layers.append(activation())
            layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(hidden_layers[-1], output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class LinearTransformer(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        num_heads: int,
        d_k: int,
        d_v: Optional[int] = None,
        d_model: Optional[int] = None,
        activation: str = "gelu",
        attn_dropout: float = 0.1,
        mlp_dropout: float = 0.1,
        dropout: float = 0.1,
        eps: float = 1e-6,
        label_embedded: bool = False,
        num_tags: Optional[int] = 0,
        label_dim: Optional[int] = 0,
    ):
        assert (num_tags ^ label_embedded), "Either label_embedded or label_dim must be True"
        assert (input_size + label_dim == d_model), "input_size + label_dim must equal d_model"

        super(LinearTransformer, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.num_heads = num_heads
        self.d_k = d_k
        self.d_v = d_v or d_k
        self.d_model = d_model or d_k * num_heads
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.mlp_dropout = mlp_dropout
        self.label_embedded = label_embedded

        if not label_embedded:
            self.label_embeddings = [
                nn.Embedding(num_tags, label_dim) for _ in range(2)
            ]
        self.layer_norms = [nn.LayerNorm(self.d_model) for _ in range(2)]

        self.attention_layer = LinearMultiHeadAttention(
            input_size=self.input_size,
            num_heads=self.num_heads,
            d_k=self.d_k,
            d_v=self.d_v,
            d_model=self.d_model,
            dropout=self.attn_dropout,
            eps=eps,
        )

        # TODO: Implement widening factor
        self.dense_layer = MLP(
            input_dim=self.d_model,
            output_dim=self.output_size,
            activation=activation,
            # dropout_rate=self.mlp_dropout
        )

    def forward(self, x, labels=None):
        # Embed labels in case labels is retained
        labels_attn = (
            self.label_embeddings[0](labels)
            if labels is not None
            else torch.zeros(x.shape[0], 0)
        )
        labels_dense = (
            self.label_embeddings[1](labels) 
            if labels is not None 
            else labels_attn
        )

        # Layer normalization
        hiddens = self.layer_norms[0](torch.cat([hiddens, labels_attn], dim=-1))
        # Multi-head attention
        attn = self.attention_layer(hiddens, hiddens, hiddens)
        # Residual connection
        hiddens = hiddens + attn
        # Layer normalization
        hiddens = self.layer_norms[1](torch.cat([hiddens, labels_dense], dim=-1))
        # Project to final dimension
        dense = self.dense_layer(hiddens)
        # Residual connection
        hiddens = hiddens + dense

        return hiddens
