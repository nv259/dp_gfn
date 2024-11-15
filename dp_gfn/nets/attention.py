import torch
import torch.nn as nn
from dp_gfn.utils.pretrains import split_into_heads


class LinearAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, eps=1e-6):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.feature_map = nn.elu() + 1.0
        self.eps = eps

        self.query_head = nn.Linear(embed_dim, embed_dim)
        self.key_head = nn.Linear(embed_dim, embed_dim)
        self.value_head = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        queries = self.query_head(query)
        keys = self.key_head(key)
        values = self.value_head(value)

        queries = split_into_heads(queries, self.num_heads)
        keys = split_into_heads(keys, self.num_heads)
        values = split_into_heads(values, self.num_heads)

        Q = self.feature_map(queries)
        K = self.feature_map(keys)

        KV = torch.einsum("nshd,nshm->nhmd", K, values)
        Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)
        V = torch.einsum("nlhd,nhmd,nlh->nlhm", Q, KV, Z)

        return V.contiguous()
