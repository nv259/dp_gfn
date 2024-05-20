from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class LinearMultiHeadAttention(nn.Module):
    def __init__(
        self,
        input_size: int, 
        num_heads: int,
        d_k: int,
        d_v: Optional[int] = None,
        d_model: Optional[int] = None,
        dropout: float = 0.1,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.input_size = input_size
        self.num_heads = num_heads
        self.d_k = d_k
        self.d_v = d_v or d_k
        self.d_model = d_model or d_k * num_heads
        self.dropout = dropout
        self.eps = eps
        
        self.query_linear = nn.Linear(input_size, self.d_k * self.num_heads)
        self.key_linear = nn.Linear(input_size, self.d_k * self.num_heads)
        self.value_linear = nn.Linear(input_size, self.d_v * self.num_heads)
        self.final_linear = nn.Linear(self.d_k * self.num_heads, self.d_model)
        
        self.projection = lambda x: F.elu(x) + 1. 
    
    def forward(self, query, key, value, mask=None):
        if mask: 
            raise NotImplementedError
        
        # Map query, key & value then project query & key
        query = self.projection(self.query_linear(query))   # bs x target x (d_k * num_heads)
        key = self.projection(self.key_linear(key))         # bs x source x (d_k * num_heads)
        value = self.value_linear(value)                    # bs x target x (d_v * num_heads)
        
        # Reshape query, key & value to [bs, (source|target), num_heads, (d_k|d_v)]
        query = query.view(query.size(0), -1, self.num_heads, self.d_k)
        key = key.view(key.size(0), -1, self.num_heads, self.d_k)
        value = value.view(value.size(0), -1, self.num_heads, self.d_v)
        
        key_values = torch.einsum('...thd,...thk->...hkd', key, value)
        normalizer = 1. / (torch.einsum('...thd,...hd->...th',
            query, torch.sum(key, axis=-3)) + self.eps)
        attn = torch.einsum('...thd,...hkd,...th->...thk',
            query, key_values, normalizer)

        attn = attn.view(attn.size(0), attn.size(1), self.num_heads * self.d_v)
        attn = self.final_linear(attn)
        
        return attn