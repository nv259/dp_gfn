from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import gc

class LinearMultiHeadAttention(nn.Module):
    """
    Linear Multi-Head Attention: a variant of multi-head attention based on the Linformer approach proposed by Wang et al. 
    This version of multi-head attention aims to reduce the computational complexity of the traditional self-attention mechanism 
    by projecting the query and key matrices before computing the attention scores.
    
    Linformer: Self-Attention with Linear Complexity. See: https://arxiv.org/pdf/2006.04768
    Attention is all you need paper. See: https://arxiv.org/abs/1706.03762
    
    Attributes:
        input_dim (int): The dimensionality of the input embeddings.
        num_heads (int): The number of attention heads.
        d_k (int): The dimensionality of the key vectors.
        d_v (Optional[int]): The dimensionality of the value vectors. Defaults to d_k if not specified.
        d_model (Optional[int]): The dimensionality of the output embeddings. Defaults to d_k * num_heads if not specified.
        dropout (float): Dropout rate applied to the attention scores. Default is 0.1.
        eps (float): A small value added to the denominator for numerical stability. Default is 1e-6.
    
    Methods:
        __init__: Initializes the linear multi-head attention mechanism with the specified parameters.

        forward: Computes the multi-head attention for the given queries, keys, and values.
            Parameters:
                query (torch.Tensor): Query tensor of shape [batch_size, target_len, input_dim].
                key (torch.Tensor): Key tensor of shape [batch_size, source_len, input_dim].
                value (torch.Tensor): Value tensor of shape [batch_size, source_len, input_dim].
                mask (Optional[torch.Tensor]): Optional mask tensor to prevent attention to certain positions (not implemented).
            Returns:
                torch.Tensor: The result of the multi-head attention mechanism, of shape [batch_size, target_len, d_model].
    """
    
    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        d_k: int,
        d_v: Optional[int] = None,
        d_model: Optional[int] = None,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.d_k = d_k
        self.d_v = d_v or d_k
        self.d_model = d_model or d_k * num_heads
        self.eps = eps

        self.query_linear = nn.Linear(input_dim, self.d_k * self.num_heads)
        self.key_linear = nn.Linear(input_dim, self.d_k * self.num_heads)
        self.value_linear = nn.Linear(input_dim, self.d_v * self.num_heads)
        self.final_linear = nn.Linear(self.d_k * self.num_heads, self.d_model)

        self.projection = lambda x: F.elu(x) + 1.0

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            raise NotImplementedError

        # Map query, key & value then project query & key
        query = self.projection(self.query_linear(query))   # bs x target x (d_k * num_heads)
        key = self.projection(self.key_linear(key))         # bs x source x (d_k * num_heads)
        value = self.value_linear(value)                    # bs x target x (d_v * num_heads)

        # Reshape query, key & value to [bs, (source|target), num_heads, (d_k|d_v)]
        query = query.reshape(query.size(0), -1, self.num_heads, self.d_k)
        key = key.reshape(key.size(0), -1, self.num_heads, self.d_k)
        value = value.reshape(value.size(0), -1, self.num_heads, self.d_v)

        key_values = torch.einsum("...thd,...thk->...hkd", key, value)
        normalizer = 1.0 / (
            torch.einsum("...thd,...hd->...th", query, torch.sum(key, axis=-3))
            + self.eps
        )
        attn = torch.einsum(
            "...thd,...hkd,...th->...thk", query, key_values, normalizer
        )
        
        # Free query, key, value, key_values from memory
        del query, key, value, key_values
        torch.cuda.empty_cache()
        gc.collect() 
        
        attn = attn.reshape(attn.size(0), attn.size(1), self.num_heads * self.d_v)
        attn = self.final_linear(attn)

        return attn
