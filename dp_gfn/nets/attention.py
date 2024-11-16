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
        self.output = nn.Linear(embed_dim, embed_dim)
        
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
        V = V.reshape(query.shape)
        
        return self.output(V.contiguous())


class RelationAwareAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, num_relations=3, dropout=0.0):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dim_per_head = embed_dim // num_heads
        self.num_relations = num_relations
        self.scale = self.dim_per_head ** -0.5
        
        self.query_head = nn.Linear(embed_dim, embed_dim)
        self.key_head = nn.Linear(embed_dim, embed_dim)
        self.value_head = nn.Linear(embed_dim, embed_dim)
        self.output = nn.Linear(embed_dim, embed_dim)
        
        self.relation_embedding_k = nn.Embedding(num_relations, self.dim_per_head, padding_idx=0)
        self.relation_embedding_v = nn.Embedding(num_relations, self.dim_per_head, padding_idx=0)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, graph_relations):
        """Generate new embeddings using attention mechanism for query, key, value and graph relations.
        
        $$ V'_i = \sum_{j} e_{ij} (V_j + E_{ij}^v) $$
        , where $e_{ij} = \textit{softmax}(Q_i.(K_j^T + E_{ij}^k) / \sqrt{d_k})$

        Args:
            query (torch.Tensor): Embeddings of query; shape [batch_size, seq_len, embed_dim]
            key (torch.Tensor): Embeddings of key; shape [batch_size, seq_len, embed_dim]
            value (torch.Tensor): Embeddings of value; shape [batch_size, seq_len, embed_dim])
            graph_relations (torch.Tensor): Adjacency matrix of graph; shape [batch_size, seq_len, seq_len]

        Returns:
            torch.Tensor: New embeddings; shape [batch_size, seq_len, embed_dim]
        """
        queries = self.query_head(query)
        keys = self.key_head(key)
        values = self.value_head(value)
        
        queries = split_into_heads(queries, self.num_heads)
        keys = split_into_heads(keys, self.num_heads)
        values = split_into_heads(values, self.num_heads)

        # b: batch size
        # s: query_seq_len (source)
        # t: key_seq_len (target)
        # h: num_heads
        # d: dim_per_head 
        QK = torch.einsum("bshd,bthd->bhst", queries, keys) # Q.K^T
        E_k = self.relation_embedding_k(graph_relations)
        QE = torch.einsum("bshd,bstd->bhst", queries, E_k)  # Q.(Ek)^T TODO: Check for potential risk (1)
        Q_KE = (QK + QE) * self.scale
        
        e = nn.Softmax(dim=-1)(Q_KE)
        e = self.dropout(e)
        
        eV = torch.einsum("bhst,bthd->bshd", e, values) # e.V  
        Ev = self.relation_embedding_v(graph_relations)
        eE = torch.einsum("bhst,bstd->bshd", e, Ev) # e.(Ev) TODO: Check for potential risk (2)
        out = eV + eE
        
        out = out.reshape(
            graph_relations.shape[0],
            graph_relations.shape[1],
            self.embed_dim
        )
         
        return self.output(out)