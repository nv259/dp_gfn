import logging
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from dp_gfn.nets.encoders import (
    DenseBlock,
    LinearTransformerBlock,
)
from dp_gfn.nets.initial_encoders import LabelScorer, PrefEncoder, StateEncoder
from dp_gfn.utils.masking import mask_logits

from hydra.utils import instantiate
import haiku as hk
import jax.numpy as jnp
import numpy as np


class DPGFlowNet(hk.Module):
    
    def __init__(self, num_variables=160, num_tags=41, node_embedding_dim=128, num_layers=5, num_heads=4, key_size=64):
        assert node_embedding_dim * 2 == key_size * num_heads, "transformer's model_size must be equal to state_embedding_dim"
        
        super().__init__()
        
        self.num_variables = num_variables
        self.num_tags = num_tags + 2    # including edge-from-ROOT & no-edge
        self.node_embdding_dim = node_embedding_dim
        
        self.num_layers = num_layers
        self.key_size = key_size
        self.num_heads = num_heads
        self.model_size = self.key_size * self.num_heads
        
        self.init_scale = 2. / self.num_layers
        self.pretrained_path = ""
        
    def __call__(self, edges_embedding, labels):
        for i in range(self.num_layers):
            edges_embedding = LinearTransformerBlock(num_heads=self.num_heads, key_size=self.key_size, embedding_size=self.node_embdding_dim, init_scale=self.init_scale, num_tags=self.num_tags)(edges_embedding, labels)
        
        logits = DenseBlock(1, init_scale=self.init_scale)(edges_embedding)
        
        return logits.squeeze(-1) 
    
    def Z(self, pref):
        return DenseBlock(output_size=self.model_size, init_scale=self.init_scale, name='Z_flow')(pref).mean(-1)
        

# class DPGFlowNet(nn.Module):
#     def __init__(
#         self, 
#         pref_encoder: PrefEncoder, 
#         state_encoder: StateEncoder, 
#         backbone: Backbone,
#         label_scorer: LabelScorer, 
#         output_logits: MLP,
#         output_Z: MLP, 
#         num_variables: int, 
#         num_tags: int, 
#         init_label_embeddings=False,
#         verbose=False,
#         *args: Any,
#         **kwargs: Any,
#     ):
#         super(DPGFlowNet, self).__init__()
#         self.num_variables = num_variables  
#         self.num_tags = num_tags + 2    # including edge-from-ROOT & no-edge
#         self.init_label_embeddings = init_label_embeddings
        
#         # 1. Initial Encoders
#         self.pref_encoder = pref_encoder
#         if verbose:
#             logging.info(f"Pref encoder initialized")
        
#         self.state_encoder = state_encoder(
#             word_embedding_dim=self.pref_encoder.word_embedding_dim,
#             num_tags=self.num_tags,
#             num_variables=self.num_variables
#         ) 
#         if verbose:
#             logging.info(f"State encoder initialized")
        
#         # TODO: Implement other architectures for backbone
#         # 2. Main encoder 
#         self.backbone = backbone(num_tags=self.num_tags)
#         if verbose:
#             logging.info(f"Backbone initialized")

#         # TODO: Re-implement for input heads and deps are from plms, maybe?
#         self.label_scorer = label_scorer(num_tags=self.num_tags)
#         if verbose:
#             logging.info(f"Label scorer initialized")
        
#         # TODO: Re-design the Z function
#         # 3. Output layers
#         self.output_logits = output_logits(output_dim=1)
#         self.output_Z_mod = output_Z(input_dim=self.pref_encoder.word_embedding_dim)
#         if verbose:
#             logging.info(f"Output layers initialized")
        
#         self.logsoftmax2 = nn.LogSoftmax(2)

#     def Z(self, pref):
#         return self.output_Z_mod(pref).sum(1)

#     def forward(self, edges, labels=None):
#         edges = self.backbone(edges, labels)
#         logits = self.output_logits(edges)
#         logits = logits.squeeze(-1)   
        
#         return logits

#     def create_initial_state(self, pref) -> torch.Tensor:
#         word_embeddings = self.pref_encoder(pref)
#         state_embeddings = self.state_encoder(word_embeddings)
        
#         # TODO: Incoporate num_words for more robust sentence embeddings
#         sentence_embeddings = word_embeddings.mean(1)      
 
#         return state_embeddings, sentence_embeddings
        
#     def Z_params(self):
#         return self.output_Z_mod.parameters()
    
#     def bert_params(self):
#         return self.pref_encoder.parameters()
    
#     def state_params(self):
#         return self.state_encoder.parameters()   
    
#     def flow_params(self):
#         return (
#             list(self.state_encoder.parameters())
#             + list(self.backbone.parameters())
#             + list(self.label_scorer.parameters())
#             + list(self.output_logits.parameters())
#         )
        