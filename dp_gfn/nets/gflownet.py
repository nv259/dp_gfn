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
        