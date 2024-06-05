import logging
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from dp_gfn.nets.encoders import (
    MLP,  # TODO: implement GNN, CNN, etc...
    LinearTransformerBlock,
    Backbone
)
from dp_gfn.nets.initial_encoders import LabelScorer, PrefEncoder, StateEncoder
from dp_gfn.utils.masking import mask_logits

from hydra.utils import instantiate
# from linear_attention_transformer import LinearAttentionTransformer


class DPGFlowNet(nn.Module):
    def __init__(
        self, 
        pref_encoder: PrefEncoder, 
        state_encoder: StateEncoder, 
        backbone: Backbone,
        label_scorer: LabelScorer, 
        output_logits: MLP,
        output_Z: MLP, 
        num_variables: int, 
        num_tags: int, 
        init_label_embeddings=False,
        verbose=False,
        *args: Any,
        **kwargs: Any,
    ):
        super(DPGFlowNet, self).__init__()
        self.num_variables = num_variables  
        self.num_tags = num_tags + 2    # including edge-from-ROOT & no-edge
        self.init_label_embeddings = init_label_embeddings
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Initial Encoders
        self.pref_encoder = pref_encoder
        if verbose:
            logging.info(f"Pref encoder initialized")
        
        self.state_encoder = state_encoder(
            word_embedding_dim=self.pref_encoder.word_embedding_dim,
        ) 
        if verbose:
            logging.info(f"State encoder initialized")
        
        # TODO: Implement other architectures for backbone
        # 2. Main encoder 
        self.backbone = backbone
        if verbose:
            logging.info(f"Backbone initialized")

        # TODO: Re-implement for input heads and deps are from plms, maybe?
        self.label_scorer = label_scorer
        if verbose:
            logging.info(f"Label scorer initialized")
        
        # TODO: Re-design the Z function
        # 3. Output layers
        self.output_logits = output_logits(output_dim=self.num_variables ** 2)
        self.output_Z_mod = output_Z(input_dim=self.pref_encoder.word_embedding_dim)
        if verbose:
            logging.info(f"Output layers initialized")
        
        self.logsoftmax2 = nn.LogSoftmax(2)

    def Z(self, pref):
        return self.output_Z_mod(pref).sum(1)

    def Z_param(self):
        return self.output_Z_mod.parameters()

    def model_params(self):
        return (
            list(self.pref_encoder.parameters())
            + list(self.state_encoder.parameters())
            + list(self.backbone.parameters())
            + list(self.label_scorer.parameters())
            + list(self.output_logits.parameters())
        )

    def forward(self, edges, labels=None, mask=None):
        edges = edges.to(self.device)
        
        edges = self.backbone(edges, labels)
        
        logits = self.logits(edges)
        logits = mask_logits(logits, mask)

        return logits
