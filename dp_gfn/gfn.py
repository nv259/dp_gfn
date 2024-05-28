import torch
import torch.nn as nn
import torch.nn.functional as F

from dp_gfn.nets.initial_encoders import LabelScorer, PrefEncoder, StateEncoder
from dp_gfn.nets.encoders import MLP, LinearTransformer  # TODO: implement GNN, CNN, etc...
from linear_attention_transformer import LinearAttentionTransformer


class DPGFlowNet(nn.Module):
    
    def __init__(self, cfg):
        super(DPGFlowNet, self).__init__()
        self.cfg = cfg
        self.cfg.num_tags += 2  # edge-from-ROOT & no-edge
            
        # 1. Initial Encoder 
        self.state_embedding = StateEncoder(
            num_variables=self.cfg.num_variables,
            num_tags=self.cfg.num_tags,
            word_embedding_dim=self.cfg.word_embedding_dim,
            node_embedding_dim=self.cfg.node_embedding_dim,
            label_embedding_dim=self.cfg.label_embedding_dim,
            hidden_layers=self.cfg.model.state_encoder.hidden_layers,
            dropout_rate=self.cfg.model.state_encoder.dropout_rate,
            activation=self.cfg.model.state_encoder.activation,
            encode_label=self.cfg.init_label_embeddings
        )
        
        self.word_embedding = PrefEncoder(
            pretrained_path=self.cfg.model.pref_encoder.pretrained_path,
            trainable_embeddings=self.cfg.model.pref_encoder.trainable,
            agg_func=self.cfg.model.pref_encoder.agg_func,
            max_word_length=self.cfg.max_word_length 
        )
        
        # 2. Main encoder (Backbone) 
        # TODO: Implement other architectures for backbone 
        self.encoder = [LinearTransformer(
            input_size=self.cfg.model.backbone.input_size,          # Input dim of the backbone 
            num_heads=self.cfg.model.backbone.num_heads,     
            d_k=self.cfg.model.backbone.d_k,
            d_v=self.cfg.model.backbone.d_v,
            d_model=self.cfg.model.backbone.d_model,                # Ouput dim of the backbone
            activation=self.cfg.model.backbone.activation,          # TODO: Specify whose these components belong to
            attn_dropout=self.cfg.model.backbone.attention.dropout,
            mlp_dropout=self.cfg.model.backbone.mlp.dropout,
            dropout=self.cfg.model.backbone.dropout,
            eps=self.cfg.model.backbone.attention.eps,
            label_embedded=self.cfg.init_label_embeddings,
            num_tags=self.cfg.num_tags,
            d_label=self.cfg.label_embedding_dim)
                        for _ in range(self.cfg.model.backbone.num_layers)]
        self.encoder = nn.Sequential(*self.encoder) 
        
        # TODO: Using embedding  
        self.label_scorer = LabelScorer(
            num_tags=self.cfg.num_tags,
            input_dim=self.cfg.model.label_scorer.node_embedding_dim,
            intermediate_dim=self.cfg.model.label_scorer.intermediate_dim,
            hidden_layers=self.cfg.model.label_scorer.hidden_layers,
            dropout_rate=self.cfg.model.label_scorer.dropout_rate,
            activation=self.cfg.model.label_scorer.activation,
            use_state_node_embeddings=self.cfg.model.label_scorer.use_state_node_embeddings
        )
        