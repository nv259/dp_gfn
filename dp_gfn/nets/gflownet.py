import torch
import torch.nn as nn
import torch.nn.functional as F

from dp_gfn.nets.encoders import (
    MLP,  # TODO: implement GNN, CNN, etc...
    LinearTransformer,
)
from dp_gfn.nets.initial_encoders import LabelScorer, PrefEncoder, StateEncoder
from dp_gfn.utils.masking import mask_logits

# from linear_attention_transformer import LinearAttentionTransformer


class DPGFlowNet(nn.Module):
    def __init__(self, cfg):
        super(DPGFlowNet, self).__init__()
        self.cfg = cfg
        self.cfg.num_tags += 2  # including edge-from-ROOT & no-edge

        # 1. Initial Encoder
        self.word_embedding = PrefEncoder(
            pretrained_path=self.cfg.model.pref_encoder.pretrained_path,
            trainable_embeddings=self.cfg.model.pref_encoder.trainable,
            agg_func=self.cfg.model.pref_encoder.agg_func,
            max_word_length=self.cfg.num_variables,
        )

        self.state_embedding = StateEncoder(
            num_variables=self.cfg.num_variables,
            num_tags=self.cfg.num_tags,
            word_embedding_dim=self.word_embedding.hidden_dim,
            node_embedding_dim=self.cfg.model.node_embedding_dim,
            label_embedding_dim=self.cfg.model.label_embedding_dim
            or self.cfg.model.node_embedding_dim,
            hidden_layers=self.cfg.model.state_encoder.hidden_layers
            or [512, 256],
            dropout_rate=self.cfg.model.state_encoder.dropout_rate,
            activation=self.cfg.model.state_encoder.activation,
            encode_label=self.cfg.init_label_embeddings,
        )

        # 2. Main encoder (Backbone)
        # TODO: Implement other architectures for backbone
        self.encoder = [
            LinearTransformer(
                input_size=self.cfg.model.backbone.in_features,  # Input dim of the backbone
                num_heads=self.cfg.model.backbone.num_heads,
                d_k=self.cfg.model.backbone.d_k,
                d_v=self.cfg.model.backbone.d_v or self.cfg.model.backbone.d_k,
                d_model=self.cfg.model.backbone.d_model,  # Ouput dim of the backbone
                activation=self.cfg.model.backbone.activation,  # TODO: Specify whose these components
                # (activation, dropout, etc...) belong to
                attn_dropout=self.cfg.model.backbone.attention.dropout
                or self.cfg.model.backbone.dropout,
                mlp_dropout=self.cfg.model.backbone.mlp.dropout
                or self.cfg.model.backbone.dropout,
                dropout=self.cfg.model.backbone.dropout,
                eps=self.cfg.model.backbone.attention.eps,
                label_embedded=self.cfg.init_label_embeddings,
                num_tags=self.cfg.num_tags,
                d_label=self.cfg.model.label_embedding_dim or self.cfg.model.node_embedding_dim,
            )
            for _ in range(self.cfg.model.backbone.num_layers)
        ]
        self.encoder = nn.Sequential(*self.encoder)

        # TODO: Re-implement for input heads and deps are from plms, maybe?
        self.label_scorer = LabelScorer(
            num_tags=self.cfg.num_tags,
            input_dim=self.cfg.model.node_embedding_dim,
            intermediate_dim=self.cfg.model.label_scorer.intermediate_dim,
            hidden_layers=self.cfg.model.label_scorer.hidden_layers,
            dropout_rate=self.cfg.model.label_scorer.dropout_rate,
            activation=self.cfg.model.label_scorer.activation,
            use_state_node_embeddings=self.cfg.model.label_scorer.use_state_node_embeddings,
        )

        self.logits = MLP(
            input_dim=self.cfg.model.backbone.d_model,
            output_dim=self.cfg.num_variables**2,
            hidden_layers=self.cfg.model.output_logits.hidden_layers,
            dropout_rate=self.cfg.model.output_logits.dropout_rate,
            activation=self.cfg.model.output_logits.activation,
        )

        # TODO: Re-design the Z function
        # Z for the overall flow of given sentences
        # self.output_Z = nn.Linear(self.word_embedding.model_embedding, self.cfg.backbone.d_model)
        self.Z_mod = MLP(
            input_dim=self.word_embedding.hidden_dim,
            output_dim=self.cfg.model.backbone.d_model,
            hidden_layers=[
                self.cfg.model.backbone.d_model + self.word_embedding.hidden_dim
            ],
            dropout_rate=self.cfg.model.output_Z.dropout_rate or 0.1,
            activation=self.cfg.model.output_Z.activation or "ReLU",
        )
        
        self.logsoftmax2 = nn.LogSoftmax(2)

    def Z(self, pref):
        return self.Z_mod(pref).sum(1)

    def Z_param(self):
        return self.output_Z.parameters()

    def model_params(self):
        return (
            list(self.word_embedding.parameters())
            + list(self.state_embedding.parameters())
            + list(self.encoder.parameters())
            + list(self.label_scorer.parameters())
            + list(self.logits.parameters())
        )

    def forward(self, edges, labels=None, mask=None):
        # Embed through backbone
        for layer in self.encoder:
            edges = layer(edges, labels)

        logits = self.logits(edges)
        logits = mask_logits(logits, mask)

        return logits
