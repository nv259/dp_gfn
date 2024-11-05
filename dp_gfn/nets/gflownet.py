import torch
from dgl.nn import EdgeGATConv
from dp_gfn.nets.encoders import MLP
from dp_gfn.utils.masking import sample_action, mask_logits
from dp_gfn.utils.pretrains import token_to_word_embeddings
from torch import nn
from transformers import AutoModel


class DPGFlowNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.bert_model = AutoModel.from_pretrained(config.initializer.pretrained_path)
        self.backbone = None
        self.Z_head = MLP(config.Z_head)

        self.mlp_dep = MLP(config.forward_head.dep)
        self.mlp_head = MLP(config.forward_head.head)

        self.mlp_backward = MLP(config.backward_head)

    def forward(self, g, mask, exp_temp, rand_coef):
        hidden = self.backbone(g, g.ndata["s0"], g.edata["x"])

        actions, log_pF = self.forward_policy(hidden, mask, exp_temp, rand_coef)
        log_pBs = self.backward_logits(hidden)

        return actions, log_pF, log_pBs

    def forward_policy(self, hidden, mask, exp_temp=1.0, rand_coef=0.0):
        dep_mask = torch.any(mask, axis=1)
        logits = self.mlp_dep(hidden)
        dep_ids, log_pF_dep = sample_action(logits, dep_mask, exp_temp, rand_coef)

        head_mask = mask[:, dep_ids]
        logits = self.mlp_head(hidden, dep_ids)
        head_ids, log_pF_head = sample_action(logits, head_mask, exp_temp, rand_coef)

        return (head_ids, dep_ids), (log_pF_head, log_pF_dep)

    def backward_logits(self, hidden):
        logits = self.mlp_backward(hidden)

        return logits

    def init_state(self, tokens, word_ids):
        config = self.config.initializer

        token_embeddings = self.bert_model(**tokens)
        word_embeddings = token_to_word_embeddings(
            token_embeddings=token_embeddings,
            word_ids=word_ids,
            agg_func=config.agg_func,
            max_word_length=config.max_word_length,
        )

        return word_embeddings

    def logZ(self, x):
        return self.Z_head(x)
