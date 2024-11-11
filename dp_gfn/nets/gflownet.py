import dgl
import torch
from torch import nn
from transformers import AutoModel

from dp_gfn.nets.encoders import MLP, BiAffine
from dp_gfn.utils.masking import sample_action, mask_logits
from dp_gfn.utils.pretrains import token_to_word_embeddings


class DPGFlowNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        config.forward_head.dep.output_sizes.insert(0, config.backbone.d_model)
        # config.forward_head.head.output_sizes.insert(0, config.backbone.d_model * 2)

        self.bert_model = AutoModel.from_pretrained(config.initializer.pretrained_path)
        if not config.initializer.trainable:
            for params in self.bert_model.parameters():
                params.requires_grad = False

        config.backbone.attr.in_feats = self.bert_model.config.hidden_size
        self.backbone = getattr(dgl.nn, config.backbone.name)(**config.backbone.attr)

        self.mlp_dep = MLP(**config.forward_head.dep)
        # self.mlp_head = MLP(**config.forward_head.head)
        self.mlp_head = BiAffine(config.backbone.d_model, 1)
        self.mlp_logZ = MLP(**config.Z_head)
        self.mlp_backward = MLP(**config.backward_head)

    def forward(self, g, mask, exp_temp, rand_coef):
        hidden = self.backbone(g, g.ndata["s0"])
        hidden = hidden.reshape(g.batch_size, -1, self.config.backbone.d_model)

        actions, log_pF = self.forward_policy(hidden, mask, exp_temp, rand_coef)
        backward_logits = self.backward_logits(hidden, ~torch.any(mask, axis=1))

        return actions, log_pF, backward_logits

    def forward_policy(self, x, mask, exp_temp=1.0, rand_coef=0.0):
        B, N, D = x.shape
        
        dep_mask = torch.any(mask, axis=1)
        logits = self.mlp_dep(x).squeeze(-1)
        dep_ids, log_pF_dep = sample_action(logits, dep_mask, exp_temp, rand_coef)

        head_mask = mask.take_along_dim(dep_ids.unsqueeze(-1), -1).squeeze(-1)
        x_deps = x.gather(1, dep_ids.unsqueeze(-1).expand(-1, -1, D)).expand(-1, N, -1)
        logits = self.mlp_head(x.view((B*N, D)), x_deps.reshape((B*N, D))).squeeze(-1)
        logits = logits.view((B, N))
        head_ids, log_pF_head = sample_action(logits, head_mask, exp_temp, rand_coef)

        return (head_ids, dep_ids), (log_pF_head, log_pF_dep)

    def backward_logits(self, x, mask):
        mask[:, 0] = False
        logits = self.mlp_backward(x)
        logits = mask_logits(logits, mask)

        return logits

    def init_state(self, input_ids, attention_mask, word_ids):
        config = self.config.initializer

        token_embeddings = self.bert_model(input_ids, attention_mask)["last_hidden_state"]
        word_embeddings = token_to_word_embeddings(
            token_embeddings=token_embeddings,
            word_ids=word_ids,
            agg_func=config.agg_func,
            max_word_length=config.max_word_length,
        )

        return word_embeddings

    def logZ(self, x):
        return self.mlp_logZ(x)
