from collections import deque
import torch
from torch import nn
from transformers import AutoModel

from dp_gfn.nets.encoders import MLP, BiAffine, TransformerEncoder
from dp_gfn.utils.masking import sample_action, mask_logits
from dp_gfn.utils.misc import get_parent_indices
from dp_gfn.utils.pretrains import token_to_word_embeddings


class DPGFlowNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_dim = config.backbone.embed_dim

        config.forward_head.dep.output_sizes.insert(0, self.hidden_dim)
        # config.forward_head.head.output_sizes.insert(0, config.backbone.d_model * 2)
        config.backward_head.output_sizes.insert(0, self.hidden_dim)

        # Initializer
        self.bert_model = AutoModel.from_pretrained(config.initializer.pretrained_path)
        if not config.initializer.trainable:
            for params in self.bert_model.parameters():
                params.requires_grad = False
        self.x2g = torch.nn.Linear(
            in_features=self.model.hidden_dim, 
            out_features=self.model.hidden_dim
        ).to(self.device)   # Mapping the mean of nodes embeddings to create virtual node
        self.intermediate = nn.Linear(self.bert_model.config.hidden_size, self.hidden_dim)
        
        # Encoder 
        self.backbone = TransformerEncoder(**config.backbone)
        
        # Predictor Heads 
        self.mlp_dep = MLP(**config.forward_head.dep)
        # self.mlp_head = MLP(**config.forward_head.head)
        self.mlp_head = BiAffine(config.backbone.embed_dim, 1)
        self.mlp_logZ = MLP(**config.Z_head)
        self.mlp_backward = MLP(**config.backward_head)

    def forward(self, node_embeddings, graph_relations, mask, attn_mask, actions=None, exp_temp=1., rand_coef=0.):
        hidden = self.backbone(node_embeddings, graph_relations, attn_mask) # TODO: attention mask

        actions, log_pF = self.forward_policy(hidden, mask, actions, exp_temp, rand_coef)
        backward_logits = self.backward_logits(hidden, ~torch.any(mask, axis=1))    # TODO: This leaves undue actions valid

        return actions, log_pF, backward_logits
    
    def trace_backward(self, node_embeddings, graph_relations, orig_graph, attn_mask=None, exp_temp=1., rand_coef=0.):
        B, N, _ = graph_relations.shape
        N = N - 1   # remove virtual nodes
        action_list = deque()
        orig_graph = orig_graph.to(torch.int).to(node_embeddings.device)
        mask = torch.any(orig_graph, dim=1).to(node_embeddings.device) 
        mask[:, 0], mask[:, -1] = False, False
        
        self.eval()
        with torch.no_grad():
            for _ in range(N - 1):
                hidden = self.backbone(node_embeddings, graph_relations, attn_mask) # FIXME: attn_mask does not reflect the reachability-of-graph
                
                backward_logits = self.backward_logits(hidden, mask)
                
                dep_ids, log_pB = sample_action(backward_logits, mask, exp_temp, rand_coef)
                head_ids = get_parent_indices(orig_graph, dep_ids)
                action_list.appendleft(torch.concat((head_ids, dep_ids), axis=1))
                
                mask[torch.arange(B), dep_ids.squeeze()] = False 
                # Remove edges in graph_relations (both directions)
                graph_relations[torch.arange(B), head_ids.squeeze(), dep_ids.squeeze()] = 0
                graph_relations[torch.arange(B), dep_ids.squeeze(), head_ids.squeeze()] = 0

                # Remove edges in orig_graph (only one direction needed)
                orig_graph[torch.arange(B), head_ids.squeeze(), dep_ids.squeeze()] = 0
                          
        action_list.append(torch.concat((head_ids, dep_ids), axis=1))   # dummy
        self.train() 
        
        return torch.stack(list(action_list), 1)
    
    def flow(self, node_embeddings):
        return self.mlp_flow(node_embeddings)

    def forward_policy(self, x, mask, actions=None, exp_temp=1.0, rand_coef=0.0):
        B, N, D = x.shape
        
        dep_mask = torch.any(mask, axis=1)
        logits = self.mlp_dep(x).squeeze(-1)
        if actions is not None:
            dep_ids = actions[:, 1].unsqueeze(1)
            logits = mask_logits(logits, dep_mask)
            log_pF_dep = logits.log_softmax(1).gather(1, dep_ids).squeeze(-1)
        else: 
            dep_ids, log_pF_dep = sample_action(logits, dep_mask, exp_temp, rand_coef)

        head_mask = mask.take_along_dim(dep_ids.unsqueeze(-1), -1).squeeze(-1)
        x_deps = x.gather(1, dep_ids.unsqueeze(-1).expand(-1, -1, D)).expand(-1, N, -1)
        logits = self.mlp_head(x.view((B*N, D)), x_deps.reshape((B*N, D))).squeeze(-1)
        logits = logits.view((B, N))
        if actions is not None:
            head_ids = actions[:, 0].unsqueeze(1)
            logits = mask_logits(logits, head_mask)
            log_pF_head = logits.log_softmax(1).gather(1, head_ids).squeeze(-1)
        else:
            head_ids, log_pF_head = sample_action(logits, head_mask, exp_temp, rand_coef)

        assert (dep_mask[:, 0] == False).all() and (dep_mask[:, -1] == False).all(), "Invalid mask"
        assert (head_mask[:, -1] == False).all(), "Invalid mask"

        return torch.concat((head_ids, dep_ids), axis=1), (log_pF_head, log_pF_dep)

    def backward_logits(self, x, mask):
        mask[:, 0], mask[:, -1] = False, False
        logits = self.mlp_backward(x).squeeze(-1)
        logits = mask_logits(logits, mask)

        return logits

    def init_state(self, input_ids, attention_mask, word_ids, use_virtual_node=False):
        config = self.config.initializer

        token_embeddings = self.bert_model(input_ids, attention_mask)["last_hidden_state"]
        word_embeddings = token_to_word_embeddings(
            token_embeddings=token_embeddings,
            word_ids=word_ids,
            agg_func=config.agg_func,
            max_word_length=max(word_ids[0]).item() + 1,
            # max_word_length=attention_mask.sum(-1).item(),
        )
        
        # Map to intermediate dimension
        word_embeddings = self.intermediate(word_embeddings)
        if use_virtual_node:
            graph_embeddings = self.x2g(word_embeddings.mean(axis=-2, keepdims=True))
            word_embeddings = torch.concat([word_embeddings, graph_embeddings], axis=1)

        return word_embeddings

    def logZ(self, x):
        return self.mlp_logZ(x)
