import torch
import torch.nn as nn
import torch.nn.functional as F

from dp_gfn.nets.encoders import MLP


class StateEncoder(nn.Module):
    def __init__(self, num_variables, num_tags, word_embedding_dim=300, node_embedding_dim=768, label_embedding_dim=128, hidden_layers=[512, 256], activation=None, share_word_representations=True):
        super(StateEncoder, self).__init__()
        self.num_variables = num_variables
        self.indices = torch.arange(0, num_variables * 2)
        
        # TODO: Implement share weight of mlp_(head|dep) between node_encoder and label_scorer
        # [mlp_node_head, mlp_node_dep, (mlp_label_head, mlp_label_dep)] 
        # self.num_mlps = 2 if share_word_representations else 4
        # self.mlps = [MLP(word_embedding_dim, node_embedding_dim, hidden_layers, activation) for _ in range(self.num_mlps)] 
        self.mlp_head = MLP(word_embedding_dim, node_embedding_dim, hidden_layers, activation)
        self.mlp_dep = MLP(word_embedding_dim, node_embedding_dim, hidden_layers, activation)
       
        # num_embeddings = len( {labels} v {edge-less} ) 
        self.label_embedding = nn.Embedding(num_tags + 1, label_embedding_dim)  
         
    def forward(self, word_embeddings, adjacency):
        wh_embeddings = self.mlp_head(word_embeddings)
        wd_embeddings = self.mlp_dep(word_embeddings)
        label_embeddings = self.label_embedding(adjacency)
        
        state_embeddings = torch.cat([wh_embeddings, wd_embeddings, label_embeddings], dim=-1)    
         
        return state_embeddings
        
        
class PrefEncoder(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """
    def __init__(self):
        super(PrefEncoder, self).__init__()
        
    def forward(self, x):
        return x


class LabelScorer(nn.Module):
    r"""__init__(self, num_tags, input_dim=768, intermediate_dim=128, hidden_layers=[512, 256], activation=None)
    
    Apply Biaffine label scorer to an input tensor (head -> dep)

    Args:
        num_tags (int): The number of expected labels in dataset
        input_dim (int, optional): The number of features in the input tensor (word embeddings). Defaults to 768.
        intermediate_dim (int, optional): The number of features in the intermediate tensor (head/dep representation). Defaults to 128. 
        
    Inputs: heads, deps
        * heads: A tensor of shape ``[batch_size, num_heads, word_emb_dim]`` representing the representations of words designated as heads in the arcs.
        * deps: A tensor of shape `[batch_size, num_deps, word_emb_dim]` representing the representations of words designated as dependents in the arcs.
   
    Outputs: 
        * lab_scores (torch.Tensor): A tensor of shape ``[batch_size, num_tags]``)
    
    Examples::
    
        >>> # Initialize the LabelScorer
        >>> label_scorer = LabelScorer(num_tags, input_dim=768, output_dim=128)

        >>> # Compute the label scores
        >>> lab_scores = label_scorer(heads, deps)

    """
    def __init__(self, num_tags, input_dim=768, intermediate_dim=128, hidden_layers=[512, 256], activation=None):
        super(LabelScorer, self).__init__()
        
        self.mlp_head = MLP(input_dim, intermediate_dim, hidden_layers, activation)
        self.mlp_dep = MLP(input_dim, intermediate_dim, hidden_layers, activation)
        
        W = torch.randn(num_tags, intermediate_dim, intermediate_dim)
        self.W = nn.Parameter(W)
        nn.init.xavier_uniform_(self.W)
        
        Wh = torch.randn(intermediate_dim, num_tags)
        self.Wh = nn.Parameter(Wh)
        nn.init.xavier_uniform_(self.Wh)
        
        Wd = torch.randn(intermediate_dim, num_tags)
        self.Wd = nn.Parameter(Wd)
        nn.init.xavier_uniform_(self.Wd)
        
        b = torch.randn(num_tags)
        self.b = nn.Parameter(b)
        nn.init.xavier_uniform_(self.b)
        
    def forward(self, heads, deps) -> torch.Tensor:
        # Map head words and dep words to corresponding intermediate representation
        lab_heads = self.mlp_head(heads)
        lab_deps = self.mlp_dep(deps)
        
        # Reshape D_L and H_L for biaffine equation
        lab_heads = lab_heads.unsqueeze(1) # [bs, 1, label_emb_dim] 
        lab_heads = lab_heads.unsqueeze(1) # [bs, 1, 1, label_emb_dim] 
        lab_deps = lab_deps.unsqueeze(1)
        lab_deps = lab_deps.unsqueeze(1)

        # Biaffine layer
        lab_scores = lab_deps @ self.W @ lab_heads.transpose(-1, -2) + lab_heads @ self.Wh + lab_deps @ self.Wd + self.b
        
        return lab_scores