import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dp_gfn.utils.pretrains import batch_token_embeddings_to_batch_word_embeddings

from transformers import AutoModel, AutoTokenizer

from dp_gfn.nets.encoders import MLP


class StateEncoder(nn.Module):
    def __init__(
        self,
        num_variables,
        num_tags=0, # FIXME: Protential error 
        word_embedding_dim=768,
        node_embedding_dim=128,
        label_embedding_dim=0,
        hidden_layers=[512, 256],
        dropout_rate=0.1,
        activation="ReLU",
        encode_label=False,
    ):
        # if (num_tags != 0) and (label_embedding_dim == 0):
        #     print(
        #         "warning: num_tags is specified but label_embedding_dim is not specified, label_embedding_dim will be set to node_embedding_dim"
        #     )
        #     label_embedding_dim = node_embedding_dim

        super(StateEncoder, self).__init__()
        self.num_variables = num_variables
        self.num_tags = num_tags + 2
        self.encode_label = encode_label                # Whether to encode the label information 
                                                        # in the state representation

        self.indices = np.arange(0, num_variables**2)
        self.head_ids = self.indices // num_variables
        self.dep_ids = self.indices % num_variables

        self.mlp_head = MLP(
            word_embedding_dim,
            node_embedding_dim,
            hidden_layers,
            dropout_rate,
            activation,
        )
        self.mlp_dep = MLP(
            word_embedding_dim,
            node_embedding_dim,
            hidden_layers,
            dropout_rate,
            activation,
        )

        if encode_label:
            # num_embeddings = len( {labels} v {edge-less} v {ROOT-edge})
            self.label_embedding = nn.Embedding(self.num_tags, label_embedding_dim)

    def forward(self, word_embeddings, adjacency):
        assert (
            word_embeddings.shape[1] <= self.num_variables
        ), "Number of word is too large"
        assert adjacency.shape[1] == adjacency.shape[2] == self.num_variables

        wh_embeddings = self.mlp_head(word_embeddings)
        wd_embeddings = self.mlp_dep(word_embeddings)

        adjacency = adjacency.reshape(adjacency.shape[0], self.num_variables**2)

        label_embeddings = (
            self.label_embedding(adjacency)
            if self.encode_label
            else torch.zeros(adjacency.shape[0], self.num_variables**2, 0)
        )

        # Prime wh and wb to create n^2 arcs by linking wh_i with corresponding wd_i
        # e.g {h_0, h_0, h_0, h_1, h_1, h_1, h_2, h_2, h_2}
        wh_embeddings = wh_embeddings[:, self.head_ids]
        # e.g {d_0, d_1, d_2, d_0, d_1, d_2, d_0, d_1, d_2}
        wd_embeddings = wd_embeddings[:, self.dep_ids]

        state_embeddings = torch.cat(
            [wh_embeddings, wd_embeddings, label_embeddings], dim=-1
        )

        return state_embeddings, adjacency


class PrefEncoder(nn.Module):
    """
    A neural network module that processes a batch of sentences
    to produce word-level embeddings instead of the usual token-level embeddings.
    The output shape is [batch_size, max_word_len, d_model].

    Args:
        pretrained_path (str, optional): Path to the pretrained language model.
        trainable (bool, optional): If True, allows fine-tuning of embeddings. Default is True.
        agg_func (str, optional): Aggregation function to combine token embeddings into word embeddings. Default is 'mean'.
        max_word_length (int, optional): Maximum number of words per sentence. Default is 160.

    Inputs:
        batch (list of str): Batch of sentences.

    Outputs:
        torch.Tensor: Word embeddings with shape [batch_size, max_word_len, d_model].
    """

    def __init__(
        self,
        pretrained_path=None,
        trainable=True,
        agg_func="mean",
        max_word_length=160,
    ):
        super(PrefEncoder, self).__init__()

        # Load pretrain language model
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
        self.bert_model = AutoModel.from_pretrained(pretrained_path)
        if trainable is False:
            for param in self.bert_model.parameters():
                param.requires_grad = False

        # Store auxiliary parameters
        self.word_embedding_dim = self.bert_model.embeddings.word_embeddings.embedding_dim
        self.agg_func = agg_func
        self.max_word_length = max_word_length

    def forward(self, batch):
        tokens = self.tokenizer(
            batch, return_tensors="pt", padding="max_length", truncation=True
        )
        token_embeddings = self.bert_model(**tokens).last_hidden_state

        word_embeddings = batch_token_embeddings_to_batch_word_embeddings(
            tokens=tokens,
            token_embeddings=token_embeddings,
            agg_func=self.agg_func,
            max_word_length=self.max_word_length,
        )

        return word_embeddings


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

    def __init__(
        self,
        num_tags,
        input_dim=768,
        intermediate_dim=128,
        hidden_layers=[512, 256],
        dropout_rate=0.1,
        activation="ReLU",
        use_pretrained_embeddings=False,
    ):
        if use_pretrained_embeddings is False:
            print(
                f"warning: use_pretrained_embeddings is False, LabelScorer will use node embeddings from state instead"
            )

        super(LabelScorer, self).__init__()
        self.num_tags = num_tags + 2
        self.use_pretrained_embeddings = use_pretrained_embeddings

        if self.use_pretrained_embeddings:
            self.mlp_head = MLP(
                input_dim, intermediate_dim, hidden_layers, dropout_rate, activation
            )
            self.mlp_dep = MLP(
                input_dim, intermediate_dim, hidden_layers, dropout_rate, activation
            )

        W = torch.randn(self.num_tags, intermediate_dim, intermediate_dim)
        self.W = nn.Parameter(W)
        nn.init.xavier_uniform_(self.W)

        Wh = torch.randn(intermediate_dim, self.num_tags)
        self.Wh = nn.Parameter(Wh)
        nn.init.xavier_uniform_(self.Wh)

        Wd = torch.randn(intermediate_dim, self.num_tags)
        self.Wd = nn.Parameter(Wd)
        nn.init.xavier_uniform_(self.Wd)

        b = torch.randn(self.num_tags)
        self.b = nn.Parameter(b)

    def forward(self, heads, deps) -> torch.Tensor:
        lab_heads = self.mlp_head(heads) if self.use_pretrained_embeddings else heads
        lab_deps = self.mlp_dep(deps) if self.use_pretrained_embeddings else deps

        # Biaffine layer
        head_scores = lab_heads @ self.Wh
        dep_scores = lab_deps @ self.Wd

        # Reshape D_L and H_L for biaffine equation
        while lab_heads.dim() <= self.W.dim():
            lab_heads = lab_heads.unsqueeze(1)
            lab_deps = lab_deps.unsqueeze(1)

        arc_scores = torch.reshape(
            lab_deps @ self.W @ lab_heads.transpose(-1, -2),
            (heads.shape[0], self.num_tags),
        )

        lab_scores = arc_scores + head_scores + dep_scores + self.b

        return lab_scores
