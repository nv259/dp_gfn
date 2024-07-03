import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dp_gfn.utils.pretrains import batch_token_embeddings_to_batch_word_embeddings

# from dp_gfn.nets.encoders import DenseBlock TODO: 

import haiku as hk
import jax.numpy as jnp
from transformers import AutoTokenizer, AutoConfig
from dp_gfn.nets.bert import BertModel


class StateEncoder(nn.Module):
    def __init__(
        self,
        num_variables,
        num_tags, # FIXME: Protential error 
        word_embedding_dim=768,
        node_embedding_dim=128,
        label_embedding_dim=0,
        hidden_layers=[512, 256],
        dropout_rate=0.1,
        activation="ReLU",
        encode_label=False,
    ):
        super(StateEncoder, self).__init__()
        self.num_variables = num_variables
        self.num_tags = num_tags
        self.node_embedding_dim = node_embedding_dim
        self.label_embedding_dim = label_embedding_dim
        self.encode_label = encode_label                # Whether to encode the label information 
                                                        # in the state representation

        self.indices = np.arange(0, num_variables**2)
        self.head_ids = self.indices // num_variables
        self.dep_ids = self.indices % num_variables

        self.mlp_head = MLP(
            word_embedding_dim,
            self.node_embedding_dim,
            hidden_layers,
            dropout_rate,
            activation,
        )
        self.mlp_dep = MLP(
            word_embedding_dim,
            self.node_embedding_dim,
            hidden_layers,
            dropout_rate,
            activation,
        )
        
        # Initial no-edge embedding   
        self.constant_label = nn.Parameter(torch.zeros(1, 1, self.label_embedding_dim))
        nn.init.xavier_uniform_(self.constant_label)
        
        if encode_label:
            self.label_embedding = nn.Embedding(self.num_tags, self.label_embedding_dim)

    def forward(self, word_embeddings, adjacency=None):
        assert (
            word_embeddings.shape[1] <= self.num_variables
        ), "Number of word is too large"
        
        wh_embeddings = self.mlp_head(word_embeddings)
        wd_embeddings = self.mlp_dep(word_embeddings)
        
        if adjacency is not None:
            assert adjacency.shape[1] == adjacency.shape[2] == self.num_variables

            adjacency = adjacency.reshape(adjacency.shape[0], self.num_variables**2)

            label_embeddings = (
                self.label_embedding(adjacency)
                if self.encode_label
                else torch.zeros(adjacency.shape[0], self.num_variables**2, 0)
            )
        else: 
            label_embeddings = self.constant_label.repeat(word_embeddings.shape[0], self.num_variables**2, 1)

        # Prime wh and wb to create n^2 arcs by linking wh_i with corresponding wd_i
        # e.g {h_0, h_0, h_0, h_1, h_1, h_1, h_2, h_2, h_2}
        wh_embeddings = wh_embeddings[:, self.head_ids]
        # e.g {d_0, d_1, d_2, d_0, d_1, d_2, d_0, d_1, d_2}
        wd_embeddings = wd_embeddings[:, self.dep_ids]

        state_embeddings = torch.cat(
            [wh_embeddings, wd_embeddings, label_embeddings], dim=-1
        )   
        
        if adjacency is not None:
            return state_embeddings, adjacency
        else:
            return state_embeddings


class PrefEncoder(hk.Module):
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
    def __init__(self, pretrained_path="bert-base-uncased", agg_func="mean", max_word_length=160): # TODO: trainable=True
        super().__init__()
        
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
        self.config = AutoConfig.from_pretrained(pretrained_path).to_dict()
        self.agg_func = agg_func
        self.max_word_length = max_word_length
        
    def __call__(self, batch, training=False):
        tokens = self.tokenizer(
            batch, return_tensors='jax', padding='max_length', truncation=True,
            add_special_tokens=False
        )
        
        token_embeddings = BertModel(self.config)(**tokens, training=training)
        
        word_embeddings = batch_token_embeddings_to_batch_word_embeddings(
            tokens=tokens,
            token_embeddings=token_embeddings,
            agg_func=self.agg_func,
            max_word_length=self.max_word_length,
        )
        
        return word_embeddings


class LabelScorer(hk.Module):
    
    def __init__(self, num_tags, init_scale, intermediate_dim=128, activation="relu"):
        super().__init__()
        
        self.num_tags = num_tags
        self.init_scale = init_scale
        self.intermediate_dim = intermediate_dim
        self.activation = activation
        
    def __call__(self, head, dep):
        w_init = hk.initializers.VarianceScaling(self.init_scale)
        
        W = hk.get_parameter(name='W', shape=(self.num_tags, self.intermediate_dim, self.intermediate_dim), init=w_init)
        Wh = hk.get_parameter(name='Wh', shape=(self.intermediate_dim, self.num_tags), init=w_init)
        Wd = hk.get_parameter(name='Wd', shape=(self.intermediate_dim, self.num_tags), init=w_init)
        b = hk.get_parameter(name='b', shape=(self.num_tags,), init=jnp.ones)

        lab_head = hk.Linear(self.intermediate_dim, w_init=w_init)(head)
        lab_dep = hk.Linear(self.intermediate_dim, w_init=w_init)(dep)

        # Biaffine layer
        head_score = lab_head @ Wh
        dep_score = lab_dep @ Wd
        arc_score = lab_dep @ W @ lab_head 
        
        lab_score = arc_score + head_score + dep_score + b
        
        return lab_score
    