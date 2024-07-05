import numpy as np

from dp_gfn.utils.pretrains import batch_token_embeddings_to_batch_word_embeddings

# from dp_gfn.nets.encoders import DenseBlock TODO: 

import haiku as hk
import jax.numpy as jnp
from transformers import AutoTokenizer, AutoConfig
from dp_gfn.nets.bert import BertModel


class StateEncoder(hk.Module):
    
    def __init__(self, num_variables=160, node_embedding_dim=128, init_scale=0.5):
        super().__init__()
        
        self.num_variables = num_variables
        self.node_embedding_dim = node_embedding_dim
        self.init_scale = init_scale

    def __call__(self, word_embeddings):
        w_init = hk.initializers.VarianceScaling(self.init_scale)
        # TODO: Whether using activation function or not?
        wh_embeddings = hk.Linear(self.node_embedding_dim, w_init=w_init)(word_embeddings)
        wd_embeddings = hk.Linear(self.node_embedding_dim, w_init=w_init)(word_embeddings)
        
        indices = jnp.arange(self.num_variables**2) 
        head_ids, dep_ids = jnp.divmod(indices, self.num_variables) 
         
        wh_embeddings = wh_embeddings[head_ids]
        wd_embeddings = wd_embeddings[dep_ids]
        
        state_embeddings = jnp.concatenate([wh_embeddings, wd_embeddings], axis=-1)
        
        return state_embeddings


class PrefEncoder(hk.Module):
    
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
    