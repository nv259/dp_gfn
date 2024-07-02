import numpy as np 
import jax.numpy as jnp
import jax 
import haiku as hk
from transformers import BertModel, BertConfig, BertTokenizer


class PretrainedWeights(object):
    
    def __init__(self, path_to_pretrained):
        self.path_to_pretrained = path_to_pretrained
        
        pretrained_model = BertModel.from_pretrained(path_to_pretrained)
        self.weights = dict(pretrained_model.named_parameters())
        
    def __getitem__(self, path_to_weight):
        if isinstance(path_to_weight, list):
            path = '.'.join(path_to_weight)
        else:
            path = path_to_weight
        
        path = path.replace('/', '.')
        path = path.replace('layer_', 'layer.')
        
        return self.weights[path].detach().numpy()
   
    def __str__(self) -> str:
        return str(self.weights.keys())


class Embeddings(hk.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
    def __call__(self, training=False, **kwargs):
        input_ids = kwargs['input_ids']
        token_type_ids = kwargs['token_type_ids']
        attention_mask = kwargs['attention_mask']
        
        # Calculate embeddings
        token_embeddings = WordEmbeddings(self.config)(input_ids)
        position_embeddings = PositionEmbeddings(self.config)()
        token_type_embeddings = TokenTypeEmbeddings(self.config)(token_type_ids)
        
        embeddings = token_embeddings + position_embeddings + token_type_embeddings
        
        # LayerNorm
        embeddings = hk.LayerNorm(
            axis=-1, 
            create_offset=True, 
            create_scale=True, 
            eps=self.config['layer_norm_eps'],
            scale_init=hk.initializers.Constant(
                pretrained_weights['embeddings.LayerNorm.weight']),
            offset_init=hk.initializers.Constant(
                pretrained_weights['embeddings.LayerNorm.bias']),
            name='LayerNorm'
        )(embeddings)
        
        if training:    # TODO: check embed_dropout_prob
            embeddings = hk.dropout(
                hk.next_rng_key(), rate=self.config['hidden_dropout_prob']
            )
        
        return embeddings
   
   
class WordEmbeddings(hk.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
    def __call__(self, input_ids, training=False):
        flat_input_ids = jnp.reshape(
            input_ids, [input_ids.shape[0] * input_ids.shape[1]]
        )
        
        # TODO: cross-check with other plms (XLM-R, RoBERTa, etc.)
        flat_input_embeddings = hk.Embed(
            vocab_size=self.config['vocab_size'], 
            embed_dim=self.config['hidden_size'],
            w_init=hk.initializers.Constant(
                pretrained_weights['embeddings.word_embeddings.weight'])
        )(flat_input_ids)
        
        token_embeddings = jnp.reshape(
            flat_input_embeddings, [input_ids.shape[0], input_ids.shape[1], self.config['hidden_size']]
        )
        
        return token_embeddings
   

# TODO: inspect position embedding type ('absolute' or else)
class PositionEmbeddings(hk.Module):
    
    def __init__(self, config, offset=0):
        super().__init__()
        self.config = config  
        self.offset = offset
        
    def __call_(self):
        position_weights = hk.get_parameter(
            "position_embeddings",
            pretrained_weights['embeddings.position_embeddings.weight'].shape,
            init=hk.initializers.Constant(
                pretrained_weights['embeddings.position_embeddings.weight']
            )
        )
        
        start = self.offset
        end = start + self.config['max_position_embeddings']
    
        return position_weights[start:end]
    
