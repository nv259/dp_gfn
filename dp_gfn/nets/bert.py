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
        
        if training:
            embeddings = hk.dropout(
                hk.next_rng_key(), rate=self.config['hidden_dropout_prob']
            )
        
        return embeddings
    