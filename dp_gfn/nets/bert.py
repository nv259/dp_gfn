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


PRETRAINED_WEIGHTS = PretrainedWeights('bert-base-uncased')


class Embeddings(hk.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
    def __call__(self, input_ids=None, token_type_ids=None, attention_mask=None, training=False):
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
                PRETRAINED_WEIGHTS['embeddings.LayerNorm.weight']),
            offset_init=hk.initializers.Constant(
                PRETRAINED_WEIGHTS['embeddings.LayerNorm.bias']),
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
                PRETRAINED_WEIGHTS['embeddings.word_embeddings.weight'])
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
        
    def __call__(self):
        position_weights = hk.get_parameter(
            "position_embeddings",
            PRETRAINED_WEIGHTS['embeddings.position_embeddings.weight'].shape,
            init=hk.initializers.Constant(
                PRETRAINED_WEIGHTS['embeddings.position_embeddings.weight']
            )
        )
        
        start = self.offset
        end = start + self.config['max_position_embeddings']
    
        return position_weights[start:end]
   

class TokenTypeEmbeddings(hk.Module):
    
    def __init__(self, config, offset=0):
        super().__init__()
        self.config = config
        
    def __call__(self, token_type_ids):
        flat_token_type_ids = jnp.reshape(
            token_type_ids, [token_type_ids.shape[0] * token_type_ids.shape[1]]
        )

        flat_token_type_embeddings = hk.Embed(
            vocab_size=self.config['type_vocab_size'],
            embed_dim=self.config['hidden_size'],
            w_init=hk.initializers.Constant(
                PRETRAINED_WEIGHTS['embeddings.token_type_embeddings.weight']
            )
        )(flat_token_type_ids)
        
        token_type_embeddings = jnp.reshape(
            flat_token_type_embeddings,
            [
                token_type_ids.shape[0],
                token_type_ids.shape[1],
                self.config['hidden_size']
            ]
        )
        
        return token_type_embeddings


class Encoder(hk.Module):
    
    def __init__(self, config, layer_num):
        super().__init__(name=f'encoder_layer_{layer_num}')
        self.config = config
        self.layer_num = layer_num
        
    def __call__(self, x, mask, training=False):
        # Feeding inputs through a multi-head attention operation
        # i.e. linear mapping -> multi-head attention -> residual connection -> LayerNorm
        attention_output = Attention(
            self.config, self.layer_num
        )(x, mask, training=training)
        
        # Inter-mediate (higher dimension)
        intermediate_output = hk.Linear(
            output_size=self.config['intermediate_size'],
            w_init=hk.initializers.Constant(
                PRETRAINED_WEIGHTS[f'encoder.layer.{self.layer_num}.intermedintermediateiate.dense.weight'].transpose()
            ),
            b_init=hk.initializers.Constant(
                PRETRAINED_WEIGHTS[f'encoder.layer.{self.layer_num}.intermediate.dense.bias']
            ),
            name="intermediate"
        )(attention_output)
        
        # TODO: Usage of approximation?
        if self.config['hidden_act'] == 'gelu':
            intermediate_output = jax.nn.gelu(intermediate_output)
        else:
            raise Exception("Hidden activation not supported")
        
        output = Output(
            self.config, self.layer_num
        ) (intermediate_output, attention_output, training=training)
        
        return output