import numpy as np 
import jax.numpy as jnp
import jax 
import haiku as hk
from transformers import AutoModel, AutoConfig, AutoTokenizer


class PretrainedWeights(object):
    
    def __init__(self, weights):
        self.weights = weights
        
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
    