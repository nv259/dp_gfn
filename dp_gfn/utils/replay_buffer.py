import numpy as np
from numpy.random import default_rng


class ReplayBuffer:
    def __init__(self, capacity, num_variables, max_priority, decay_rate=0.1):
        self.capacity = capacity
        self.num_variables = num_variables
        self.max_priority = max_priority
        self.decay_rate = decay_rate
        
        dtype = np.dtype([
            ('input_ids', np.int_, (512, )),
            ('attention_mask', np.int_, (512, )),
            ('word_ids', np.int_, (512, )),
            ('adjacency', np.int_, (self.num_variables, self.num_variables)),
            ('mask', np.uint8, (self.num_variables, self.num_variables)),
            ('action', np.uint8, (2, )),
            ('next_mask', np.uint8, (self.num_variables, self.num_variables)),
            ('next_adjacency', np.int_, (self.num_variables, self.num_variables)),
            # ('gold', np.int_, (self.capacity, self.capacity, )),
            ('reward', np.int_, (1,)),  # -1, 0, C
            # ('num_words', np.int_, (1,)),
            ('priority', np.float64, (1,)),
        ])
        
        self._replay = np.zeros((capacity, ), dtype=dtype) 
        self._syn_replay = np.zeros((capacity, ), dtype=dtype)
        self._index = 0
        self._is_full = False
        
    def __len__(self):
        return self.capacity if self._is_full else self._index
    
    def reset(self, ):
        pass
    
    def sample(self, batch_size, rng=default_rng()):
        indices = rng.choice(len(self), size=batch_size, replace=False, p=self._replay['priority'] / self._replay['priority'].sum())
        samples = self._replay[indices]
        self._replay[indices]['priority'] -= self.decay_rate
    
        indices = rng.choice(len(self), size=batch_size, replace=False, p=self._syn_replay['priority'] / self._syn_replay['priority'].sum())
        syn_samples = self._syn_replay[indices]
        self._syn_replay[indices]['priority'] -= self.decay_rate
            
        return {
            'input_ids': np.concat([samples['input_ids'], syn_samples['input_ids']], axis=0),
            'attention_mask': np.concat([samples['attention_mask'], syn_samples['attention_mask']], axis=0),
            'word_ids': np.concat([samples['word_ids'], syn_samples['word_ids']], axis=0),
            'adjacency': np.concat([samples['adjacency'], syn_samples['adjacency']], axis=0),
            'mask': np.concat([samples['mask'], syn_samples['mask']], axis=0),
            'action': np.concat([samples['action'], syn_samples['action']], axis=0),
            'next_mask': np.concat([samples['next_mask'], syn_samples['next_mask']], axis=0),
            'next_adjacency': np.concat([samples['next_adjacency'], syn_samples['next_adjacency']], axis=0),
            'reward': np.concat([samples['reward'], syn_samples['reward']], axis=0),
        }    
        
    def add(self, ):
        pass 
    
       
       