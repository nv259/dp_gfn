import torch
from torch.distributions import Categorical
import torch.nn.functional as F
import haiku as hk
import jax
import jax.numpy as jnp

from dp_gfn.nets.encoders import MLP 
from dp_gfn.nets.initial_encoders import Biaffine
from dp_gfn.utils import masking
from torch import nn
from transformers import AutoModel, AutoTokenizer
from dp_gfn.utils.pretrains import token_to_word_embeddings
from dgl.nn import EdgeGATConv


class DPGFlowNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config  
        
        self.bert_model = AutoModel.from_pretrained(config.initializer.pretrained_path)
        self.backbone = None
        self.Z_head = MLP(config.Z_head)
        self.forward_dep = MLP(config.forward_head.dep)
        self.forward_head = MLP(config.forward_head.head)
        self.backward = MLP(config.backward_head)
        
    def forward(self, g, mask, exp_temp, rand_coef):
        hidden = self.backbone(g, g.ndata['s0'], g.edata['s0'])
        
        actions, log_pF = self.forward_policy(hidden, mask, exp_temp, rand_coef)
        log_pBs = self.backward_policy(hidden, mask)
        
        return actions, log_pF, log_pBs   
     
    def forward_policy(self, hidden, mask, exp_temp=1.0, rand_coef=0.0):
        dep_mask = torch.any(mask, axis=1)
        logits = self.forward_dep(hidden)
        dep_ids, log_pF_dep = masking.sample_action(logits, dep_mask, exp_temp, rand_coef)
        
        head_mask = mask[:, dep_ids]
        logits = self.forward_head(hidden, dep_ids)
        head_ids, log_pF_head = masking.sample_action(logits, head_mask, exp_temp, rand_coef) 
        
        return (head_ids, dep_ids), (log_pF_head, log_pF_dep)

    def init_state(self, tokens, word_ids):
        config = self.config.initializer
        
        token_embeddings = self.bert_model(**tokens)
        word_embeddings = token_to_word_embeddings(
            token_embeddings=token_embeddings,
            word_ids=word_ids,
            agg_func=config.agg_func,
            max_word_length=config.max_word_length
        )
        
        return word_embeddings
    
    def logZ(self, x):
        return self.Z_head(x)
     
    
class DPGFlowNet(hk.Module):
    """`GFlowNet` model used in DP-GFlowNet.

    The model uses 3 neural network architectures with the same backbone based on Linear Transformer, i.e.,
    an architecture with 3 heads to estimate the Forward Policy P_F(G' | G) = P_F_dep(w' | G) * P_F_head(w | w', G)
    and Backward Policy P_B(G | G'). The backbone is inspired from Linear Transformer (modified to imbue key, value
    with edge's information) and built by stacking `num_layers` Linear Transformers upon themselves. Each P_F heads
    is represented by a Biaffine layers, whereas P_B is approximated using a 2-layers MLP.

    This model is then vmapped inside `gflownet_fn`. Initializing parameters and inputs are described in the vmapped version,
    therefore below is description of how `__call__` function works.

    Args:
        - key (jnp.DeviceArray) : PRNGKey used for sampling actions, including:
            - Choosing dependent conditioned on input graph G
            - Choosing head to create a link to previous chosen dependent.

        - x (jnp.DeviceArray) : Abbreviation of `node_embeddings` in other docstrings, representing G as a list of nodes; positted shape [..., num_variables, model_size]
        - labels-mask-delta : Auxiliary variables for proposed sampling strategy.

    Returns:
        : The estimating flow is as follow:
        (1) Encoding G using `num_layers` of `LinearTransformerBlock`,
        (2) Extract the viable nodes that can be choosed as dependent using mask,
        (3) Sample dependent node from log_pi_dep, which is computed by applying Biaffine on encoded embeddings given in step1,
        (4) Extract the viable nodes that can be choosed as head for the previously sampled dependent in step3,
        (5) Sample head node from log_pi_head, which is computed by applying Biaffine on encoded embeddings given in step1,
        (6) Computing entire Backward Policy pB(- | G) for all G' \in Par(G)
    """

    def __init__(
        self,
        num_variables=100,
        num_tags=41,
        num_layers=5,
        num_heads=4,
        key_size=64,
        model_size=None,
        name="gfn",
    ):
        super().__init__(name=name)

        self.num_variables = num_variables
        self.num_tags = num_tags + 1  # including no-edge type

        self.num_layers = num_layers
        self.key_size = key_size
        self.num_heads = num_heads
        self.model_size = (
            model_size if model_size is not None else self.num_heads * self.key_size
        )

        self.init_scale = 2.0 / self.num_layers

    def __call__(self, key, x, labels, mask, delta):
        for _ in range(self.num_layers):
            x = LinearTransformerBlock(
                num_heads=self.num_heads,
                key_size=self.key_size,
                model_size=self.model_size,
                init_scale=self.init_scale,
                num_tags=self.num_tags,
            )(x, labels)

        dep_mask = jnp.any(mask, axis=0)
        log_pi_dep = self.dep_policy(x[1:], dep_mask[1:])
        key, dep_id, log_pF_dep = masking.sample_action(key, log_pi_dep, dep_mask[1:], delta)
        # Offset dep_id by 1 since ROOT cannot be chosen as a dependant
        dep_id = dep_id + 1

        head_mask = mask[:, dep_id]
        log_pi_head = self.head_policy(x, dep_id, head_mask)
        key, head_id, log_pF_head = masking.sample_action(key, log_pi_head, head_mask, delta)
        
        log_pBs = self.backward_policy(x[1:], labels[1:].astype(jnp.bool_))

        return key, (dep_id, head_id), (log_pF_dep, log_pF_head), log_pBs

    def dep_policy(self, node_embeddings, mask):
        """Compute the log policy of nodes being chosen as a dependent, given the graph. The length of list of variables is offset by 1 when comparing with self.num_variables, since ROOT cannot be chosen as a dependent.

        Args:
            graph_emb (jnp.DeviceArray): Embeddings of the graph; shape [..., embedding_dim]
            node_embeddings (jnp.DeviceArray): List of embeddings of graph's nodes; shape [..., num_variables - 1, embedding_dim]
            mask (np.ndarray): Mask of the valid dependent nodes [..., num_variables - 1]

        Returns:
            (jnp.DeviceArray): Policy of nodes being chosen as a dependent P(dependent | G); shape [..., num_variables - 1]
        """ 
        logits = hk.nets.MLP([256, 128, 1])(node_embeddings)
        logits = logits.squeeze(-1)
        log_pi = log_policy(logits, mask)

        return log_pi

    def head_policy(self, node_embeddings, node_id, mask):
        """Compute the log policy of nodes being chosen as a head, given the graph and a dependent node

        Args:
            node_embeddings (jnp.DeviceArray): List of embeddings of graph's nodes; shape [..., num_variables, embedding_dim]
            node_id (jnp.DeviceArray): Specified dependent node_id which needs to find a head; shape [..., ]
            mask (np.ndarray): Mask of the valid head nodes; shape [..., num_variables]

        Returns:
            (jnp.DeviceArray): Policy of nodes being chosen as a head P(head | dependent, G); shape [..., num_variables]
        """
        deps = jnp.repeat(
            node_embeddings[jnp.newaxis, node_id], node_embeddings.shape[-2], axis=-2
        )

        logits = jax.vmap(
            Biaffine(num_tags=1, init_scale=self.init_scale), in_axes=(0, 0)
        )(node_embeddings, deps)
        logits = logits.squeeze(-1)
        log_pi = log_policy(logits, mask)

        return log_pi

    def backward_policy(self, node_embeddings, mask):
        """Compute the log backward policy between two states, which is equivalent to remove a node (and its incoming edge)

        Args:
            node_embeddings (jnp.DeviceArray): List of embeddings of graph's nodes; shape [..., num_variables, embedding_dim]
            mask (np.ndarray): Mask of the valid actions; shape [..., num_variables]

        Returns:
            jnp.DeviceArray: Backward policy of states P(G' | G), where each G' equals to G but has exactly one node removed; shape [..., num_variables]
        """
        logits = hk.nets.MLP([256, 128, 1])(node_embeddings)
        logits = logits.squeeze(-1)
        log_pB = log_policy(logits, mask)

        return log_pB


def log_policy(logits, mask):
    """Generate the log version of Policy from the input logits and filtering mask.

    Args:
        logits (jnp.DeviceArray): Input logits
        mask (np.ndarray): Mask of the valid actions

    Returns:
        jnp.DeviceArray: Log Policy with shape similar to logits
    """
    mask = mask.reshape(logits.shape)
    logits = masking.mask_logits(logits, mask)
    log_pi = jax.nn.log_softmax(logits, axis=-1)

    return log_pi


def gflownet_fn(
    key,
    node_embeddings,
    labels,
    mask,
    num_tags,
    num_layers,
    num_heads,
    key_size,
    delta=0.0,
):
    """Function wrapping the main model (`DPGFlowNet`) used.

    Args:
        key (jnp.DeviceArray): Random key for forward sampling
        node_embeddings (jnp.DeviceArray): List of nodes' embeddings; shape [..., num_variables, embedding_dim]
        labels (jnp.DeviceArray): Labels of incoming links w.r.t their dependent nodes; shape [..., num_variables]
        mask (jnp.DeviceArray): The mask for the valid actions (viable edges) that can be taken; shape [..., num_variables, num_variables]
        num_tags (int): Total number of link's tags (labels) in the dataset # TODO: Currently not used
        num_layers (int): Number of layers in the backbone
        num_heads (int): Number of independent attention heads
        key_size (int): The size of keys and queries used for attention
        delta (jnp.DeviceArray[float], optional): A float or array of floats for the mean of the random variables used in action sampling progress. Defaults to 0.0.; shape [..., ]

    Returns:
    ---
        - key (jnp.DeviceArray): New random key (PRNGKey)
        - action (tuple(jnp.DeviceArray, jnp.DeviceArray)): Tuple of (`dependent_id`, `head_id`) representing an action of adding one edge from `head_id` to `dependent_id`
        - log_pF_dep (jnp.DeviceArray): Log Forward Policy of choosing a node as dependent at `dependent_id` (`action[0]`); shape [.., ]
        - log_pF_head (jnp.DeviceArray): Log Forward Policy of choosing a node as head to the chosen dependent at `head_id` (`action[1]`); shape [..., ]
        - log_pBs (jnp.DeviceArray): Log Backward Policy of given state (`node_embeddings`); shape [..., num_variables - 1]. Here the size is deduced by 1 to exclude ROOT.
    """
    num_variables = int(node_embeddings.shape[0])

    key, action, (log_pF_dep, log_pF_head), log_pBs = DPGFlowNet(
        num_variables=num_variables,
        num_tags=num_tags,
        num_layers=num_layers,
        num_heads=num_heads,
        key_size=key_size,
    )(key, node_embeddings, labels, mask, delta)

    return key, action, (log_pF_dep, log_pF_head), log_pBs


def output_total_flow_fn(sent):
    """Estimate partion function (total flow) of a sentence.

    Args:
        sent (jnp.DeviceArray): Sentence embeddings; shape [..., embedding_dim]

    Returns:
        jnp.DeviceArray: Total flow of the sentence; shape [..., 1]
    """
    return hk.get_parameter("logZ", shape=(1,), init=jnp.ones)
