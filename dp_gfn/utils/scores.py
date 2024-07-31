import jax.numpy as jnp


def unlabeled_graph_edit_distance(predict, gold):
    # Retain edges only 
    predict = predict.astype(bool)
    gold = gold.astype(bool)
    
    ged = (predict != gold).sum(-1).sum(-1) / (2 * gold.sum(-1).sum(-1))
    
    reward = jnp.exp(1. - ged)
    
    return reward


# TODO: Implement bayesian graph edit distance
def bayesian_graph_edit_distance(predict, gold):
    pass 