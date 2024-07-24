import jax.numpy as jnp


def reward_fn(predict, gold):
    ged = graph_edit_distance(predict, gold)
    reward = jnp.exp(1. - ged)
    
    return reward


def graph_edit_distance(predict, gold):
    predict = predict.astype(bool)
    gold = gold.astype(bool)

    return (predict != gold).sum(-1).sum(-1)
    

def bayesian_graph_edit_distance(predict, gold):
    pass 