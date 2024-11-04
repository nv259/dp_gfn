import haiku as hk
import jax.numpy as jnp



class Biaffine(hk.Module):
    def __init__(self, num_tags, init_scale=None, intermediate_dim=128):
        super().__init__()

        self.num_tags = num_tags
        self.init_scale = init_scale
        self.intermediate_dim = intermediate_dim

    def __call__(self, head, dep):
        w_init = (
            hk.initializers.RandomNormal()
            if self.init_scale is None
            else hk.initializers.VarianceScaling(self.init_scale)
        )

        W = hk.get_parameter(
            name="W",
            shape=(self.num_tags, self.intermediate_dim, self.intermediate_dim),
            init=w_init,
        )
        Wh = hk.get_parameter(
            name="Wh", shape=(self.intermediate_dim, self.num_tags), init=w_init
        )
        Wd = hk.get_parameter(
            name="Wd", shape=(self.intermediate_dim, self.num_tags), init=w_init
        )
        b = hk.get_parameter(name="b", shape=(self.num_tags,), init=jnp.ones)

        lab_head = hk.Linear(self.intermediate_dim, w_init=w_init)(head)
        lab_dep = hk.Linear(self.intermediate_dim, w_init=w_init)(dep)

        # Biaffine layer
        head_score = lab_head @ Wh
        dep_score = lab_dep @ Wd
        arc_score = lab_dep @ W @ lab_head

        lab_score = arc_score + head_score + dep_score + b

        return lab_score


def label_score_fn(head, dep, num_tags):
    lab_score = Biaffine(num_tags=num_tags)(head, dep)

    return lab_score
