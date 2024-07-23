import numpy as np
import jax.numpy as jnp
import jax
import haiku as hk
from transformers import BertModel, BertConfig
from dp_gfn.utils.pretrains import split_into_heads


class PretrainedWeights(object):
    def __init__(self, path_to_pretrained):
        self.path_to_pretrained = path_to_pretrained

        pretrained_model = BertModel.from_pretrained(path_to_pretrained)
        self.weights = dict(pretrained_model.named_parameters())

    def __getitem__(self, path_to_weight):
        if isinstance(path_to_weight, list):
            path = ".".join(path_to_weight)
        else:
            path = path_to_weight

        path = path.replace("/", ".")
        path = path.replace("layer_", "layer.")

        return self.weights[path].detach().numpy()

    def __str__(self) -> str:
        return str(self.weights.keys())


PRETRAINED_WEIGHTS = PretrainedWeights("bert-base-uncased")
CONFIG = BertConfig("bert-base-uncased")


class Embeddings(hk.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def __call__(self, input_ids=None, token_type_ids=None, training=False, **kwargs):
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
            eps=self.config["layer_norm_eps"],
            scale_init=hk.initializers.Constant(
                PRETRAINED_WEIGHTS["embeddings.LayerNorm.weight"]
            ),
            offset_init=hk.initializers.Constant(
                PRETRAINED_WEIGHTS["embeddings.LayerNorm.bias"]
            ),
            name="LayerNorm",
        )(embeddings)

        if training: 
            embeddings = hk.dropout(
                hk.next_rng_key(), rate=self.config["hidden_dropout_prob"]
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
            vocab_size=self.config["vocab_size"],
            embed_dim=self.config["hidden_size"],
            w_init=hk.initializers.Constant(
                PRETRAINED_WEIGHTS["embeddings.word_embeddings.weight"]
            ),
        )(flat_input_ids)

        token_embeddings = jnp.reshape(
            flat_input_embeddings,
            [input_ids.shape[0], input_ids.shape[1], self.config["hidden_size"]],
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
            PRETRAINED_WEIGHTS["embeddings.position_embeddings.weight"].shape,
            init=hk.initializers.Constant(
                PRETRAINED_WEIGHTS["embeddings.position_embeddings.weight"]
            ),
        )

        start = self.offset
        end = start + self.config["max_position_embeddings"]

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
            vocab_size=self.config["type_vocab_size"],
            embed_dim=self.config["hidden_size"],
            w_init=hk.initializers.Constant(
                PRETRAINED_WEIGHTS["embeddings.token_type_embeddings.weight"]
            ),
        )(flat_token_type_ids)

        token_type_embeddings = jnp.reshape(
            flat_token_type_embeddings,
            [
                token_type_ids.shape[0],
                token_type_ids.shape[1],
                self.config["hidden_size"],
            ],
        )

        return token_type_embeddings


class EncoderLayer(hk.Module):
    def __init__(self, config, layer_num):
        super().__init__(name=f"encoder_layer_{layer_num}")
        self.config = config
        self.layer_num = layer_num

    def __call__(self, x, mask, training=False):
        # Feeding inputs through a multi-head attention operation
        # i.e. linear mapping -> multi-head attention -> residual connection -> LayerNorm
        attention_output = Attention(self.config, self.layer_num)(
            x, mask, training=training
        )

        # Inter-mediate (higher dimension)
        intermediate_output = hk.Linear(
            output_size=self.config["intermediate_size"],
            w_init=hk.initializers.Constant(
                PRETRAINED_WEIGHTS[
                    f"encoder.layer.{self.layer_num}.intermediate.dense.weight"
                ].transpose()
            ),
            b_init=hk.initializers.Constant(
                PRETRAINED_WEIGHTS[
                    f"encoder.layer.{self.layer_num}.intermediate.dense.bias"
                ]
            ),
            name="intermediate",
        )(attention_output)

        # TODO: Usage of approximation?
        if self.config["hidden_act"] == "gelu":
            intermediate_output = jax.nn.gelu(intermediate_output)
        else:
            raise Exception("Hidden activation not supported")

        output = Output(self.config, self.layer_num)(
            intermediate_output, attention_output, training=training
        )

        return output


class Attention(hk.Module):
    def __init__(self, config, layer_num):
        super().__init__()
        self.config = config
        self.layer_num = layer_num

    def __call__(self, x, mask, training=False):
        query = split_into_heads(
            hk.Linear(
                output_size=self.config["hidden_size"],
                w_init=hk.initializers.Constant(
                    PRETRAINED_WEIGHTS[
                        f"encoder.layer.{self.layer_num}.attention.self.query.weight"
                    ].transpose()
                ),
                b_init=hk.initializers.Constant(
                    PRETRAINED_WEIGHTS[
                        f"encoder.layer.{self.layer_num}.attention.self.query.bias"
                    ]
                ),
                name="query",
            )(x),
            self.config["num_attention_heads"],
        )

        key = split_into_heads(
            hk.Linear(
                output_size=self.config["hidden_size"],
                w_init=hk.initializers.Constant(
                    PRETRAINED_WEIGHTS[
                        f"encoder.layer.{self.layer_num}.attention.self.key.weight"
                    ].transpose()
                ),
                b_init=hk.initializers.Constant(
                    PRETRAINED_WEIGHTS[
                        f"encoder.layer.{self.layer_num}.attention.self.key.bias"
                    ]
                ),
                name="key",
            )(x),
            self.config["num_attention_heads"],
        )

        value = split_into_heads(
            hk.Linear(
                output_size=self.config["hidden_size"],
                w_init=hk.initializers.Constant(
                    PRETRAINED_WEIGHTS[
                        f"encoder.layer.{self.layer_num}.attention.self.value.weight"
                    ].transpose()
                ),
                b_init=hk.initializers.Constant(
                    PRETRAINED_WEIGHTS[
                        f"encoder.layer.{self.layer_num}.attention.self.value.bias"
                    ]
                ),
                name="value",
            )(x),
            self.config["num_attention_heads"],
        )

        # Scaled-dot Production Self Attention
        # b: batch
        # s: query_seq_len (source)
        # t: key_seq_len (target)
        # n: num_attention_heads
        # h: hidden_size_per_head
        attention_logits = jnp.einsum("bsnh, btnh -> bnst", query, key)  # Q.K^T
        attention_logits /= np.sqrt(query.shape[-1])  # dot_product / sqrt(d_k)

        # Masking
        attention_logits += jnp.reshape(
            (1 - mask) * -1e9, [mask.shape[0], 1, 1, mask.shape[1]]
        )

        attention_weights = jax.nn.softmax(
            attention_logits, axis=-1
        )  # Softmax(Q.K^T / sqrt(d_k))
        per_head_attention_output = jnp.einsum(
            "btnh, bnst -> bsnh", value, attention_weights
        )

        attention_output = jnp.reshape(
            per_head_attention_output,
            [
                per_head_attention_output.shape[0],
                per_head_attention_output.shape[1],
                -1,
            ],
        )

        # Output dense
        attention_output = hk.Linear(
            output_size=self.config["hidden_size"],
            w_init=hk.initializers.Constant(
                PRETRAINED_WEIGHTS[
                    f"encoder.layer.{self.layer_num}.attention.output.dense.weight"
                ].transpose()
            ),
            b_init=hk.initializers.Constant(
                PRETRAINED_WEIGHTS[
                    f"encoder.layer.{self.layer_num}.attention.output.dense.bias"
                ]
            ),
            name="output_dense",
        )(attention_output)

        if training:
            attention_output = hk.dropout(
                rng=hk.next_rng_key(),
                rate=self.config["attention_probs_dropout_prob"],
                x=attention_output,
            )(attention_output)

        # Add & Norm
        attention_output = attention_output + x
        attention_output = hk.LayerNorm(
            axis=-1,
            create_scale=True,
            create_offset=True,
            scale_init=hk.initializers.Constant(
                PRETRAINED_WEIGHTS[
                    f"encoder.layer.{self.layer_num}.attention.output.LayerNorm.weight"
                ]
            ),
            offset_init=hk.initializers.Constant(
                PRETRAINED_WEIGHTS[
                    f"encoder.layer.{self.layer_num}.attention.output.LayerNorm.bias"
                ]
            ),
            name="output_LayerNorm",
        )(attention_output)

        return attention_output


class Output(hk.Module):
    def __init__(self, config, layer_num):
        super().__init__()
        self.config = config
        self.layer_num = layer_num

    def __call__(self, intermediate_output, attention_output, training=False):
        output = hk.Linear(
            output_size=self.config["hidden_size"],
            w_init=hk.initializers.Constant(
                PRETRAINED_WEIGHTS[
                    f"encoder.layer.{self.layer_num}.output.dense.weight"
                ].transpose()
            ),
            b_init=hk.initializers.Constant(
                PRETRAINED_WEIGHTS[f"encoder.layer.{self.layer_num}.output.dense.bias"]
            ),
        )(intermediate_output)

        # Add & Norm
        output = output + attention_output
        output = hk.LayerNorm(
            axis=-1,
            create_offset=True,
            create_scale=True,
            scale_init=hk.initializers.Constant(
                PRETRAINED_WEIGHTS[
                    f"encoder.layer.{self.layer_num}.output.LayerNorm.weight"
                ]
            ),
            offset_init=hk.initializers.Constant(
                PRETRAINED_WEIGHTS[
                    f"encoder.layer.{self.layer_num}.output.LayerNorm.bias"
                ]
            ),
        )(output)

        return output


class BertModel(hk.Module):
    def __init__(self, config, name="BertModel"):
        super().__init__(name=name)
        self.config = config

    def __call__(
        self, input_ids=None, token_type_ids=None, attention_mask=None, training=False
    ):
        x = Embeddings(self.config)(
            input_ids=input_ids, token_type_ids=token_type_ids, training=training
        )
        mask = attention_mask.astype(jnp.float32)

        for layer_num in range(self.config["num_hidden_layers"]):
            x = EncoderLayer(self.config, layer_num)(x, mask, training=training)

        return x


def get_bert_token_embeddings_fn(
    self, input_ids, token_type_ids, attention_mask, training=False
):
    config = self.config.model.pref_encoder.pretrained_path
    token_embeddings = BertModel(config)(
        input_ids, token_type_ids, attention_mask, training=training
    )

    return token_embeddings
