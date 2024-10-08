import jax.numpy as jnp


def token_embeddings_to_word_embeddings(
    tokens, token_embeddings, batch_idx, agg_func="mean", max_word_length=160
):
    word_embeddings = []
    start, end = 0, 0

    for word_idx in set(tokens.word_ids(batch_idx)):
        # If token is not a valid word then break
        if word_idx is None and end != 0:
            break

        # TODO: Address [CLS] token case

        start, end = tokens.word_to_tokens(batch_idx, word_idx)

        try:
            if agg_func == "first":
                word_embedding = token_embeddings[batch_idx, start]
            elif agg_func == "last":
                word_embedding = token_embeddings[batch_idx, end - 1]
            else:
                word_embedding = getattr(jnp, agg_func)(
                    token_embeddings[batch_idx, start:end], axis=0
                )
        except:
            raise NotImplementedError

        word_embeddings.append(word_embedding)

    word_embeddings = jnp.stack(word_embeddings)
    # Padding to max_word_length
    word_embeddings = jnp.concatenate(
        [
            word_embeddings,
            token_embeddings[
                batch_idx, end : max_word_length - word_embeddings.shape[0] + end
            ],
        ]
    )

    return word_embeddings


def batch_token_embeddings_to_batch_word_embeddings(
    tokens, token_embeddings, agg_func="mean", max_word_length=160
):
    batch_size = token_embeddings.shape[0]
    batch_word_embeddings = []

    for batch_idx in range(batch_size):
        batch_word_embeddings.append(
            token_embeddings_to_word_embeddings(
                tokens=tokens,
                token_embeddings=token_embeddings,
                batch_idx=batch_idx,
                agg_func=agg_func,
                max_word_length=max_word_length,
            )
        )

    batch_word_embeddings = jnp.stack(batch_word_embeddings)

    return batch_word_embeddings


def get_pretrained_parameters(pretrained_weights, keyword="."):
    res = []

    for key in pretrained_weights.weights.keys():
        if keyword in key:
            res.append(key)

    return res


def split_into_heads(x, num_heads):
    return jnp.reshape(
        x,
        [
            x.shape[0],  # batch_size
            x.shape[1],  # seq_len
            num_heads,
            x.shape[2] // num_heads,
        ],
    )


def create_position_ids_from_input_ids(input_ids, padding_idx=1):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.

    Args:
        input_ids: jnp.ndarray
        padding_idx: int

    Returns: jnp.ndarray
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = (input_ids != padding_idx).astype("i4")

    if mask.ndim > 2:
        mask = mask.reshape((-1, mask.shape[-1]))
        incremental_indices = jnp.cumsum(mask, axis=1).astype("i4") * mask
        incremental_indices = incremental_indices.reshape(input_ids.shape)
    else:
        incremental_indices = jnp.cumsum(mask, axis=1).astype("i4") * mask

    return incremental_indices.astype("i4") + padding_idx
