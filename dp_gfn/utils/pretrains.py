import jax.numpy as jnp
import torch


def mean_token_embeddings(token_embeddings, word_ids, max_word_length):
    batch_size, _, embedding_dim = token_embeddings.shape
    mask = word_ids >= 0
    word_embeddings = torch.zeros(
        (batch_size, max_word_length, embedding_dim),
        dtype=token_embeddings.dtype,
        device=token_embeddings.device,
    )

    # Sum all tokens' embeddings for each word ids
    word_embeddings.scatter_add_(
        1,
        word_ids.clamp(min=0).unsqueeze(-1).expand(-1, -1, embedding_dim),
        token_embeddings * mask.unsqueeze(-1).float(),
    )

    subtoken_counts = torch.zeros_like(word_embeddings[:, :, 0])
    subtoken_counts.scatter_add_(1, word_ids.clamp(min=0), mask.float())
    word_embeddings = torch.where(
        subtoken_counts.unsqueeze(-1) > 0,
        word_embeddings / subtoken_counts.unsqueeze(-1),
        torch.zeros_like(word_embeddings),
    )

    return word_embeddings


def get_word_ids_from_tokens(tokens, mask_value=-100):
    word_ids_list = [tokens.word_ids(i) for i in range(2)]
    word_ids = torch.tensor(
        [
            [mask_value if val is None else val for val in word_ids]
            for word_ids in word_ids_list
        ]
    )
    
    return word_ids


def token_embeddings_to_word_embeddings(
    token_embeddings: torch.Tensor, 
    word_ids: torch.Tensor, 
    agg_func="mean", 
    max_word_length=160
):
    if agg_func == "first":
        pass
    elif agg_func == "last":
        pass
    elif agg_func == "mean":
        word_embeddings = mean_token_embeddings(token_embeddings, word_ids, max_word_length)
        
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
