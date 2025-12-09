# copy from https://github.com/NVIDIA/TileGym/blob/main/src/tilegym/ops/cutile/rope.py

import cuda.tile as ct
import torch


# Type aliases for constants
ConstInt = ct.Constant[int]

@ct.kernel
def rope_kernel(
    q,
    k,
    cos,
    sin,
    cos_bs: ConstInt,
    seq_len: ConstInt,
    TILE_QH: ConstInt,
    TILE_KH: ConstInt,
    TILE_HD: ConstInt,
):
    # q size: (bsz, seq_len, num_q_heads, 2, head_dim)
    # k size: (bsz, seq_len, num_kv_heads, 2, head_dim)
    # cos size: (1, seq_len, *, head_dim) or (bsz, seq_len, , head_dim)
    cos_bs = cos.shape[0]

    bid = ct.bid(0)
    batch_idx = bid // seq_len
    row_idx = bid % seq_len
    cos_batch_idx = 0 if cos_bs == 1 else batch_idx

    # ####################################################################
    # Load cos and sin values
    # ####################################################################
    cos_row = ct.load(
        cos, index=(cos_batch_idx, row_idx, 0, 0), shape=(1, 1, 1, TILE_HD)
    ).reshape((1, TILE_HD))
    sin_row = ct.load(
        sin, index=(cos_batch_idx, row_idx, 0, 0), shape=(1, 1, 1, TILE_HD)
    ).reshape((1, TILE_HD))

    # ####################################################################
    # Process Q tensor
    # ####################################################################
    q_tile_1 = ct.load(
        q,
        index=(batch_idx, 0, row_idx, 0, 0),
        shape=(1, TILE_QH, 1, 1, TILE_HD),
    ).reshape((TILE_QH, TILE_HD))
    q_tile_2 = ct.load(
        q,
        index=(batch_idx, 0, row_idx, 1, 0),
        shape=(1, TILE_QH, 1, 1, TILE_HD),
    ).reshape((TILE_QH, TILE_HD))
    # y = [x1, x2] * [cos, cos] + [-x2, x1] * [sin, sin]
    new_q_tile_1 = q_tile_1 * cos_row - q_tile_2 * sin_row
    new_q_tile_2 = q_tile_2 * cos_row + q_tile_1 * sin_row
    ct.store(
        q,
        index=(batch_idx, 0, row_idx, 0, 0),
        tile=new_q_tile_1.reshape((1, TILE_QH, 1, 1, TILE_HD)).astype(
            q.dtype
        ),
    )
    ct.store(
        q,
        index=(batch_idx, 0, row_idx, 1, 0),
        tile=new_q_tile_2.reshape((1, TILE_QH, 1, 1, TILE_HD)).astype(
            q.dtype
        ),
    )

    # ####################################################################
    # Process K tensor
    # ####################################################################
    k_tile_1 = ct.load(
        k,
        index=(batch_idx, 0, row_idx, 0, 0),
        shape=(1, TILE_KH, 1, 1, TILE_HD),
    ).reshape((TILE_KH, TILE_HD))
    k_tile_2 = ct.load(
        k,
        index=(batch_idx, 0, row_idx, 1, 0),
        shape=(1, TILE_KH, 1, 1, TILE_HD),
    ).reshape((TILE_KH, TILE_HD))
    # y = [x1, x2] * [cos, cos] + [-x2, x1] * [sin, sin]
    new_k_tile_1 = k_tile_1 * cos_row - k_tile_2 * sin_row
    new_k_tile_2 = k_tile_2 * cos_row + k_tile_1 * sin_row
    ct.store(
        k,
        index=(batch_idx, 0, row_idx, 0, 0),
        tile=new_k_tile_1.reshape((1, TILE_KH, 1, 1, TILE_HD)).astype(
            k.dtype
        ),
    )
    ct.store(
        k,
        index=(batch_idx, 0, row_idx, 1, 0),
        tile=new_k_tile_2.reshape((1, TILE_KH, 1, 1, TILE_HD)).astype(
            k.dtype
        ),
    )


def next_power_of_2(n: int):
    """Return the smallest power of 2 greater than or equal to n"""
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n |= n >> 32
    n += 1
    return n


def rope_forward(q, k, cos, sin):
    """
    Apply rotary position encoding in forward pass

    Args:
        q: [bsz, n_q_head, seq_len, head_dim] - Query tensor
        k: [bsz, n_kv_head, seq_len, head_dim] - Key tensor
        cos: [1, seq_len, head_dim] or [bsz, seq_len, head_dim] - Cosine values
        sin: [1, seq_len, head_dim] or [bsz, seq_len, head_dim] - Sine values

    Returns:
        Query and key tensors with RoPE applied
    """
    # Calculate padded dimensions
    batch_size, n_q_head, seq_len, head_dim = q.shape
    print(q.shape, k.shape)
    n_kv_head = k.shape[1]
    q = q.reshape(batch_size, n_q_head, seq_len, 2, head_dim // 2)
    k = k.reshape(batch_size, n_kv_head, seq_len, 2, head_dim // 2)
    assert (
        cos.shape[-1] == head_dim // 2 or cos.shape[-1] == head_dim
    ), f"cos.shape[-1]: {cos.shape[-1]}, head_dim: {head_dim}"
    original_cos_shape = cos.shape
    original_sin_shape = sin.shape
    if cos.shape[-1] == head_dim:
        cos = cos.reshape(cos.shape[0], seq_len, 2, head_dim // 2)
        sin = sin.reshape(sin.shape[0], seq_len, 2, head_dim // 2)
    else:
        cos = cos.reshape(cos.shape[0], seq_len, 1, head_dim // 2)
        sin = sin.reshape(sin.shape[0], seq_len, 1, head_dim // 2)

    half_head_dim = q.shape[-1]
    TILE_HD = next_power_of_2(half_head_dim)
    TILE_QH = next_power_of_2(n_q_head)
    TILE_KH = next_power_of_2(n_kv_head)

    n_row = batch_size * seq_len
    grid = (n_row, 1, 1)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        rope_kernel,
        (
            q,
            k,
            cos,
            sin,
            cos.shape[0],
            seq_len,
            TILE_QH,
            TILE_KH,
            TILE_HD,
        ),
    )

    return (
        q.reshape(batch_size, n_q_head, seq_len, head_dim),
        k.reshape(batch_size, n_kv_head, seq_len, head_dim),
        cos.reshape(original_cos_shape),
        sin.reshape(original_sin_shape),
    )


class TileRopeFunction(torch.autograd.Function):
    """
    CUDA Tile implementation of the Rotary Positional Embedding (RoPE) operation. Please note that
    this implements the HuggingFace Llama & Mistral version, whose rotation matrix is slightly different
    than the original RoPE paper.

    Please find the corresponding HuggingFace implementation here:
    https://github.com/huggingface/transformers/blob/v4.40.2/src/transformers/models/llama/modeling_llama.py#L184

    For more details about the rotation matrix used here, please refer to:
    https://discuss.huggingface.co/t/is-llama-rotary-embedding-implementation-correct/44509/2
    """

    @staticmethod
    def forward(
        ctx, q, k, cos, sin, position_ids=None, unsqueeze_dim=1
    ):
        """
        q size: (bsz, n_q_head, seq_len, head_dim)
        k size: (bsz, n_kv_head, seq_len, head_dim)
        cos size: (1, seq_len, head_dim) or (bsz, seq_len, head_dim)
        sin size: (1, seq_len, head_dim) or (bsz, seq_len, head_dim)
        """
        q, k, cos, sin = rope_forward(q, k, cos, sin)
        ctx.save_for_backward(cos, sin)
        return q, k

    @staticmethod
    def backward(ctx, dq, dk):
        """
        Backward pass not yet implemented
        """
        raise NotImplementedError(
            "Backward pass is not implemented for TileRopeFunction"
        )


def apply_rope_base(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """
    Applies Rotary Positional Embedding (RoPE) operation to query and key states.

    Args:
        q: [bsz, n_q_head, seq_len, head_dim] - Query tensor
        k: [bsz, n_kv_head, seq_len, head_dim] - Key tensor
        cos: [1, seq_len, head_dim] or [bsz, seq_len, head_dim] - Cosine tensor
        sin: [1, seq_len, head_dim] or [bsz, seq_len, head_dim] - Sine tensor
        position_ids: Optional - Position IDs tensor, default None
        unsqueeze_dim: Optional - Dimension to unsqueeze, default 1

    Returns:
        Query and key tensor pair with RoPE applied
    """
    return TileRopeFunction.apply(
        q, k, cos, sin, position_ids, unsqueeze_dim
    )


def get_apply_rope_func(model='llama'):
    if model == 'llama':
        return apply_rope_base
    elif model == 'deepseek':
        def wrapper(q, k, freqs_cis):
            cos, sin = freqs_cis.real, freqs_cis.imag

            b, h, s, d = q.shape
            q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

            b, h, s, d = k.shape
            k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

            return apply_rope_base(q, k, cos, sin)

        return wrapper

    else:
        raise ValueError(f"Unsupported model: {model}")


def apply_rotary_emb_torch(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox_style: bool,
) -> torch.Tensor:
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    if is_neox_style:
        x1, x2 = torch.chunk(x, 2, dim=-1)
    else:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    if is_neox_style:
        return torch.cat((o1, o2), dim=-1)
    else:
        return torch.stack((o1, o2), dim=-1).flatten(-2)

# copy from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/rotary_embedding/base.py

def rope_vllm_torch(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor | None,
    head_size: int,
    rotary_dim: int,
    cos_sin_cache: torch.Tensor,
    is_neox_style: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """A PyTorch-native implementation of forward()."""
    positions = positions.flatten()
    num_tokens = positions.shape[0]
    cos_sin = cos_sin_cache.index_select(0, positions)
    cos, sin = cos_sin.chunk(2, dim=-1)

    query_shape = query.shape
    query = query.view(num_tokens, -1, head_size)
    query_rot = query[..., :rotary_dim]
    query_pass = query[..., rotary_dim:]
    query_rot = apply_rotary_emb_torch(query_rot, cos, sin, is_neox_style)
    query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

    # key may be None in some cases, e.g. cross-layer KV sharing
    if key is not None:
        key_shape = key.shape
        key = key.view(num_tokens, -1, head_size)
        key_rot = key[..., :rotary_dim]
        key_pass = key[..., rotary_dim:]
        key_rot = apply_rotary_emb_torch(key_rot, cos, sin, is_neox_style)
        key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
    return query, key

# llama3-8b config
max_seq_len, num_heads, head_size, num_key_value_heads = (8192, 32, 128, 8)

num_tokens = 1023

query_shape = (num_tokens, num_heads, head_size)
key_shape = (num_tokens, num_key_value_heads, head_size)
cos_sin_cache_shape = (max_seq_len, head_size)

DEVICE = torch.cuda.current_device()
query = torch.randn(*query_shape, device=DEVICE)
key = torch.randn(*key_shape, device=DEVICE)
cos_sin_cache = torch.randn(*cos_sin_cache_shape, device=DEVICE)
positions = torch.arange(num_tokens, device=DEVICE)
rotary_dim = head_size
torch_output = rope_vllm_torch(positions, query, key, head_size, rotary_dim, cos_sin_cache, True)
apply_rope = get_apply_rope_func(model='llama')

cos_sin = cos_sin_cache.index_select(0, positions)
cos, sin = cos_sin.chunk(2, dim=-1)
cutile_query = query.permute(1, 0, 2).unsqueeze(0)
cutile_key = key.permute(1, 0, 2).unsqueeze(0)
cutile_cos = cos.unsqueeze(0)
cutile_sin = sin.unsqueeze(0)
cutile_output = apply_rope(cutile_query, cutile_key, cutile_cos, cutile_sin)

torch_query_output = torch_output[0]
cutile_query_output = cutile_output[0].permute(0, 2, 1, 3).reshape(query_shape)

if torch.allclose(cutile_query_output, torch_query_output, atol=1e-2, rtol=0):
    print("✅ cuTile query and Torch match")
else:
    print("❌ cuTile query and Torch differ")

torch_key_output = torch_output[1]
cutile_key_output = cutile_output[1].permute(0, 2, 1, 3).reshape(key_shape)

if torch.allclose(cutile_key_output, torch_key_output, atol=1e-2, rtol=0):
    print("✅ cuTile key and Torch match")
else:
    print("❌ cuTile key and Torch differ")
