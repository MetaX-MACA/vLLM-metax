# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import torch
import triton
import triton.language as tl


# -------- inv_rope ---------
@triton.jit
def _inv_rope_kernel(
    o_ptr,
    positions_ptr,
    cos_sin_cache_ptr,
    out_ptr,
    num_tokens,
    heads_per_group: tl.constexpr,
    o_stride_token,
    o_stride_head,
    cache_stride_pos,
    out_stride_group,
    out_stride_token,
    out_stride_d,
    CHUNKS_PER_HEAD: tl.constexpr,
    QUANT_GROUP_SIZE: tl.constexpr,
    ROPE_START: tl.constexpr,
    HALF_ROPE: tl.constexpr,
):
    # int64: stride multiply overflows int32 past num_tokens=32768.
    pid_token = tl.program_id(0).to(tl.int64)
    pid_gh = tl.program_id(1).to(tl.int64)

    g = pid_gh // heads_per_group
    head_in_group = pid_gh % heads_per_group
    global_head = pid_gh

    HEAD_DIM: tl.constexpr = CHUNKS_PER_HEAD * QUANT_GROUP_SIZE
    offsets = tl.arange(0, HEAD_DIM)

    input_base = o_ptr + pid_token * o_stride_token + global_head * o_stride_head

    x = tl.load(input_base + offsets).to(tl.float32)

    # RoPE starts at absolute offset:
    # default: nope_dim=448, rope_dim=64, head_dim=512
    # quant_group_size=128, chunks_per_head=4, rope_start=64
    # rope_abs_start = 3 * 128 + 64 = 448
    rope_abs_start: tl.constexpr = (CHUNKS_PER_HEAD - 1) * QUANT_GROUP_SIZE + ROPE_START

    pos = tl.load(positions_ptr + pid_token)
    cache_base = cos_sin_cache_ptr + pos * cache_stride_pos

    is_rope = offsets >= rope_abs_start
    rope_local = offsets - rope_abs_start

    x_partner = tl.load(
        input_base + (offsets ^ 1),
        mask=is_rope,
        other=0.0,
    ).to(tl.float32)

    cs_idx = tl.maximum(rope_local >> 1, 0)

    cos_v = tl.load(cache_base + cs_idx, mask=is_rope, other=1.0)
    sin_v = tl.load(cache_base + HALF_ROPE + cs_idx, mask=is_rope, other=0.0)

    # inverse RoPE:
    # even: x_even * cos + x_odd  * sin
    # odd : x_odd  * cos - x_even * sin
    x_add = x * cos_v + x_partner * sin_v
    x_sub = x * cos_v - x_partner * sin_v

    is_even = (rope_local & 1) == 0
    rotated = tl.where(is_even, x_add, x_sub)
    y = tl.where(is_rope, rotated, x)

    out_base = (
        out_ptr
        + g * out_stride_group
        + pid_token * out_stride_token
        + head_in_group * HEAD_DIM
    )

    tl.store(out_base + offsets * out_stride_d, y)


def inv_rope(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int = 448,
    rope_dim: int = 64,
    quant_group_size: int = 128,
) -> torch.Tensor:
    """Fused inverse RoPE without quantization.

    Args:
        o:
            Attention output, shape [num_tokens, num_heads, head_dim].
            dtype can be bf16/fp16/fp32.
        positions:
            Token positions, shape [num_tokens], int64/int32.
        cos_sin_cache:
            Precomputed cos||sin cache, shape [max_pos, rope_dim], fp32.
        n_groups:
            Number of output groups.
        heads_per_group:
            Number of heads per group.
        nope_dim:
            Non-RoPE dimensions per head.
        rope_dim:
            RoPE dimensions per head.
        quant_group_size:
            Kept only for compatibility with original layout assumptions.
            It no longer means quantization group size here.

    Returns:
        out:
            Shape [num_tokens, n_groups, heads_per_group * head_dim].
            dtype is same as input o.
    """
    num_tokens, num_heads, head_dim = o.shape

    assert num_heads == n_groups * heads_per_group
    assert head_dim == nope_dim + rope_dim
    assert head_dim % quant_group_size == 0
    assert nope_dim % quant_group_size == (quant_group_size - rope_dim)
    assert rope_dim % 2 == 0
    assert cos_sin_cache.shape[-1] == rope_dim
    assert cos_sin_cache.dtype == torch.float32

    d = heads_per_group * head_dim
    chunks_per_head = head_dim // quant_group_size

    out_buf = torch.empty(
        (n_groups, num_tokens, d),
        dtype=o.dtype,
        device=o.device,
    )

    grid = (num_tokens, n_groups * heads_per_group)

    _inv_rope_kernel[grid](
        o,
        positions,
        cos_sin_cache,
        out_buf,
        num_tokens,
        heads_per_group,
        o.stride(0),
        o.stride(1),
        cos_sin_cache.stride(0),
        out_buf.stride(0),
        out_buf.stride(1),
        out_buf.stride(2),
        CHUNKS_PER_HEAD=chunks_per_head,
        QUANT_GROUP_SIZE=quant_group_size,
        ROPE_START=nope_dim % quant_group_size,
        HALF_ROPE=rope_dim // 2,
        num_warps=1,
        num_stages=1,
    )

    return out_buf.transpose(0, 1)


# ---------------------------


def apply_rope_gptj_last_k(
    x: torch.Tensor, positions: torch.Tensor, cos_sin_cache: torch.Tensor
) -> torch.Tensor:
    """GPT-J-style (interleaved-pair) RoPE on the LAST rope_dim elements.

    x: [..., head_dim] float32
    positions: [num_tokens] int64 (positions[i] corresponds to x[i, ...])
    cos_sin_cache: [max_pos, rope_dim] float (cos|sin layout)

    Returns rotated x (same shape/dtype).
    """
    rope_dim = cos_sin_cache.shape[-1]
    half = rope_dim // 2
    head_dim = x.shape[-1]
    nope_dim = head_dim - rope_dim

    # Gather cos/sin for each token position: [num_tokens, rope_dim]
    cs = cos_sin_cache[positions].to(torch.float32)  # [N, rope_dim]
    cos = cs[..., :half]  # [N, half]
    sin = cs[..., half:]  # [N, half]

    # Reshape leading dims so we can broadcast: x shape [..., head_dim].
    # Bring token dim to front; assume x is [num_tokens, ..., head_dim].
    # We rely on positions being per-token and all other dims sharing the same pos.
    rope = x[..., nope_dim:].float()  # [..., rope_dim]
    # Make rope pairs: reshape last dim to [half, 2]
    shape = rope.shape
    rope = rope.reshape(*shape[:-1], half, 2)
    even = rope[..., 0]  # [..., half]
    odd = rope[..., 1]

    # Broadcast cos/sin over any heads dim in between.  cos/sin are [N, half].
    # Add singleton dims for intermediate axes.
    for _ in range(rope.ndim - 3):
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

    new_even = even * cos - odd * sin
    new_odd = even * sin + odd * cos
    rope_rotated = torch.stack((new_even, new_odd), dim=-1).reshape(shape)

    out = x.clone().float()
    out[..., nope_dim:] = rope_rotated
    return out.to(x.dtype)


def rmsnorm_no_weight(x: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm with no learnable weight, matching
    `RMSNorm(head_dim, has_weight=False)`."""
    orig_dtype = x.dtype
    xf = x.float()
    variance = xf.pow(2).mean(dim=-1, keepdim=True)
    return (xf * torch.rsqrt(variance + eps)).to(orig_dtype)


@triton.jit
def insert_k_cache_bf16_kernel(
    # Input tensors
    k_ptr,  # [num_tokens, 512] bf16
    slot_mapping_ptr,  # [num_tokens] int64
    # Output tensor
    k_cache_ptr,  # [num_blocks, block_stride] bf16 (flattened view)
    # Dimensions
    num_tokens,
    head_dim: tl.constexpr,  # 512
    cache_block_size: tl.constexpr,  # tokens per paged-cache block
    token_stride: tl.constexpr,  # head_dim elements per token
    block_stride: tl.constexpr,  # elements per block (block_size * head_dim)
):
    """
    Insert BF16 K tensor into paged K cache (no quantization).

    K Cache block layout (block_size tokens):
    - Each block contains block_size * head_dim bf16 elements
    - Each token occupies head_dim consecutive bf16 elements

    One program per token.
    """
    pid = tl.program_id(0)

    if pid >= num_tokens:
        return

    # Get slot mapping
    slot_idx = tl.load(slot_mapping_ptr + pid)
    if slot_idx == -1:
        return

    block_idx = slot_idx // cache_block_size
    pos_in_block = slot_idx % cache_block_size

    # Input pointer for this token
    input_row_ptr = k_ptr + pid * head_dim

    # Cache pointer as bf16
    # int64 to handle large block indices
    cache_block_ptr = k_cache_ptr + block_idx.to(tl.int64) * block_stride

    # Token pointer: each token occupies head_dim bf16 elements
    token_ptr = cache_block_ptr + pos_in_block * token_stride

    # Load and store all head_dim elements
    # Process in chunks of 16 for better memory efficiency
    for offset in range(0, head_dim, 16):
        offsets = offset + tl.arange(0, 16)
        mask = offsets < head_dim

        # Load bf16 values (Triton automatically handles bf16)
        vals = tl.load(input_row_ptr + offsets, mask=mask, other=0.0)

        # Store directly to cache (no quantization)
        tl.store(token_ptr + offsets, vals, mask=mask)


def insert_k_cache_bf16(
    k: torch.Tensor,  # [num_tokens, 512] bf16
    k_cache: torch.Tensor,  # [num_blocks, block_stride_elements] bf16
    slot_mapping: torch.Tensor,  # [num_tokens] int64
    block_size: int = 64,
):
    """
    Insert BF16 K tensor into paged K cache (no quantization).

    K Cache is a 2D tensor of bf16 with shape [num_blocks, block_stride_elements].
    Each block contains block_size * head_dim bf16 elements.
    Each token occupies head_dim consecutive bf16 elements within its block.

    Args:
        k: Input K tensor [num_tokens, head_dim] in bf16
        k_cache: Output cache [num_blocks, block_stride_elements] in bf16
        slot_mapping: Maps token index to cache slot [-1 means skip]
        block_size: Number of tokens per cache block
    """
    assert k.dim() == 2 and k.shape[1] == 512, (
        f"K must be [num_tokens, 512], got {k.shape}"
    )
    assert k.dtype == torch.bfloat16, f"K must be bf16, got {k.dtype}"
    assert k_cache.dtype == torch.bfloat16, f"K cache must be bf16, got {k_cache.dtype}"
    assert k_cache.dim() == 2, f"K cache must be 2D, got {k_cache.dim()}"

    HEAD_DIM = 512
    TOKEN_STRIDE = HEAD_DIM  # elements per token

    # NOTE: When using DP, slot_mapping.shape[0] can be less than k.shape[0] due to
    # padding. Always use slot_mapping.shape[0] as the token count.
    num_tokens = slot_mapping.shape[0]
    block_stride = k_cache.stride(0)  # elements per block

    # Verify cache has enough space
    expected_elements_per_block = block_size * HEAD_DIM
    assert k_cache.size(1) >= expected_elements_per_block, (
        f"Cache block too small: need {expected_elements_per_block} elements, "
        f"got {k_cache.size(1)}"
    )

    grid = (num_tokens,)

    insert_k_cache_bf16_kernel[grid](
        k,
        slot_mapping,
        k_cache,
        num_tokens,
        head_dim=HEAD_DIM,
        cache_block_size=block_size,
        token_stride=TOKEN_STRIDE,
        block_stride=block_stride,
    )


def fused_deepseek_v4_qnorm_rope_kv_rope_insert(
    q, kv, k_cache, slot_mapping, positions, cos_sin_cache, eps, bs
):
    torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_insert(
        q, kv, k_cache, slot_mapping, positions, cos_sin_cache, eps, bs
    )


# -- fused_indexer_q_rope_int8_quant -----


@triton.jit
def _round_to_nearest(x):
    # round-half-away-from-zero
    return tl.where(x >= 0.0, tl.floor(x + 0.5), tl.ceil(x - 0.5))


@triton.jit
def _quantize_int8(x, scale):
    q = tl.div_rn(x, scale)
    q = _round_to_nearest(q)
    q = tl.maximum(tl.minimum(q, 127.0), -127.0)
    return q.to(tl.int8)


@triton.jit
def _get_cos_sin(
    cos_sin_cache_ptr,
    cos_sin_cache_stride,
    pos,
    HALF_ROT_DIM: tl.constexpr,
):
    block = tl.arange(0, HALF_ROT_DIM)
    cos = tl.load(cos_sin_cache_ptr + pos * cos_sin_cache_stride + block)
    cos = cos.to(tl.float32)
    sin = tl.load(cos_sin_cache_ptr + pos * cos_sin_cache_stride + block + HALF_ROT_DIM)
    sin = sin.to(tl.float32)
    return cos, sin


@triton.jit
def _fused_indexer_q_rope_int8_quant_kernel(
    pos_ptr,
    # Index Q RoPE
    index_q_ptr,
    index_q_stride0,
    index_q_stride1,
    index_q_cos_sin_ptr,
    index_q_cos_sin_stride,
    INDEX_Q_HALF_ROT_DIM: tl.constexpr,
    # Index Q Quantize
    index_q_int8_ptr,
    index_q_int8_stride0,
    index_q_int8_stride1,
    INDEX_Q_HEAD_DIM: tl.constexpr,
    # Index weights
    index_weights_ptr,
    index_weights_stride,
    index_weights_softmax_scale,
    index_weights_head_scale,
    index_weights_out_ptr,
    index_weights_out_stride,
):
    INDEX_Q_ROT_DIM: tl.constexpr = 2 * INDEX_Q_HALF_ROT_DIM
    INDEX_Q_NOPE_DIM: tl.constexpr = INDEX_Q_HEAD_DIM - INDEX_Q_ROT_DIM
    tl.static_assert(INDEX_Q_NOPE_DIM >= 0)

    tok_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    pos = tl.load(pos_ptr + tok_idx)
    cos, sin = _get_cos_sin(
        index_q_cos_sin_ptr,
        index_q_cos_sin_stride,
        pos,
        INDEX_Q_HALF_ROT_DIM,
    )

    half_offset = tl.arange(0, INDEX_Q_HALF_ROT_DIM)
    base_ptr = index_q_ptr + tok_idx * index_q_stride0 + head_idx * index_q_stride1

    # GPT-J interleaved RoPE on dims [NOPE_DIM, HEAD_DIM)
    rot_base = base_ptr + INDEX_Q_NOPE_DIM
    x_even = tl.load(rot_base + half_offset * 2).to(tl.float32)
    x_odd = tl.load(rot_base + half_offset * 2 + 1).to(tl.float32)

    r_even = x_even * cos - x_odd * sin
    r_odd = x_odd * cos + x_even * sin

    # Keep same numeric convention as original FP8 path:
    # fp32 -> bf16 -> fp32 before absmax / quant.
    r_even = r_even.to(tl.bfloat16).to(tl.float32)
    r_odd = r_odd.to(tl.bfloat16).to(tl.float32)

    amax = tl.maximum(tl.max(tl.abs(r_even)), tl.max(tl.abs(r_odd)))

    if INDEX_Q_NOPE_DIM > 0:
        nope_offset = tl.arange(0, INDEX_Q_NOPE_DIM)
        x_nope = tl.load(base_ptr + nope_offset).to(tl.float32)
        amax = tl.maximum(amax, tl.max(tl.abs(x_nope)))

    # INT8 symmetric scale.
    index_q_scale = tl.where(amax > 0.0, amax / 127.0, 1.0)

    int8_base_ptr = (
        index_q_int8_ptr
        + tok_idx * index_q_int8_stride0
        + head_idx * index_q_int8_stride1
    )

    if INDEX_Q_NOPE_DIM > 0:
        tl.store(
            int8_base_ptr + nope_offset,
            _quantize_int8(x_nope, index_q_scale),
        )

    int8_rot_base = int8_base_ptr + INDEX_Q_NOPE_DIM

    tl.store(
        int8_rot_base + half_offset * 2,
        _quantize_int8(r_even, index_q_scale),
    )

    tl.store(
        int8_rot_base + half_offset * 2 + 1,
        _quantize_int8(r_odd, index_q_scale),
    )

    # INT8 weight-fold contract:
    #   q_int8 approximately represents q / index_q_scale
    #   q_original approximately q_int8 * index_q_scale
    #
    # Since the Q scale is not returned as a separate tensor, fold it into
    # index_weights_out, same as the old FP8 path.
    index_weights = tl.load(
        index_weights_ptr + tok_idx * index_weights_stride + head_idx
    ).to(tl.float32)

    index_weights *= index_q_scale
    index_weights *= index_weights_softmax_scale
    index_weights *= index_weights_head_scale

    tl.store(
        index_weights_out_ptr + tok_idx * index_weights_out_stride + head_idx,
        index_weights,
    )


def fused_indexer_q_rope_int8_quant(
    positions: torch.Tensor,
    index_q: torch.Tensor,
    index_q_cos_sin_cache: torch.Tensor,
    # Index weights
    index_weights: torch.Tensor,
    index_weights_softmax_scale: float,
    index_weights_head_scale: float,
) -> tuple[
    torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    torch.Tensor,
]:
    """Fused RoPE + quantize Q for the sparse indexer.

    Weight-fold semantics:

    INT8:
        q_int8     : (T, H, HEAD_DIM) int8, per-token-per-head scalar scale
                     is NOT stored — folded into weights below.
        weights_out = weights * q_scale * softmax_scale * head_scale

        Quantization:
            q_scale = max(abs(q_rope)) / 127
            q_int8  = clamp(round(q_rope / q_scale), -127, 127)

        Dequantization contract:
            q_rope ≈ q_int8.float() * q_scale

    """
    assert positions.ndim == 1
    assert index_q.ndim == 3
    assert index_q_cos_sin_cache.ndim == 2

    num_tokens = positions.shape[0]
    num_index_q_heads = index_q.shape[1]
    index_q_head_dim = index_q.shape[2]

    index_weights_out = torch.empty_like(index_weights, dtype=torch.float32)

    index_q_int8 = torch.empty_like(index_q, dtype=torch.int8)

    _fused_indexer_q_rope_int8_quant_kernel[(num_tokens, num_index_q_heads)](
        positions,
        index_q,
        index_q.stride(0),
        index_q.stride(1),
        index_q_cos_sin_cache,
        index_q_cos_sin_cache.stride(0),
        index_q_cos_sin_cache.shape[-1] // 2,
        index_q_int8,
        index_q_int8.stride(0),
        index_q_int8.stride(1),
        index_q_head_dim,
        index_weights,
        index_weights.stride(0),
        index_weights_softmax_scale,
        index_weights_head_scale,
        index_weights_out,
        index_weights_out.stride(0),
        num_warps=1,
    )

    return index_q_int8, index_weights_out


# ----------------------------------------


# -------------------gather_k_cache--------------------


@triton.jit
def _gather_k_cache_kernel(
    out_ptr,
    out_stride0: tl.constexpr,
    out_stride1: tl.constexpr,
    k_cache_ptr,
    seq_lens_ptr,
    block_table_ptr,
    offset: tl.constexpr,
    gather_lens_ptr,
    # constexpr
    max_blocks_per_seq: tl.constexpr,
    cache_block_size: tl.constexpr,
    head_size: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    token_worker_id = tl.program_id(1)
    dim_block_id = tl.program_id(2)

    num_token_workers = tl.num_programs(1)

    seq_len = tl.load(seq_lens_ptr + batch_idx)

    if gather_lens_ptr is not None:
        gather_len = tl.load(gather_lens_ptr + batch_idx)
    else:
        gather_len = seq_len

    start_pos = seq_len - gather_len

    dim_offsets = dim_block_id * BLOCK_D + tl.arange(0, BLOCK_D)
    dim_mask = dim_offsets < head_size

    for i in range(token_worker_id, gather_len, num_token_workers):
        pos = start_pos + i

        block_in_seq = pos // cache_block_size
        pos_in_block = pos % cache_block_size

        block_table_row_ptr = block_table_ptr + batch_idx * max_blocks_per_seq
        physical_block_idx = tl.load(block_table_row_ptr + block_in_seq)

        # k_cache layout:
        # [num_blocks, cache_block_size, head_size]
        k_ptr = (
            k_cache_ptr
            + physical_block_idx.to(tl.int64) * cache_block_size * head_size
            + pos_in_block * head_size
            + dim_offsets
        )

        out_ptr_cur = (
            out_ptr + batch_idx * out_stride0 + (offset + i) * out_stride1 + dim_offsets
        )

        vals = tl.load(k_ptr, mask=dim_mask, other=0.0)
        tl.store(out_ptr_cur, vals, mask=dim_mask)


def gather_k_cache(
    # [num_reqs, max_num_tokens, head_size]
    out: torch.Tensor,
    # [num_blocks, block_size, head_size]
    k_cache: torch.Tensor,
    # [num_reqs]
    seq_lens: torch.Tensor,
    # [num_reqs] or None
    gather_lens: torch.Tensor | None,
    # [num_reqs, max_blocks_per_seq]
    block_table: torch.Tensor,
    block_size: int,
    offset: int,
) -> None:
    num_reqs = seq_lens.shape[0]
    head_size = k_cache.shape[2]

    if gather_lens is not None:
        assert gather_lens.is_cuda
        assert gather_lens.shape == seq_lens.shape

    NUM_TOKEN_WORKERS = 128
    BLOCK_D = triton.next_power_of_2(head_size)

    _gather_k_cache_kernel[(num_reqs, NUM_TOKEN_WORKERS, 1)](
        out,
        out.stride(0),
        out.stride(1),
        k_cache,
        seq_lens,
        block_table,
        offset,
        gather_lens,
        max_blocks_per_seq=block_table.shape[-1],
        cache_block_size=block_size,
        head_size=head_size,
        BLOCK_D=BLOCK_D,
    )


# -------------------------------------------------------


@triton.jit
def _gather_k_kernel_bf16(
    out_ptr,
    out_stride0,
    out_stride1,
    k_cache_ptr,
    seq_lens_ptr,
    block_table_ptr,
    offset,
    gather_lens_ptr,
    # Constants
    max_blocks_per_seq: tl.constexpr,
    head_dim: tl.constexpr,  # 512
    cache_block_size: tl.constexpr,  # Block size in tokens
    token_stride: tl.constexpr,  # Number of bf16 elements per token (head_dim)
    block_stride: tl.constexpr,  # Total bytes per block (int32 stride)
):
    batch_idx = tl.program_id(0)
    worker_id = tl.program_id(1)
    num_workers = tl.num_programs(1)

    seq_len = tl.load(seq_lens_ptr + batch_idx)
    if gather_lens_ptr is not None:
        gather_len = tl.load(gather_lens_ptr + batch_idx)
    else:
        gather_len = seq_len
    start_pos = seq_len - gather_len

    for i in range(worker_id, gather_len, num_workers):
        pos = start_pos + i

        # Calculate block and position
        block_in_seq = pos // cache_block_size
        pos_in_block = pos % cache_block_size

        # Get physical block
        block_table_row_ptr = block_table_ptr + batch_idx * max_blocks_per_seq
        physical_block_idx = tl.load(block_table_row_ptr + block_in_seq)

        # Pointer to the start of the block (bf16 pointer)
        cache_block_ptr = k_cache_ptr + physical_block_idx.to(tl.int64) * block_stride
        cache_block_ptr = cache_block_ptr.to(tl.pointer_type(tl.bfloat16))

        # Pointer to the token's data in the block
        token_ptr = cache_block_ptr + pos_in_block * token_stride

        # Output pointer
        output_row_ptr = out_ptr + batch_idx * out_stride0 + (offset + i) * out_stride1

        # Load all head_dim bf16 values directly
        offsets = tl.arange(0, head_dim)
        mask = offsets < head_dim
        kv_vals = tl.load(token_ptr + offsets, mask=mask, other=0.0)

        # Store as bf16 (or keep as float32 if needed)
        tl.store(output_row_ptr + offsets, kv_vals.to(tl.bfloat16), mask=mask)


def gather_k_cache_bf16(
    # [num_reqs, max_num_tokens, head_size]
    out: torch.Tensor,
    # [num_blocks, block_size, head_size] as bf16
    k_cache: torch.Tensor,
    # [num_reqs]
    seq_lens: torch.Tensor,
    # [num_reqs]
    gather_lens: torch.Tensor | None,
    # [num_reqs, max_blocks_per_seq]
    block_table: torch.Tensor,
    block_size: int,
    offset: int,
) -> None:
    HEAD_DIM = 512  # Total head dimension
    TOKEN_STRIDE = HEAD_DIM  # Number of bf16 elements per token
    NUM_WORKERS = 128

    num_reqs = seq_lens.shape[0]

    _gather_k_kernel_bf16[(num_reqs, NUM_WORKERS)](
        out,
        out.stride(0),
        out.stride(1),
        k_cache,
        seq_lens,
        block_table,
        offset,
        gather_lens,
        max_blocks_per_seq=block_table.shape[-1],
        head_dim=HEAD_DIM,
        cache_block_size=block_size,
        token_stride=TOKEN_STRIDE,
        block_stride=k_cache.stride(0),  # Stride between blocks in elements (not bytes)
    )
