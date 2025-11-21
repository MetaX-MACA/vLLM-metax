# 2025 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import vllm
from vllm_metax_plugin.patch.hook_registry import register_patch
from vllm.logger import init_logger

import torch
import triton
from vllm.attention.ops.blocksparse_attention.blocksparse_attention_kernel import _fwd_kernel_batch_inference

logger = init_logger(__name__)

def blocksparse_flash_attn_varlen_fwd(
        q,
        k,
        v,  # (#tokens, n_heads, head_size)
        cu_seqlens_k,
        cu_seqlens_q,
        sm_scale,
        sparse_layout,
        *,
        block_size=64,
        q_block_size=None,
        max_seqlen=None):
    # split q to blocks

    assert isinstance(sparse_layout, (list, tuple))

    _, n_heads, head_size = q.shape
    batch_size = cu_seqlens_k.size(0) - 1
    q_block_size = q_block_size or block_size

    assert q.dim() == k.dim() == v.dim() == 3
    assert q.size(1) % k.size(1) == 0
    assert q.size(2) == k.size(2)
    # TODO(linxihui): allow k, v to have different head_size
    assert k.shape == v.shape
    assert cu_seqlens_k.dim() == 1

    q_k_ratio = q.size(1) // k.size(1)

    if cu_seqlens_q is None:
        if q.size(0) == batch_size:  # decoding only
            cu_seqlens_q = torch.arange(
                0,
                batch_size + 1,
                dtype=cu_seqlens_k.dtype,
                device=cu_seqlens_k.device,
            )
        elif q.size(0) == k.size(0):
            cu_seqlens_q = cu_seqlens_k
        else:
            raise ValueError("cu_seqlens_q must be specified\
                    if it mix of prefilling and decoding.")
    else:
        assert cu_seqlens_k.size(0) == cu_seqlens_q.size(0)

    # switch to use cpu to avoid too many kernel launches when iterated over
    q_lens = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).cpu()
    k_lens = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).cpu()

    assert torch.logical_or(q_lens == 1, k_lens == q_lens).all(), (
        "length of q should either be 1 (decoding) or same as k (prefilling).")

    if max_seqlen:
        assert k_lens.max() <= max_seqlen

    n_blocks = (q_lens + q_block_size - 1) // q_block_size

    q_batch_ids = torch.tensor(
        [i for i, n in enumerate(n_blocks) for _ in range(n)],
        dtype=cu_seqlens_q.dtype,
        device=cu_seqlens_q.device,
    )
    q_start_sids = torch.tensor(
        [i * q_block_size for n in n_blocks for i in range(n)],
        dtype=cu_seqlens_q.dtype,
        device=cu_seqlens_q.device,
    )

    out = q.new_empty(q.shape)
    cu_seqlens_q = cu_seqlens_q.contiguous()
    cu_seqlens_k = cu_seqlens_k.contiguous()

    layout_crow_indices, layout_col_indices = sparse_layout
    block_d = triton.next_power_of_2(head_size)

    decoding_only = (q_lens == 1).all().item()
    grid = (len(q_start_sids), n_heads, 1)

    _fwd_kernel_batch_inference[grid](
        q,
        k,
        v,
        out,
        sm_scale,
        cu_seqlens_q[:-1],
        cu_seqlens_q[1:],
        cu_seqlens_k[:-1],
        cu_seqlens_k[1:],
        q_batch_ids,
        q_start_sids,
        0,
        *q.stride(),
        0,
        *k.stride(),
        0,
        *v.stride(),
        0,
        *out.stride(),
        layout_crow_indices,
        layout_col_indices,
        *layout_crow_indices.stride(),
        *layout_col_indices.stride(),
        q_k_ratio,
        HAS_BATCH_DIM=False,
        D_HEAD=head_size,
        BLOCK_M=q_block_size,
        BLOCK_N=block_size,
        BLOCK_D=block_d,
        BLOCK_M_LOADING=(16 if decoding_only else
                         q_block_size),  # smaller for decoding
        EVEN_D=block_d == head_size,
        num_warps=1 if decoding_only else 4,
        num_stages=1)

    return out

vllm.attention.ops.blocksparse_attention.blocksparse_attention_kernel.blocksparse_flash_attn_varlen_fwd = blocksparse_flash_attn_varlen_fwd
register_patch("vllm.attention.ops.blocksparse_attention.blocksparse_attention_kernel", "blocksparse_flash_attn_varlen_fwd", blocksparse_flash_attn_varlen_fwd)
