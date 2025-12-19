# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This file contains ops for ViT attention to be compatible with torch.compile
as there are operations here not supported by torch.compile (for instance,
`.item()` in flash attention)

Using these ops and wrapping vision blocks with `torch.compile` can speed up
throughput in vision models by ~5% relative on H100, and improve token
latencies by ~7% (see qwen2_5_vl for example usage)

To use these ops, you must have a recent version of PyTorch installed (>= 2.4.0)
"""

# -----------------------------------
# Use Maca version of flash attention

import einops
import torch

from vllm.utils.torch_utils import direct_register_custom_op


def flash_attn_maxseqlen_wrapper(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: torch.Tensor,
    batch_size: int,
    is_rocm_aiter: bool,
) -> torch.Tensor:
    # -----------------------------------
    # Meca version of flash attention
    from vllm_metax.attention.utils.fa_utils import flash_attn_varlen_func

    q, k, v = (einops.rearrange(x, "b s ... -> (b s) ...") for x in [q, k, v])
    output = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen.item(),
        max_seqlen_k=max_seqlen.item(),
        dropout_p=0.0,
        causal=False,
    )
    context_layer = einops.rearrange(
        output, "(b s) h d -> s b (h d)", b=batch_size
    ).contiguous()
    return context_layer


def flash_attn_maxseqlen_wrapper_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: torch.Tensor,
    batch_size: int,
    is_rocm_aiter: bool,
) -> torch.Tensor:
    b, s, h, d = q.shape
    return torch.empty((s, b, h * d), dtype=q.dtype, device=q.device)


direct_register_custom_op(
    op_name="maca_flash_attn_maxseqlen_wrapper",
    op_func=flash_attn_maxseqlen_wrapper,
    fake_impl=flash_attn_maxseqlen_wrapper_fake,
)


def vit_flash_attn_wrapper(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: torch.Tensor,
    batch_size: int,
    is_rocm_aiter: bool,
) -> torch.Tensor:
    return torch.ops.vllm.maca_flash_attn_maxseqlen_wrapper(
        q, k, v, cu_seqlens, max_seqlen, batch_size, is_rocm_aiter
    )
