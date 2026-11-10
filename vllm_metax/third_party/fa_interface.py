# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Normalize the installed MetaX FA2/FA3 APIs to vLLM's inference contract."""

from __future__ import annotations

from typing import Any

import torch

from vllm_metax.utils.math_util import is_power_of_2

try:
    import flash_attn

    FA2_UNAVAILABLE_REASON = None
except ImportError as e:
    FA2_UNAVAILABLE_REASON = str(e)
FA2_AVAILABLE = FA2_UNAVAILABLE_REASON is None

try:
    import flash_attn_3

    FA3_UNAVAILABLE_REASON = None
except ImportError as e:
    FA3_UNAVAILABLE_REASON = str(e)
FA3_AVAILABLE = FA3_UNAVAILABLE_REASON is None

FA4_UNAVAILABLE_REASON = "MetaX FlashAttention 4 is not implemented"
FA4_AVAILABLE = False
DEFAULT_FA_VERSION = 2

FA2_SUPPORTED_HDIM = (32, 64, 96, 128, 160, 192, 256, 512)
FA3_SUPPORTED_HDIM_FP8 = (64, 96, 128, 192, 256)
FA3_SUPPORTED_HDIM_BF16 = (64, 128, 192, 256)


def is_fa_version_supported(fa_version: int | None) -> bool:
    if fa_version == 2:
        return FA2_AVAILABLE
    if fa_version == 3:
        return FA3_AVAILABLE
    return False


def fa_version_unsupported_reason(fa_version: int | None) -> str | None:
    if fa_version == 2:
        return FA2_UNAVAILABLE_REASON
    if fa_version == 3:
        return FA3_UNAVAILABLE_REASON
    if fa_version == 4:
        return FA4_UNAVAILABLE_REASON
    return f"Unsupported FlashAttention version: {fa_version}"


def is_fa_geometry_supported(
    fa_version: int,
    head_size: int | None = None,
    head_size_v: int | None = None,
    *,
    is_fp8: bool = False,
    block_size: int | None = None,
) -> bool:
    """Check exact kernel dimensions; callers must account for their own padding."""
    if block_size is not None and not is_power_of_2(block_size):
        return False
    sizes: tuple[int, ...]
    if fa_version == 2 and not is_fp8:
        sizes = FA2_SUPPORTED_HDIM
    elif fa_version == 3:
        sizes = FA3_SUPPORTED_HDIM_FP8 if is_fp8 else FA3_SUPPORTED_HDIM_BF16
    else:
        return False
    if head_size is None:
        return head_size_v is None or head_size_v in sizes
    if head_size_v is None:
        head_size_v = head_size
    return (head_size == head_size_v and head_size in sizes) or (
        head_size,
        head_size_v,
    ) == (192, 128)


def _normalize_result(result: Any, out: torch.Tensor | None, return_lse: bool) -> Any:
    """Return a tensor or (output, LSE), honoring the caller's output buffer."""
    if return_lse and (not isinstance(result, tuple) or len(result) < 2):
        raise RuntimeError("MetaX FlashAttention did not return softmax LSE")
    result_out = result[0] if isinstance(result, tuple) else result
    if out is not None:
        if out.shape != result_out.shape or out.dtype != result_out.dtype:
            raise ValueError(
                "MetaX FlashAttention out must match output shape and dtype"
            )
        if out.data_ptr() != result_out.data_ptr():
            out.copy_(result_out)
        result_out = out
    return (result_out, result[1]) if return_lse else result_out


def _get_cu_seqlens_k(cu_seqlens_k, seqused_k):
    if cu_seqlens_k is not None:
        return cu_seqlens_k
    result = torch.empty(
        seqused_k.shape[0] + 1, dtype=torch.int32, device=seqused_k.device
    )
    result[:1].zero_()
    torch.cumsum(seqused_k, dim=0, dtype=torch.int32, out=result[1:])
    return result


def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x



def flash_attn_varlen_func(
    q,
    k,
    v,
    max_seqlen_q,
    cu_seqlens_q,
    max_seqlen_k,
    cu_seqlens_k=None,
    seqused_k=None,
    q_v=None,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=None,
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    block_table=None,
    return_softmax_lse=False,
    out=None,
    scheduler_metadata=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    num_splits: int = 0,
    output_scale=None,
    fa_version: int = DEFAULT_FA_VERSION,
    s_aux=None,
    cp_world_size=1,
    cp_rank=0,
    cp_tot_seqused_k=None,
    mask_mod=None,
    block_sparse_tensors=None,
    aux_tensors=None,
    aux_tensor_leading_dims=None,
    dynamic_causal=None,
    **kwargs: Any,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Run packed or paged attention, returning LSE in [heads, total_q] layout.

    Exactly one of cu_seqlens_k and seqused_k supplies per-request KV lengths.
    FA2 supports paged K/V; FA3 requires packed K/V and ignores block_table.
    On a paged call, cumulative lengths describe logical lengths, not cache offsets.
    Unsupported optional arguments are ignored for unified API callers.
    return_attn_probs is a legacy alias for return_softmax_lse; probabilities are
    not part of the normalized inference result.
    """
    if (cu_seqlens_k is None) == (seqused_k is None):
        raise ValueError("Provide exactly one of cu_seqlens_k and seqused_k")
    paged_kv = fa_version == 2 and block_table is not None
    if q.ndim != 3 or k.ndim != (4 if paged_kv else 3) or v.ndim != k.ndim:
        raise ValueError(
            "MetaX varlen requires packed Q/K/V, or paged K/V with FA2 block_table"
        )
    return_lse = return_softmax_lse or return_attn_probs
    real_window_size = (-1, -1) if window_size is None else tuple(window_size)
    if len(real_window_size) != 2:
        raise ValueError("window_size must contain two elements")
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]

    if fa_version == 2:
        result = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=_get_cu_seqlens_k(cu_seqlens_k, seqused_k),
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=real_window_size,
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
            return_attn_probs=return_lse,
            block_table=block_table,
            s_aux=s_aux,
        )
    else:
        result = flash_attn_3.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=_get_cu_seqlens_k(cu_seqlens_k, seqused_k),
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            qv=q_v,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=real_window_size,
            softcap=softcap,
            num_splits=num_splits,
            deterministic=deterministic,
            return_attn_probs=return_lse,
        )
    return _normalize_result(result, out, return_lse)


def flash_attn_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k=None,
    v=None,
    q_v=None,
    cache_seqlens=None,
    cache_batch_idx=None,
    cache_leftpad=None,
    block_table=None,
    cu_seqlens_q=None,
    cu_seqlens_k_new=None,
    max_seqlen_q=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    softmax_scale=None,
    causal=False,
    window_size=None,
    softcap=0.0,
    alibi_slopes=None,
    scheduler_metadata=None,
    num_splits: int = 0,
    s_aux=None,
    return_softmax_lse=False,
    out=None,
    fa_version: int = DEFAULT_FA_VERSION,
    rotary_cos=None,
    rotary_sin=None,
    rotary_interleaved=True,
    rotary_seqlens=None,
    attention_chunk: int = 0,
    pack_gqa=None,
    sm_margin: int = 0,
    **kwargs: Any,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Run cache attention; FA2 uses batched Q, FA3 also accepts flattened Q."""
    real_window_size = (-1, -1) if window_size is None else tuple(window_size)
    if len(real_window_size) != 2:
        raise ValueError("window_size must contain two elements")
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    common = dict(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        k=k,
        v=v,
        rotary_cos=rotary_cos,
        rotary_sin=rotary_sin,
        rotary_interleaved=rotary_interleaved,
        cache_seqlens=cache_seqlens,
        cache_batch_idx=cache_batch_idx,
        cache_leftpad=cache_leftpad,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=real_window_size,
        softcap=softcap,
        num_splits=num_splits,
    )
    if fa_version == 2:
        result = flash_attn.flash_attn_with_kvcache(
            **common,
            block_table=block_table,
            alibi_slopes=alibi_slopes,
            s_aux=s_aux,
        )
    else:
        result = flash_attn_3.flash_attn_with_kvcache(
            **common,
            qv=q_v,
            page_table=block_table,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k_new=cu_seqlens_k_new,
            max_seqlen_q=max_seqlen_q,
            rotary_seqlens=rotary_seqlens,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            attention_chunk=attention_chunk,
            pack_gqa=pack_gqa,
            sm_margin=sm_margin,
            return_softmax_lse=return_softmax_lse,
        )
    return _normalize_result(result, out, return_softmax_lse and fa_version == 3)
