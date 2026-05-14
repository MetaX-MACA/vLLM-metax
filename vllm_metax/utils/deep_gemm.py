# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility wrapper for DeepGEMM API changes.

Users of vLLM should always import **only** these wrappers.
"""

from __future__ import annotations

import importlib
import os
from typing import Any, Callable, NoReturn

import torch

import vllm.envs as envs
from vllm.utils.import_utils import has_deep_gemm
from vllm.utils.deep_gemm import (
    is_deep_gemm_supported,
)


def _missing(*_: Any, **__: Any) -> NoReturn:
    """Placeholder for unavailable DeepGEMM backend."""
    raise RuntimeError(
        "DeepGEMM backend is not available. Please install the `deep_gemm` "
        "package to enable BF16 kernels."
    )


_bf16_mqa_logits_impl: Callable[..., Any] | None = None
_bf16_paged_mqa_logits_impl: Callable[..., Any] | None = None
_get_num_blocks_paged_mqa_logits_metadata_impl: Callable[..., Any] | None = None
_int8_mqa_logits_impl: Callable[..., Any] | None = None
_int8_paged_mqa_logits_impl: Callable[..., Any] | None = None
_bf16_einsum: Callable[..., Any] | None = None
_tf32_hc_prenorm_gemm_impl: Callable[..., Any] | None = None

# _layz_init for:
#   - bf16_mqa_logits
#   - bf16_paged_mqa_logits.
def _lazy_init() -> None:
    """Import deep_gemm and resolve symbols on first use."""
    global _fp8_mqa_logits_impl, _fp8_paged_mqa_logits_impl
    global _bf16_mqa_logits_impl, _bf16_paged_mqa_logits_impl
    global _int8_mqa_logits_impl, _int8_paged_mqa_logits_impl
    global _bf16_einsum
    global _get_num_blocks_paged_mqa_logits_metadata_impl
    global _int8_mqa_logits_impl, _int8_paged_mqa_logits_impl
    global _bf16_einsum
    global _tf32_hc_prenorm_gemm_impl

    # fast path
    if (
        _fp8_mqa_logits_impl is not None
        or _fp8_paged_mqa_logits_impl is not None
        or _bf16_mqa_logits_impl is not None
        or _bf16_paged_mqa_logits_impl is not None
        or _get_num_blocks_paged_mqa_logits_metadata_impl is not None
        or _int8_mqa_logits_impl is not None
        or _int8_paged_mqa_logits_impl is not None
        or _bf16_einsum is not None
        or _tf32_hc_prenorm_gemm_impl is not None
    ):
        return
    global \
        _bf16_mqa_logits_impl, \
        _bf16_paged_mqa_logits_impl, \
        _get_num_blocks_paged_mqa_logits_metadata_impl, \
        _int8_mqa_logits_impl, \
        _int8_paged_mqa_logits_impl, \
        _bf16_einsum

    if not has_deep_gemm():
        return

    # Set up deep_gemm cache path
    DEEP_GEMM_JIT_CACHE_ENV_NAME = "DG_JIT_CACHE_DIR"
    if not os.environ.get(DEEP_GEMM_JIT_CACHE_ENV_NAME, None):
        os.environ[DEEP_GEMM_JIT_CACHE_ENV_NAME] = os.path.join(
            envs.VLLM_CACHE_ROOT, "deep_gemm"
        )

    _dg = importlib.import_module("deep_gemm")

    _bf16_mqa_logits_impl = getattr(_dg, "bf16_mqa_logits", None)
    _bf16_paged_mqa_logits_impl = getattr(_dg, "bf16_paged_mqa_logits", None)
    _int8_mqa_logits_impl = getattr(_dg, "int8_mqa_logits", None)
    _int8_paged_mqa_logits_impl = getattr(_dg, "int8_paged_mqa_logits", None)
    _bf16_einsum = getattr(_dg, "einsum", None)
    _tf32_hc_prenorm_gemm_impl = getattr(_dg, "tf32_hc_prenorm_gemm", None)
    _get_num_blocks_paged_mqa_logits_metadata_impl = getattr(
        _dg, "get_num_blocks_paged_mqa_logits_metadata", None
    )

def get_num_blocks_paged_mqa_logits_metadata(num_sms: int) -> int:
    """Get scheduling metadata buffer size for paged MQA logits.

    Args:
        num_sms: Number of SMs available.

    Returns:
        Backend-specific tensor shape[0] consumed by `bf16_paged_mqa_logits` to
        schedule work across SMs.
    """
    _lazy_init()
    if _get_num_blocks_paged_mqa_logits_metadata_impl is None:
        return num_sms
    return _get_num_blocks_paged_mqa_logits_metadata_impl(num_sms)

def bf16_mqa_logits(
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
) -> torch.Tensor:
    """Compute FP8 MQA logits for a single sequence without KV paging.

    Args:
        q: Query tensor of shape [M, H, D]. Casted to
            `torch.float8_e4m3fn` by caller.
        kv: Tuple `(k_fp8, k_scales)` where `k_fp8` has shape [N, D] with
            dtype `torch.float8_e4m3fn` and `k_scales` has shape [N] (or
            [N, 1]) with dtype `torch.float32`.
        weights: weights of shape [M, H], dtype `torch.float32`.
        cu_seqlen_ks: Start indices (inclusive) for valid K per query position,
            shape [M], dtype int32.
        cu_seqlen_ke: End indices (exclusive) for valid K per query position,
            shape [M], dtype int32.

    Returns:
        Logits tensor of shape [M, N], dtype `torch.float32`.
    """
    _lazy_init()
    if _bf16_mqa_logits_impl is None:
        return _missing()
    return _bf16_mqa_logits_impl(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke)


def bf16_paged_mqa_logits(
    q_bf16: torch.Tensor,
    kv_cache_bf16: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    schedule_metadata: torch.Tensor,
    max_model_len: int,
    clean_logits: bool = True,
) -> torch.Tensor:
    """Compute BF16 MQA logits using paged KV-cache.

    Args:
        q_bf16: Query tensor of shape [B, next_n, H, D]. Casted to
            `torch.float16` by caller.
        kv_cache_bf16: Paged KV-cache in packed BF16+scale layout with shape
            [num_blocks, block_size, 1, D+4], dtype `torch.uint8`. The last
            4 bytes per (block,pos) store the `float` dequant scale.
        weights: Tensor of shape [B * next_n, H], dtype `torch.float32`.
        context_lens: Tensor of shape [B], dtype int32; effective context length
            for each batch element.
        block_tables: Tensor of shape [B, max_blocks], dtype int32; maps logical
            block indices to physical blocks in the paged cache.
        schedule_metadata: Returned by `get_paged_mqa_logits_metadata`;
            used to distribute work across SMs.
        max_model_len: Maximum sequence length used to size the logits output.

    Returns:
        Logits tensor of shape [B * next_n, max_model_len], dtype
        `torch.float32`.
    """
    _lazy_init()
    if _bf16_paged_mqa_logits_impl is None:
        return _missing()
    return _bf16_paged_mqa_logits_impl(
        q_bf16,
        kv_cache_bf16,
        weights,
        context_lens,
        block_tables,
        schedule_metadata,
        max_model_len,
        clean_logits=True,
    )


def int8_mqa_logits(
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    clean_logits: bool = True,
) -> torch.Tensor:
    """Compute FP8 MQA logits for a single sequence without KV paging.

    Args:
        q: Query tensor of shape [M, H, D]. Casted to
            `torch.float8_e4m3fn` by caller.
        kv: Tuple `(k_fp8, k_scales)` where `k_fp8` has shape [N, D] with
            dtype `torch.float8_e4m3fn` and `k_scales` has shape [N] (or
            [N, 1]) with dtype `torch.float32`.
        weights: weights of shape [M, H], dtype `torch.float32`.
        cu_seqlen_ks: Start indices (inclusive) for valid K per query position,
            shape [M], dtype int32.
        cu_seqlen_ke: End indices (exclusive) for valid K per query position,
            shape [M], dtype int32.

    Returns:
        Logits tensor of shape [M, N], dtype `torch.float32`.
    """
    _lazy_init()
    if _int8_mqa_logits_impl is None:
        return _missing()
    return _int8_mqa_logits_impl(
        q, kv, weights, cu_seqlen_ks, cu_seqlen_ke, clean_logits
    )


def int8_paged_mqa_logits(
    q_bf16: torch.Tensor,
    kv_cache_bf16: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    schedule_metadata: torch.Tensor,
    max_model_len: int,
    clean_logits: bool = True,
) -> torch.Tensor:
    """Compute BF16 MQA logits using paged KV-cache.

    Args:
        q_bf16: Query tensor of shape [B, next_n, H, D]. Casted to
            `torch.float16` by caller.
        kv_cache_bf16: Paged KV-cache in packed BF16+scale layout with shape
            [num_blocks, block_size, 1, D+4], dtype `torch.uint8`. The last
            4 bytes per (block,pos) store the `float` dequant scale.
        weights: Tensor of shape [B * next_n, H], dtype `torch.float32`.
        context_lens: Tensor of shape [B], dtype int32; effective context length
            for each batch element.
        block_tables: Tensor of shape [B, max_blocks], dtype int32; maps logical
            block indices to physical blocks in the paged cache.
        schedule_metadata: Returned by `get_paged_mqa_logits_metadata`;
            used to distribute work across SMs.
        max_model_len: Maximum sequence length used to size the logits output.

    Returns:
        Logits tensor of shape [B * next_n, max_model_len], dtype
        `torch.float32`.
    """
    _lazy_init()
    if _int8_paged_mqa_logits_impl is None:
        return _missing()
    return _int8_paged_mqa_logits_impl(
        q_bf16,
        kv_cache_bf16,
        weights,
        context_lens,
        block_tables,
        schedule_metadata,
        max_model_len,
        clean_logits=True,
    )


def bf16_einsum(*args, **kwargs):
    _lazy_init()
    if _bf16_einsum is None:
        return _missing(*args, **kwargs)
    return _bf16_einsum(*args, **kwargs)


def tf32_hc_prenorm_gemm(
    x: torch.Tensor,
    fn: torch.Tensor,
    out: torch.Tensor,
    sqrsum: torch.Tensor,
    num_split: int,
) -> torch.Tensor:
    """
    Perform the following computation:
        out = x.float() @ fn.T
        sqrsum = x.float().square().sum(-1)

    See the caller function for shape requirement
    """
    _lazy_init()
    if _tf32_hc_prenorm_gemm_impl is None:
        return _missing()
    return _tf32_hc_prenorm_gemm_impl(
        x,
        fn,
        out,
        sqrsum,
        num_split,
        "torch"
    )

__all__ = [
    "has_deep_gemm",
    "bf16_mqa_logits",
    "bf16_paged_mqa_logits",
    "is_deep_gemm_supported",
    "int8_mqa_logits",
    "int8_paged_mqa_logits",
    "bf16_einsum",
    "tf32_hc_prenorm_gemm",
    "get_num_blocks_paged_mqa_logits_metadata",
]
