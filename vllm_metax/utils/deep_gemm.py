# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility wrapper for DeepGEMM API changes.

Users of vLLM should always import **only** these wrappers.
"""

from __future__ import annotations

import functools
import importlib
import os
from typing import Any, Callable, NoReturn

import torch

import vllm.envs as envs
from vllm.utils.import_utils import has_deep_gemm


def is_deep_gemm_supported() -> bool:
    """Return `True` if DeepGEMM is supported on the current platform.
    Currently, only Hopper and Blackwell GPUs are supported.
    """
    return envs.VLLM_USE_DEEP_GEMM and has_deep_gemm()


def _missing(*_: Any, **__: Any) -> NoReturn:
    """Placeholder for unavailable DeepGEMM backend."""
    raise RuntimeError(
        "DeepGEMM backend is not available. Please install the `deep_gemm` "
        "package to enable BF16 kernels."
    )


_bf16_mqa_logits_impl: Callable[..., Any] | None = None
_bf16_paged_mqa_logits_impl: Callable[..., Any] | None = None


def _lazy_init() -> None:
    """Import deep_gemm and resolve symbols on first use."""
    global _bf16_mqa_logits_impl, _bf16_paged_mqa_logits_impl

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


@functools.cache
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


__all__ = [
    "bf16_mqa_logits",
    "bf16_paged_mqa_logits",
    "is_deep_gemm_supported",
]
