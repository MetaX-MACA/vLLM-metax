# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility wrapper for DeepGEMM API changes.

Users of vLLM should always import **only** these wrappers.
"""

import os
import functools

from enum import Enum
from typing import Any, Callable, NoReturn

import torch

import vllm.envs as envs
from vllm.utils.import_utils import has_deep_gemm
from vllm.utils.deep_gemm import _import_deep_gemm, is_deep_gemm_supported, logger
import vllm_metax.envs as mx_envs


class DeepGemmQuantScaleFMT(Enum):
    # Float32 scales in Float32 tensor
    FLOAT32 = 0
    # Compute float32 scales and ceil the scales to UE8M0.
    # Keep the scales in Float32 tensor.
    FLOAT32_CEIL_UE8M0 = 1
    # Compute float32 scales and ceil the scales to UE8M0.
    # Pack the scales into a int32 tensor where each int32
    # element contains 4 scale values.
    UE8M0 = 2

    @classmethod
    def init_oracle_cache(cls) -> None:
        """
        MetaX does not support E8M0 for now, so we will always use FLOAT32 for DeepGEMM.
        """
        cached = getattr(cls, "_oracle_cache", None)
        if cached is not None:
            return

        use_e8m0 = (
            envs.VLLM_USE_DEEP_GEMM_E8M0
            and is_deep_gemm_supported()
            and (_fp8_gemm_nt_impl is not None)
        )
        if not use_e8m0:
            cls._oracle_cache = cls.FLOAT32  # type: ignore
            return

        cls._oracle_cache = cls.UE8M0  # type: ignore

    @classmethod
    def from_oracle(cls) -> "DeepGemmQuantScaleFMT":
        """Return the pre-initialized oracle decision"""
        cached = getattr(cls, "_oracle_cache", None)
        assert cached is not None, "DeepGemmQuantScaleFMT oracle cache not initialized"
        return cached


@functools.cache
def is_deep_gemm_e8m0_used() -> bool:
    """Return `True` if vLLM is configured to use DeepGEMM "
    "E8M0 scale on a Hopper or Blackwell-class GPU.
    """
    if not is_deep_gemm_supported():
        logger.debug_once(
            "DeepGEMM E8M0 disabled: DeepGEMM not supported on this system."
        )
        return False

    _lazy_init()

    if _fp8_gemm_nt_impl is None:
        logger.info_once("DeepGEMM E8M0 disabled: _fp8_gemm_nt_impl not found")
        return False

    if envs.VLLM_USE_DEEP_GEMM_E8M0:
        logger.info_once("DeepGEMM E8M0 enabled on current platform.")
        return True

    logger.info_once("DeepGEMM E8M0 disabled on current configuration.")
    return False


def _missing(*_: Any, **__: Any) -> NoReturn:
    """Placeholder for unavailable DeepGEMM backend."""
    raise RuntimeError(
        "DeepGEMM backend is not available. Please install the `deep_gemm` "
        "package to enable BF16 kernels."
    )


_grouped_impl: Callable[..., Any] | None = None
_fp8_grouped_masked_impl: Callable[..., Any] | None = None
_fp8_gemm_nt_impl: Callable[..., Any] | None = None
_bf16_mqa_logits_impl: Callable[..., Any] | None = None
_bf16_paged_mqa_logits_impl: Callable[..., Any] | None = None
_get_num_blocks_paged_mqa_logits_metadata_impl: Callable[..., Any] | None = None
_int8_grouped_masked_impl: Callable[..., Any] | None = None
_int8_mqa_logits_impl: Callable[..., Any] | None = None
_int8_paged_mqa_logits_impl: Callable[..., Any] | None = None
_bf16_einsum: Callable[..., Any] | None = None
_tf32_hc_prenorm_gemm_impl: Callable[..., Any] | None = None
_fp8_mqa_logits_impl: Callable[..., Any] | None = None
_fp8_paged_mqa_logits_impl: Callable[..., Any] | None = None
_get_mk_alignment_for_contiguous_layout_impl: Callable[..., Any] | None = None
_transform_sf_into_required_layout_impl: Callable[..., Any] | None = None


# _layz_init for:
#   - bf16_mqa_logits
#   - bf16_paged_mqa_logits.
def _lazy_init() -> None:
    """Import deep_gemm and resolve symbols on first use."""
    global _grouped_impl, _fp8_grouped_masked_impl, _int8_grouped_masked_impl
    global _fp8_gemm_nt_impl
    global _bf16_mqa_logits_impl, _bf16_paged_mqa_logits_impl
    global _int8_mqa_logits_impl, _int8_paged_mqa_logits_impl
    global _get_num_blocks_paged_mqa_logits_metadata_impl
    global _bf16_einsum
    global _tf32_hc_prenorm_gemm_impl
    global _fp8_mqa_logits_impl, _fp8_paged_mqa_logits_impl
    global _get_mk_alignment_for_contiguous_layout_impl
    global _transform_sf_into_required_layout_impl

    # fast path
    if (
        _bf16_mqa_logits_impl is not None
        or _bf16_paged_mqa_logits_impl is not None
        or _get_num_blocks_paged_mqa_logits_metadata_impl is not None
        or _int8_mqa_logits_impl is not None
        or _int8_paged_mqa_logits_impl is not None
        or _bf16_einsum is not None
        or _tf32_hc_prenorm_gemm_impl is not None
        or _fp8_mqa_logits_impl is not None
        or _fp8_paged_mqa_logits_impl is not None
        or _get_mk_alignment_for_contiguous_layout_impl is not None
        or _transform_sf_into_required_layout_impl is not None
        or _fp8_gemm_nt_impl is not None
        or _grouped_impl is not None
        or _fp8_grouped_masked_impl is not None
    ):
        return

    if not has_deep_gemm():
        return

    # Set up deep_gemm cache path
    DEEP_GEMM_JIT_CACHE_ENV_NAME = "DG_JIT_CACHE_DIR"
    if not os.environ.get(DEEP_GEMM_JIT_CACHE_ENV_NAME, None):
        os.environ[DEEP_GEMM_JIT_CACHE_ENV_NAME] = os.path.join(
            envs.VLLM_CACHE_ROOT, "deep_gemm"
        )

    _dg = _import_deep_gemm()
    if _dg is None:
        return

    _bf16_mqa_logits_impl = getattr(_dg, "bf16_mqa_logits", None)
    _bf16_paged_mqa_logits_impl = getattr(_dg, "bf16_paged_mqa_logits", None)
    _get_num_blocks_paged_mqa_logits_metadata_impl = getattr(
        _dg, "get_num_blocks_paged_mqa_logits_metadata", None
    )
    _int8_mqa_logits_impl = getattr(_dg, "int8_mqa_logits", None)
    _int8_paged_mqa_logits_impl = getattr(_dg, "int8_paged_mqa_logits", None)
    _bf16_einsum = getattr(_dg, "einsum", None)
    _tf32_hc_prenorm_gemm_impl = getattr(_dg, "tf32_hc_prenorm_gemm", None)
    _fp8_mqa_logits_impl = getattr(_dg, "fp8_mqa_logits", None)
    _fp8_paged_mqa_logits_impl = getattr(_dg, "fp8_paged_mqa_logits", None)
    _fp8_gemm_nt_impl = getattr(_dg, "fp8_gemm_nt", None)
    _get_mk_alignment_for_contiguous_layout_impl = getattr(
        _dg, "get_mk_alignment_for_contiguous_layout", None
    )
    _transform_sf_into_required_layout_impl = getattr(
        _dg, "transform_sf_into_required_layout", None
    )
    _grouped_impl = getattr(_dg, "m_grouped_fp8_gemm_nt_contiguous", None)
    _fp8_grouped_masked_impl = getattr(_dg, "fp8_m_grouped_gemm_nt_masked", None)
    _int8_grouped_masked_impl = getattr(_dg, "int8_m_grouped_gemm_nt_masked", None)
    DeepGemmQuantScaleFMT.init_oracle_cache()


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
        clean_logits=clean_logits,
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
        q,
        kv,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        clean_logits,
        backend="triton" if mx_envs.VLLM_METAX_SUPPORTS_FP8 else "tilelang",
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
        clean_logits,
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
        backend="torch" if mx_envs.VLLM_METAX_SUPPORTS_FP8 else "mctlassEx",
    )


def fp8_mqa_logits(
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    clean_logits: bool,
) -> torch.Tensor:
    """Compute MQA logits for a single sequence without KV paging.

    Unified FP8/FP4 dispatch — the underlying DeepGEMM kernel takes
    ``q = (values, scales_or_None)`` where ``scales`` is None for FP8 Q
    (per-token scale is folded into ``weights``) and a packed block-scale
    tensor for MXFP4 Q.

    Args:
        q: Tuple ``(q_values, q_scale)``. FP8 path: q_values is [M, H, D]
            float8_e4m3fn and q_scale is None (per-token scale is folded
            into ``weights``). FP4 path: q_values is packed uint8 and
            q_scale is the companion block-scale tensor.
        kv: Tuple `(k_packed, k_scales)` — FP8 layout is [N, D]
            float8_e4m3fn plus fp32 scales [N]; FP4 layout is packed uint8.
        weights: weights of shape [M, H], dtype `torch.float32`.
        cu_seqlen_ks: Start indices (inclusive) for valid K per query
            position, shape [M], dtype int32.
        cu_seqlen_ke: End indices (exclusive) for valid K per query
            position, shape [M], dtype int32.
        clean_logits: Whether to clean the unfilled logits into `-inf`.

    Returns:
        Logits tensor of shape [M, N], dtype `torch.float32`.
    """
    _lazy_init()
    if _fp8_mqa_logits_impl is None:
        return _missing()
    return _fp8_mqa_logits_impl(
        q,
        kv,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        clean_logits=clean_logits,
    )


def fp8_paged_mqa_logits(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    schedule_metadata: torch.Tensor,
    max_model_len: int,
    clean_logits: bool,
) -> torch.Tensor:
    """Compute MQA logits using a paged KV-cache.

    Unified FP8/FP4 dispatch — the underlying DeepGEMM kernel takes
    ``q = (values, scales_or_None)``; pass ``(q_tensor, None)`` for the FP8
    path and ``(q_values, q_scale)`` for MXFP4.

    Args:
        q: Tuple ``(q_values, q_scale)``. FP8 path: q_values is
            [B, next_n, H, D] float8_e4m3fn and q_scale is None. FP4 path:
            q_values is packed uint8 and q_scale is the companion
            block-scale tensor.
        kv_cache: Paged KV-cache. FP8 layout is [num_blocks, block_size, 1,
            D+4], dtype `torch.uint8`, with the last 4 bytes per (block, pos)
            storing the float dequant scale.
        weights: Tensor of shape [B * next_n, H], dtype `torch.float32`.
        context_lens: Tensor of shape [B], dtype int32; effective context length
            for each batch element.
        block_tables: Tensor of shape [B, max_blocks], dtype int32; maps logical
            block indices to physical blocks in the paged cache.
        schedule_metadata: Returned by `get_paged_mqa_logits_metadata`;
            used to distribute work across SMs.
        max_model_len: Maximum sequence length used to size the logits output.
        clean_logits: Whether to clean the unfilled logits into `-inf`.

    Returns:
        Logits tensor of shape [B * next_n, max_model_len], dtype
        `torch.float32`.
    """
    _lazy_init()
    if _fp8_paged_mqa_logits_impl is None:
        return _missing()
    return _fp8_paged_mqa_logits_impl(
        q,
        kv_cache,
        weights,
        context_lens,
        block_tables,
        schedule_metadata,
        max_model_len,
        clean_logits=clean_logits,
    )


def fp8_gemm_nt(*args, **kwargs):
    _lazy_init()
    if _fp8_gemm_nt_impl is None:
        return _missing(*args, **kwargs)
    if "is_deep_gemm_e8m0_used" in kwargs:
        use_ue8m0 = kwargs["is_deep_gemm_e8m0_used"]
        del kwargs["is_deep_gemm_e8m0_used"]
    else:
        use_ue8m0 = is_deep_gemm_e8m0_used()
    return _fp8_gemm_nt_impl(*args, disable_ue8m0_cast=not use_ue8m0, **kwargs)


def get_mk_alignment_for_contiguous_layout() -> list[int]:
    _lazy_init()
    if _get_mk_alignment_for_contiguous_layout_impl is None:
        return _missing()
    mk_align_size = _get_mk_alignment_for_contiguous_layout_impl()
    return [mk_align_size, mk_align_size]


def transform_sf_into_required_layout(*args, **kwargs):
    _lazy_init()
    if _transform_sf_into_required_layout_impl is None:
        return _missing(*args, **kwargs)
    return _transform_sf_into_required_layout_impl(
        *args, disable_ue8m0_cast=not is_deep_gemm_e8m0_used(), **kwargs
    )


def m_grouped_fp8_gemm_nt_contiguous(*args, **kwargs):
    _lazy_init()
    if _grouped_impl is None:
        return _missing(*args, **kwargs)
    return _grouped_impl(
        *args, disable_ue8m0_cast=not is_deep_gemm_e8m0_used(), **kwargs
    )


def fp8_m_grouped_gemm_nt_masked(*args, **kwargs):
    _lazy_init()
    if _fp8_grouped_masked_impl is None:
        return _missing(*args, **kwargs)
    return _fp8_grouped_masked_impl(*args, **kwargs)


def int8_m_grouped_gemm_nt_masked(*args, **kwargs):
    _lazy_init()
    if _int8_grouped_masked_impl is None:
        return _missing(*args, **kwargs)
    return _int8_grouped_masked_impl(*args, **kwargs)


def dummy():
    pass


__all__ = [
    "bf16_mqa_logits",
    "bf16_paged_mqa_logits",
    "get_num_blocks_paged_mqa_logits_metadata",
    "is_deep_gemm_supported",
    "int8_mqa_logits",
    "int8_paged_mqa_logits",
    "bf16_einsum",
    "tf32_hc_prenorm_gemm",
    "fp8_mqa_logits",
    "fp8_paged_mqa_logits",
    "get_mk_alignment_for_contiguous_layout",
    "transform_sf_into_required_layout",
    "m_grouped_fp8_gemm_nt_contiguous",
    "fp8_m_grouped_gemm_nt_masked",
]
