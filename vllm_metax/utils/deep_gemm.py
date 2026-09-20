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
    kv: torch.Tensor,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
) -> torch.Tensor:
    """BF16 Q [M,H,D], KV [N,D]; sum weighted ReLU scores over heads.

    The installed MetaX DeepGEMM non-paged BF16 kernel is numerically
    incorrect on C500. Use our GPU fallback until that kernel is validated.
    """
    from vllm_metax.v1.attention.ops.bf16_mqa_logits import bf16_mqa_logits as fallback

    return fallback(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke)


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
    """BF16 paged logits, including caches with gaps between physical pages.

    Q is [B,S,H,D], KV is BF16 [pages,page_size,1,D] with no scale tail.
    Context lengths are [B] or [B,S]; weights are [B*S,H].
    """
    if not kv_cache_bf16.is_contiguous():
        from vllm_metax.v1.attention.ops.bf16_mqa_logits import (
            bf16_mqa_logits as fallback,
        )

        return fallback(
            q_bf16,
            kv_cache_bf16,
            weights,
            context_lens,
            context_lens,
            block_table=block_tables,
            max_model_len=max_model_len,
            clean_logits=clean_logits,
        )
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
    """Compute INT8 MQA logits for a single sequence without KV paging.

    Args:
        q: INT8 query tensor with shape [M, H, D]. Its scale is folded into
            ``weights`` by the caller.
        kv: Tuple ``(k_int8, k_scales)`` where ``k_int8`` has shape [N, D]
            and ``k_scales`` has shape [N] (or [N, 1]) with dtype FP32.
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


def _split_int8_paged_kv_cache(
    kv_cache: torch.Tensor,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a block-segregated INT8 cache without losing its block stride."""
    if kv_cache.dtype not in (torch.uint8, torch.int8):
        raise TypeError(
            f"INT8 paged MQA requires a uint8 or int8 KV cache; got {kv_cache.dtype}."
        )
    if kv_cache.ndim != 4:
        raise ValueError(
            "INT8 paged MQA expects KV cache shape "
            "[num_blocks, block_size, 1, head_dim + 4]; "
            f"got {tuple(kv_cache.shape)}."
        )

    num_blocks, block_size, num_kv_heads, fused_width = kv_cache.shape
    if num_kv_heads != 1 or fused_width != head_dim + 4:
        raise ValueError(
            "INT8 paged MQA expects one KV head and a four-byte FP32 scale "
            f"tail per token; got shape={tuple(kv_cache.shape)}, "
            f"head_dim={head_dim}."
        )
    if kv_cache.stride(-1) != 1 or kv_cache.stride(1) != fused_width:
        raise ValueError(
            "INT8 paged MQA requires each physical cache page to contain "
            "contiguous logical bytes; "
            f"got stride={kv_cache.stride()}."
        )

    # Physical page layout is block-segregated, not token-interleaved:
    # [block_size * head_dim INT8 values][block_size FP32 scales].
    # Keep stride(0), which may span a packed multi-layer block slab in 0.27.
    fused_flat = kv_cache.view(torch.int8).view(num_blocks, block_size * fused_width)
    value_bytes = fused_flat[:, : block_size * head_dim]
    scale_bytes = fused_flat[:, block_size * head_dim :]
    values = value_bytes.view(num_blocks, block_size, head_dim)
    scales = scale_bytes.view(num_blocks, block_size, 4).view(torch.float32)
    return values, scales.squeeze(-1)


def _int8_paged_mqa_logits_strided(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    schedule_metadata: torch.Tensor,
    max_model_len: int,
    clean_logits: bool,
) -> torch.Tensor:
    """Run the DeepGEMM Triton kernel on a packed, block-strided KV view."""
    from deep_gemm.kernels.mqa.int8.triton_paged import (
        int8_paged_mqa_logits_kernel,
    )

    if q.ndim != 4 or q.dtype != torch.int8 or not q.is_contiguous():
        raise ValueError(
            "INT8 paged MQA requires contiguous q with shape "
            f"[batch, next_n, heads, head_dim]; got {tuple(q.shape)}, "
            f"dtype={q.dtype}, stride={q.stride()}."
        )
    if not q.is_cuda or not kv_cache.is_cuda or kv_cache.device != q.device:
        raise ValueError("INT8 paged MQA requires device-resident Q and KV tensors.")
    if (
        not weights.is_cuda
        or weights.device != q.device
        or weights.dtype not in (torch.float32, torch.bfloat16)
        or not weights.is_contiguous()
    ):
        raise ValueError(
            "INT8 paged MQA weights must be contiguous FP32 or BF16 tensors."
        )
    if (
        not context_lens.is_cuda
        or not block_tables.is_cuda
        or not schedule_metadata.is_cuda
        or context_lens.device != q.device
        or block_tables.device != q.device
        or schedule_metadata.device != q.device
        or context_lens.dtype != torch.int32
        or not context_lens.is_contiguous()
        or schedule_metadata.dtype != torch.int32
        or not schedule_metadata.is_contiguous()
        or block_tables.dtype != torch.int32
    ):
        raise ValueError(
            "INT8 paged MQA context lengths, block tables, and schedule "
            "metadata must use INT32; context lengths and schedule metadata "
            "must be contiguous."
        )
    if max_model_len <= 0:
        raise ValueError(f"max_model_len must be positive; got {max_model_len}.")

    batch_size, next_n, num_heads, head_dim = q.shape
    kv_values, kv_scales = _split_int8_paged_kv_cache(kv_cache, head_dim)
    num_blocks, block_size, _ = kv_values.shape

    rows = batch_size * next_n
    logits = torch.full(
        (rows, max_model_len),
        float("-inf") if clean_logits else 0.0,
        device=q.device,
        dtype=torch.float32,
    )
    if context_lens.ndim == 2:
        context_caps = context_lens[:, next_n - 1].contiguous()
        q_limits = (context_lens.view(-1) - 1).contiguous()
    else:
        context_caps = context_lens
        next_offsets = torch.arange(next_n, device=q.device, dtype=torch.int32)
        q_limits = (context_caps[:, None] - next_n + next_offsets[None, :]).reshape(-1)

    if num_heads % 64 == 0:
        block_h = 64
    elif num_heads % 32 == 0:
        block_h = 32
    else:
        block_h = 64
    num_worker_blocks = schedule_metadata.shape[0] - 1

    int8_paged_mqa_logits_kernel[(num_worker_blocks,)](
        q,
        kv_values,
        kv_scales,
        weights,
        context_caps,
        q_limits,
        block_tables,
        schedule_metadata,
        logits,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        kv_values.stride(0),
        kv_values.stride(1),
        kv_values.stride(2),
        kv_scales.stride(0),
        kv_scales.stride(1),
        weights.stride(0),
        weights.stride(1),
        block_tables.stride(0),
        block_tables.stride(1),
        context_caps.stride(0),
        q_limits.stride(0),
        schedule_metadata.stride(0),
        schedule_metadata.stride(1),
        logits.stride(0),
        logits.stride(1),
        batch_size,
        num_blocks,
        block_tables.shape[1],
        max_model_len,
        num_worker_blocks,
        NEXT_N=next_n,
        NUM_HEADS=num_heads,
        HEAD_DIM=head_dim,
        BLOCK_KV=block_size,
        BLOCK_H=block_h,
        BLOCK_GROUP=4,
        NEED_HEAD_MASK=num_heads % block_h != 0,
        num_warps=4,
        num_stages=3,
        pipeline="cpasync",
        scenario="storeCoalesce",
    )
    return logits


def int8_paged_mqa_logits(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    schedule_metadata: torch.Tensor,
    max_model_len: int,
    clean_logits: bool = True,
) -> torch.Tensor:
    """Compute INT8 MQA logits using a block-segregated paged KV cache.

    Args:
        q: Contiguous INT8 query tensor with shape [B, next_n, H, D]. Its
            per-token/head scale is folded into ``weights`` by the caller.
        kv_cache: Paged cache with logical shape
            [num_blocks, block_size, 1, D+4] and dtype uint8/int8. Each page
            stores all ``block_size * D`` INT8 values first, followed by all
            ``block_size`` FP32 scales. ``stride(0)`` may span a packed
            multi-layer block slab.
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
    # if not kv_cache.is_contiguous():
    #     return _int8_paged_mqa_logits_strided(
    #         q,
    #         kv_cache,
    #         weights,
    #         context_lens,
    #         block_tables,
    #         schedule_metadata,
    #         max_model_len,
    #         clean_logits,
    #     )
    return _int8_paged_mqa_logits_impl(
        q,
        kv_cache,
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
