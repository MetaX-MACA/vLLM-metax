# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Model-local MiniMax preprocessing across installed MACA operator ABIs."""

import torch

# Import upstream registrations before inspecting the actually selected binary.
from vllm import _custom_ops as _ops  # noqa: F401

# NOTE(metax): Fail early for an unknown native ABI rather than retrying a
# possibly mutating operator after a TypeError in the token execution path.
_ARGUMENTS = tuple(
    argument.name
    for argument in torch.ops._C.fused_minimax_m3_qknorm_rope_kv_insert.default._schema.arguments
)
_HAS_KV_SCALES = _ARGUMENTS[-2:] == ("kv_k_scale", "kv_v_scale")
_BASE_ARGUMENTS = (
    "qkv",
    "q_norm_weight",
    "k_norm_weight",
    "cos_sin_cache",
    "positions",
    "num_heads",
    "num_kv_heads",
    "rotary_dim",
    "eps",
    "index_q_norm_weight",
    "index_k_norm_weight",
    "num_index_heads",
    "slot_mapping",
    "index_slot_mapping",
    "kv_cache",
    "index_cache",
    "block_size",
    "q_out",
    "index_q_out",
    "kv_cache_dtype",
    "skip_index_branch",
    "q_fp8_out",
    "q_fp8_scale",
)
if _ARGUMENTS not in (_BASE_ARGUMENTS, _BASE_ARGUMENTS + ("kv_k_scale", "kv_v_scale")):
    raise RuntimeError(f"Unsupported MiniMax preprocessing ABI: {_ARGUMENTS}")


def fused_minimax_m3_qknorm_rope_kv_insert(
    qkv: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    rotary_dim: int,
    eps: float,
    index_q_norm_weight: torch.Tensor | None = None,
    index_k_norm_weight: torch.Tensor | None = None,
    num_index_heads: int = 0,
    slot_mapping: torch.Tensor | None = None,
    index_slot_mapping: torch.Tensor | None = None,
    kv_cache: torch.Tensor | None = None,
    index_cache: torch.Tensor | None = None,
    block_size: int = 0,
    q_out: torch.Tensor | None = None,
    index_q_out: torch.Tensor | None = None,
    kv_cache_dtype: str = "auto",
    skip_index_branch: bool = False,
    q_fp8_out: torch.Tensor | None = None,
    q_fp8_scale: float = 1.0,
    kv_k_scale: torch.Tensor | None = None,
    kv_v_scale: torch.Tensor | None = None,
) -> None:
    """Fused MiniMax-M3 attention pre-processing (in-place).

    Applies Gemma RMSNorm + partial NeoX RoPE to ``qkv`` in place. ``qkv`` is a
    single fused tensor:

    - dense layer (``num_index_heads == 0``): ``[q | k | v]``;
    - sparse layer (``num_index_heads > 0``): ``[q | k | v | index_q |
      index_k]`` — the index branch is read straight out of ``qkv``.

    When ``kv_cache`` is given (sparse serving), also scatter-inserts the
    normed/roped k & v into the paged KV cache by ``slot_mapping`` and the
    index key into ``index_cache`` by ``index_slot_mapping``. ``kv_cache_dtype``
    selects the cache storage/conversion path. If
    ``index_slot_mapping`` is omitted, ``slot_mapping`` is used for both caches.

    If ``q_out`` / ``index_q_out`` (contiguous ``[N, nq*128]`` / ``[N,
    niq*128]``) are given, the normed/roped q / index_q are written there
    instead of in place — folding the de-interleave into this kernel's store so
    callers skip a separate ``.contiguous()`` copy before the SM100 sparse
    attention's flat TMA descriptor.

    If ``q_fp8_out`` is given, the same normalized q is also written in FP8
    E4M3 using ``q_fp8_scale`` as its dequantization scale. Without ``q_out``,
    q is then written only in FP8 and the q slice of ``qkv`` is left as is.

    ``kv_cache_dtype="nvfp4"`` quantizes k/v into the packed HND
    ``[num_blocks, 2 * num_kv_heads, block_size, 72]`` uint8 cache using the
    device dequantization scales ``kv_k_scale``/``kv_v_scale``. Slot
    ``2 * head + side`` (K = 0, V = 1) holds that head's E2M1 data followed by
    its E4M3 block scales, so every head is one contiguous run of the page.

    When ``skip_index_branch`` is true, sparse rows still keep their packed
    ``[index_q | index_k]`` tail, but the kernel only processes the main q/k/v
    branches and main KV cache. This is used by MiniMax-M3 index-topk reuse
    layers that consume top-k block ids selected by an earlier sparse layer.
    """
    args = (
        qkv,
        q_norm_weight,
        k_norm_weight,
        cos_sin_cache,
        positions,
        num_heads,
        num_kv_heads,
        rotary_dim,
        eps,
        index_q_norm_weight,
        index_k_norm_weight,
        num_index_heads,
        slot_mapping,
        index_slot_mapping,
        kv_cache,
        index_cache,
        block_size,
        q_out,
        index_q_out,
        kv_cache_dtype,
        skip_index_branch,
        q_fp8_out,
        q_fp8_scale,
    )
    # NOTE(metax): MCOPLIB 0.4.14 exposes the pre-NVFP4-KV 23-argument ABI;
    # source kernels expose v0.32's 25 arguments. Inspect the loaded operator,
    # not the package version or USE_PRECOMPILED_KERNEL preference.
    if _HAS_KV_SCALES:
        torch.ops._C.fused_minimax_m3_qknorm_rope_kv_insert(
            *args, kv_k_scale, kv_v_scale
        )
    else:
        if (
            kv_k_scale is not None
            or kv_v_scale is not None
            or kv_cache_dtype == "nvfp4"
        ):
            raise ValueError(
                "The loaded MiniMax kernel does not support NVFP4 KV scales"
            )
        torch.ops._C.fused_minimax_m3_qknorm_rope_kv_insert(*args)
