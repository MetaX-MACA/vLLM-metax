# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Grouped block-32 FP8 output projection with FP32 scales.

Requires fp8_einsum and fused inverse RoPE support for block size 32.
Weight scales are decoded once during weight post-processing. The fused
quantizer produces power-of-two activation scales stored as FP32.
"""

from collections.abc import Callable

import torch
from torch import nn

from vllm.model_executor.kernels.linear.mxfp8.flashinfer import (
    FlashInferCutedslMxfp8LinearKernel,
)
from vllm.model_executor.layers.fusion.quant_activation import (
    QuantizedActivation,
    get_input_quant_key,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Dynamic
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import fp8_einsum

from vllm_metax.registry.quant_config.fp8 import MctlassFp8LinearMethod

from vllm_metax.models.deepseek_v4.metax.ops.o_proj import (
    deep_gemm_fp8_o_proj as vllm_fp8_o_proj,
)
from vllm_metax.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import (
    fused_inv_rope_fp8_quant,
)


def _fused_wo_a_max_tokens(layer: nn.Module) -> int:
    max_tokens = getattr(layer, "_fused_wo_a_token_limit", None)
    if max_tokens is None:
        # Additional token tiles only pay off for smaller grids.
        ctas_per_tile = layer.n_local_heads * (layer.o_lora_rank // 128)
        max_tokens = 96 if ctas_per_tile <= 128 else 32
        layer._fused_wo_a_token_limit = max_tokens
    return max_tokens


def _can_fuse_wo_a(layer: nn.Module) -> bool:
    """Whether the attention layer matches the fused WO-A kernel's layout."""
    method = getattr(layer.wo_b, "scheme", layer.wo_b.quant_method)
    return (
        current_platform.is_device_capability_family(100)
        # WO-A is FP8 with per-32 scales, and WO-B takes MXFP8 input directly.
        and layer._einsum_recipe == (1, 1, 32)
        and get_input_quant_key(layer.wo_b) == kMxfp8Dynamic
        and isinstance(
            getattr(method, "kernel", None), FlashInferCutedslMxfp8LinearKernel
        )
        # A group's heads form one portable (<= 8 CTA) cluster.
        and layer.n_local_heads // layer.n_local_groups <= 8
        and (layer.nope_head_dim, layer.rope_head_dim) == (448, 64)
        and layer.o_lora_rank % 128 == 0
    )


def register_dsv41_o_proj_warmup(layer: nn.Module) -> None:
    """Warm the fused WO-A token counts ``dsv41_o_proj`` can dispatch."""
    # Draft models load outside the warmup registry; their layers share the
    # target's attention shapes, so the target's registration covers them too.
    if _can_fuse_wo_a(layer):
        from .fused_wo_a import _FUSED_WO_A_KERNEL

        _FUSED_WO_A_KERNEL.register_warmup(
            max_tokens=_fused_wo_a_max_tokens(layer),
            n_groups=layer.n_local_groups,
            heads_per_group=layer.n_local_heads // layer.n_local_groups,
            o_lora_rank=layer.o_lora_rank,
        )


def dsv41_o_proj(
    layer: nn.Module,
    attn_out: torch.Tensor,
    positions: torch.Tensor,
    wo_b_proj: Callable[..., torch.Tensor] | None = None,
) -> torch.Tensor:
    """``deep_gemm_fp8_o_proj`` with WO-A fused for small SM100/SM103 batches;
    ``wo_b_proj`` replaces ``layer._wo_b_proj``."""
    wo_b_proj = wo_b_proj or layer._wo_b_proj
    o = attn_out[:, : layer.n_local_heads, :]
    cos_sin_cache = layer.rotary_emb.cos_sin_cache
    if 1 <= o.shape[0] <= _fused_wo_a_max_tokens(layer) and _can_fuse_wo_a(layer):
        from .fused_wo_a import _FUSED_WO_A_KERNEL

        q, scales = _FUSED_WO_A_KERNEL(
            x=o,
            positions=positions,
            rope=cos_sin_cache,
            weight=layer.wo_a.weight,
            weight_scale=layer.wo_a.weight_scale,
        )
        return wo_b_proj(
            QuantizedActivation(q, scales, o.dtype, q.shape, kMxfp8Dynamic)
        )
    return deep_gemm_fp8_o_proj(
        o,
        positions,
        cos_sin_cache,
        layer.wo_a,
        wo_b_proj,
        n_groups=layer.n_local_groups,
        heads_per_group=layer.n_local_heads // layer.n_local_groups,
        nope_dim=layer.nope_head_dim,
        rope_dim=layer.rope_head_dim,
        o_lora_rank=layer.o_lora_rank,
        einsum_recipe=layer._einsum_recipe,
        tma_aligned_scales=layer._tma_aligned_scales,
    )


def mctlass_fp8_o_proj(
    o,
    positions,
    cos_sin_cache,
    wo_a,
    wo_b,
    *,
    n_groups,
    heads_per_group,
    nope_dim,
    rope_dim,
    o_lora_rank,
):
    k = heads_per_group * (nope_dim + rope_dim)
    if n_groups < 1 or k % 32 or o_lora_rank % 32:
        raise ValueError(
            "Grouped block-32 projection requires positive groups and aligned K/D"
        )
    if wo_a.weight.shape != (n_groups * o_lora_rank, k):
        raise ValueError("Unexpected local wo_a weight shape")
    scale_rows = o_lora_rank // 32
    if wo_a.weight_scale.shape != (n_groups * scale_rows, k // 32):
        raise ValueError("Unexpected compact wo_a scale shape")

    o_fp8, o_scale = fused_inv_rope_fp8_quant(
        o,
        positions,
        cos_sin_cache,
        n_groups=n_groups,
        heads_per_group=heads_per_group,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
        quant_group_size=32,
        tma_aligned_scales=False,
    )
    if wo_a.weight_scale.dtype != torch.float32:
        raise ValueError("Block-32 einsum requires FP32 weight scales")
    weight = wo_a.weight.view(n_groups, o_lora_rank, k)
    weight_scale = wo_a.weight_scale.view(n_groups, scale_rows, k // 32)
    z = torch.empty(
        (o.shape[0], n_groups, o_lora_rank),
        device=o.device,
        dtype=wo_a.quant_method.fp8_linear.config.out_dtype,
    )
    fp8_einsum(
        "bhr,hdr->bhd",
        (o_fp8, o_scale),
        (weight, weight_scale),
        z,
        recipe=(1, 32, 32),
    )
    return wo_b(z.flatten(1))


def deep_gemm_fp8_o_proj(
    o,
    positions,
    cos_sin_cache,
    wo_a,
    wo_b,
    *,
    n_groups,
    heads_per_group,
    nope_dim,
    rope_dim,
    o_lora_rank,
    einsum_recipe,
    tma_aligned_scales,
):
    if not isinstance(wo_a.quant_method, MctlassFp8LinearMethod):
        return vllm_fp8_o_proj(
            o,
            positions,
            cos_sin_cache,
            wo_a,
            wo_b,
            n_groups=n_groups,
            heads_per_group=heads_per_group,
            nope_dim=nope_dim,
            rope_dim=rope_dim,
            o_lora_rank=o_lora_rank,
            einsum_recipe=einsum_recipe,
            tma_aligned_scales=tma_aligned_scales,
        )
    else:
        return mctlass_fp8_o_proj(
            o,
            positions,
            cos_sin_cache,
            wo_a,
            wo_b,
            n_groups=n_groups,
            heads_per_group=heads_per_group,
            nope_dim=nope_dim,
            rope_dim=rope_dim,
            o_lora_rank=o_lora_rank,
        )
