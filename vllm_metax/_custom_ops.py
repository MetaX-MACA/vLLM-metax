# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# ---------------------------------------------------
# Note:
#
# Here we only maintain the custom ops that are:
#
#   - modified
#   - newly added
#
# in vllm_metax compared to vllm.
#
# When *adding* new custom ops, make sure you checked the
# latest vllm/_custom_ops.py first to avoid adding duplicates.
# ---------------------------------------------------

from typing import TYPE_CHECKING

import torch
import vllm.envs as envs
import vllm_metax.envs as mx_envs

# Import upstream custom-op registrations first. MetaX overrides the GPTQ fake
# impl below because the MetaX kernel exposes a different schema.
import vllm._custom_ops as _vllm_custom_ops  # noqa: F401
from vllm.platforms import current_platform

current_platform.import_kernels()


if TYPE_CHECKING:

    def register_fake(name, allow_override=False):
        return lambda fn: fn
else:
    try:
        from torch.library import register_fake
    except ImportError:
        from torch.library import impl_abstract as register_fake


def _register_fake_impl(
    op_name: str,
    fake_impl,
    *,
    allow_override: bool = False,
) -> None:
    try:
        register_fake(op_name, allow_override=allow_override)(fake_impl)
    except TypeError:
        register_fake(op_name)(fake_impl)


def awq_gemm(
    input: torch.Tensor,
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    split_k_iters: int,
    temp_space: torch.Tensor,
    dtype_bf16: bool,
) -> torch.Tensor:
    if envs.VLLM_USE_TRITON_AWQ:
        from vllm.model_executor.layers.quantization.awq_triton import awq_gemm_triton

        return awq_gemm_triton(input, qweight, scales, qzeros, split_k_iters)
    return torch.ops._C.awq_gemm(
        input, qweight, scales, qzeros, split_k_iters, temp_space, dtype_bf16
    )


# awq to gptq 4bit conversion
def awq_to_gptq_4bit(qweight: torch.Tensor) -> torch.Tensor:
    if envs.VLLM_USE_TRITON_AWQ:
        return qweight
    return torch.ops._C.awq_to_gptq_4bit(qweight)


# gptq
def gptq_gemm(
    a: torch.Tensor,
    b_q_weight: torch.Tensor,
    b_gptq_qzeros: torch.Tensor,
    b_gptq_scales: torch.Tensor,
    b_g_idx: torch.Tensor,
    use_exllama: bool,
    bit: int,
    group_size: int,
    perm_space: torch.Tensor,
    temp_space: torch.Tensor,
    dtype_bf16: bool,
) -> torch.Tensor:
    return torch.ops._C.gptq_gemm(
        a,
        b_q_weight,
        b_gptq_qzeros,
        b_gptq_scales,
        b_g_idx,
        use_exllama,
        bit,
        group_size,
        perm_space,
        temp_space,
        dtype_bf16,
    )


if hasattr(torch.ops._C, "gptq_gemm"):

    def _gptq_gemm_fake(
        a: torch.Tensor,
        b_q_weight: torch.Tensor,
        b_gptq_qzeros: torch.Tensor,
        b_gptq_scales: torch.Tensor,
        b_g_idx: torch.Tensor,
        use_exllama: bool,
        bit: int,
        group_size: int,
        perm_space: torch.Tensor,
        temp_space: torch.Tensor,
        dtype_bf16: bool,
    ) -> torch.Tensor:
        return torch.empty(
            (a.size(0), b_q_weight.size(1)), dtype=a.dtype, device=a.device
        )

    # Override the upstream vLLM fake impl: MetaX exposes a different
    # gptq_gemm schema with extra workspace arguments.
    _register_fake_impl(
        "_C::gptq_gemm",
        _gptq_gemm_fake,
        allow_override=True,
    )


def indexer_k_quant_and_cache(
    k: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    quant_block_size: int,
    kv_cache_dtype: str,
) -> None:
    if kv_cache.dtype in (torch.bfloat16, torch.float16):
        torch.ops._C_cache_ops.indexer_k_cache(k, kv_cache, slot_mapping)
    else:
        torch.ops._C_cache_ops.indexer_k_quant_and_cache(
            k, kv_cache, slot_mapping, quant_block_size, kv_cache_dtype
        )


def cp_gather_indexer_k_quant_cache(
    kv_cache: torch.Tensor,
    dst_k: torch.Tensor,
    dst_scale: torch.Tensor,
    block_table: torch.Tensor,
    cu_seq_lens: torch.Tensor,
) -> None:
    if dst_k.dtype in (torch.bfloat16, torch.float16) or dst_scale is None:
        torch.ops._C_cache_ops.cp_gather_indexer_k_cache(
            kv_cache, dst_k, block_table, cu_seq_lens
        )
    else:
        torch.ops._C_cache_ops.cp_gather_indexer_k_quant_cache(
            kv_cache, dst_k, dst_scale, block_table, cu_seq_lens
        )


def grouped_topk(
    scores: torch.Tensor,
    num_expert_group: int,
    topk_group: int,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
    bias: torch.Tensor,
    scoring_func: int = 0,
):
    """
    Perform grouped top-k routing for mixture of experts.

    Args:
        scores: Raw inputs (logits if scoring_func=1, scores if scoring_func=0)
        num_expert_group: Number of expert groups
        topk_group: Number of groups to select
        topk: Number of experts to select per token
        renormalize: Whether to renormalize the output weights
        routed_scaling_factor: Scaling factor for routing weights
        bias: Bias tensor (e_score_correction_bias). Always fused in kernel.
        scoring_func: 0=none (no activation), 1=sigmoid
    """
    if scores.shape[0] == 0:
        return (
            scores.new_empty((0, topk), dtype=torch.float32),
            scores.new_empty((0, topk), dtype=torch.int32),
        )
    return torch.ops._moe_C.grouped_topk(
        scores,
        num_expert_group,
        topk_group,
        topk,
        renormalize,
        routed_scaling_factor,
        bias,
        scoring_func,
    )


def sgl_fused_moe_gate_opt(
    gating_outputs: torch.Tensor,
    correction_bias: torch.Tensor,
    out_routing_weights: torch.Tensor,
    out_selected_experts: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: int,
    topk_group: int,
    num_shared_experts: int | None = None,
    scale_factor: float | None = None,
) -> int:
    return torch.ops.sgl_kernel.fused_moe_gate_opt.default(
        gating_outputs,
        correction_bias,
        out_routing_weights,
        out_selected_experts,
        topk,
        renormalize,
        num_expert_group,
        topk_group,
        num_shared_experts,
        scale_factor,
    )


def swiglustep_and_mul(
    act_out: torch.Tensor,
    act_in: torch.Tensor,
    clamp_limit: float = 7.0,
    alpha: float = 1.0,
    beta: float = 0.0,
) -> None:
    if not mx_envs.USE_PRECOMPILED_KERNEL:
        # NOTE(metax): The target clamp op has different semantics and no
        # SWIGLUSTEP flag. Use the complete upstream activation instead.
        if alpha != 1.0 or beta != 0.0:
            raise NotImplementedError(
                "Source SWIGLUSTEP supports alpha=1.0 and beta=0.0"
            )
        from vllm.model_executor.layers.activation import swiglustep_and_mul_triton

        swiglustep_and_mul_triton(act_out, act_in, clamp_limit)
        return
    return torch.ops._C.silu_and_mul_with_clamp(
        act_out,
        act_in,
        float(clamp_limit),
        float(alpha),
        float(beta),
        True,
    )


def swiglu_step_and_mul_per_block_quant(
    input: torch.Tensor,
    group_size: int = 128,
    quant_dtype: torch.dtype = torch.float8_e4m3fn,
    clamp_limit: float = 7.0,
    alpha: float = 1.0,
    beta: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens = input.shape[0]
    hidden_size = input.shape[-1] // 2

    if not mx_envs.USE_PRECOMPILED_KERNEL:
        # NOTE(metax): MCOPLIB's fused activation/quantizer is not part of the
        # native extension. Preserve activation rounding before group quant.
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            per_token_group_quant_fp8,
        )

        activated = torch.empty(
            (num_tokens, hidden_size), device=input.device, dtype=input.dtype
        )
        swiglustep_and_mul(activated, input, clamp_limit, alpha, beta)
        return per_token_group_quant_fp8(
            activated,
            group_size,
            dtype=quant_dtype,
            column_major_scales=False,
            tma_aligned_scales=False,
            use_ue8m0=False,
        )

    output = torch.empty(
        (num_tokens, hidden_size), device=input.device, dtype=quant_dtype
    )
    scales = torch.empty(
        (num_tokens, hidden_size // group_size),
        device=input.device,
        dtype=torch.float32,
    )

    torch.ops._C.swiglu_step_and_mul_per_block_quant(
        output,
        input,
        float(clamp_limit),
        scales,
        group_size,
        None,
        False,
        float(alpha),
        float(beta),
    )

    return output, scales
