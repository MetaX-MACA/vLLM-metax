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
#mport contextlib
# When *adding* new custom ops, make sure you checked the
# latest vllm/_custom_ops.py first to avoid adding duplicates.
# ---------------------------------------------------
import contextlib
from typing import TYPE_CHECKING

import torch
import vllm.envs as envs
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op, is_torch_equal_or_newer

import vllm_metax.envs as mx_envs

logger = init_logger(__name__)

mctlass_op = None
with contextlib.suppress(ImportError):
    if mx_envs.MACA_VLLM_ENABLE_MCTLASS_PYTHON_API and mctlass_op is None:
        import mctlassEx
        mctlass_op = mctlassEx.mctlassExHandleWrapper()

def mctlassEx_w8a8_scaled_mm_azp(
                      out: torch.Tensor,
                      a: torch.Tensor,
                      b: torch.Tensor,
                      scale_a: torch.Tensor,
                      scale_b: torch.Tensor,
                      bias: torch.Tensor | None = None, 
                      azp_adj: torch.Tensor | None = None,
                      azp: torch.Tensor | None = None) -> torch.Tensor:
    stream = torch.cuda.current_stream().cuda_stream
    mctlass_op.mctlass_w8a8_scaled_mm_azp(a, b, out, scale_a, scale_b.T, 
                                          bias, azp_adj, azp, stream)
    return out

def mctlassEx_w8a8_scaled_mm_azp_fake(
                      out: torch.Tensor,
                      a: torch.Tensor,
                      b: torch.Tensor,
                      scale_a: torch.Tensor,
                      scale_b: torch.Tensor,
                      bias: torch.Tensor | None = None, 
                      azp_adj: torch.Tensor | None = None,
                      azp: torch.Tensor | None = None) -> torch.Tensor:
    return out

direct_register_custom_op(
    op_name="mctlassEx_w8a8_scaled_mm_azp",
    op_func=mctlassEx_w8a8_scaled_mm_azp,
    mutates_args=["out"],
    fake_impl=mctlassEx_w8a8_scaled_mm_azp_fake,
    tags=(
        ()
        if is_torch_equal_or_newer("2.7.0")
        else (torch.Tag.needs_fixed_stride_order,)
    ),
)


def mctlassEx_fused_moe_get_kernel_m(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, topk: int) -> int:
    qa = a.to(torch.int8)
    qb = b.to(torch.int8)
    c1 = c.view(-1, c.size(-1)).contiguous()
    return mctlass_op.mctlass_fuse_moe_get_kernel_m(qa, qb, c1, topk)


def mctlassEx_fused_moe_gemm(a: torch.Tensor,
                        b: torch.Tensor,
                        c: torch.Tensor,
                        a_scales: torch.Tensor,
                        b_scales: torch.Tensor,
                        topk_weights: torch.Tensor,
                        token_ids: torch.Tensor,
                        expert_ids: torch.Tensor,
                        num_tokens_post_padded: torch.Tensor,
                        EM: int,
                        topk: int,
                        mul_routed_weight: bool) -> torch.Tensor:
    # TODO: need mctlass to fix it 
    stream = torch.cuda.current_stream().cuda_stream
    c1 = c.view(-1, c.size(-1)).contiguous()
    mctlass_op.mctlass_fuse_moe_gemm(a, b, c1, a_scales, b_scales, 
                                     topk_weights, token_ids, 
                                     expert_ids, num_tokens_post_padded, EM, topk, 
                                     mul_routed_weight, stream)
    return c1.reshape(c.shape)

def mctlassEx_fused_moe_gemm_fake(a: torch.Tensor,
                        b: torch.Tensor,
                        c: torch.Tensor,
                        a_scales: torch.Tensor,
                        b_scales: torch.Tensor,
                        topk_weights: torch.Tensor,
                        token_ids: torch.Tensor,
                        expert_ids: torch.Tensor,
                        num_tokens_post_padded: torch.Tensor,
                        EM: int,
                        topk: int,
                        mul_routed_weight: bool) -> torch.Tensor:
    return c
    
direct_register_custom_op(
    op_name="mctlassEx_fused_moe_gemm",
    op_func=mctlassEx_fused_moe_gemm,
    mutates_args=["c"],
    fake_impl=mctlassEx_fused_moe_gemm_fake,
    tags=(
        ()
        if is_torch_equal_or_newer("2.7.0")
        else (torch.Tag.needs_fixed_stride_order,)
    ),
)

def cutlass_moe_mm_gemm_kernel_m_w8a8(num_valid_tokens: int, N: int, K: int, group: int) -> int:
    return torch.ops._C.cutlass_moe_mm_gemm_kernel_m_w8a8(num_valid_tokens, N, K, group)

def cutlass_moe_mm_w8a8(a: torch.Tensor,
                        b: torch.Tensor,
                        c: torch.Tensor,
                        a_scales: torch.Tensor,
                        b_scales: torch.Tensor,
                        moe_weight: torch.Tensor,
                        token_ids: torch.Tensor,
                        expert_ids: torch.Tensor,
                        num_tokens_post_padded: torch.Tensor,
                        N: int,
                        K: int,
                        EM: int,
                        num_valid_tokens: int,
                        topk: int,
                        mul_routed_weight: bool
                        ) -> torch.Tensor:
    
    if mx_envs.MACA_VLLM_ENABLE_MCTLASS_PYTHON_API:
        torch.ops.vllm.mctlassEx_fused_moe_gemm(a, b, c, a_scales, b_scales,
                        moe_weight, token_ids, expert_ids, num_tokens_post_padded,
                        EM, topk, mul_routed_weight)
    else:
        torch.ops._C.cutlass_moe_mm_w8a8(a, b, c, a_scales, b_scales,
                        moe_weight, token_ids, expert_ids, num_tokens_post_padded,
                        N, K, EM, num_valid_tokens, topk, mul_routed_weight)

def mctlassEx_fused_moe_w4a8_get_kernel_m(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, 
                                          num_experts: int,
                                          batch_size: int,
                                          N: int,
                                          K: int,
                                          num_valid_tokens: int,
                                          topk: int,
                                          group_size: int) -> int:
    return mctlass_op.mctlass_fuse_moe_get_kernel_m_basic(a, b, c, num_experts, batch_size, N, K, 
                                                          num_valid_tokens, topk, group_size)

def mctlassEx_fused_moe_w4a8_gemm(a: torch.Tensor,
                        b: torch.Tensor,
                        c: torch.Tensor,
                        a_scales: torch.Tensor,
                        b_scales: torch.Tensor,
                        topk_weights: torch.Tensor,
                        token_ids: torch.Tensor,
                        expert_ids: torch.Tensor,
                        num_tokens_post_padded: torch.Tensor,
                        num_experts: int,
                        batch_size: int,
                        N: int,
                        K: int,
                        num_valid_tokens: int,
                        EM: int,
                        topk: int,
                        mul_routed_weight: bool,
                        group_size: int) -> torch.Tensor:

    stream = torch.cuda.current_stream().cuda_stream
    mctlass_op.mctlass_fuse_moe_gemm_basic(a, b, c, a_scales, b_scales, topk_weights, 
                                           token_ids, expert_ids, num_tokens_post_padded, 
                                           num_experts, batch_size, N, K, num_valid_tokens, 
                                           EM, topk, mul_routed_weight, group_size)
    return c

def mctlassEx_fused_moe_w4a8_gemm_fake(a: torch.Tensor,
                        b: torch.Tensor,
                        c: torch.Tensor,
                        a_scales: torch.Tensor,
                        b_scales: torch.Tensor,
                        topk_weights: torch.Tensor,
                        token_ids: torch.Tensor,
                        expert_ids: torch.Tensor,
                        num_tokens_post_padded: torch.Tensor,
                        num_experts: int,
                        batch_size: int,
                        N: int,
                        K: int,
                        num_valid_tokens: int,
                        EM: int,
                        topk: int,
                        mul_routed_weight: bool,
                        group_size: int) -> torch.Tensor:
    return c

direct_register_custom_op(
    op_name="mctlassEx_fused_moe_w4a8_gemm",
    op_func=mctlassEx_fused_moe_w4a8_gemm,
    mutates_args=["c"],
    fake_impl=mctlassEx_fused_moe_w4a8_gemm_fake,
    tags=(
        ()
        if is_torch_equal_or_newer("2.7.0")
        else (torch.Tag.needs_fixed_stride_order,)
    ),
)

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


def gptq_shuffle(q_weight: torch.Tensor, q_perm: torch.Tensor, bit: int) -> None:
    torch.ops._C.gptq_shuffle(q_weight, q_perm, bit)


def fused_moe_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    tileConfig: int,
) -> None:
    torch.ops._moe_C.fused_moe_kernel(
        A,
        B,
        C,
        topk_weights,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        mul_routed_weight,
        top_k,
        tileConfig,
    )


def indexer_k_quant_and_cache(
    k: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    quant_block_size: int,
    kv_cache_dtype: str,
) -> None:
    if k.dtype in (torch.bfloat16, torch.float16):
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


def top_k_per_row(
    logits: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    topk_indices: torch.Tensor,
    num_rows: int,
) -> None:
    torch.ops._C.top_k_per_row(
        logits,
        row_starts,
        row_ends,
        topk_indices,
        num_rows,
        logits.stride(0),
        logits.stride(1),
    )


def top_k_per_row_decode(
    logits: torch.Tensor,
    next_n: int,
    seq_lens: torch.Tensor,
    topk_indices: torch.Tensor,
    num_rows: int,
) -> None:
    torch.ops._C.top_k_per_row_decode(
        logits,
        next_n,
        seq_lens,
        topk_indices,
        num_rows,
        logits.stride(0),
        logits.stride(1),
    )


def cutlass_moe_bf16_mm(out: torch.Tensor,
                        a: torch.Tensor,
                        b: torch.Tensor,
                        moe_weight: torch.Tensor,
                        token_ids: torch.Tensor, 
                        expert_ids: torch.Tensor, 
                        num_tokens_post_padded: torch.Tensor, 
                        num_valid_tokens: int, 
                        topk: int, 
                        mul_routed_weight: bool) -> torch.Tensor:

    return torch.ops._C.cutlass_moe_bf16_mm(out, a, b, moe_weight, token_ids, expert_ids, 
                        num_tokens_post_padded, num_valid_tokens, topk, mul_routed_weight)

def cutlass_scaled_mm(a: torch.Tensor,
                      b: torch.Tensor,
                      scale_a: torch.Tensor,
                      scale_b: torch.Tensor,
                      out_dtype: torch.dtype,
                      bias: torch.Tensor | None = None) -> torch.Tensor:
    """
    `cutlass_scaled_mm` implements a fused version of
        `output = torch.mm((scale_a * a), (scale_b * b)).to(out_dtype)`
    where scale_a * a and scale_b * b are implemented using numpy-style
    broadcasting.

    In order to support blockwise scaling like found in DeepSeek V3 we also
    support extended "group" broadcast rules. We extend the numpy-style
    broadcasting rules with the following rule:
        "if the extent of a dimension in the source shape is between 1 and
        corresponding extent in the target shape we repeat each element along
        that dimension  src_shape[dim] // target_shape[dim] times consecutively"
    example if we have:
          a = [[1, 2], and target_shape = (2, 4)
               [3, 4]]
    then we would expand a to:
          a = [[1, 1, 2, 2],
               [3, 3, 4, 4]]
    currently we only support the case:
        scale_a.shape * [1, 128] == a.shape
        scale_b.shape * [128, 128] == b.shape
    """
    assert (out_dtype is torch.bfloat16 or out_dtype is torch.float16)
    assert bias is None or bias.shape[0] == b.shape[
        1] and bias.dtype == out_dtype

    m = a.shape[0]
    n = b.shape[1]

    cutlass_compatible_b = (b.shape[0] % 16 == 0 and b.shape[1] % 16 == 0)
    if current_platform.is_rocm() or not cutlass_compatible_b:
        triton_scaled_mm_module = importlib.import_module(
            "vllm.model_executor.layers.quantization.compressed_tensors."
            "triton_scaled_mm")
        triton_scaled_mm = triton_scaled_mm_module.triton_scaled_mm
        return triton_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias)

    out = torch.empty((m, n), dtype=out_dtype, device=a.device)

    if mx_envs.MACA_VLLM_ENABLE_MCTLASS_PYTHON_API:
        torch.ops.vllm.mctlassEx_w8a8_scaled_mm_azp(out, a, b, scale_a, scale_b, bias)
    else:
        torch.ops._C.cutlass_scaled_mm(out, a, b, scale_a, scale_b, bias)

    return out


def cutlass_scaled_mm_azp(a: torch.Tensor,
                          b: torch.Tensor,
                          scale_a: torch.Tensor,
                          scale_b: torch.Tensor,
                          out_dtype: torch.dtype,
                          azp_adj: torch.Tensor,
                          azp: torch.Tensor | None = None,
                          bias: torch.Tensor | None = None) -> torch.Tensor:
    """
    :param azp_adj: In the per-tensor case, this should include the azp.
    Always per-channel.
    :param azp: Only set in the per-token case. Per-token if set.
    """
    assert (b.shape[0] % 16 == 0 and b.shape[1] % 16 == 0)
    assert (out_dtype is torch.bfloat16 or out_dtype is torch.float16)
    assert bias is None or bias.numel(
    ) == b.shape[1] and bias.dtype == out_dtype
    assert azp is None or azp.numel() == a.shape[0]

    m = a.shape[0]
    n = b.shape[1]

    out = torch.empty((m, n), dtype=out_dtype, device=a.device)

    if False and mx_envs.MACA_VLLM_ENABLE_MCTLASS_PYTHON_API:
        # not support
        pass
    else:
        torch.ops._C.cutlass_scaled_mm_azp(out, a, b, scale_a, scale_b, azp_adj,
                                        azp, bias)
    return out

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
    if not current_platform.is_cuda_alike():
        raise NotImplementedError(
            "The fused grouped_topk kernel is only available on CUDA platforms"
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