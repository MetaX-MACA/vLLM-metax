# SPDX-License-Identifier: Apache-2.0
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

import torch
import vllm.envs as envs


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
