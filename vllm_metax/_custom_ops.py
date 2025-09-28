# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import vllm.envs as envs


def awq_gemm(input: torch.Tensor, qweight: torch.Tensor, qzeros: torch.Tensor,
             scales: torch.Tensor, split_k_iters: int,
             temp_space: torch.Tensor, dtype_bf16: bool) -> torch.Tensor:
    if envs.VLLM_USE_TRITON_AWQ:
        from vllm.model_executor.layers.quantization.awq_triton import (
            awq_gemm_triton)
        return awq_gemm_triton(input, qweight, scales, qzeros, split_k_iters)
    return torch.ops._C.awq_gemm(input, qweight, scales, qzeros, split_k_iters,
                                 temp_space, dtype_bf16)


# awq to gptq 4bit conversion
def awq_to_gptq_4bit(qweight: torch.Tensor) -> torch.Tensor:
    if envs.VLLM_USE_TRITON_AWQ:
        return qweight
    return torch.ops._C.awq_to_gptq_4bit(qweight)


# gptq
def gptq_gemm(a: torch.Tensor, b_q_weight: torch.Tensor,
              b_gptq_qzeros: torch.Tensor, b_gptq_scales: torch.Tensor,
              b_g_idx: torch.Tensor, use_exllama: bool, bit: int,
              group_size: int, perm_space: torch.Tensor,
              temp_space: torch.Tensor, dtype_bf16: bool) -> torch.Tensor:
    return torch.ops._C.gptq_gemm(a, b_q_weight, b_gptq_qzeros, b_gptq_scales,
                                  b_g_idx, use_exllama, bit, group_size,
                                  perm_space, temp_space, dtype_bf16)


def fused_moe_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor,
                     topk_weights: torch.Tensor, topk_ids: torch.Tensor,
                     sorted_token_ids: torch.Tensor, expert_ids: torch.Tensor,
                     num_tokens_post_padded: torch.Tensor,
                     mul_routed_weight: bool, top_k: int,
                     tileConfig: int) -> None:
    torch.ops._moe_C.fused_moe_kernel(A, B, C, topk_weights, topk_ids,
                                      sorted_token_ids, expert_ids,
                                      num_tokens_post_padded,
                                      mul_routed_weight, top_k, tileConfig)
