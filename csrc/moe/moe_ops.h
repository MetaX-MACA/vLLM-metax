#pragma once

#include <torch/all.h>

void topk_softmax(torch::Tensor& topk_weights, torch::Tensor& topk_indices,
                  torch::Tensor& token_expert_indices,
                  torch::Tensor& gating_output);

void moe_sum(torch::Tensor& input, torch::Tensor& output);

void moe_align_block_size(torch::Tensor topk_ids, int64_t num_experts,
                          int64_t block_size, torch::Tensor sorted_token_ids,
                          torch::Tensor experts_ids,
                          torch::Tensor num_tokens_post_pad);

void fused_moe_kernel(const torch::Tensor& A, const torch::Tensor& B, const torch::Tensor& C,
                    const torch::Tensor& topk_weights, const torch::Tensor& topk_ids,
                    const torch::Tensor& sorted_token_ids, const torch::Tensor& expert_ids,
                    const torch::Tensor& num_tokens_post_padded, bool mul_routed_weight, int64_t top_k, int64_t tileConfig);

bool moe_permute_unpermute_supported();

void shuffle_rows(const torch::Tensor& input_tensor,
                  const torch::Tensor& dst2src_map,
                  torch::Tensor& output_tensor);