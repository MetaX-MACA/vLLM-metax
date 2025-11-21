// 2025 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved. 
#include <c10/core/ScalarType.h>
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include "core/registration.h"

void moe_permute(const torch::Tensor& input,
                 const torch::Tensor& topk_ids,  // [n_token, topk]
                 const torch::Tensor& token_expert_indices,
                 const std::optional<torch::Tensor>& expert_map,
                 int64_t n_expert, int64_t n_local_expert, int64_t topk,
                 const std::optional<int64_t>& align_block_size,
                 torch::Tensor& permuted_input,
                 torch::Tensor& expert_first_token_offset,
                 torch::Tensor& inv_permuted_idx,  // [n_token, topk]
                 torch::Tensor& permuted_idx,      // [permute_size]
                 torch::Tensor& m_indices) {
  TORCH_CHECK(false, "moe_unpermute is not supported on MACA");
}

void moe_unpermute(
    const torch::Tensor& permuted_hidden_states,  // [n_token * topk, hidden]
    const torch::Tensor& topk_weights,            // [n_token, topk]
    const torch::Tensor& inv_permuted_idx,        // [n_token, topk]
    const std::optional<torch::Tensor>&
        expert_first_token_offset,  // [n_local_expert+1]
    int64_t topk,
    torch::Tensor& hidden_states  // [n_token, hidden]
) {
  TORCH_CHECK(false, "moe_unpermute is not supported on MACA");
}

void shuffle_rows(const torch::Tensor& input_tensor,
                  const torch::Tensor& dst2src_map,
                  torch::Tensor& output_tensor) {
  TORCH_CHECK(false, "shuffle_rows is not supported on MACA");
}

bool moe_permute_unpermute_supported() { return false; }

TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, CUDA, m) {
  m.impl("moe_permute", &moe_permute);
  m.impl("moe_unpermute", &moe_unpermute);
}
