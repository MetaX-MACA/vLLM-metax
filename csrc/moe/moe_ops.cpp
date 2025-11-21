// 2025 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved. 
#include "moe_ops.h"

#include <ATen/cuda/CUDAContext.h>
#include <mcblas.h>
#include <maca_fp16.h>

mcblasStatus_t mcblasFusedMoe(
    mcStream_t stream, const void* a_ptr, const void* b_ptr, void* c_ptr,
    const int* sorted_token_ids_ptr, const int* expert_ids_ptr,
    const int* num_tokens_post_padded, int N, int K, int num_valid_tokens,
    int sorted_token_ids_len, int stride_am, int stride_ak, int stride_be,
    int stride_bk, int stride_bn, int stride_cm, int stride_cn, int top_k,
    bool mul_routed_weight, const float* topk_weights_ptr,
    macaDataType compute_type, int tileConfig = 0);

void fused_moe_kernel(const torch::Tensor& A, const torch::Tensor& B,
                      const torch::Tensor& C, const torch::Tensor& topk_weights,
                      const torch::Tensor& topk_ids,
                      const torch::Tensor& sorted_token_ids,
                      const torch::Tensor& expert_ids,
                      const torch::Tensor& num_tokens_post_padded,
                      bool mul_routed_weight, int64_t top_k,
                      int64_t tileConfig) {
  assert(topk_weights.stride(1) == 1);
  assert(sorted_token_ids.stride(0) == 1);

  auto stream = at::cuda::getCurrentCUDAStream();
  macaDataType compute_type =
      (A.dtype() == at::ScalarType::BFloat16) ? MACA_R_16BF : MACA_R_16F;
  mcblasFusedMoe(stream, A.data_ptr(), B.data_ptr(), C.data_ptr(),
                 sorted_token_ids.data_ptr<int>(), expert_ids.data_ptr<int>(),
                 num_tokens_post_padded.data_ptr<int>(), B.size(1), B.size(2),
                 topk_ids.numel(), sorted_token_ids.size(0), A.stride(0),
                 A.stride(1), B.stride(0), B.stride(2), B.stride(1),
                 C.stride(1), C.stride(2), static_cast<int>(top_k),
                 mul_routed_weight, topk_weights.data_ptr<float>(),
                 compute_type, static_cast<int>(tileConfig));
}