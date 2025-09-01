#include <cudaTypedefs.h>

#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include "cutlass_extensions/common.hpp"

void cutlass_scaled_mm_sm75(torch::Tensor& c, torch::Tensor const& a,
                            torch::Tensor const& b,
                            torch::Tensor const& a_scales,
                            torch::Tensor const& b_scales,
                            std::optional<torch::Tensor> const& bias);

void cutlass_scaled_mm_azp_sm75(torch::Tensor& c, torch::Tensor const& a,
                                torch::Tensor const& b,
                                torch::Tensor const& a_scales,
                                torch::Tensor const& b_scales,
                                torch::Tensor const& azp_adj,
                                std::optional<torch::Tensor> const& azp,
                                std::optional<torch::Tensor> const& bias);


bool cutlass_scaled_mm_supports_fp8(int64_t cuda_device_capability) {
  return false;
}

bool cutlass_scaled_mm_supports_block_fp8(int64_t cuda_device_capability) {
  return false;
}

bool cutlass_group_gemm_supported(int64_t cuda_device_capability) {
  return false;
}

void cutlass_scaled_mm(torch::Tensor& c, torch::Tensor const& a,
                       torch::Tensor const& b, torch::Tensor const& a_scales,
                       torch::Tensor const& b_scales,
                       std::optional<torch::Tensor> const& bias) {
  cutlass_scaled_mm_sm75(c, a, b, a_scales, b_scales, bias);
}

void cutlass_moe_mm(
    torch::Tensor& out_tensors, torch::Tensor const& a_tensors,
    torch::Tensor const& b_tensors, torch::Tensor const& a_scales,
    torch::Tensor const& b_scales, torch::Tensor const& expert_offsets,
    torch::Tensor const& problem_sizes, torch::Tensor const& a_strides,
    torch::Tensor const& b_strides, torch::Tensor const& c_strides,
    bool per_act_token, bool per_out_ch) {
  int32_t version_num = get_sm_version_num();
  TORCH_CHECK_NOT_IMPLEMENTED(
      false,
      "No compiled cutlass_scaled_mm for CUDA device capability: ", version_num,
      ". Required capability: 90");
}

void get_cutlass_moe_mm_data(
    const torch::Tensor& topk_ids, torch::Tensor& expert_offsets,
    torch::Tensor& problem_sizes1, torch::Tensor& problem_sizes2,
    torch::Tensor& input_permutation, torch::Tensor& output_permutation,
    const int64_t num_experts, const int64_t n, const int64_t k,
    const std::optional<torch::Tensor>& blockscale_offsets) {
  int32_t version_num = get_sm_version_num();
  TORCH_CHECK_NOT_IMPLEMENTED(
      false,
      "No compiled get_cutlass_moe_mm_data: no cutlass_scaled_mm kernel for "
      "CUDA device capability: ",
      version_num, ". Required capability: 90");
}

void get_cutlass_pplx_moe_mm_data(torch::Tensor& expert_offsets,
                                  torch::Tensor& problem_sizes1,
                                  torch::Tensor& problem_sizes2,
                                  const torch::Tensor& expert_num_tokens,
                                  const int64_t num_local_experts,
                                  const int64_t padded_m, const int64_t n,
                                  const int64_t k) {
  int32_t version_num = get_sm_version_num();
  TORCH_CHECK_NOT_IMPLEMENTED(
      false,
      "No compiled get_cutlass_pplx_moe_mm_data: no cutlass_scaled_mm kernel "
      "for CUDA device capability: ",
      version_num, ". Required capability: 90");
}

void cutlass_scaled_mm_azp(torch::Tensor& c, torch::Tensor const& a,
                           torch::Tensor const& b,
                           torch::Tensor const& a_scales,
                           torch::Tensor const& b_scales,
                           torch::Tensor const& azp_adj,
                           std::optional<torch::Tensor> const& azp,
                           std::optional<torch::Tensor> const& bias) {
  // Checks for conformality
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && c.dim() == 2);
  TORCH_CHECK(c.size(0) == a.size(0) && a.size(1) == b.size(0) &&
              b.size(1) == c.size(1));
  TORCH_CHECK(a_scales.numel() == 1 || a_scales.numel() == a.size(0));
  TORCH_CHECK(b_scales.numel() == 1 || b_scales.numel() == b.size(1));

  // Check for strides and alignment
  TORCH_CHECK(a.stride(1) == 1 && c.stride(1) == 1);  // Row-major
  TORCH_CHECK(b.stride(0) == 1);                      // Column-major
  TORCH_CHECK(c.stride(0) % 16 == 0 &&
              b.stride(1) % 16 == 0);  // 16 Byte Alignment
  TORCH_CHECK(a_scales.is_contiguous() && b_scales.is_contiguous());

  // bias, azp, azp_adj are all 1d
  // bias and azp_adj have n elements, azp has m elements
  if (bias) {
    TORCH_CHECK(bias->numel() == b.size(1) && bias->is_contiguous());
  }
  if (azp) {
    TORCH_CHECK(azp->numel() == a.size(0) && azp->is_contiguous());
  }
  TORCH_CHECK(azp_adj.numel() == b.size(1) && azp_adj.is_contiguous());

  // azp & bias types
  TORCH_CHECK(azp_adj.dtype() == torch::kInt32);
  TORCH_CHECK(!azp || azp->dtype() == torch::kInt32);
  TORCH_CHECK(!bias || bias->dtype() == c.dtype(),
              "currently bias dtype must match output dtype ", c.dtype());

  at::cuda::OptionalCUDAGuard const device_guard(device_of(a));

  if (!bias) {
    // mctlass not support None bias

    int32_t n = b.size(1);
    int32_t batchsize = 1;
    if (a.dim() == 3 && b.dim() == 3) {
      // a.size = [batch_size, M, K], b.size = [batch_size, K, N]
      n = b.size(2);
      batchsize = a.size(0);
    }
    auto options = torch::TensorOptions()
                     .dtype(c.dtype())
                     .device(a.device());
    torch::Tensor zero_bias = torch::zeros({batchsize,  n}, options);
    cutlass_scaled_mm_azp_sm75(c, a, b, a_scales, b_scales, azp_adj, azp, zero_bias);
  } else {
    cutlass_scaled_mm_azp_sm75(c, a, b, a_scales, b_scales, azp_adj, azp, bias);
  }

}
