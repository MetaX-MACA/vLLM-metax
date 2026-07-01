#include <cuda.h>

#include <cstdint>
#include <optional>

#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include "libtorch_stable/cutlass_extensions/common.hpp"
#include "libtorch_stable/torch_utils.h"

#ifdef ENABLE_SCALED_MM_C2X
void cutlass_scaled_mm_sm75(torch::stable::Tensor& c,
                            torch::stable::Tensor const& a,
                            torch::stable::Tensor const& b,
                            torch::stable::Tensor const& a_scales,
                            torch::stable::Tensor const& b_scales,
                            std::optional<torch::stable::Tensor> const& bias);

void cutlass_scaled_mm_azp_sm75(
    torch::stable::Tensor& c, torch::stable::Tensor const& a,
    torch::stable::Tensor const& b, torch::stable::Tensor const& a_scales,
    torch::stable::Tensor const& b_scales, torch::stable::Tensor const& azp_adj,
    std::optional<torch::stable::Tensor> const& azp,
    std::optional<torch::stable::Tensor> const& bias);
#endif

bool cutlass_scaled_mm_supports_fp8(int64_t cuda_device_capability) {
#ifdef ENABLE_SCALED_MM_C2X
  return cuda_device_capability >= 75 && cuda_device_capability < 90;
#else
  return false;
#endif
}

bool cutlass_scaled_mm_supports_block_fp8(int64_t) { return false; }

bool cutlass_group_gemm_supported(int64_t) { return false; }

void cutlass_scaled_mm(torch::stable::Tensor& c, torch::stable::Tensor const& a,
                       torch::stable::Tensor const& b,
                       torch::stable::Tensor const& a_scales,
                       torch::stable::Tensor const& b_scales,
                       std::optional<torch::stable::Tensor> const& bias) {
#ifdef ENABLE_SCALED_MM_C2X
  const torch::stable::accelerator::DeviceGuard device_guard(
      a.get_device_index());
  cutlass_scaled_mm_sm75(c, a, b, a_scales, b_scales, bias);
#else
  int32_t version_num = get_sm_version_num();
  STD_TORCH_CHECK_NOT_IMPLEMENTED(
      false,
      "No compiled cutlass_scaled_mm for CUDA device capability: ", version_num,
      ". Required capability: 75");
#endif
}

void cutlass_moe_mm(torch::stable::Tensor&, torch::stable::Tensor const&,
                    torch::stable::Tensor const&, torch::stable::Tensor const&,
                    torch::stable::Tensor const&, torch::stable::Tensor const&,
                    torch::stable::Tensor const&, torch::stable::Tensor const&,
                    torch::stable::Tensor const&, torch::stable::Tensor const&,
                    bool, bool) {
  int32_t version_num = get_sm_version_num();
  STD_TORCH_CHECK_NOT_IMPLEMENTED(
      false,
      "No compiled cutlass_moe_mm for CUDA device capability: ", version_num,
      ". Required capability: 90");
}

void get_cutlass_moe_mm_data(const torch::stable::Tensor&,
                             torch::stable::Tensor&, torch::stable::Tensor&,
                             torch::stable::Tensor&, torch::stable::Tensor&,
                             torch::stable::Tensor&, const int64_t,
                             const int64_t, const int64_t,
                             const std::optional<torch::stable::Tensor>&,
                             const bool) {
  int32_t version_num = get_sm_version_num();
  STD_TORCH_CHECK_NOT_IMPLEMENTED(
      false,
      "No compiled get_cutlass_moe_mm_data: no cutlass_moe_mm kernel for CUDA "
      "device capability: ",
      version_num, ". Required capability: 90");
}

void get_cutlass_moe_mm_problem_sizes_from_expert_offsets(
    const torch::stable::Tensor&, torch::stable::Tensor&,
    torch::stable::Tensor&, const int64_t, const int64_t, const bool) {
  int32_t version_num = get_sm_version_num();
  STD_TORCH_CHECK_NOT_IMPLEMENTED(
      false,
      "No compiled get_cutlass_moe_mm_problem_sizes_from_expert_offsets: no "
      "cutlass_moe_mm kernel for CUDA device capability: ",
      version_num, ". Required capability: 90");
}

void get_cutlass_batched_moe_mm_data(torch::stable::Tensor&,
                                     torch::stable::Tensor&,
                                     torch::stable::Tensor&,
                                     const torch::stable::Tensor&,
                                     const int64_t, const int64_t,
                                     const int64_t, const int64_t) {
  int32_t version_num = get_sm_version_num();
  STD_TORCH_CHECK_NOT_IMPLEMENTED(
      false,
      "No compiled get_cutlass_batched_moe_mm_data: no cutlass_moe_mm kernel "
      "for CUDA device capability: ",
      version_num, ". Required capability: 90");
}

void cutlass_scaled_mm_azp(torch::stable::Tensor& c,
                           torch::stable::Tensor const& a,
                           torch::stable::Tensor const& b,
                           torch::stable::Tensor const& a_scales,
                           torch::stable::Tensor const& b_scales,
                           torch::stable::Tensor const& azp_adj,
                           std::optional<torch::stable::Tensor> const& azp,
                           std::optional<torch::stable::Tensor> const& bias) {
#ifdef ENABLE_SCALED_MM_C2X
  // Checks for conformality
  STD_TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && c.dim() == 2);
  STD_TORCH_CHECK(c.size(0) == a.size(0) && a.size(1) == b.size(0) &&
                  b.size(1) == c.size(1));
  STD_TORCH_CHECK(a_scales.numel() == 1 || a_scales.numel() == a.size(0));
  STD_TORCH_CHECK(b_scales.numel() == 1 || b_scales.numel() == b.size(1));

  // Check for strides and alignment
  STD_TORCH_CHECK(a.stride(1) == 1 && c.stride(1) == 1);  // Row-major
  STD_TORCH_CHECK(b.stride(0) == 1);                      // Column-major
  STD_TORCH_CHECK(c.stride(0) % 16 == 0 &&
                  b.stride(1) % 16 == 0);  // 16 Byte Alignment
  STD_TORCH_CHECK(a_scales.is_contiguous() && b_scales.is_contiguous());

  // bias, azp, azp_adj are all 1d.
  // bias and azp_adj have n elements, azp has m elements.
  if (bias) {
    STD_TORCH_CHECK(bias->numel() == b.size(1) && bias->is_contiguous());
  }
  if (azp) {
    STD_TORCH_CHECK(azp->numel() == a.size(0) && azp->is_contiguous());
  }
  STD_TORCH_CHECK(azp_adj.numel() == b.size(1) && azp_adj.is_contiguous());

  // azp and bias types
  STD_TORCH_CHECK(azp_adj.scalar_type() == torch::headeronly::ScalarType::Int);
  STD_TORCH_CHECK(!azp ||
                  azp->scalar_type() == torch::headeronly::ScalarType::Int);
  STD_TORCH_CHECK(!bias || bias->scalar_type() == c.scalar_type(),
                  "currently bias dtype must match output dtype ",
                  c.scalar_type());

  const torch::stable::accelerator::DeviceGuard device_guard(
      a.get_device_index());

  std::optional<torch::stable::Tensor> effective_bias = bias;
  if (!effective_bias) {
    int64_t n = b.size(1);
    int64_t batchsize = 1;
    if (a.dim() == 3 && b.dim() == 3) {
      // a.size = [batch_size, M, K], b.size = [batch_size, K, N]
      n = b.size(2);
      batchsize = a.size(0);
    }
    effective_bias = torch::stable::new_zeros(c, {batchsize, n});
  }

  cutlass_scaled_mm_azp_sm75(c, a, b, a_scales, b_scales, azp_adj, azp,
                             effective_bias);
#else
  int32_t version_num = get_sm_version_num();
  STD_TORCH_CHECK_NOT_IMPLEMENTED(
      false, "No compiled cutlass_scaled_mm_azp for CUDA device capability: ",
      version_num, ". Required capability: 75");
#endif
}