#include <cudaTypedefs.h>

#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include "cutlass_extensions/common.hpp"

bool cutlass_sparse_scaled_mm_supported(int64_t cuda_device_capability) {
  return false;
}

void cutlass_scaled_sparse_mm(torch::Tensor& c, torch::Tensor const& a,
                              torch::Tensor const& bt_nzs,
                              torch::Tensor const& bt_meta,
                              torch::Tensor const& a_scales,
                              torch::Tensor const& b_scales,
                              std::optional<torch::Tensor> const& bias) {
  int32_t version_num = get_sm_version_num();

  TORCH_CHECK_NOT_IMPLEMENTED(
      false,
      "No compiled cutlass_scaled_sparse_mm for a compute capability less than "
      "CUDA device capability: ",
      version_num);
}

std::vector<torch::Tensor> cutlass_sparse_compress(torch::Tensor const& a) {
  int32_t version_num = get_sm_version_num();

  TORCH_CHECK_NOT_IMPLEMENTED(
      false,
      "No compiled cutlass_sparse_compress for a compute capability less than "
      "CUDA device capability: ",
      version_num);
}
