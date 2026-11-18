// SPDX-License-Identifier: Apache-2.0
// 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd.
//
// NOTE(metax): Stable Torch 2.13 from_blob has no storage deleter overload.
// This single translation unit uses the exact installed ATen ABI to retain
// host-storage ownership. The rest of the extension keeps its stable ABI.
#include <ATen/ATen.h>
#include <torch/version.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <cuda_runtime_api.h>

#include <cstring>
#include <memory>

#if TORCH_VERSION_MAJOR != 2 || TORCH_VERSION_MINOR != 13
  #error "The MACA UVA ownership bridge must be reviewed for this Torch version"
#endif

AtenTensorHandle metax_cuda_view_with_owner(AtenTensorHandle input) {
  const auto& cpu = *reinterpret_cast<at::Tensor*>(input);
  TORCH_CHECK(cpu.device().is_cpu(), "Input tensor must be on CPU");
  TORCH_CHECK(cpu.layout() == at::kStrided,
              "UVA views require strided storage");
  const auto options = cpu.options().device(at::kCUDA);
  if (cpu.numel() == 0) {
    return reinterpret_cast<AtenTensorHandle>(
        new at::Tensor(at::empty(cpu.sizes(), options)));
  }

  if (cpu.is_pinned()) {
    void* device_ptr = nullptr;
    auto error = cudaHostGetDevicePointer(&device_ptr, cpu.data_ptr(), 0);
    TORCH_CHECK(error == cudaSuccess,
                "cudaHostGetDevicePointer failed: ", cudaGetErrorString(error));
    auto view = at::from_blob(
        device_ptr, cpu.sizes(), cpu.strides(), [owner = cpu](void*) {},
        options);
    return reinterpret_cast<AtenTensorHandle>(new at::Tensor(std::move(view)));
  }

  auto contiguous = cpu.contiguous();
  void* host_ptr = nullptr;
  auto error =
      cudaHostAlloc(&host_ptr, contiguous.nbytes(), cudaHostAllocMapped);
  TORCH_CHECK(error == cudaSuccess,
              "cudaHostAlloc failed: ", cudaGetErrorString(error));
  // Share ownership with the storage deleter, including constructor failures.
  auto free_host = [](void* ptr) { cudaFreeHost(ptr); };
  std::shared_ptr<void> owner(host_ptr, free_host);
  std::memcpy(host_ptr, contiguous.const_data_ptr(), contiguous.nbytes());
  void* device_ptr = nullptr;
  error = cudaHostGetDevicePointer(&device_ptr, host_ptr, 0);
  TORCH_CHECK(error == cudaSuccess,
              "cudaHostGetDevicePointer failed: ", cudaGetErrorString(error));
  auto view = at::from_blob(
      device_ptr, contiguous.sizes(), contiguous.strides(), [owner](void*) {},
      options);
  return reinterpret_cast<AtenTensorHandle>(new at::Tensor(std::move(view)));
}
