// SPDX-License-Identifier: Apache-2.0
// 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Custom element-wise addition kernel: out = a + b
// Supports float32, float16, and bfloat16 dtypes.

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "cuda_compat.h"
#include "dispatch_utils.h"

namespace vllm {

// Vectorized load/store helper: packs multiple scalar elements into a vector
// to improve memory throughput. The vector width is chosen per dtype so that
// each thread processes 16 bytes per memory transaction.
template <typename scalar_t>
struct vec_type { using type = scalar_t; };

template <>
struct vec_type<at::Half> { using type = uint4; };

template <>
struct vec_type<at::BFloat16> { using type = uint4; };

template <>
struct vec_type<float> { using type = float4; };

// Number of scalar elements per vector load.
template <typename scalar_t>
constexpr int vec_elems() {
  if constexpr (std::is_same_v<scalar_t, float>) return 4;
  if constexpr (std::is_same_v<scalar_t, at::Half>) return 8;
  if constexpr (std::is_same_v<scalar_t, at::BFloat16>) return 8;
  return 1;
}

// ---- scalar fallback (used when numel is not a multiple of vector width) ----

template <typename scalar_t>
__global__ void custom_add_scalar_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    const int64_t numel) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numel) {
    out[idx] = a[idx] + b[idx];
  }
}

// ---- vectorized kernel (highest throughput path) ----

template <typename scalar_t, typename vec_t, int VEC_ELEMS>
__global__ void custom_add_vec_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    const int64_t numel) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t vec_idx = idx * VEC_ELEMS;
  if (vec_idx + VEC_ELEMS <= numel) {
    const vec_t a_vec = *reinterpret_cast<const vec_t*>(a + vec_idx);
    const vec_t b_vec = *reinterpret_cast<const vec_t*>(b + vec_idx);
    vec_t out_vec;
    auto* a_arr = reinterpret_cast<const scalar_t*>(&a_vec);
    auto* b_arr = reinterpret_cast<const scalar_t*>(&b_vec);
    auto* o_arr = reinterpret_cast<scalar_t*>(&out_vec);
    #pragma unroll
    for (int i = 0; i < VEC_ELEMS; ++i) {
      o_arr[i] = a_arr[i] + b_arr[i];
    }
    *reinterpret_cast<vec_t*>(out + vec_idx) = out_vec;
  } else {
    // Handle tail elements that don't fill a full vector.
    for (int64_t i = vec_idx; i < numel; ++i) {
      out[i] = a[i] + b[i];
    }
  }
}

}  // namespace vllm

// ---- host entry point (global scope — matches ops.h declaration) ----

void custom_add(
    torch::Tensor& out,
    const torch::Tensor& a,
    const torch::Tensor& b) {

  // Device checks
  TORCH_CHECK(a.is_cuda(), "custom_add: input a must be on CUDA device");
  TORCH_CHECK(b.is_cuda(), "custom_add: input b must be on CUDA device");
  TORCH_CHECK(out.is_cuda(), "custom_add: output tensor must be on CUDA device");
  TORCH_CHECK(a.get_device() == b.get_device(),
              "custom_add: inputs must be on the same device");

  // Shape checks
  TORCH_CHECK(a.sizes() == b.sizes(),
              "custom_add: shape mismatch between inputs");
  TORCH_CHECK(out.sizes() == a.sizes(),
              "custom_add: output shape must match input shape");

  const int64_t numel = a.numel();
  if (numel == 0) return;

  at::cuda::CUDAGuard device_guard{a.get_device()};
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  constexpr int kBlockSize = 256;
  const int64_t grid = (numel + kBlockSize - 1) / kBlockSize;

#ifndef NDEBUG
  printf("[CUSTOM_ADD] CUDA kernel launch: numel=%ld, "
         "dtype=%.*s, device=%d\n",
         numel,
         static_cast<int>(a.dtype().name().size()), a.dtype().name().data(),
         static_cast<int>(a.get_device()));
#endif

  // Dispatch by dtype. For 16-bit types, use the vectorized path (8 elements
  // per thread); for float32, use 4 per thread.
  // Note: double is dispatched via the scalar kernel since vec_elems<double>()
  // returns 1 (no vectorized specialization needed in vLLM workloads).
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      a.scalar_type(),
      "custom_add",
      [&] {
        using vec_t = typename vllm::vec_type<scalar_t>::type;
        constexpr int velems = vllm::vec_elems<scalar_t>();
        if (numel % velems == 0) {
          int64_t vec_grid = (numel / velems + kBlockSize - 1) / kBlockSize;
          vllm::custom_add_vec_kernel<scalar_t, vec_t, velems>
              <<<vec_grid, kBlockSize, 0, stream>>>(out.data_ptr<scalar_t>(),
                                                      a.data_ptr<scalar_t>(),
                                                      b.data_ptr<scalar_t>(),
                                                      numel);
        } else {
          vllm::custom_add_scalar_kernel<scalar_t>
              <<<grid, kBlockSize, 0, stream>>>(out.data_ptr<scalar_t>(),
                                                  a.data_ptr<scalar_t>(),
                                                  b.data_ptr<scalar_t>(),
                                                  numel);
        }
      });
}
