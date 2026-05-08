/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 *
 * Horizontally-fused DeepseekV4-MLA kernel:
 *   - Q side:  per-head RMSNorm (no weight) + GPT-J RoPE on last ROPE_DIM
 *   - KV side: GPT-J RoPE on last ROPE_DIM + BF16 paged cache insert
 */

#include <cmath>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <type_traits>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/cuda.h>

#include "cuda_compat.h"
#include "dispatch_utils.h"
#include "type_convert.cuh"

#ifndef FINAL_MASK
  #define FINAL_MASK 0xffffffffu
#endif

namespace vllm {
namespace deepseek_v4_fused_ops {

namespace {
inline int getSMVersion() {
  auto* props = at::cuda::getCurrentDeviceProperties();
  return props->major * 10 + props->minor;
}
}  // namespace

// ────────────────────────────────────────────────────────────────────────────
// Constants
// ────────────────────────────────────────────────────────────────────────────
constexpr int kHeadDim = 512;
constexpr int kRopeDim = 64;
constexpr int kNopeDim = kHeadDim - kRopeDim;  // 448
constexpr int kTokenDataBytes = kHeadDim * 2;

// Per-warp layout:  32 lanes × 16 elems/lane = 512 elems = HEAD_DIM.
constexpr int kNumLanes = 32;
constexpr int kElemsPerLane = kHeadDim / kNumLanes;  // 16

// ────────────────────────────────────────────────────────────────────────────
// Small inline helpers
// ────────────────────────────────────────────────────────────────────────────
template <typename T>
__device__ __forceinline__ float warpSum(float val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    val += __shfl_xor_sync(FINAL_MASK, val, mask, 32);
  }
  return val;
}

// ────────────────────────────────────────────────────────────────────────────
// Kernel
// ────────────────────────────────────────────────────────────────────────────
template <typename scalar_t_in>
__global__ void fusedDeepseekV4QNormRopeKVRopeInsertKernel(
    scalar_t_in* __restrict__ q_inout,      // [N, H, 512] bf16, in place
    scalar_t_in const* __restrict__ kv_in,  // [N, 512] bf16
    scalar_t_in* __restrict__ k_cache,  // [num_blocks, block_stride_elements]
                                        // bf16
    int64_t const* __restrict__ slot_mapping,  // [num_tokens_insert] i64
    int64_t const* __restrict__ position_ids,  // [N] i64
    float const* __restrict__ cos_sin_cache,   // [max_pos, 64] fp32
    float const eps,
    int const num_tokens_full,    // = q.size(0) = kv.size(0)
    int const num_tokens_insert,  // = slot_mapping.size(0), ≤ num_tokens_full
    int const num_heads_q,        // H
    int const cache_block_size,   // tokens per paged-cache block
    int const
        kv_block_stride) {  // elements per paged-cache block (bf16 elements)
#if (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 800) && !defined(USE_ROCM)
  if constexpr (std::is_same_v<scalar_t_in, c10::BFloat16>) {
    return;
  } else {
#endif
    using Converter = vllm::_typeConvert<scalar_t_in>;

    int const warpsPerBlock = blockDim.x / 32;
    int const warpId = threadIdx.x / 32;
    int const laneId = threadIdx.x % 32;
    int const globalWarpIdx = blockIdx.x * warpsPerBlock + warpId;

    int const total_slots_per_token = num_heads_q + 1;
    int const tokenIdx = globalWarpIdx / total_slots_per_token;
    int const slotIdx = globalWarpIdx % total_slots_per_token;
    if (tokenIdx >= num_tokens_full) return;

    bool const isKV = (slotIdx == num_heads_q);
    if (isKV && tokenIdx >= num_tokens_insert) return;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    cudaGridDependencySynchronize();
#endif

    int const dim_base = laneId * kElemsPerLane;

    // ── Load 16 bf16 → 16 fp32 registers ──────────────────────────────────
    float elements[kElemsPerLane];
    float sumOfSquares = 0.0f;

    scalar_t_in const* src_ptr;
    if (isKV) {
      src_ptr = kv_in + static_cast<int64_t>(tokenIdx) * kHeadDim + dim_base;
    } else {
      int64_t const q_row_offset =
          (static_cast<int64_t>(tokenIdx) * num_heads_q + slotIdx) * kHeadDim +
          dim_base;
      src_ptr = q_inout + q_row_offset;
    }

    uint4 v0 = *reinterpret_cast<uint4 const*>(src_ptr);
    uint4 v1 = *reinterpret_cast<uint4 const*>(src_ptr + 8);

    {
      typename Converter::packed_hip_type const* p0 =
          reinterpret_cast<typename Converter::packed_hip_type const*>(&v0);
      typename Converter::packed_hip_type const* p1 =
          reinterpret_cast<typename Converter::packed_hip_type const*>(&v1);
#pragma unroll
      for (int i = 0; i < 4; i++) {
        float2 f2 = Converter::convert(p0[i]);
        elements[2 * i] = f2.x;
        elements[2 * i + 1] = f2.y;
      }
#pragma unroll
      for (int i = 0; i < 4; i++) {
        float2 f2 = Converter::convert(p1[i]);
        elements[8 + 2 * i] = f2.x;
        elements[8 + 2 * i + 1] = f2.y;
      }
    }

    // ── Q branch: RMSNorm (no weight) ────────────────────────────────────
    if (!isKV) {
#pragma unroll
      for (int i = 0; i < kElemsPerLane; i++) {
        sumOfSquares += elements[i] * elements[i];
      }
      sumOfSquares = warpSum<float>(sumOfSquares);
      float const rms_rcp =
          rsqrtf(sumOfSquares / static_cast<float>(kHeadDim) + eps);
#pragma unroll
      for (int i = 0; i < kElemsPerLane; i++) {
        elements[i] = elements[i] * rms_rcp;
      }
    }

    // ── GPT-J RoPE on dims [NOPE_DIM, HEAD_DIM) ───────────────────────────
    bool const is_rope_lane = dim_base >= kNopeDim;
    if (is_rope_lane) {
      int64_t const pos = position_ids[tokenIdx];
      constexpr int kHalfRope = kRopeDim / 2;  // 32
      float const* cos_ptr = cos_sin_cache + pos * kRopeDim;
      float const* sin_ptr = cos_ptr + kHalfRope;

      int const rope_local_base = dim_base - kNopeDim;
#pragma unroll
      for (int p = 0; p < kElemsPerLane / 2; p++) {
        int const pair_dim = rope_local_base + 2 * p;
        int const half_idx = pair_dim / 2;
        float const cos_v = VLLM_LDG(cos_ptr + half_idx);
        float const sin_v = VLLM_LDG(sin_ptr + half_idx);
        float const x_even = elements[2 * p];
        float const x_odd = elements[2 * p + 1];
        elements[2 * p] = x_even * cos_v - x_odd * sin_v;
        elements[2 * p + 1] = x_even * sin_v + x_odd * cos_v;
      }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Q branch: cast to bf16 and store back in place.
    // ═══════════════════════════════════════════════════════════════════════
    if (!isKV) {
      uint4 out0, out1;
      typename Converter::packed_hip_type* po0 =
          reinterpret_cast<typename Converter::packed_hip_type*>(&out0);
      typename Converter::packed_hip_type* po1 =
          reinterpret_cast<typename Converter::packed_hip_type*>(&out1);
#pragma unroll
      for (int i = 0; i < 4; i++) {
        po0[i] = Converter::convert(
            make_float2(elements[2 * i], elements[2 * i + 1]));
      }
#pragma unroll
      for (int i = 0; i < 4; i++) {
        po1[i] = Converter::convert(
            make_float2(elements[8 + 2 * i], elements[8 + 2 * i + 1]));
      }
      scalar_t_in* dst =
          q_inout +
          (static_cast<int64_t>(tokenIdx) * num_heads_q + slotIdx) * kHeadDim +
          dim_base;
      *reinterpret_cast<uint4*>(dst) = out0;
      *reinterpret_cast<uint4*>(dst + 8) = out1;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
      cudaTriggerProgrammaticLaunchCompletion();
#endif
      return;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // KV branch: store as BF16 (directly, no reinterpret_cast needed)
    // ═══════════════════════════════════════════════════════════════════════
    int64_t const slot_id = slot_mapping[tokenIdx];
    if (slot_id < 0) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
      cudaTriggerProgrammaticLaunchCompletion();
#endif
      return;
    }

    int64_t const block_idx = slot_id / cache_block_size;
    int64_t const pos_in_block = slot_id % cache_block_size;

    scalar_t_in* block_base =
        k_cache + block_idx * static_cast<int64_t>(kv_block_stride);
    scalar_t_in* token_ptr = block_base + pos_in_block * kHeadDim;
#pragma unroll
    for (int i = 0; i < kElemsPerLane; i++) {
      elements[i] = Converter::convert(Converter::convert(elements[i]));
    }

    // Convert fp32 to bf16 and store directly
    uint4 out0, out1;
    typename Converter::packed_hip_type* po0 =
        reinterpret_cast<typename Converter::packed_hip_type*>(&out0);
    typename Converter::packed_hip_type* po1 =
        reinterpret_cast<typename Converter::packed_hip_type*>(&out1);
#pragma unroll
    for (int i = 0; i < 4; i++) {
      po0[i] =
          Converter::convert(make_float2(elements[2 * i], elements[2 * i + 1]));
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {
      po1[i] = Converter::convert(
          make_float2(elements[8 + 2 * i], elements[8 + 2 * i + 1]));
    }
    scalar_t_in* bf16_dst = token_ptr + dim_base;
    *reinterpret_cast<uint4*>(bf16_dst) = out0;
    *reinterpret_cast<uint4*>(bf16_dst + 8) = out1;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    cudaTriggerProgrammaticLaunchCompletion();
#endif
#if (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 800) && !defined(USE_ROCM)
  }
#endif
}

// ────────────────────────────────────────────────────────────────────────────
// Launch wrapper
// ────────────────────────────────────────────────────────────────────────────
template <typename scalar_t_in>
void launchFusedDeepseekV4QNormRopeKVRopeInsert(
    scalar_t_in* q_inout, scalar_t_in const* kv_in, scalar_t_in* k_cache,
    int64_t const* slot_mapping, int64_t const* position_ids,
    float const* cos_sin_cache, float const eps, int const num_tokens_full,
    int const num_tokens_insert, int const num_heads_q,
    int const cache_block_size, int const kv_block_stride,
    cudaStream_t stream) {
  constexpr int kBlockSize = 256;
  constexpr int kWarpsPerBlock = kBlockSize / 32;
  int64_t const total_warps =
      static_cast<int64_t>(num_tokens_full) * (num_heads_q + 1);
  int const grid =
      static_cast<int>((total_warps + kWarpsPerBlock - 1) / kWarpsPerBlock);

  fusedDeepseekV4QNormRopeKVRopeInsertKernel<scalar_t_in>
      <<<grid, kBlockSize, 0, stream>>>(
          q_inout, kv_in, k_cache, slot_mapping, position_ids, cos_sin_cache,
          eps, num_tokens_full, num_tokens_insert, num_heads_q,
          cache_block_size, kv_block_stride);
}

}  // namespace deepseek_v4_fused_ops
}  // namespace vllm

// ────────────────────────────────────────────────────────────────────────────
// Torch op wrapper
// ────────────────────────────────────────────────────────────────────────────
void fused_deepseek_v4_qnorm_rope_kv_rope_insert(
    torch::Tensor& q,         // [N, H, 512] bf16, in place
    torch::Tensor const& kv,  // [N, 512] bf16 (read-only)
    torch::Tensor& k_cache,   // [num_blocks, block_stride_elements] bf16
    torch::Tensor const& slot_mapping,   // [N] int64
    torch::Tensor const& position_ids,   // [N] int64
    torch::Tensor const& cos_sin_cache,  // [max_pos, rope_dim] fp32
    double eps, int64_t cache_block_size) {
  TORCH_CHECK(q.is_cuda() && q.is_contiguous(), "q must be contiguous CUDA");
  TORCH_CHECK(kv.is_cuda() && kv.is_contiguous(), "kv must be contiguous CUDA");
  TORCH_CHECK(k_cache.is_cuda(), "k_cache must be CUDA");
  TORCH_CHECK(slot_mapping.is_cuda() && slot_mapping.dtype() == torch::kInt64,
              "slot_mapping must be int64 CUDA");
  TORCH_CHECK(position_ids.is_cuda() && position_ids.dtype() == torch::kInt64,
              "position_ids must be int64 CUDA");
  TORCH_CHECK(cos_sin_cache.is_cuda(), "cos_sin_cache must be CUDA");
  TORCH_CHECK(q.dim() == 3 && q.size(2) == 512, "q shape [N, H, 512]");
  TORCH_CHECK(kv.dim() == 2 && kv.size(1) == 512, "kv shape [N, 512]");
  TORCH_CHECK(q.dtype() == kv.dtype(), "q and kv dtype must match");
  TORCH_CHECK(k_cache.dtype() == q.dtype(),
              "k_cache dtype must match q/kv (bfloat16)");
  TORCH_CHECK(cos_sin_cache.dtype() == torch::kFloat32,
              "cos_sin_cache must be float32");
  TORCH_CHECK(k_cache.dim() == 2,
              "k_cache shape [num_blocks, block_stride_elements]");

  int const num_tokens_full = static_cast<int>(q.size(0));
  int const num_tokens_insert = static_cast<int>(slot_mapping.size(0));
  int const num_heads_q = static_cast<int>(q.size(1));
  int const cache_block_size_i = static_cast<int>(cache_block_size);
  int const kv_block_stride = static_cast<int>(k_cache.stride(0));

  at::cuda::OptionalCUDAGuard device_guard(device_of(q));
  auto stream = at::cuda::getCurrentCUDAStream();

  VLLM_DISPATCH_HALF_TYPES(
      q.scalar_type(), "fused_deepseek_v4_qnorm_rope_kv_insert", [&] {
        using qkv_scalar_t = scalar_t;
        vllm::deepseek_v4_fused_ops::launchFusedDeepseekV4QNormRopeKVRopeInsert<
            qkv_scalar_t>(
            reinterpret_cast<qkv_scalar_t*>(q.data_ptr()),
            reinterpret_cast<qkv_scalar_t const*>(kv.data_ptr()),
            reinterpret_cast<qkv_scalar_t*>(k_cache.data_ptr()),
            reinterpret_cast<int64_t const*>(slot_mapping.data_ptr()),
            reinterpret_cast<int64_t const*>(position_ids.data_ptr()),
            cos_sin_cache.data_ptr<float>(), static_cast<float>(eps),
            num_tokens_full, num_tokens_insert, num_heads_q, cache_block_size_i,
            kv_block_stride, stream);
      });
}