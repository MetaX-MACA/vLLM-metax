#ifndef CONCAT_MLA_Q_CUH_
#define CONCAT_MLA_Q_CUH_

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "cuda_vec_utils.cuh"

namespace vllm {

// =============================================================================
// concat_mla_q Kernel for DeepSeek V3.2 MLA Attention
// =============================================================================
//
// Concatenates ql_nope [num_tokens, num_heads, NOPE_DIM] and
// q_pe [num_tokens, num_heads, 64] into
// q_out [num_tokens, num_heads, NOPE_DIM + 64]
//
// Currently instantiated only for NOPE_DIM=512 (DeepSeek V3.2 MLA)
// RoPE dimension is hardcoded to 64
//
// Architecture-specific optimizations:
//   SM100+ (Blackwell): Uses 256-bit PTX ld/st with .cs cache hints
//   SM80 (Metax C500): Uses 128-bit __ldg() vector loads (no PTX assembly)
//
// Memory access pattern:
//   - One warp per (token, head) pair
//   - Coalesced 128/256-bit vector loads within each warp
//   - NoPE: 512 FP16/BF16 elements per warp
//   - RoPE: 64 FP16/BF16 elements per warp
//
// =============================================================================
template <typename DType, int NOPE_DIM>
__global__ void ConcatMLAQKernel(
    DType* __restrict__ q_out, const DType* __restrict__ ql_nope,
    const DType* __restrict__ q_pe, const int num_tokens, const int num_heads,
    const int64_t out_stride_0, const int64_t out_stride_1,
    const int64_t nope_stride_0, const int64_t nope_stride_1,
    const int64_t pe_stride_0, const int64_t pe_stride_1) {
  // Each warp handles one (token_id, head_id) pair
  const int flat_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  if (flat_warp_id >= num_tokens * num_heads) return;

  const int token_id = flat_warp_id / num_heads;
  const int head_id = flat_warp_id % num_heads;
  const int lane_id = threadIdx.x & 31;

  // ========== Architecture Selection ==========
  // VLLM_256B_PTX_ENABLED is 1 for SM100+ with CUDA 12.9+
  // For SM80 (Metax C500), it is 0
  constexpr bool use_256b_ptx = VLLM_256B_PTX_ENABLED;

  // ========== NoPE Part Copy ==========
  // Calculate number of vector loads needed
  // For use_256b_ptx=true:  512 * 2 / (32 * 32) = 1 (256-bit = 16 FP16)
  // For use_256b_ptx=false: 512 * 2 / (16 * 32) = 2 (128-bit = 8 FP16)
  constexpr int nope_vec_loads =
      NOPE_DIM * sizeof(DType) /
      (VecTraits<use_256b_ptx>::ARCH_MAX_VEC_SIZE * 32);

  // Calculate base pointers using stride parameters
  const DType* nope_src =
      ql_nope + token_id * nope_stride_0 + head_id * nope_stride_1;
  DType* nope_dst = q_out + token_id * out_stride_0 + head_id * out_stride_1;

  // Vectorized load/store loop
#pragma unroll
  for (int i = 0; i < nope_vec_loads; i++) {
    const int offset = i * 32 + lane_id;

    if constexpr (use_256b_ptx) {
      // ========== SM100+: 256-bit PTX with .cs cache hint ==========
      st256_cs(reinterpret_cast<u32x8_t*>(nope_dst) + offset,
               ld256_cs(reinterpret_cast<const u32x8_t*>(nope_src) + offset));
    } else {
      // ========== SM80 (Metax C500): 128-bit using __ldg() ==========
      int4 val = __ldg(reinterpret_cast<const int4*>(nope_src) + offset);
      reinterpret_cast<int4*>(nope_dst)[offset] = val;
    }
  }

  const int* rope_src = reinterpret_cast<const int*>(
      q_pe + token_id * pe_stride_0 + head_id * pe_stride_1);
  int* rope_dst = reinterpret_cast<int*>(q_out + token_id * out_stride_0 +
                                         head_id * out_stride_1 + NOPE_DIM);

  if constexpr (use_256b_ptx) {
    // SM100+: Use PTX with .cs cache hint
    st32_cs(rope_dst + lane_id, ld32_cs(rope_src + lane_id));
  } else {
    // SM80: Use standard __ldg() - no PTX assembly
    int rope_val = __ldg(rope_src + lane_id);
    rope_dst[lane_id] = rope_val;
  }
}

}  // namespace vllm

#endif  // CONCAT_MLA_Q_CUH_