// 2025 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
/*
 * Adapted from
 * https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/kernels/decoder_masked_multihead_attention/decoder_masked_multihead_attention_template.hpp
 * Copyright (c) 2023, The vLLM team.
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>

#include "attention_dtypes.h"
#include "attention_utils.cuh"

#ifdef USE_ROCM
  #include <hip/hip_bf16.h>
  #include "../quantization/fp8/amd/quant_utils.cuh"
typedef __hip_bfloat16 __nv_bfloat16;
#else
  #include "../quantization/fp8/nvidia/quant_utils.cuh"
#endif

#ifndef USE_ROCM
  #define WARP_SIZE 32
#else
  #define WARP_SIZE warpSize
#endif

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define DIVIDE_ROUND_UP(a, b) (((a) + (b) - 1) / (b))

typedef __NATIVE_VECTOR__(2, float) v2f;
typedef __NATIVE_VECTOR__(2, _Float16) v2h;
typedef __NATIVE_VECTOR__(4, float) v4f;

namespace vllm {

// Utility function for attention softmax.
template <int NUM_WARPS>
inline __device__ float block_sum(float* red_smem, float sum) {
  // Decompose the thread index into warp / lane.
  int warp = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x & (WARP_SIZE - 1);

  // Compute the sum per warp.
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
    sum += VLLM_SHFL_XOR_SYNC(sum, mask);
  }

  // Warp leaders store the data to shared memory.
  if (lane == 0) {
    red_smem[warp] = sum;
  }

  // Make sure the data is in shared memory.
  __syncthreads();

  // The warps compute the final sums.
  if (lane < NUM_WARPS) {
    sum = red_smem[lane];
  }

  // Parallel reduction inside the warp.
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    sum += VLLM_SHFL_XOR_SYNC(sum, mask);
  }

  // Broadcast to other threads.
  return VLLM_SHFL_SYNC(sum, 0);
}

template<int NUM_WARPS>
inline __device__ float mxblock_sum(float* red_smem, float sum) {
  // Decompose the thread index into warp / lane.
  int warp = threadIdx.x >> 6;
  int lane = threadIdx.x & (MXWARP_SIZE - 1);

  // Compute the sum per warp.
#pragma unroll
  for (int mask = MXWARP_SIZE / 2; mask >= 1; mask /= 2) {
    sum += MXVLLM_SHFL_XOR_SYNC(sum, mask);
  }

  // Warp leaders store the data to shared memory.
  if (lane == 0) {
    red_smem[warp] = sum;
  }

  // Make sure the data is in shared memory.
  __syncthreads();

  // The warps compute the final sums.
  if (lane < NUM_WARPS) {
    sum = red_smem[lane];
  }
 // Parallel reduction inside the warp.
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    sum += MXVLLM_SHFL_XOR_SYNC(sum, mask);
  }

  // Broadcast to other threads.
  return MXVLLM_SHFL_SYNC(sum, 0);
}
template<typename scalar_t>
__device__  float __forceinline__ atten_mul(scalar_t *a, float b, int j) {
  printf("not support\n");
}

template<>
__device__ float __forceinline__ atten_mul(uint16_t *a, float b, int j) {
    return __half2float(*((half*)a + j)) * __half2float(__float2half(b));
}

template<>
__device__ float __forceinline__ atten_mul(__nv_bfloat16 *a, float b, int j) {
    return __bfloat162float(*(a + j)) * __bfloat162float(__float2bfloat16(b));
}

template<typename scalar_t>
__device__  float __forceinline__ atten_mul_opt(scalar_t *a, float b, int j) {
  printf("not support\n");
}

template<>
__device__ float __forceinline__ atten_mul_opt(uint16_t *a, float b, int j) {
    return __half2float(*((half*)a + j)) * b;
}

template<>
__device__ float __forceinline__ atten_mul_opt(__nv_bfloat16 *a, float b, int j) {
    return __bfloat162float(*(a + j)) * b;
}

template<typename scalar_t>
__device__  void __forceinline__ atten_mul_opt2(scalar_t *a, float b, int j, float &r0, float &r1) {
  printf("not support\n");
}
template<>
__device__ void __forceinline__ atten_mul_opt2(uint16_t *a, float b, int j, float &r0, float &r1) {
    v2f vacc; vacc[0] = r0; vacc[1] = r1;
    v2f vb; vb[0] = b; vb[1] = b;
    v2h a_2h = *(v2h*)(a + j);
    v2f va = __builtin_mxc_cvt_pk_f16tof32(a_2h);
    vacc = __builtin_mxc_pk_fma_f32(va, vb, vacc);
    r0 = vacc[0]; r1 = vacc[1];
}

template<>
__device__ void __forceinline__ atten_mul_opt2(__nv_bfloat16 *a, float b, int j, float &r0, float &r1) {
    v2f vacc; vacc[0] = r0; vacc[1] = r1;
    v2f vb; vb[0] = b; vb[1] = b;
    v2f va; va[0] = __bfloat162float(*(a + j)); va[1] = __bfloat162float(*(a + j + 1));
    vacc = __builtin_mxc_pk_fma_f32(va, vb, vacc);
    r0 = vacc[0]; r1 = vacc[1];
}

template<typename scalar_t, typename cache_t>
__device__ float __forceinline__ atten_dot(scalar_t* a, cache_t *b ,int i){
  printf("not support\n");
}

template<>
__device__ float __forceinline__ atten_dot(uint16_t* a, uint16_t *b ,int i){
  return __half2float(*((half*)a + i)) * __half2float(*((half*)b + i));
}

template<>
__device__ float __forceinline__ atten_dot(float* a, uint16_t *b ,int i) {
  return *(a + i) * __half2float(*((half*)b + i));
}

template<>
__device__ float __forceinline__ atten_dot(__nv_bfloat16* a, __nv_bfloat16 *b ,int i){
  return __bfloat162float(a[i]) * __bfloat162float(b[i]);
}

template<>
__device__ float __forceinline__ atten_dot(float *a, __nv_bfloat16* b, int i) {
  return *(a + i) * __bfloat162float(b[i]);
}

template<typename scalar_t, typename cache_t, typename T, int Vec_size>
__device__ void __forceinline__ atten_dot(scalar_t &v1, cache_t &v2, T&qk) {
  printf("not support\n");
}

template<>
__device__  void __forceinline__ atten_dot<Float8_, uint4,v2f, 8>(Float8_ &v1, uint4 &v2,v2f &vdst) {
    v2h *ptr_v2 = (v2h*)&v2;
    v4f* ptr_v1 = (v4f*)&v1;
    v4f reg_v1_0 = ptr_v1[0], reg_v1_1 = ptr_v1[1];
    v2f v1_2f, v2_2f;
    v2h v2_2h;
    v1_2f[0] = reg_v1_0[0];                  v1_2f[1] = reg_v1_0[1];
    v2_2h = ptr_v2[0];
    v2_2f = __builtin_mxc_cvt_pk_f16tof32(v2_2h);
    vdst = __builtin_mxc_pk_fma_f32(v1_2f, v2_2f, vdst);
    v1_2f[0] = reg_v1_0[2];                  v1_2f[1] = reg_v1_0[3];
    v2_2h = ptr_v2[1];
    v2_2f = __builtin_mxc_cvt_pk_f16tof32(v2_2h);
    vdst = __builtin_mxc_pk_fma_f32(v1_2f, v2_2f, vdst);
    v1_2f[0] = reg_v1_1[0];                  v1_2f[1] = reg_v1_1[1];
    v2_2h = ptr_v2[2];
    v2_2f = __builtin_mxc_cvt_pk_f16tof32(v2_2h);
    vdst = __builtin_mxc_pk_fma_f32(v1_2f, v2_2f, vdst);
    v1_2f[0] = reg_v1_1[2];                  v1_2f[1] = reg_v1_1[3];
    v2_2h = ptr_v2[3];
    v2_2f = __builtin_mxc_cvt_pk_f16tof32(v2_2h);
    vdst = __builtin_mxc_pk_fma_f32(v1_2f, v2_2f, vdst);
}
template<>
__device__ void __forceinline__ atten_dot<Float8_, bf16_8_t, v2f, 8>(Float8_ &v1, bf16_8_t &v2, v2f &vdst) {
    __nv_bfloat16 * ptr_v2 = (__nv_bfloat16*)&v2;
    v2f v1_2f, v2_2f;
    v1_2f[0] = v1.x.x;                  v1_2f[1] = v1.x.y;
    v2_2f[0] = __bfloat162float(ptr_v2[0]); v2_2f[1] = __bfloat162float(ptr_v2[1]);
    vdst = __builtin_mxc_pk_fma_f32(v1_2f, v2_2f, vdst);
    v1_2f[0] = v1.y.x;                  v1_2f[1] = v1.y.y;
    v2_2f[0] = __bfloat162float(ptr_v2[2]); v2_2f[1] = __bfloat162float(ptr_v2[3]);
    vdst = __builtin_mxc_pk_fma_f32(v1_2f, v2_2f, vdst);
    v1_2f[0] = v1.z.x;                  v1_2f[1] = v1.z.y;
    v2_2f[0] = __bfloat162float(ptr_v2[4]); v2_2f[1] = __bfloat162float(ptr_v2[5]);
    vdst = __builtin_mxc_pk_fma_f32(v1_2f, v2_2f, vdst);
    v1_2f[0] = v1.w.x;                  v1_2f[1] = v1.w.y;
    v2_2f[0] = __bfloat162float(ptr_v2[6]); v2_2f[1] = __bfloat162float(ptr_v2[7]);
    vdst = __builtin_mxc_pk_fma_f32(v1_2f, v2_2f, vdst);
}

template<typename T, typename Vec_T0, typename Vec_T1, int Vec_size>
__device__ __forceinline__ void convert(Vec_T0 & src , Vec_T1 &dst){
    printf("not support\n");
}

template<>
__device__ __forceinline__ void convert<uint16_t, uint4, Float8_, 8>(uint4 & src , Float8_ &dst) {
  half * ptr_src = (half *)&src;
  dst.x.x = __half2float(ptr_src[0]);
  dst.x.y = __half2float(ptr_src[1]);
  dst.y.x = __half2float(ptr_src[2]);
  dst.y.y = __half2float(ptr_src[3]);
  dst.z.x = __half2float(ptr_src[4]);
  dst.z.y = __half2float(ptr_src[5]);
  dst.w.x = __half2float(ptr_src[6]);
  dst.w.y = __half2float(ptr_src[7]);
}
template<>
__device__ __forceinline__ void convert<__nv_bfloat16, bf16_8_t, Float8_, 8>(bf16_8_t & src , Float8_ &dst) {
  __nv_bfloat16 * ptr_src = (__nv_bfloat16 *)&src;
  dst.x.x = __bfloat162float(ptr_src[0]);
  dst.x.y = __bfloat162float(ptr_src[1]);
  dst.y.x = __bfloat162float(ptr_src[2]);
  dst.y.y = __bfloat162float(ptr_src[3]);
  dst.z.x = __bfloat162float(ptr_src[4]);
  dst.z.y = __bfloat162float(ptr_src[5]);
  dst.w.x = __bfloat162float(ptr_src[6]);
  dst.w.y = __bfloat162float(ptr_src[7]);
}

template<typename T>
__device__ __forceinline__ float convert(float src){
  printf("not support\n");
}

template<>
__device__ __forceinline__ float convert<uint16_t>(float src) {
   return __half2float(__float2half(src));
}

template<>
__device__ __forceinline__ float convert<__nv_bfloat16>(float src) {
   return __bfloat162float(__float2bfloat16(src));
}

template<typename scalar_t>
__device__ void to_v2f(scalar_t& a, scalar_t& b, v2f& vdst){
printf("not support\n");
}

template<>
__device__ void to_v2f(uint16_t& a, uint16_t& b, v2f &vdst) {
  v2h f_d;
  _Float16 * ptr_a = (_Float16*)&f_d;
  ptr_a[0] = *(_Float16*)&a;
  ptr_a[1] = *(_Float16*)&b;
  vdst = __builtin_mxc_cvt_pk_f16tof32(f_d);
}

template<>
__device__ void to_v2f(__nv_bfloat16& a, __nv_bfloat16& b, v2f &vdst) {
  vdst[0] = __bfloat162float(a);
  vdst[1] = __bfloat162float(b);
}

// TODO(woosuk): Merge the last two dimensions of the grid.
// Grid: (num_heads, num_seqs, max_num_partitions).
template <typename scalar_t, typename cache_t, int HEAD_SIZE, int BLOCK_SIZE,
          int NUM_THREADS, vllm::Fp8KVCacheDataType KV_DTYPE,
          bool IS_BLOCK_SPARSE,
          int PARTITION_SIZE = 0>  // Zero means no partitioning.
__device__ void paged_attention_kernel(
    float* __restrict__ exp_sums,  // [num_seqs, num_heads, max_num_partitions]
    float* __restrict__ max_logits,  // [num_seqs, num_heads,
                                     // max_num_partitions]
    scalar_t* __restrict__ out,  // [num_seqs, num_heads, max_num_partitions,
                                 // head_size]
    const scalar_t* __restrict__ q,       // [num_seqs, num_heads, head_size]
    const cache_t* __restrict__ k_cache,  // [num_blocks, num_kv_heads,
                                          // head_size/x, block_size, x]
    const cache_t* __restrict__ v_cache,  // [num_blocks, num_kv_heads,
                                          // head_size, block_size]
    const int num_kv_heads,               // [num_heads]
    const float scale,
    const int* __restrict__ block_tables,  // [num_seqs, max_num_blocks_per_seq]
    const int* __restrict__ seq_lens,      // [num_seqs]
    const int max_num_blocks_per_seq,
    const float* __restrict__ alibi_slopes,  // [num_heads]
    const int q_stride, const int kv_block_stride, const int kv_head_stride,
    const float* k_scale, const float* v_scale, const int tp_rank,
    const int blocksparse_local_blocks, const int blocksparse_vert_stride,
    const int blocksparse_block_size, const int blocksparse_head_sliding_step) {
  const int seq_idx = blockIdx.y;
  const int partition_idx = blockIdx.z;
  const int max_num_partitions = gridDim.z;
  const int blockDim_x = blockDim.x;
  constexpr bool USE_PARTITIONING = PARTITION_SIZE > 0;
  const int seq_len = seq_lens[seq_idx];
  if (USE_PARTITIONING && partition_idx * PARTITION_SIZE >= seq_len) {
    // No work to do. Terminate the thread block.
    return;
  }
  const int num_seq_blocks = DIVIDE_ROUND_UP(seq_len, BLOCK_SIZE);
  const int num_blocks_per_partition = USE_PARTITIONING ? PARTITION_SIZE / BLOCK_SIZE : num_seq_blocks;

  // [start_block_idx, end_block_idx) is the range of blocks to process.
  const int start_block_idx = USE_PARTITIONING ? partition_idx * num_blocks_per_partition : 0;
  const int end_block_idx = MIN(start_block_idx + num_blocks_per_partition, num_seq_blocks);
  const int num_blocks = end_block_idx - start_block_idx;

  // [start_token_idx, end_token_idx) is the range of tokens to process.
  const int start_token_idx = start_block_idx * BLOCK_SIZE;
  const int end_token_idx = MIN(start_token_idx + num_blocks * BLOCK_SIZE, seq_len);
  const int num_tokens = end_token_idx - start_token_idx;

  constexpr int THREAD_GROUP_SIZE = MAX(WARP_SIZE / BLOCK_SIZE, 1);
  constexpr int NUM_THREAD_GROUPS = NUM_THREADS / THREAD_GROUP_SIZE; // Note: This assumes THREAD_GROUP_SIZE divides NUM_THREADS
  assert(NUM_THREADS % THREAD_GROUP_SIZE == 0);
  constexpr int NUM_TOKENS_PER_THREAD_GROUP = DIVIDE_ROUND_UP(BLOCK_SIZE, WARP_SIZE);
  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  const int thread_idx = threadIdx.x;
  const int warp_idx = thread_idx / WARP_SIZE;
  const int lane = thread_idx % WARP_SIZE;

  const int head_idx = blockIdx.x;
  const int num_heads = gridDim.x;
  const int num_queries_per_kv = num_heads / num_kv_heads;
  const int kv_head_idx = head_idx / num_queries_per_kv;
  const float alibi_slope = alibi_slopes == nullptr ? 0.f : alibi_slopes[head_idx];
  constexpr int VEC_SIZE = 16 / sizeof(scalar_t);
  using K_vec = typename Vec<scalar_t, VEC_SIZE>::Type;
  using Q_vec = typename Vec<scalar_t, VEC_SIZE>::Type;
  using Q_vec_l = typename FloatVec<Q_vec>::Type;
#ifdef ENABLE_FP8_E5M2
  using Quant_vec = typename Vec<cache_t, VEC_SIZE>::Type;
#endif

  constexpr int NUM_ELEMS_PER_THREAD = HEAD_SIZE / THREAD_GROUP_SIZE;
  constexpr int NUM_VECS_PER_THREAD = NUM_ELEMS_PER_THREAD / VEC_SIZE;

  const int thread_group_idx = thread_idx / THREAD_GROUP_SIZE;
  const int thread_group_offset = thread_idx % THREAD_GROUP_SIZE;

  // Load the query to registers.
  // Each thread in a thread group has a different part of the query.
  // For example, if the the thread group size is 4, then the first thread in the group
  // has 0, 4, 8, ... th vectors of the query, and the second thread has 1, 5, 9, ...
  // th vectors of the query, and so on.
  // NOTE(woosuk): Because q is split from a qkv tensor, it may not be contiguous.
  const scalar_t* q_ptr = q + seq_idx * q_stride + head_idx * HEAD_SIZE;
  __shared__ Q_vec_l q_vecs[THREAD_GROUP_SIZE][NUM_VECS_PER_THREAD];
#pragma unroll
  for (int i = thread_group_idx; i < NUM_VECS_PER_THREAD; i += NUM_THREAD_GROUPS) {
    Q_vec_l dst;
    const int vec_idx = thread_group_offset + i * THREAD_GROUP_SIZE;
    Q_vec reg = *reinterpret_cast<const Q_vec*>(q_ptr + vec_idx * VEC_SIZE);
    convert<scalar_t, Q_vec, Q_vec_l, VEC_SIZE>(reg, q_vecs[thread_group_offset][i]);
  }
  __syncthreads(); // TODO(naed90): possible speedup if this is replaced with a memory wall right before we use q_vecs

  // Memory planning.
  extern __shared__ char shared_mem[];
  // NOTE(woosuk): We use FP32 for the softmax logits for better accuracy.
  float* logits = reinterpret_cast<float*>(shared_mem);
  // Workspace for reduction.
  __shared__ float red_smem[2 * NUM_WARPS];

  // Each thread group fetches x elements from the key at a time.
  constexpr int x = 16;
  float qk_max = -FLT_MAX;

  // Iterate over the key blocks.
  // Each warp fetches a block of keys for each iteration.
  // Each thread group in a warp fetches a key from the block, and computes
  // dot product with the query.
  const int* block_table = block_tables + seq_idx * max_num_blocks_per_seq;

  // blocksparse specific vars
  int bs_block_offset;
  int q_bs_block_id;
  if constexpr (IS_BLOCK_SPARSE) {
    // const int num_blocksparse_blocks = DIVIDE_ROUND_UP(seq_len,
    // blocksparse_block_size);
    q_bs_block_id = (seq_len - 1) / blocksparse_block_size;
    if (blocksparse_head_sliding_step >= 0)
      // sliding on q heads
      bs_block_offset =
          (tp_rank * num_heads + head_idx) * blocksparse_head_sliding_step + 1;
    else
      // sliding on kv heads
      bs_block_offset = (tp_rank * num_kv_heads + kv_head_idx) *
                            (-blocksparse_head_sliding_step) +
                        1;
  }

  int block_idx0 = start_block_idx + warp_idx;
  int kv_offset0, kv_offset1;
  K_vec load_k[NUM_VECS_PER_THREAD];
  K_vec compute_k[NUM_VECS_PER_THREAD];
  
  int k_offset[NUM_VECS_PER_THREAD];
  kv_offset0 = block_table[block_idx0];
  if(block_idx0 + NUM_WARPS < end_block_idx) {
    kv_offset1 = block_table[block_idx0 + NUM_WARPS];
  }
  #pragma unroll
  for(int j = 0; j < NUM_VECS_PER_THREAD; j++) {
    const int vec_idx = (thread_group_offset + j * THREAD_GROUP_SIZE)*VEC_SIZE;
    const int offset1 = vec_idx >> 4;
    const int offset2 = vec_idx & 15;
    k_offset[j] = offset1 * BLOCK_SIZE * x + offset2;
  }
  if constexpr (IS_BLOCK_SPARSE) {
      const int k_bs_block_id = block_idx0 * BLOCK_SIZE / blocksparse_block_size;
      const bool is_remote =
          ((k_bs_block_id + bs_block_offset) % blocksparse_vert_stride == 0);
      const bool is_local =
          (k_bs_block_id > q_bs_block_id - blocksparse_local_blocks);
      if(is_remote || is_local) {
          for (int i = 0; i < NUM_TOKENS_PER_THREAD_GROUP; i++) {
            const int physical_block_offset = (thread_group_idx + i * WARP_SIZE) % BLOCK_SIZE;
            const int token_idx = block_idx0 * BLOCK_SIZE + physical_block_offset;
            const cache_t* k_ptr = k_cache + static_cast<int64_t>(kv_offset0) * kv_block_stride
                                              + kv_head_idx * kv_head_stride
                                              + physical_block_offset * x;
            
        #pragma unroll
            for (int j = 0; j < NUM_VECS_PER_THREAD; j++) {
              load_k[j] = *reinterpret_cast<const K_vec*>(k_ptr + k_offset[j]);
            }
          }    
      } 
  } else {
    for (int i = 0; i < NUM_TOKENS_PER_THREAD_GROUP; i++) {
      const int physical_block_offset = (thread_group_idx + i * WARP_SIZE) % BLOCK_SIZE;
      const int token_idx = block_idx0 * BLOCK_SIZE + physical_block_offset;
      const cache_t* k_ptr = k_cache + static_cast<int64_t>(kv_offset0) * kv_block_stride
                                        + kv_head_idx * kv_head_stride
                                        + physical_block_offset * x;
      
  #pragma unroll
      for (int j = 0; j < NUM_VECS_PER_THREAD; j++) {
        load_k[j] = *reinterpret_cast<const K_vec*>(k_ptr + k_offset[j]);
      }
    }
  } 
  for (int block_idx = block_idx0; block_idx < end_block_idx; block_idx += NUM_WARPS) {
    for(int i = 0; i < NUM_TOKENS_PER_THREAD_GROUP; i++) {
      const int physical_block_offset = (thread_group_idx + i * WARP_SIZE) % BLOCK_SIZE;
      const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;
      #pragma unroll
      for(int j = 0; j < NUM_VECS_PER_THREAD; j++) {
        compute_k[j] = load_k[j];
      }
      if(block_idx < end_block_idx - NUM_WARPS) {
          kv_offset0 = kv_offset1;
          int nblock_idx = block_idx + NUM_WARPS;
          if(block_idx < end_block_idx - (NUM_WARPS << 1)) {
            kv_offset1 = block_table[block_idx + (NUM_WARPS<<1)];
          }
          if constexpr (IS_BLOCK_SPARSE) {
            const int k_bs_block_id = nblock_idx * BLOCK_SIZE / blocksparse_block_size;
            const bool is_remote =
                ((k_bs_block_id + bs_block_offset) % blocksparse_vert_stride == 0);
            const bool is_local =
                (k_bs_block_id > q_bs_block_id - blocksparse_local_blocks);
            if(is_remote || is_local) {
              const cache_t* k_ptr = k_cache + static_cast<int64_t>(kv_offset0) * kv_block_stride
                                      + kv_head_idx * kv_head_stride
                                      + physical_block_offset * x;
              #pragma unroll NUM_VECS_PER_THREAD
              for(int j = 0; j < NUM_VECS_PER_THREAD; j++) {
                  load_k[j] = *reinterpret_cast<const K_vec*>(k_ptr + k_offset[j]);
              }  
            }
          } else {
            const cache_t* k_ptr = k_cache + static_cast<int64_t>(kv_offset0) * kv_block_stride
                                      + kv_head_idx * kv_head_stride
                                      + physical_block_offset * x;
            #pragma unroll NUM_VECS_PER_THREAD
            for(int j = 0; j < NUM_VECS_PER_THREAD; j++) {
                load_k[j] = *reinterpret_cast<const K_vec*>(k_ptr + k_offset[j]);
            }
          } 
    }
    if constexpr (IS_BLOCK_SPARSE) {
      const int k_bs_block_id = block_idx * BLOCK_SIZE / blocksparse_block_size;
      const bool is_remote =
          ((k_bs_block_id + bs_block_offset) % blocksparse_vert_stride == 0);
      const bool is_local =
          (k_bs_block_id > q_bs_block_id - blocksparse_local_blocks);
      if (!is_remote && !is_local) {
        for (int i = 0; i < NUM_TOKENS_PER_THREAD_GROUP; i++) {
          const int physical_block_offset =
              (thread_group_idx + i * WARP_SIZE) % BLOCK_SIZE;
          const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;

          if (thread_group_offset == 0) {
            // NOTE(linxihui): assign very large number to skipped tokens to
            // avoid contribution to the sumexp softmax normalizer. This will
            // not be used at computing sum(softmax*v) as the blocks will be
            // skipped.
            logits[token_idx - start_token_idx] = -FLT_MAX;
          }
        }
        continue;
      }
    }

      // Compute dot product.
      // This includes a reduction across the threads in the same thread group.
      // Compute the parallel products for Q*K^T (treat vector lanes separately).
      float qk = 0.0f;
      v2f f2_qk = {0,0};
      #pragma unroll
      for(int j = 0; j < NUM_VECS_PER_THREAD; j++) {
	atten_dot<Q_vec_l, K_vec, v2f, VEC_SIZE>(q_vecs[thread_group_offset][j], compute_k[j],f2_qk);
      }
      qk = f2_qk[0] + f2_qk[1];
  
      #pragma unroll
      for (int mask = THREAD_GROUP_SIZE / 2; mask >= 1; mask /= 2) {
        qk += VLLM_SHFL_XOR_SYNC(qk, mask);
      }
      qk = scale * qk;
      // Add the ALiBi bias if slopes are given.
      qk += (alibi_slope != 0) ? alibi_slope * (token_idx - seq_len + 1) : 0;
      
      if (thread_group_offset == 0) {
        // Store the partial reductions to shared memory.
        // NOTE(woosuk): It is required to zero out the masked logits.
        const bool mask = token_idx >= seq_len;
        logits[token_idx - start_token_idx] = mask ? 0.f : qk;
        // Update the max value.
        qk_max = mask ? qk_max : fmaxf(qk_max, qk);
      }
    }
  }

  // Perform reduction across the threads in the same warp to get the
  // max qk value for each "warp" (not across the thread block yet).
  // The 0-th thread of each thread group already has its max qk value.
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= THREAD_GROUP_SIZE; mask /= 2) {
    qk_max = fmaxf(qk_max, VLLM_SHFL_XOR_SYNC(qk_max, mask));
  }
  if (lane == 0) {
    red_smem[warp_idx] = qk_max;
  }
  __syncthreads();
  // TODO(woosuk): Refactor this part.
  // Get the max qk value for the sequence.
  qk_max = lane < NUM_WARPS ? red_smem[lane] : -FLT_MAX;
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    qk_max = fmaxf(qk_max, VLLM_SHFL_XOR_SYNC(qk_max, mask));
  }
  // Broadcast the max qk value to all threads.
  qk_max = VLLM_SHFL_SYNC(qk_max, 0);

  // Get the sum of the exp values.
  float exp_sum = 0.f;
  for (int i = thread_idx; i < num_tokens; i += NUM_THREADS) {
    float val = __builtin_expf(logits[i] - qk_max);
    logits[i] = val;
    exp_sum += val;
  }
  exp_sum = block_sum<NUM_WARPS>(&red_smem[NUM_WARPS], exp_sum);

  // Compute softmax.
  const float inv_sum = __builtin_mxc_rcpf(exp_sum + 1e-6f);
  for (int i = thread_idx; i < num_tokens; i += NUM_THREADS) {
    logits[i] *= inv_sum;
  }
  __syncthreads();

  // If partitioning is enabled, store the max logit and exp_sum.
  if (USE_PARTITIONING && thread_idx == 0) {
    float* max_logits_ptr = max_logits + seq_idx * num_heads * max_num_partitions
                                       + head_idx * max_num_partitions
                                       + partition_idx;
    *max_logits_ptr = qk_max;
    float* exp_sums_ptr = exp_sums + seq_idx * num_heads * max_num_partitions
                                   + head_idx * max_num_partitions
                                   + partition_idx;
    *exp_sums_ptr = exp_sum;
  }
  constexpr int V_VEC_SIZE = 16 / sizeof(scalar_t);
  constexpr int NUM_V_VECS_PER_THREAD = HEAD_SIZE / V_VEC_SIZE; 
  constexpr int NUM_COLS_PER_ITER = MAX(WARP_SIZE / NUM_V_VECS_PER_THREAD,1);
  constexpr int NUM_VALID_THREAD = NUM_COLS_PER_ITER * NUM_V_VECS_PER_THREAD;
  constexpr int NUM_LGT_PER_COL = (BLOCK_SIZE + NUM_COLS_PER_ITER - 1) / NUM_COLS_PER_ITER;
  constexpr int NUM_LANE = NUM_WARPS * NUM_COLS_PER_ITER;
  using V_vec = typename Vec<scalar_t, V_VEC_SIZE>::Type;
  using L_vec = typename Vec<scalar_t, V_VEC_SIZE>::Type;
  using Float_L_vec = typename FloatVec<L_vec>::Type;
  const int physical_block_offset = lane / NUM_V_VECS_PER_THREAD;
  const int laneid = lane % NUM_V_VECS_PER_THREAD;
  V_vec v_vecs[NUM_LGT_PER_COL];
  V_vec v_prev_vecs[NUM_LGT_PER_COL];
  float accs[V_VEC_SIZE];
  float reg_log[NUM_LGT_PER_COL];
  float reg_prev_log[NUM_LGT_PER_COL];
  #pragma unroll
  for(int i = 0; i < V_VEC_SIZE; i++) {
    accs[i] = 0.0f;
  }
  int token_idx, kv_stride, block_offset;
  kv_stride = BLOCK_SIZE * HEAD_SIZE ;
  kv_offset0 = block_table[block_idx0];
  block_offset = NUM_COLS_PER_ITER * HEAD_SIZE;
  if(block_idx0 + NUM_WARPS < end_block_idx) {
    kv_offset1 = block_table[block_idx0 + NUM_WARPS];
  }
  token_idx = block_idx0 * BLOCK_SIZE + physical_block_offset;
  const cache_t *v_ptr0 = v_cache + kv_head_idx * kv_stride + physical_block_offset * HEAD_SIZE; 
  const cache_t* v_ptr = v_ptr0 + static_cast<int64_t>(kv_offset0) * kv_block_stride;
  float *ptr_logits = logits + token_idx - start_token_idx;
  if(lane < NUM_VALID_THREAD) {
    if constexpr (IS_BLOCK_SPARSE) {
          int v_bs_block_id = block_idx0 * BLOCK_SIZE / blocksparse_block_size;
          if (((v_bs_block_id + bs_block_offset) % blocksparse_vert_stride == 0) ||
              ((v_bs_block_id > q_bs_block_id - blocksparse_local_blocks))) {
              if(block_idx0 == num_seq_blocks - 1) {
              #pragma unroll
                for(int i = 0; i < NUM_LGT_PER_COL; i++) {
                  if(token_idx + i * NUM_COLS_PER_ITER < seq_len ) {
                    const int idx = laneid * V_VEC_SIZE + i * block_offset;
                    v_prev_vecs[i] = *reinterpret_cast<const V_vec*>(v_ptr + idx);
                    reg_prev_log[i] = ptr_logits[i * NUM_COLS_PER_ITER];
                  }
                }
              } else {
                #pragma unroll
                for(int i = 0; i < NUM_LGT_PER_COL; i++) {
                  if(token_idx + i * NUM_COLS_PER_ITER < seq_len ) {
                    const int idx = laneid * V_VEC_SIZE + i * block_offset;
                    v_prev_vecs[i] = *reinterpret_cast<const V_vec*>(v_ptr + idx);
                    reg_prev_log[i] = ptr_logits[i * NUM_COLS_PER_ITER];
                  }
                }
              }
          }
    }
    else {
      if(block_idx0 == num_seq_blocks - 1) {
        #pragma unroll
          for(int i = 0; i < NUM_LGT_PER_COL; i++) {
	    if(token_idx + i * NUM_COLS_PER_ITER < seq_len ) {
              const int idx = laneid * V_VEC_SIZE + i * block_offset;
              v_prev_vecs[i] = *reinterpret_cast<const V_vec*>(v_ptr + idx);
              reg_prev_log[i] = ptr_logits[i * NUM_COLS_PER_ITER];
            }
          }
        } else {
          #pragma unroll
          for(int i = 0; i < NUM_LGT_PER_COL; i++) {
	    if(token_idx + i * NUM_COLS_PER_ITER < seq_len ) {
              const int idx = laneid * V_VEC_SIZE + i * block_offset;
              v_prev_vecs[i] = *reinterpret_cast<const V_vec*>(v_ptr + idx);
              reg_prev_log[i] = ptr_logits[i * NUM_COLS_PER_ITER];
            }
          }
        }
    }
    
    for(int block_idx = block_idx0; block_idx < end_block_idx; block_idx += NUM_WARPS) {
        int next_block = block_idx + NUM_WARPS;
        int nnext_block = next_block + NUM_WARPS;
        for(int i = 0; i < NUM_LGT_PER_COL; i++) {
            v_vecs[i] = v_prev_vecs[i];
            reg_log[i] = reg_prev_log[i];
        }
        if(next_block < end_block_idx) {
            kv_offset0 = kv_offset1;
            if(nnext_block < end_block_idx) {
            kv_offset1 = block_table[nnext_block];
            }
            token_idx = next_block * BLOCK_SIZE + physical_block_offset;
            const cache_t* v_ptr = v_ptr0 + static_cast<int64_t>(kv_offset0) * kv_block_stride;
            ptr_logits = logits + token_idx - start_token_idx;
            if constexpr (IS_BLOCK_SPARSE) {
            int v_bs_block_id = next_block * BLOCK_SIZE / blocksparse_block_size;
            if (((v_bs_block_id + bs_block_offset) % blocksparse_vert_stride == 0) ||
                ((v_bs_block_id > q_bs_block_id - blocksparse_local_blocks))) {
                    if(next_block == num_seq_blocks - 1) {
                    #pragma unroll
                    for(int i = 0; i < NUM_LGT_PER_COL; i++) {
                    if(token_idx + i * NUM_COLS_PER_ITER < seq_len && i * NUM_COLS_PER_ITER + physical_block_offset < BLOCK_SIZE) {
                        const int idx = laneid * V_VEC_SIZE + i * block_offset;
                        v_prev_vecs[i] = *reinterpret_cast<const V_vec*>(v_ptr + idx);
                        reg_prev_log[i] = ptr_logits[i * NUM_COLS_PER_ITER];
                    }
                    }
                } else {
                    #pragma unroll
                    for(int i = 0; i < NUM_LGT_PER_COL; i++) {
                    if(token_idx + i * NUM_COLS_PER_ITER < seq_len && i * NUM_COLS_PER_ITER + physical_block_offset < BLOCK_SIZE) {
                        const int idx = laneid * V_VEC_SIZE + i * block_offset;
                        v_prev_vecs[i] = *reinterpret_cast<const V_vec*>(v_ptr + idx);
                        reg_prev_log[i] = ptr_logits[i * NUM_COLS_PER_ITER];
                    }
                    }
                }
            }
            } else {
            if(next_block == num_seq_blocks - 1) {
                #pragma unroll
                for(int i = 0; i < NUM_LGT_PER_COL; i++) {
                if(token_idx + i * NUM_COLS_PER_ITER < seq_len && i * NUM_COLS_PER_ITER + physical_block_offset < BLOCK_SIZE) {
                    const int idx = laneid * V_VEC_SIZE + i * block_offset;
                    v_prev_vecs[i] = *reinterpret_cast<const V_vec*>(v_ptr + idx);
                    reg_prev_log[i] = ptr_logits[i * NUM_COLS_PER_ITER];
                }
                }
            } else {
                #pragma unroll
                for(int i = 0; i < NUM_LGT_PER_COL; i++) {
                if(token_idx + i * NUM_COLS_PER_ITER < seq_len && i * NUM_COLS_PER_ITER + physical_block_offset < BLOCK_SIZE) {
                    const int idx = laneid * V_VEC_SIZE + i * block_offset;
                    v_prev_vecs[i] = *reinterpret_cast<const V_vec*>(v_ptr + idx);
                    reg_prev_log[i] = ptr_logits[i * NUM_COLS_PER_ITER];
                }
                }
            }
            }
        }
        
      if constexpr (IS_BLOCK_SPARSE) {
        int v_bs_block_id = block_idx * BLOCK_SIZE / blocksparse_block_size;
        if (!((v_bs_block_id + bs_block_offset) % blocksparse_vert_stride == 0) &&
            !((v_bs_block_id > q_bs_block_id - blocksparse_local_blocks))) {
          continue;
        }
      }
      
      token_idx = block_idx * BLOCK_SIZE + physical_block_offset;
      for(int i = 0; i < NUM_LGT_PER_COL; i++) {
        if(token_idx + i * NUM_COLS_PER_ITER < seq_len && i * NUM_COLS_PER_ITER + physical_block_offset < BLOCK_SIZE) {
          scalar_t* v_vec_ptr = reinterpret_cast<scalar_t*>(&v_vecs[i]);
	  #pragma unroll
          for(int j = 0; j < V_VEC_SIZE; j+=2) {
            atten_mul_opt2(v_vec_ptr, reg_log[i], j, accs[j],accs[j + 1]);
          }
        }
      } 
    }
  }
  __syncthreads();
  //need move
  float* out_smem = reinterpret_cast<float*>(shared_mem);
  for(int i = threadIdx.x; i < NUM_WARPS * NUM_COLS_PER_ITER * HEAD_SIZE; i += blockDim_x) {
    out_smem[i] = 0.0f;
  }
  __syncthreads();

  if(lane < NUM_VALID_THREAD) {
    float*ptr_out_smem = out_smem + warp_idx * HEAD_SIZE*NUM_COLS_PER_ITER + physical_block_offset * HEAD_SIZE + laneid* V_VEC_SIZE;
    for(int i = 0; i < V_VEC_SIZE; i++) {
      ptr_out_smem[i] = accs[i];
    }
  }
  __syncthreads();
  scalar_t* out_ptr = out + seq_idx * num_heads * max_num_partitions * HEAD_SIZE
                        + head_idx * max_num_partitions * HEAD_SIZE
                        + partition_idx * HEAD_SIZE;
  if(threadIdx.x < HEAD_SIZE) {
    int length = NUM_LANE * HEAD_SIZE;
      float r = 0;
      for(int i = threadIdx.x; i < length; i += HEAD_SIZE) {
        r += out_smem[i];
      }
      from_float(*(out_ptr + threadIdx.x), r);
  }
}

template <typename scalar_t, typename cache_t, int HEAD_SIZE, int BLOCK_SIZE,
          int NUM_THREADS, vllm::Fp8KVCacheDataType KV_DTYPE,
          bool IS_BLOCK_SPARSE,
          int PARTITION_SIZE = 0>  // Zero means no partitioning.
__device__ __forceinline__ void paged_attention_kernel_32N(
    float* __restrict__ exp_sums,  // [num_seqs, num_heads, max_num_partitions]
    float* __restrict__ max_logits,  // [num_seqs, num_heads,
                                     // max_num_partitions]
    scalar_t* __restrict__ out,  // [num_seqs, num_heads, max_num_partitions,head_size]
    const scalar_t* __restrict__ q,       // [num_seqs, num_heads, head_size]
    const cache_t* __restrict__ k_cache,  // [num_blocks, num_kv_heads, head_size/x, block_size, x]->[num_blocks, num_kv_heads, head_size/16, block_size, 16]
    const cache_t* __restrict__ v_cache,  // [num_blocks, num_kv_heads, head_size, block_size]->[num_blocks, num_kv_heads, block_size, head_size]
    const int num_kv_heads,               // [num_heads]
    const float scale,
    const int* __restrict__ block_tables,  // [num_seqs, max_num_blocks_per_seq]
    const int* __restrict__ seq_lens,      // [num_seqs]
    const int max_num_blocks_per_seq,
    const float* __restrict__ alibi_slopes,  // [num_heads]
    const int q_stride, const int kv_block_stride, const int kv_head_stride,
    const float* k_scale, const float* v_scale, const int tp_rank, const int blocksparse_local_blocks,
    const int blocksparse_vert_stride, const int blocksparse_block_size,
    const int blocksparse_head_sliding_step,const int max_num_partitions,const int num_heads) {
  const int seq_idx = blockIdx.y;
  const int partition_idx = blockIdx.z;
  // const int max_num_partitions = gridDim.z;
  constexpr bool USE_PARTITIONING = PARTITION_SIZE > 0;
  const int seq_len = seq_lens[seq_idx];
  if (USE_PARTITIONING && partition_idx * PARTITION_SIZE >= seq_len) {
    // No work to do. Terminate the thread block.
    return;
  }
  const int num_seq_blocks = DIVIDE_ROUND_UP(seq_len, BLOCK_SIZE);
  const int num_blocks_per_partition = USE_PARTITIONING ? PARTITION_SIZE / BLOCK_SIZE : num_seq_blocks;

  // [start_block_idx, end_block_idx) is the range of blocks to process.
  const int start_block_idx = USE_PARTITIONING ? partition_idx * num_blocks_per_partition : 0;
  const int end_block_idx = MIN(start_block_idx + num_blocks_per_partition, num_seq_blocks);
  const int num_blocks = end_block_idx - start_block_idx;

  // [start_token_idx, end_token_idx) is the range of tokens to process.
  const int start_token_idx = start_block_idx * BLOCK_SIZE;
  const int end_token_idx = MIN(start_token_idx + num_blocks * BLOCK_SIZE, seq_len);
  const int num_tokens = end_token_idx - start_token_idx;
  constexpr int THREAD_GROUP_SIZE = MAX(MXWARP_SIZE / BLOCK_SIZE, 1);
  constexpr int NUM_THREAD_GROUPS = NUM_THREADS / THREAD_GROUP_SIZE; // Note: This assumes THREAD_GROUP_SIZE divides NUM_THREADS
  // assert(NUM_THREADS % THREAD_GROUP_SIZE == 0);
  constexpr int NUM_TOKENS_PER_THREAD_GROUP = DIVIDE_ROUND_UP(BLOCK_SIZE, MXWARP_SIZE);
  constexpr int NUM_WARPS = NUM_THREADS >> 6;
  const int thread_idx = threadIdx.x;
  const int warp_idx = thread_idx >> 6;
  const int lane = thread_idx & 63;

  const int head_idx = blockIdx.x;
  //const int num_heads = gridDim.x;
  const int num_queries_per_kv = num_heads / num_kv_heads;
  const int kv_head_idx = head_idx / num_queries_per_kv;
  const float alibi_slope = alibi_slopes == nullptr ? 0.f : alibi_slopes[head_idx];
  // A vector type to store a part of a key or a query.
  // The vector size is configured in such a way that the threads in a thread group
  // fetch or compute 16 bytes at a time.
  // For example, if the size of a thread group is 4 and the data type is half,
  // then the vector size is 16 / (4 * sizeof(half)) == 2.
  constexpr int VEC_SIZE = 16 / sizeof(scalar_t);
  using K_vec = typename Vec<scalar_t, VEC_SIZE>::Type;
  using Q_vec = typename Vec<scalar_t, VEC_SIZE>::Type;
  using Q_vec_l = typename FloatVec<Q_vec>::Type;
#ifdef ENABLE_FP8_E5M2
  using Quant_vec = typename Vec<cache_t, VEC_SIZE>::Type;
#endif
  
  constexpr int NUM_ELEMS_PER_THREAD = HEAD_SIZE / THREAD_GROUP_SIZE;
  constexpr int NUM_VECS_PER_THREAD = NUM_ELEMS_PER_THREAD / VEC_SIZE;

  const int thread_group_idx = thread_idx / THREAD_GROUP_SIZE;
  const int thread_group_offset = thread_idx % THREAD_GROUP_SIZE;

  // Load the query to registers.
  // Each thread in a thread group has a different part of the query.
  // For example, if the the thread group size is 4, then the first thread in the group
  // has 0, 4, 8, ... th vectors of the query, and the second thread has 1, 5, 9, ...
  // th vectors of the query, and so on.
  // NOTE(woosuk): Because q is split from a qkv tensor, it may not be contiguous.
  const scalar_t* q_ptr = q + seq_idx * q_stride + head_idx * HEAD_SIZE;
  __shared__ Q_vec_l q_vecs[THREAD_GROUP_SIZE][NUM_VECS_PER_THREAD];
#pragma unroll
  for (int i = thread_group_idx; i < NUM_VECS_PER_THREAD; i += NUM_THREAD_GROUPS) {
    Q_vec_l dst;
    const int vec_idx = thread_group_offset + i * THREAD_GROUP_SIZE;
    Q_vec reg = *reinterpret_cast<const Q_vec*>(q_ptr + vec_idx * VEC_SIZE); 
    convert<scalar_t, Q_vec, Q_vec_l, VEC_SIZE>(reg, q_vecs[thread_group_offset][i]);
  }
  __syncthreads(); // TODO(naed90): possible speedup if this is replaced with a memory wall right before we use q_vecs
  // Memory planning.
  extern __shared__ char shared_mem[];
  // NOTE(woosuk): We use FP32 for the softmax logits for better accuracy.
  float* logits = reinterpret_cast<float*>(shared_mem);
  // Workspace for reduction.
  __shared__ float red_smem[2 * NUM_WARPS];

  // Each thread group fetches x elements from the key at a time.
  constexpr int x = 16;
  float qk_max = -FLT_MAX;

  // Iterate over the key blocks.
  // Each warp fetches a block of keys for each iteration.
  // Each thread group in a warp fetches a key from the block, and computes
  // dot product with the query.
  const int* block_table = block_tables + seq_idx * max_num_blocks_per_seq;
  // blocksparse specific vars
  int bs_block_offset;
  int q_bs_block_id;
  if constexpr (IS_BLOCK_SPARSE) {
    // const int num_blocksparse_blocks = DIVIDE_ROUND_UP(seq_len,
    // blocksparse_block_size);
    q_bs_block_id = (seq_len - 1) / blocksparse_block_size;
    if (blocksparse_head_sliding_step >= 0)
      // sliding on q heads
      bs_block_offset =
          (tp_rank * num_heads + head_idx) * blocksparse_head_sliding_step + 1;
    else
      // sliding on kv heads
      bs_block_offset = (tp_rank * num_kv_heads + kv_head_idx) *
                            (-blocksparse_head_sliding_step) +
                        1;
  }
  int block_idx0 = start_block_idx + warp_idx;
  int kv_offset0, kv_offset1;
  K_vec load_k[NUM_VECS_PER_THREAD];
  K_vec compute_k[NUM_VECS_PER_THREAD];
  int k_offset[NUM_VECS_PER_THREAD];
  kv_offset0 = block_table[block_idx0];
  if(block_idx0 + NUM_WARPS < end_block_idx) {
    kv_offset1 = block_table[block_idx0 + NUM_WARPS];
  }
  #pragma unroll
  for(int j = 0; j < NUM_VECS_PER_THREAD; j++) {
    const int vec_idx = (thread_group_offset + j * THREAD_GROUP_SIZE)*VEC_SIZE;
    const int offset1 = vec_idx >> 4;
    const int offset2 = vec_idx & 15;
    k_offset[j] = offset1 * BLOCK_SIZE * x + offset2;
  }
  const cache_t* ptr_k_cache = k_cache + kv_head_idx * kv_head_stride;
  if constexpr (IS_BLOCK_SPARSE) {
      const int k_bs_block_id = block_idx0 * BLOCK_SIZE / blocksparse_block_size;
      const bool is_remote =
          ((k_bs_block_id + bs_block_offset) % blocksparse_vert_stride == 0);
      const bool is_local =
          (k_bs_block_id > q_bs_block_id - blocksparse_local_blocks);
      if(is_remote || is_local) {
          for (int i = 0; i < NUM_TOKENS_PER_THREAD_GROUP; i++) {
            const int physical_block_offset = (thread_group_idx + i * MXWARP_SIZE) & (BLOCK_SIZE - 1);
            const cache_t* k_ptr = ptr_k_cache + static_cast<int64_t>(kv_offset0) * kv_block_stride + physical_block_offset * x;
        #pragma unroll
            for (int j = 0; j < NUM_VECS_PER_THREAD; j++) {
              load_k[j] = *reinterpret_cast<const K_vec*>(k_ptr + k_offset[j]);
            }
          }    
      } 
  } else {
    for (int i = 0; i < NUM_TOKENS_PER_THREAD_GROUP; i++) {
        const int physical_block_offset = (thread_group_idx + i * MXWARP_SIZE) & (BLOCK_SIZE - 1) ;
        const int token_idx = block_idx0 * BLOCK_SIZE + physical_block_offset;
        const cache_t* k_ptr = ptr_k_cache + static_cast<int64_t>(kv_offset0) * kv_block_stride + physical_block_offset * x;
    #pragma unroll
        for (int j = 0; j < NUM_VECS_PER_THREAD; j++) {
          load_k[j] = *reinterpret_cast<const K_vec*>(k_ptr + k_offset[j]);
        } 
    }
  }
  for (int block_idx = block_idx0; block_idx < end_block_idx; block_idx += NUM_WARPS) {  
    for(int i = 0; i < NUM_TOKENS_PER_THREAD_GROUP; i++) {
      const int physical_block_offset = (thread_group_idx + i * MXWARP_SIZE) & (BLOCK_SIZE - 1);
      const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;
      #pragma unroll
      for(int j = 0; j < NUM_VECS_PER_THREAD; j++) {
        compute_k[j] = load_k[j];
      }
      if(block_idx < end_block_idx - NUM_WARPS) {
          kv_offset0 = kv_offset1;
	  int nblock_idx = block_idx + NUM_WARPS;
          if(block_idx < end_block_idx - (NUM_WARPS << 1)) {
            kv_offset1 = block_table[block_idx + (NUM_WARPS<<1)];
          }
          if constexpr (IS_BLOCK_SPARSE) {
            const int k_bs_block_id = nblock_idx * BLOCK_SIZE / blocksparse_block_size;
            const bool is_remote =
                ((k_bs_block_id + bs_block_offset) % blocksparse_vert_stride == 0);
            const bool is_local =
                (k_bs_block_id > q_bs_block_id - blocksparse_local_blocks);
            if(is_remote || is_local) {
              const cache_t* k_ptr = ptr_k_cache + static_cast<int64_t>(kv_offset0) * kv_block_stride + physical_block_offset * x;
              #pragma unroll NUM_VECS_PER_THREAD
              for(int j = 0; j < NUM_VECS_PER_THREAD; j++) {
                  load_k[j] = *reinterpret_cast<const K_vec*>(k_ptr + k_offset[j]);
              }  
            }
          } else {
            const cache_t* k_ptr = ptr_k_cache + static_cast<int64_t>(kv_offset0) * kv_block_stride + physical_block_offset * x;
            #pragma unroll NUM_VECS_PER_THREAD
            for(int j = 0; j < NUM_VECS_PER_THREAD; j++) {
                load_k[j] = *reinterpret_cast<const K_vec*>(k_ptr + k_offset[j]);
            }
          }
      }
      if constexpr (IS_BLOCK_SPARSE) {
        const int k_bs_block_id = block_idx * BLOCK_SIZE / blocksparse_block_size;
        const bool is_remote =
            ((k_bs_block_id + bs_block_offset) % blocksparse_vert_stride == 0);
        const bool is_local =
            (k_bs_block_id > q_bs_block_id - blocksparse_local_blocks);
        if (!is_remote && !is_local) {
            for (int i = 0; i < NUM_TOKENS_PER_THREAD_GROUP; i++) {
            const int physical_block_offset =
                (thread_group_idx + i * MXWARP_SIZE) & (BLOCK_SIZE - 1);
            const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;

            if (thread_group_offset == 0) {
                // NOTE(linxihui): assign very large number to skipped tokens to
                // avoid contribution to the sumexp softmax normalizer. This will
                // not be used at computing sum(softmax*v) as the blocks will be
                // skipped.
                logits[token_idx - start_token_idx] = -FLT_MAX;
            }
            }
            continue;
        }
    }
      // Compute dot product.
      // This includes a reduction across the threads in the same thread group.
      // Compute the parallel products for Q*K^T (treat vector lanes separately).
      float qk = 0.0f;
      v2f f2_qk = {0,0};
      #pragma unroll
      for(int j = 0; j < NUM_VECS_PER_THREAD; j++) {
        atten_dot<Q_vec_l, K_vec, v2f, VEC_SIZE>(q_vecs[thread_group_offset][j], compute_k[j],f2_qk);
      }
      qk = f2_qk[0] + f2_qk[1];
      #pragma unroll
      for (int mask = THREAD_GROUP_SIZE / 2; mask >= 1; mask /= 2) {
        qk += MXVLLM_SHFL_XOR_SYNC(qk, mask);
      }
      qk = scale * qk;
      // Add the ALiBi bias if slopes are given.
      qk += (alibi_slope != 0) ? alibi_slope * (token_idx - seq_len + 1) : 0;

      if (thread_group_offset == 0) {
        // Store the partial reductions to shared memory.
        // NOTE(woosuk): It is required to zero out the masked logits.
        const bool mask = token_idx >= seq_len;
        logits[token_idx - start_token_idx] = mask ? 0.f : qk;
        // Update the max value.
        qk_max = mask ? qk_max : fmaxf(qk_max, qk);
      }
    }
  }
  // Perform reduction across the threads in the same warp to get the
  // max qk value for each "warp" (not across the thread block yet).
  // The 0-th thread of each thread group already has its max qk value.
  #pragma unroll
  for (int mask = MXWARP_SIZE / 2; mask >= THREAD_GROUP_SIZE; mask /= 2) {
    qk_max = fmaxf(qk_max, MXVLLM_SHFL_XOR_SYNC(qk_max, mask));
  }
  if (lane == 0) {
    red_smem[warp_idx] = qk_max;
  }
  __syncthreads();

  // TODO(woosuk): Refactor this part.
  // Get the max qk value for the sequence.
  qk_max = lane < NUM_WARPS ? red_smem[lane] : -FLT_MAX;
  #pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    qk_max = fmaxf(qk_max, MXVLLM_SHFL_XOR_SYNC(qk_max, mask));
  }
  // Broadcast the max qk value to all threads.
  qk_max = MXVLLM_SHFL_SYNC(qk_max, 0);

  // Get the sum of the exp values.
  float exp_sum = 0.f;
  for (int i = thread_idx; i < num_tokens; i += NUM_THREADS) {
    float val = __builtin_expf(logits[i] - qk_max);
    logits[i] = val;
    exp_sum += val;
  }
  exp_sum = mxblock_sum<NUM_WARPS>(&red_smem[NUM_WARPS], exp_sum);

  // Compute softmax.
  const float inv_sum = __builtin_mxc_rcpf(exp_sum + 1e-6f);
  for (int i = thread_idx; i < num_tokens; i += NUM_THREADS) {
    float val = logits[i];
    val *= inv_sum;
    logits[i] = convert<cache_t>(val);
  }
  __syncthreads();

  // If partitioning is enabled, store the max logit and exp_sum.
  if (USE_PARTITIONING && thread_idx == 0) {
    float* max_logits_ptr = max_logits + seq_idx * num_heads * max_num_partitions
                                       + head_idx * max_num_partitions
                                       + partition_idx;
    *max_logits_ptr = qk_max;
    float* exp_sums_ptr = exp_sums + seq_idx * num_heads * max_num_partitions
                                   + head_idx * max_num_partitions
                                   + partition_idx;
    *exp_sums_ptr = exp_sum;
  }
  constexpr int V_VEC_SIZE = 16 / sizeof(scalar_t);
  constexpr int NUM_V_VECS_PER_THREAD = HEAD_SIZE / V_VEC_SIZE; 
  constexpr int NUM_COLS_PER_ITER = MAX(MXWARP_SIZE / NUM_V_VECS_PER_THREAD , 1); 
  constexpr int NUM_LGT_PER_COL = BLOCK_SIZE / NUM_COLS_PER_ITER; 
  constexpr int NUM_LANE = NUM_WARPS * NUM_COLS_PER_ITER;
  using V_vec = typename Vec<scalar_t, V_VEC_SIZE>::Type;
  using L_vec = typename Vec<scalar_t, V_VEC_SIZE>::Type;
  const int physical_block_offset = lane / NUM_V_VECS_PER_THREAD;
  const int laneid = lane % NUM_V_VECS_PER_THREAD;
  V_vec v_vecs[NUM_LGT_PER_COL];
  V_vec v_prev_vecs[NUM_LGT_PER_COL];
  float accs[V_VEC_SIZE];
  float reg_log[NUM_LGT_PER_COL];
  float reg_prev_log[NUM_LGT_PER_COL];
  #pragma unroll
  for(int i = 0; i < V_VEC_SIZE; i++) {
    accs[i] = 0.0f;
  }
  int token_idx, kv_stride, block_offset;
  kv_stride = BLOCK_SIZE * HEAD_SIZE ;
  kv_offset0 = block_table[block_idx0];
  block_offset = NUM_COLS_PER_ITER * HEAD_SIZE;
  if(block_idx0 + NUM_WARPS < end_block_idx) {
    kv_offset1 = block_table[block_idx0 + NUM_WARPS];
  }
  token_idx = block_idx0 * BLOCK_SIZE + physical_block_offset;
  const cache_t *v_ptr0 = v_cache + kv_head_idx * kv_stride + physical_block_offset * HEAD_SIZE; 
  const cache_t* v_ptr = v_ptr0 + static_cast<int64_t>(kv_offset0) * kv_block_stride;
  float *ptr_logits = logits + token_idx - start_token_idx;
  if constexpr (IS_BLOCK_SPARSE) {
        int v_bs_block_id = block_idx0 * BLOCK_SIZE / blocksparse_block_size;
        if (((v_bs_block_id + bs_block_offset) % blocksparse_vert_stride == 0) ||
            ((v_bs_block_id > q_bs_block_id - blocksparse_local_blocks))) {
            #pragma unroll
            for(int i = 0; i < NUM_LGT_PER_COL; i++) {
                if(token_idx + i * NUM_COLS_PER_ITER < seq_len ) {
                const int idx = laneid * V_VEC_SIZE + i * block_offset;
                v_prev_vecs[i] = *reinterpret_cast<const V_vec*>(v_ptr + idx);
                reg_prev_log[i] = ptr_logits[i * NUM_COLS_PER_ITER];
                }
            }
        }
    } else {
          #pragma unroll
          for(int i = 0; i < NUM_LGT_PER_COL; i++) {
            if(token_idx + i * NUM_COLS_PER_ITER < seq_len ) {
              const int idx = laneid * V_VEC_SIZE + i * block_offset;
              v_prev_vecs[i] = *reinterpret_cast<const V_vec*>(v_ptr + idx);
              reg_prev_log[i] = ptr_logits[i * NUM_COLS_PER_ITER];
            }
          }
    }
  for(int block_idx = block_idx0; block_idx < end_block_idx; block_idx += NUM_WARPS) {
    int next_block = block_idx + NUM_WARPS;
    int nnext_block = next_block + NUM_WARPS;
    for(int i = 0; i < NUM_LGT_PER_COL; i++) {
      v_vecs[i] = v_prev_vecs[i];
      reg_log[i] = reg_prev_log[i];
    }
    if(next_block < end_block_idx) {
      kv_offset0 = kv_offset1;
      if(nnext_block < end_block_idx) {
        kv_offset1 = block_table[nnext_block];
      }
      token_idx = next_block * BLOCK_SIZE + physical_block_offset;
      const cache_t* v_ptr = v_ptr0 + static_cast<int64_t>(kv_offset0) * kv_block_stride;
      ptr_logits = logits + token_idx - start_token_idx;
      if constexpr (IS_BLOCK_SPARSE) {
        int v_bs_block_id = next_block * BLOCK_SIZE / blocksparse_block_size;
        if (((v_bs_block_id + bs_block_offset) % blocksparse_vert_stride == 0) ||
            ((v_bs_block_id > q_bs_block_id - blocksparse_local_blocks))) {
                #pragma unroll
                for(int i = 0; i < NUM_LGT_PER_COL; i++) {
                if(token_idx + i * NUM_COLS_PER_ITER < seq_len) {
                    const int idx = laneid * V_VEC_SIZE + i * block_offset;
                    v_prev_vecs[i] = *reinterpret_cast<const V_vec*>(v_ptr + idx);
                    reg_prev_log[i] = ptr_logits[i * NUM_COLS_PER_ITER];
                }
            }
        }
        } else  {
            #pragma unroll
            for(int i = 0; i < NUM_LGT_PER_COL; i++) {
              const int idx = laneid * V_VEC_SIZE + i * block_offset;
              v_prev_vecs[i] = *reinterpret_cast<const V_vec*>(v_ptr + idx);
              reg_prev_log[i] = ptr_logits[i * NUM_COLS_PER_ITER];
            }
        }
    }
    if constexpr (IS_BLOCK_SPARSE) {
        int v_bs_block_id = block_idx * BLOCK_SIZE / blocksparse_block_size;
        if (!((v_bs_block_id + bs_block_offset) % blocksparse_vert_stride == 0) &&
            !((v_bs_block_id > q_bs_block_id - blocksparse_local_blocks))) {
          continue;
        }
    }
    
    token_idx = block_idx * BLOCK_SIZE + physical_block_offset;
    float *ptr_logits = logits + token_idx - start_token_idx;
    for(int i = 0; i < NUM_LGT_PER_COL; i++) {
      if(token_idx + i * NUM_COLS_PER_ITER < seq_len) {
        scalar_t* v_vec_ptr = reinterpret_cast<scalar_t*>(&v_vecs[i]);
        #pragma unroll
        for(int j = 0; j < V_VEC_SIZE; j+=2) {
          atten_mul_opt2(v_vec_ptr, reg_log[i], j, accs[j],accs[j + 1]);
        }
      }
    } 
  }
  __syncthreads();
  float* out_smem = reinterpret_cast<float*>(shared_mem);
  float*ptr_out_smem = out_smem + warp_idx * HEAD_SIZE*NUM_COLS_PER_ITER + physical_block_offset * HEAD_SIZE + laneid* V_VEC_SIZE;
  #pragma unroll
  for(int i = 0; i < V_VEC_SIZE; i++) {
    ptr_out_smem[i] = accs[i];
  }
   __syncthreads();
  
  scalar_t* out_ptr = out + seq_idx * num_heads * max_num_partitions * HEAD_SIZE
                        + head_idx * max_num_partitions * HEAD_SIZE
                        + partition_idx * HEAD_SIZE;
  if(threadIdx.x < HEAD_SIZE) {
    int length = NUM_LANE * HEAD_SIZE;
      float r = 0;
      for(int i = threadIdx.x; i < length; i += HEAD_SIZE) {
        r += out_smem[i];
      }
      from_float(*(out_ptr + threadIdx.x), r);
  }
}

template <typename scalar_t, typename cache_t, int HEAD_SIZE, int BLOCK_SIZE,
          int NUM_THREADS, vllm::Fp8KVCacheDataType KV_DTYPE,
          bool IS_BLOCK_SPARSE,
	  int PARTITION_SIZE = 512>  // Zero means no partitioning.
__device__ __forceinline__ void paged_attention_kernel_32N_final(
    float* __restrict__ exp_sums,  // [num_seqs, num_heads, max_num_partitions]
    float* __restrict__ max_logits,  // [num_seqs, num_heads,
                                     // max_num_partitions]
    int* __restrict__ block_count,          // [num_seqs, num_heads]
    scalar_t* __restrict__ out,  // [num_seqs, num_heads, max_num_partitions,
                                 // head_size]
    scalar_t* __restrict__ final_out,      // [num_seqs, num_heads, head_size]
    const scalar_t* __restrict__ q,       // [num_seqs, num_heads, head_size]
    const cache_t* __restrict__ k_cache,  // [num_blocks, num_kv_heads, head_size/x, block_size, x]->[num_blocks, num_kv_heads, head_size/16, block_size, 16]
    const cache_t* __restrict__ v_cache,  // [num_blocks, num_kv_heads, head_size, block_size]->[num_blocks, num_kv_heads, block_size, head_size]
    const int num_kv_heads,               // [num_heads]
    const float scale,
    const int* __restrict__ block_tables,  // [num_seqs, max_num_blocks_per_seq]
    const int* __restrict__ seq_lens,      // [num_seqs]
    const int max_num_blocks_per_seq,
    const float* __restrict__ alibi_slopes,  // [num_heads]
    const int q_stride, const int kv_block_stride, const int kv_head_stride,
    const float* k_scale, const float* v_scale, const int tp_rank, const int blocksparse_local_blocks,
    const int blocksparse_vert_stride, const int blocksparse_block_size,
    const int blocksparse_head_sliding_step, const int max_num_partitions,
    const int blockDim_x,
    const int num_heads,
    const int grid_dim_y,
    const bool count_init_once) {
  const int seq_idx = blockIdx.y;
  const int partition_idx = blockIdx.z;
  // const int max_num_partitions = gridDim.z;
  // const int blockDim_x = blockDim.x;
  const int head_idx = blockIdx.x;
  // const int num_heads = gridDim.x;
  // const int grid_dim_y = gridDim.y;
  const int seq_len = seq_lens[seq_idx];

  const int num_seq_blocks = DIVIDE_ROUND_UP(seq_len, BLOCK_SIZE);
  const int num_blocks_per_partition = PARTITION_SIZE / BLOCK_SIZE;

  // [start_block_idx, end_block_idx) is the range of blocks to process.
  const int start_block_idx = partition_idx * num_blocks_per_partition;
  const int end_block_idx = MIN(start_block_idx + num_blocks_per_partition, num_seq_blocks);
  const int num_blocks = end_block_idx - start_block_idx;

  // [start_token_idx, end_token_idx) is the range of tokens to process.
  const int start_token_idx = start_block_idx * BLOCK_SIZE;
  const int end_token_idx = MIN(start_token_idx + num_blocks * BLOCK_SIZE, seq_len);
  const int num_tokens = end_token_idx - start_token_idx;

  constexpr int THREAD_GROUP_SIZE = MAX(MXWARP_SIZE / BLOCK_SIZE, 1);
  constexpr int NUM_THREAD_GROUPS = NUM_THREADS / THREAD_GROUP_SIZE; // Note: This assumes THREAD_GROUP_SIZE divides NUM_THREADS
  constexpr int NUM_TOKENS_PER_THREAD_GROUP = DIVIDE_ROUND_UP(BLOCK_SIZE, MXWARP_SIZE);
  constexpr int NUM_WARPS = NUM_THREADS >> 6;
  const int thread_idx = threadIdx.x;
  const int warp_idx = thread_idx >> 6;
  const int lane = thread_idx & 63;

  const int num_queries_per_kv = num_heads / num_kv_heads;
  const int kv_head_idx = head_idx / num_queries_per_kv;
  const float alibi_slope = alibi_slopes == nullptr ? 0.f : alibi_slopes[head_idx];
  int offset0 = seq_idx * num_heads;
  int offset1 = offset0 * HEAD_SIZE;
  int offset2 = head_idx * HEAD_SIZE;
  int offset3 = head_idx * max_num_partitions;
  int offset4 = offset0 * max_num_partitions;
  int offset5 = seq_idx * max_num_blocks_per_seq;
  int num_warps2 = NUM_WARPS << 1;

  // The vector size is configured in such a way that the threads in a thread group
  // fetch or compute 16 bytes at a time.
  // For example, if the size of a thread group is 4 and the data type is half,
  // then the vector size is 16 / (4 * sizeof(half)) == 2.
  constexpr int VEC_SIZE = 16 / sizeof(scalar_t);
  using K_vec = typename Vec<scalar_t, VEC_SIZE>::Type;
  using Q_vec = typename Vec<scalar_t, VEC_SIZE>::Type;
  using Q_vec_l = typename FloatVec<Q_vec>::Type;
#ifdef ENABLE_FP8_E5M2
  using Quant_vec = typename Vec<cache_t, VEC_SIZE>::Type;
#endif

  constexpr int NUM_ELEMS_PER_THREAD = HEAD_SIZE / THREAD_GROUP_SIZE;
  constexpr int NUM_VECS_PER_THREAD = NUM_ELEMS_PER_THREAD / VEC_SIZE;

  const int thread_group_idx = thread_idx / THREAD_GROUP_SIZE;
  const int thread_group_offset = thread_idx % THREAD_GROUP_SIZE;

  // Load the query to registers.
  // Each thread in a thread group has a different part of the query.
  // For example, if the the thread group size is 4, then the first thread in the group
  // has 0, 4, 8, ... th vectors of the query, and the second thread has 1, 5, 9, ...
  // th vectors of the query, and so on.
  // NOTE(woosuk): Because q is split from a qkv tensor, it may not be contiguous.
  const scalar_t* q_ptr = q + seq_idx * q_stride + head_idx * HEAD_SIZE;
  __shared__ Q_vec_l q_vecs[THREAD_GROUP_SIZE][NUM_VECS_PER_THREAD];
#pragma unroll
  for (int i = thread_group_idx; i < NUM_VECS_PER_THREAD; i += NUM_THREAD_GROUPS) {
    const int vec_idx = thread_group_offset + i * THREAD_GROUP_SIZE;
    Q_vec reg = *reinterpret_cast<const Q_vec*>(q_ptr + vec_idx * VEC_SIZE);
    convert<scalar_t, Q_vec, Q_vec_l, VEC_SIZE>(reg, q_vecs[thread_group_offset][i]);
  }
  __syncthreads();  // TODO(naed90): possible speedup if this is replaced with a
                    // memory wall right before we use q_vecs

  // Memory planning.
  extern __shared__ char shared_mem[];
  float* red_smem = reinterpret_cast<float*>(shared_mem);
  int * block_table_smem = reinterpret_cast<int*>(red_smem + num_warps2);
  float * logits = reinterpret_cast<float*>(block_table_smem + 512 + num_warps2);

  // Each thread group fetches x elements from the key at a time.
  constexpr int x = 16;
  float qk_max = -FLT_MAX;

  // blocksparse specific vars
  int bs_block_offset;
  int q_bs_block_id;
  if constexpr (IS_BLOCK_SPARSE) {
    // const int num_blocksparse_blocks = DIVIDE_ROUND_UP(seq_len,
    // blocksparse_block_size);
    q_bs_block_id = (seq_len - 1) / blocksparse_block_size;
    if (blocksparse_head_sliding_step >= 0)
      // sliding on q heads
      bs_block_offset =
          (tp_rank * num_heads + head_idx) * blocksparse_head_sliding_step + 1;
    else
      // sliding on kv heads
      bs_block_offset = (tp_rank * num_kv_heads + kv_head_idx) *
                            (-blocksparse_head_sliding_step) +
                        1;
  }

  // Iterate over the key blocks.
  // Each warp fetches a block of keys for each iteration.
  // Each thread group in a warp fetches a key from the block, and computes
  // dot product with the query.
  const int* block_table = block_tables + offset5 + start_block_idx;
  //load block_table to share memory
  for(int i = threadIdx.x; i < num_blocks; i += blockDim_x){
    block_table_smem[i] = block_table[i];
  }
  int block_idx0 = start_block_idx + warp_idx;
  __syncthreads();
  int kv_offset0, kv_offset1;
  K_vec load_k[NUM_VECS_PER_THREAD];
  K_vec compute_k[NUM_VECS_PER_THREAD];
  int k_offset[NUM_VECS_PER_THREAD];
  kv_offset0 = block_table_smem[block_idx0 - start_block_idx];
  kv_offset1 = block_table_smem[block_idx0 + NUM_WARPS - start_block_idx];
  #pragma unroll
  for(int j = 0; j < NUM_VECS_PER_THREAD; j++) {
    const int vec_idx = (thread_group_offset + j * THREAD_GROUP_SIZE)*VEC_SIZE;
    const int offset1 = vec_idx >> 4;
    const int offset2 = vec_idx & 15;
    k_offset[j] = offset1 * BLOCK_SIZE * x + offset2;
  }

  const cache_t* ptr_k_cache = k_cache + kv_head_idx * kv_head_stride;
  if constexpr (IS_BLOCK_SPARSE) {
      const int k_bs_block_id = block_idx0 * BLOCK_SIZE / blocksparse_block_size;
      const bool is_remote =
          ((k_bs_block_id + bs_block_offset) % blocksparse_vert_stride == 0);
      const bool is_local =
          (k_bs_block_id > q_bs_block_id - blocksparse_local_blocks);
      if(is_remote || is_local) {
          for (int i = 0; i < NUM_TOKENS_PER_THREAD_GROUP; i++) {
            const int physical_block_offset = (thread_group_idx + i * MXWARP_SIZE) & (BLOCK_SIZE - 1);
            const cache_t* k_ptr = ptr_k_cache + static_cast<int64_t>(kv_offset0) * kv_block_stride + physical_block_offset * x;
        #pragma unroll
            for (int j = 0; j < NUM_VECS_PER_THREAD; j++) {
              load_k[j] = *reinterpret_cast<const K_vec*>(k_ptr + k_offset[j]);
            }
          }    
      } 
  } else {
    for (int i = 0; i < NUM_TOKENS_PER_THREAD_GROUP; i++) {
      const int physical_block_offset = (thread_group_idx + i * MXWARP_SIZE) & (BLOCK_SIZE - 1);
      const cache_t* k_ptr = ptr_k_cache + static_cast<int64_t>(kv_offset0) * kv_block_stride + physical_block_offset * x;
    #pragma unroll
      for (int j = 0; j < NUM_VECS_PER_THREAD; j++) {
        load_k[j] = *reinterpret_cast<const K_vec*>(k_ptr + k_offset[j]);
      }
    }
  }
  
  for (int block_idx = block_idx0; block_idx < end_block_idx; block_idx += NUM_WARPS) {
    for(int i = 0; i < NUM_TOKENS_PER_THREAD_GROUP; i++) {
      const int physical_block_offset = (thread_group_idx + i * MXWARP_SIZE) & (BLOCK_SIZE - 1);
      const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;
      #pragma unroll
      for(int j = 0; j < NUM_VECS_PER_THREAD; j++) {
        compute_k[j] = load_k[j];
      }
      if(block_idx < end_block_idx - NUM_WARPS) {
          kv_offset0 = kv_offset1;
	  int nblock_idx = block_idx + NUM_WARPS;
          kv_offset1 = block_table_smem[block_idx + num_warps2 - start_block_idx];
          if constexpr (IS_BLOCK_SPARSE) {
            const int k_bs_block_id = nblock_idx * BLOCK_SIZE / blocksparse_block_size;
            const bool is_remote =
                ((k_bs_block_id + bs_block_offset) % blocksparse_vert_stride == 0);
            const bool is_local =
                (k_bs_block_id > q_bs_block_id - blocksparse_local_blocks);
            if(is_remote || is_local) {
              const cache_t* k_ptr = ptr_k_cache + static_cast<int64_t>(kv_offset0) * kv_block_stride + physical_block_offset * x;
              #pragma unroll NUM_VECS_PER_THREAD
              for(int j = 0; j < NUM_VECS_PER_THREAD; j++) {
                  load_k[j] = *reinterpret_cast<const K_vec*>(k_ptr + k_offset[j]);
              }  
            }
          } else {
            const cache_t* k_ptr = ptr_k_cache + static_cast<int64_t>(kv_offset0) * kv_block_stride + physical_block_offset * x;
            #pragma unroll NUM_VECS_PER_THREAD
            for(int j = 0; j < NUM_VECS_PER_THREAD; j++) {
                load_k[j] = *reinterpret_cast<const K_vec*>(k_ptr + k_offset[j]);
            }
          }
      }
      if constexpr (IS_BLOCK_SPARSE) {
      const int k_bs_block_id = block_idx * BLOCK_SIZE / blocksparse_block_size;
      const bool is_remote =
          ((k_bs_block_id + bs_block_offset) % blocksparse_vert_stride == 0);
      const bool is_local =
          (k_bs_block_id > q_bs_block_id - blocksparse_local_blocks);
      if (!is_remote && !is_local) {
        for (int i = 0; i < NUM_TOKENS_PER_THREAD_GROUP; i++) {
          const int physical_block_offset =
              (thread_group_idx + i * MXWARP_SIZE) & (BLOCK_SIZE - 1);
          const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;

          if (thread_group_offset == 0) {
            // NOTE(linxihui): assign very large number to skipped tokens to
            // avoid contribution to the sumexp softmax normalizer. This will
            // not be used at computing sum(softmax*v) as the blocks will be
            // skipped.
            logits[token_idx - start_token_idx] = -FLT_MAX;
          }
        }
        continue;
      }
    }

      // Compute dot product.
      // This includes a reduction across the threads in the same thread group.
      // Compute the parallel products for Q*K^T (treat vector lanes separately).
      float qk = 0.0f;
      v2f f2_qk = {0,0};
      #pragma unroll
      for(int j = 0; j < NUM_VECS_PER_THREAD; j++) {
        atten_dot<Q_vec_l, K_vec, v2f, VEC_SIZE>(q_vecs[thread_group_offset][j], compute_k[j],f2_qk);
      }
      qk = f2_qk[0] + f2_qk[1];
      #pragma unroll
      for (int mask = THREAD_GROUP_SIZE / 2; mask >= 1; mask /= 2) {
        qk += MXVLLM_SHFL_XOR_SYNC(qk, mask);
      }

      qk = scale * qk;
      // Add the ALiBi bias if slopes are given.
      qk += (alibi_slope != 0) ? alibi_slope * (token_idx - seq_len + 1) : 0;

      if (thread_group_offset == 0) {
        // Store the partial reductions to shared memory.
        // NOTE(woosuk): It is required to zero out the masked logits.
        const bool mask = token_idx >= seq_len;
        logits[token_idx - start_token_idx] = mask ? 0.f : qk;
        // Update the max value.
        qk_max = mask ? qk_max : fmaxf(qk_max, qk);
      }
    }
  }

  // Perform reduction across the threads in the same warp to get the
  // max qk value for each "warp" (not across the thread block yet).
  // The 0-th thread of each thread group already has its max qk value.
#pragma unroll
  for (int mask = MXWARP_SIZE / 2; mask >= THREAD_GROUP_SIZE; mask /= 2) {
    qk_max = fmaxf(qk_max, MXVLLM_SHFL_XOR_SYNC(qk_max, mask));
  }
  if (lane == 0) {
    red_smem[warp_idx] = qk_max;
  }
  __syncthreads();

  // TODO(woosuk): Refactor this part.
  // Get the max qk value for the sequence.
  qk_max = lane < NUM_WARPS ? red_smem[lane] : -FLT_MAX;
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    qk_max = fmaxf(qk_max, MXVLLM_SHFL_XOR_SYNC(qk_max, mask));
  }
  // Broadcast the max qk value to all threads.
  qk_max = MXVLLM_SHFL_SYNC(qk_max, 0);

  // Get the sum of the exp values.
  float exp_sum = 0.f;
  for (int i = thread_idx; i < num_tokens; i += NUM_THREADS) {
    float val = __builtin_expf(logits[i] - qk_max);
    logits[i] = val;
    exp_sum += val;
  }
  exp_sum = mxblock_sum<NUM_WARPS>(&red_smem[NUM_WARPS], exp_sum);

  // Compute softmax.
  const float inv_sum = __builtin_mxc_rcpf(exp_sum + 1e-6f);
  for (int i = thread_idx; i < num_tokens; i += NUM_THREADS) {
    float val = logits[i];
    val *= inv_sum;
    logits[i] = convert<cache_t>(val);
  }
  __syncthreads();

  // If partitioning is enabled, store the max logit and exp_sum.
  if (thread_idx == 0) {
    float* max_logits_ptr = max_logits + offset4
                                       + offset3
                                       + partition_idx;
    *max_logits_ptr = qk_max;
    float* exp_sums_ptr = exp_sums + offset4
                                   + offset3
                                   + partition_idx;
    *exp_sums_ptr = exp_sum;
  }

  constexpr int V_VEC_SIZE = 16 / sizeof(scalar_t);
  constexpr int NUM_V_VECS_PER_THREAD = HEAD_SIZE / V_VEC_SIZE; 
  constexpr int NUM_COLS_PER_ITER = MAX(MXWARP_SIZE / NUM_V_VECS_PER_THREAD , 1); 
  constexpr int NUM_LGT_PER_COL = BLOCK_SIZE / NUM_COLS_PER_ITER; 
  constexpr int NUM_LANE = NUM_WARPS * NUM_COLS_PER_ITER;
  using V_vec = typename Vec<scalar_t, V_VEC_SIZE>::Type;
  using L_vec = typename Vec<scalar_t, V_VEC_SIZE>::Type;
  const int physical_block_offset = lane / NUM_V_VECS_PER_THREAD;
  const int laneid = lane % NUM_V_VECS_PER_THREAD;
  V_vec v_vecs[NUM_LGT_PER_COL];
  V_vec v_prev_vecs[NUM_LGT_PER_COL];
  float accs[V_VEC_SIZE];
  float reg_log[NUM_LGT_PER_COL];
  float reg_prev_log[NUM_LGT_PER_COL];

  #pragma unroll
  for(int i = 0; i < V_VEC_SIZE; i++) {
    accs[i] = 0.0f;
  }
  int token_idx, kv_stride, block_offset;
  kv_stride = BLOCK_SIZE * HEAD_SIZE ;
  kv_offset0 = block_table_smem[block_idx0 - start_block_idx];
  block_offset = NUM_COLS_PER_ITER * HEAD_SIZE;
  kv_offset1 = block_table_smem[block_idx0 + NUM_WARPS - start_block_idx];
  token_idx = block_idx0 * BLOCK_SIZE + physical_block_offset;
  const cache_t *v_ptr0 = v_cache + kv_head_idx * kv_stride + physical_block_offset * HEAD_SIZE;
  const cache_t* v_ptr = v_ptr0 + static_cast<int64_t>(kv_offset0) * kv_block_stride;
  float *ptr_logits = logits + token_idx - start_token_idx;

  if constexpr (IS_BLOCK_SPARSE) {
      int v_bs_block_id = block_idx0 * BLOCK_SIZE / blocksparse_block_size;
      if (((v_bs_block_id + bs_block_offset) % blocksparse_vert_stride == 0) ||
          ((v_bs_block_id > q_bs_block_id - blocksparse_local_blocks))) {
          #pragma unroll
          for(int i = 0; i < NUM_LGT_PER_COL; i++) {
              if(token_idx + i * NUM_COLS_PER_ITER < seq_len ) {
              const int idx = laneid * V_VEC_SIZE + i * block_offset;
              v_prev_vecs[i] = *reinterpret_cast<const V_vec*>(v_ptr + idx);
              reg_prev_log[i] = ptr_logits[i * NUM_COLS_PER_ITER];
              }
          }
      }
  } else {
        #pragma unroll
        for(int i = 0; i < NUM_LGT_PER_COL; i++) {
          if(token_idx + i * NUM_COLS_PER_ITER < seq_len ) {
            const int idx = laneid * V_VEC_SIZE + i * block_offset;
            v_prev_vecs[i] = *reinterpret_cast<const V_vec*>(v_ptr + idx);
            reg_prev_log[i] = ptr_logits[i * NUM_COLS_PER_ITER];
          }
        }
    }

  for(int block_idx = block_idx0; block_idx < end_block_idx; block_idx += NUM_WARPS) {
    int next_block = block_idx + NUM_WARPS;
    int nnext_block = next_block + NUM_WARPS;
    for(int i = 0; i < NUM_LGT_PER_COL; i++) {
      v_vecs[i] = v_prev_vecs[i]; reg_log[i] = reg_prev_log[i];
    }
    if(next_block < end_block_idx) {
      kv_offset0 = kv_offset1;
      kv_offset1 = block_table_smem[nnext_block - start_block_idx];
      token_idx = next_block * BLOCK_SIZE + physical_block_offset;
      const cache_t* v_ptr = v_ptr0 + static_cast<int64_t>(kv_offset0) * kv_block_stride;
      ptr_logits = logits + token_idx - start_token_idx;
      if constexpr (IS_BLOCK_SPARSE) {
        int v_bs_block_id = next_block * BLOCK_SIZE / blocksparse_block_size;
        if (((v_bs_block_id + bs_block_offset) % blocksparse_vert_stride == 0) ||
            ((v_bs_block_id > q_bs_block_id - blocksparse_local_blocks))) {
              if(next_block == num_seq_blocks - 1) {
                #pragma unroll
                for(int i = 0; i < NUM_LGT_PER_COL; i++) {
                    if(token_idx + i * NUM_COLS_PER_ITER < seq_len) {
                        const int idx = laneid * V_VEC_SIZE + i * block_offset;
                        v_prev_vecs[i] = *reinterpret_cast<const V_vec*>(v_ptr + idx);
                        reg_prev_log[i] = ptr_logits[i * NUM_COLS_PER_ITER];
                    }
                }
              } else {
                #pragma unroll
                for(int i = 0; i < NUM_LGT_PER_COL; i++) {
                    const int idx = laneid * V_VEC_SIZE + i * block_offset;
                    v_prev_vecs[i] = *reinterpret_cast<const V_vec*>(v_ptr + idx);
                    reg_prev_log[i] = ptr_logits[i * NUM_COLS_PER_ITER];
                }
              }
        }
        } else {
          if(next_block == num_seq_blocks - 1) {
            #pragma unroll
            for(int i = 0; i < NUM_LGT_PER_COL; i++) {
                if(token_idx + i * NUM_COLS_PER_ITER < seq_len) {
                const int idx = laneid * V_VEC_SIZE + i * block_offset;
                v_prev_vecs[i] = *reinterpret_cast<const V_vec*>(v_ptr + idx);
                reg_prev_log[i] = ptr_logits[i * NUM_COLS_PER_ITER];
              }
            }
          } else {
            #pragma unroll
            for(int i = 0; i < NUM_LGT_PER_COL; i++) {
              const int idx = laneid * V_VEC_SIZE + i * block_offset;
              v_prev_vecs[i] = *reinterpret_cast<const V_vec*>(v_ptr + idx);
              reg_prev_log[i] = ptr_logits[i * NUM_COLS_PER_ITER];
            }
          }
        }
    }
    if constexpr (IS_BLOCK_SPARSE) {
        int v_bs_block_id = block_idx * BLOCK_SIZE / blocksparse_block_size;
        if (!((v_bs_block_id + bs_block_offset) % blocksparse_vert_stride == 0) &&
            !((v_bs_block_id > q_bs_block_id - blocksparse_local_blocks))) {
          continue;
        }
    }

    token_idx = block_idx * BLOCK_SIZE + physical_block_offset;
    float *ptr_logits = logits + token_idx - start_token_idx;
    for(int i = 0; i < NUM_LGT_PER_COL; i++) {
      if(token_idx + i * NUM_COLS_PER_ITER < seq_len) {
        scalar_t* v_vec_ptr = reinterpret_cast<scalar_t*>(&v_vecs[i]);
        #pragma unroll
        for(int j = 0; j < V_VEC_SIZE; j+=2) {
          atten_mul_opt2(v_vec_ptr, reg_log[i], j, accs[j],accs[j + 1]);
        }
      }
    } 
  }

  __syncthreads();
  float* out_smem = reinterpret_cast<float*>(shared_mem);
  float*ptr_out_smem = out_smem + warp_idx * HEAD_SIZE*NUM_COLS_PER_ITER + physical_block_offset * HEAD_SIZE + laneid* V_VEC_SIZE;
  #pragma unroll
  for(int i = 0; i < V_VEC_SIZE; i++) {
    ptr_out_smem[i] = accs[i];
  }
   __syncthreads();
  const int num_partitions = DIVIDE_ROUND_UP(seq_len, PARTITION_SIZE);

  if(partition_idx * PARTITION_SIZE < seq_len) {
    scalar_t* out_ptr = out + (offset4 + offset3 + partition_idx) * HEAD_SIZE;
    if(threadIdx.x < HEAD_SIZE) {
      int length = NUM_LANE * HEAD_SIZE;
      float r = 0;
      for(int i = threadIdx.x; i < length; i += HEAD_SIZE) {
        r += out_smem[i];
      }
      from_float(*(out_ptr + threadIdx.x), r);
    }
  }

  __syncthreads();
  bool last_block = false;
  if(threadIdx.x == blockDim_x - 1) {
    if(atomicAdd(block_count + head_idx * grid_dim_y + seq_idx, 1) == max_num_partitions - 1) {
      last_block = true;
    }
  }
  if (__syncthreads_or(last_block)) {
      if(count_init_once) {
        if(threadIdx.x == blockDim_x - 2){
          *(block_count + head_idx * grid_dim_y + seq_idx) = 0;
        }
      }
      if(num_partitions == 1) {
        scalar_t* out_ptr = final_out + offset1 + offset2;
        const scalar_t* tmp_out_ptr = out + (offset4 + offset3) * HEAD_SIZE;
        V_vec* ptr_vec_out = (V_vec*)out_ptr;
        V_vec* ptr_vec_in = (V_vec*)tmp_out_ptr;
        int num = HEAD_SIZE / V_VEC_SIZE;
        for (int i = threadIdx.x; i < num; i += blockDim_x) {
          ptr_vec_out[i] = ptr_vec_in[i];
        }
	return;
      }
      // Load max logits to shared memory.
      float* shared_max_logits = reinterpret_cast<float*>(shared_mem);
      float * red_smem = shared_max_logits + (num_partitions << 1 );
      const float* max_logits_ptr = max_logits + offset4 + offset3;
      float max_logit = -FLT_MAX;
      for (int i = threadIdx.x; i < num_partitions; i += blockDim_x) {
        const float l = max_logits_ptr[i];
        shared_max_logits[i] = l;
      }
      for(int i = threadIdx.x; i < num_partitions; i += blockDim_x) {
        max_logit = fmaxf(max_logit, shared_max_logits[i]);
      }
      __syncthreads();
      // Get the global max logit.
      // Reduce within the warp.
      #pragma unroll
      for (int mask = MXWARP_SIZE / 2; mask >= 1; mask /= 2) {
        max_logit = fmaxf(max_logit, MXVLLM_SHFL_XOR_SYNC(max_logit, mask));
      }
      
      if (lane == 0) {
        red_smem[warp_idx] = max_logit;
      }
      __syncthreads();
      // Reduce across warps.
      max_logit = lane < NUM_WARPS ? red_smem[lane] : -FLT_MAX;
    #pragma unroll
      for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
        max_logit = fmaxf(max_logit, MXVLLM_SHFL_XOR_SYNC(max_logit, mask));
      }
      // Broadcast the max value to all threads.
      max_logit = MXVLLM_SHFL_SYNC(max_logit, 0);

      // Load rescaled exp sums to shared memory.
      float* shared_exp_sums = reinterpret_cast<float*>(shared_mem + sizeof(float) * num_partitions);
      const float* exp_sums_ptr = exp_sums + offset4 + offset3;

      float global_exp_sum = 0.0f;  
      float * out_sm_ptr = reinterpret_cast<float*>(shared_mem + sizeof(float) * num_partitions * 2);
      for(int i = threadIdx.x; i < num_partitions; i += blockDim_x) {
        out_sm_ptr[i] = exp_sums_ptr[i];
      }

      for (int i = threadIdx.x; i < num_partitions; i += blockDim_x) {
        float l = shared_max_logits[i];
        float rescaled_exp_sum = out_sm_ptr[i] * __builtin_expf(l - max_logit);
        global_exp_sum += rescaled_exp_sum;
        shared_exp_sums[i] = rescaled_exp_sum;
      }
      __syncthreads();

      global_exp_sum = mxblock_sum<NUM_WARPS>(&red_smem[NUM_WARPS], global_exp_sum);
      const float inv_global_exp_sum = __builtin_mxc_rcpf(global_exp_sum + 1e-6f);
      
      for(int i = threadIdx.x; i < num_partitions; i += blockDim_x) {
          shared_exp_sums[i] = shared_exp_sums[i] * inv_global_exp_sum;
      }

      // Aggregate tmp_out to out.
      scalar_t* out_ptr = final_out + offset1 + offset2;
      const scalar_t* tmp_out_ptr = out + (offset1 + offset2) * max_num_partitions;
      scalar_t * out_sm_ptr_2 = reinterpret_cast<scalar_t*>(out_sm_ptr);
      int buffer_size = num_partitions * HEAD_SIZE / V_VEC_SIZE;
      for(int i = threadIdx.x; i < buffer_size; i += blockDim_x) {
          int offset = i * V_VEC_SIZE;
          V_vec reg = *reinterpret_cast<const V_vec*>(tmp_out_ptr + offset);
          *(V_vec *)(out_sm_ptr_2  + offset) = reg;
      }
      __syncthreads();
      
      if(threadIdx.x < HEAD_SIZE) {
        scalar_t * ptr_sm_out_ptr = out_sm_ptr_2 + threadIdx.x;
        float acc = 0.0f;
        int num_partitions2 = num_partitions >> 1 << 1;
        int j = 0;
        v2f vacc; vacc[0] = 0.0f; vacc[1] = 0.0f;
        for(; j < num_partitions2; j += 2) {
          v2f va; 
          scalar_t a0, a1;
          a0 = *(ptr_sm_out_ptr); ptr_sm_out_ptr += HEAD_SIZE;
          a1 = *(ptr_sm_out_ptr); ptr_sm_out_ptr += HEAD_SIZE;
          to_v2f(a0,a1,va);
          v2f vb; 
          vb[0] = shared_exp_sums[j]; vb[1] = shared_exp_sums[j + 1];
          vacc = __builtin_mxc_pk_fma_f32(va, vb, vacc);
        }
        acc = vacc[0] + vacc[1];
        for (; j < num_partitions; ++j) {
          acc += to_float(*ptr_sm_out_ptr) * shared_exp_sums[j];
          ptr_sm_out_ptr += HEAD_SIZE;
        }
        from_float(out_ptr[threadIdx.x], acc);
      }
  }
}

// Grid: (num_heads, num_seqs, 1).
template <typename scalar_t, typename cache_t, int HEAD_SIZE, int BLOCK_SIZE,
          int NUM_THREADS, vllm::Fp8KVCacheDataType KV_DTYPE,
          bool IS_BLOCK_SPARSE>
__global__ void paged_attention_v1_kernel(
    scalar_t* __restrict__ out,           // [num_seqs, num_heads, head_size]
    const scalar_t* __restrict__ q,       // [num_seqs, num_heads, head_size]
    const cache_t* __restrict__ k_cache,  // [num_blocks, num_kv_heads,
                                          // head_size/x, block_size, x]
    const cache_t* __restrict__ v_cache,  // [num_blocks, num_kv_heads,
                                          // head_size, block_size]
    const int num_kv_heads,               // [num_heads]
    const float scale,
    const int* __restrict__ block_tables,  // [num_seqs, max_num_blocks_per_seq]
    const int* __restrict__ seq_lens,      // [num_seqs]
    const int max_num_blocks_per_seq,
    const float* __restrict__ alibi_slopes,  // [num_heads]
    const int q_stride, const int kv_block_stride, const int kv_head_stride,
    const float* k_scale, const float* v_scale, const int tp_rank,
    const int blocksparse_local_blocks, const int blocksparse_vert_stride,
    const int blocksparse_block_size, const int blocksparse_head_sliding_step) {
  paged_attention_kernel<scalar_t, cache_t, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS,
                         KV_DTYPE, IS_BLOCK_SPARSE>(
      /* exp_sums */ nullptr, /* max_logits */ nullptr, out, q, k_cache,
      v_cache, num_kv_heads, scale, block_tables, seq_lens,
      max_num_blocks_per_seq, alibi_slopes, q_stride, kv_block_stride,
      kv_head_stride, k_scale, v_scale, tp_rank, blocksparse_local_blocks,
      blocksparse_vert_stride, blocksparse_block_size,
      blocksparse_head_sliding_step);
}

template <typename scalar_t, typename cache_t, int HEAD_SIZE, int BLOCK_SIZE,
          int NUM_THREADS, vllm::Fp8KVCacheDataType KV_DTYPE,
          bool IS_BLOCK_SPARSE>
__global__ void paged_attention_v1_32N_kernel(
    scalar_t* __restrict__ out,           // [num_seqs, num_heads, head_size]
    const scalar_t* __restrict__ q,       // [num_seqs, num_heads, head_size]
    const cache_t* __restrict__ k_cache,  // [num_blocks, num_kv_heads,
                                          // head_size/x, block_size, x]
    const cache_t* __restrict__ v_cache,  // [num_blocks, num_kv_heads,
                                          // head_size, block_size]
    const int num_kv_heads,               // [num_heads]
    const float scale,
    const int* __restrict__ block_tables,  // [num_seqs, max_num_blocks_per_seq]
    const int* __restrict__ seq_lens,      // [num_seqs]
    const int max_num_blocks_per_seq,
    const float* __restrict__ alibi_slopes,  // [num_heads]
    const int q_stride, const int kv_block_stride, const int kv_head_stride,
    const float* k_scale, const float* v_scale, const int tp_rank, const int blocksparse_local_blocks,
    const int blocksparse_vert_stride, const int blocksparse_block_size,
    const int blocksparse_head_sliding_step,const int max_num_partitions,const int num_heads) {
  paged_attention_kernel_32N<scalar_t, cache_t, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS,
                         KV_DTYPE, IS_BLOCK_SPARSE>(
      /* exp_sums */ nullptr, /* max_logits */ nullptr, out, q, k_cache,
      v_cache, num_kv_heads, scale, block_tables, seq_lens,
      max_num_blocks_per_seq, alibi_slopes, q_stride, kv_block_stride,
      kv_head_stride, k_scale, v_scale, tp_rank, blocksparse_local_blocks,
      blocksparse_vert_stride, blocksparse_block_size,
      blocksparse_head_sliding_step,max_num_partitions,num_heads);
}

// Grid: (num_heads, num_seqs, max_num_partitions).
template <typename scalar_t, typename cache_t, int HEAD_SIZE, int BLOCK_SIZE,
          int NUM_THREADS, vllm::Fp8KVCacheDataType KV_DTYPE,
          bool IS_BLOCK_SPARSE,
          int PARTITION_SIZE>
__global__ void paged_attention_v2_kernel(
    float* __restrict__ exp_sums,  // [num_seqs, num_heads, max_num_partitions]
    float* __restrict__ max_logits,       // [num_seqs, num_heads,
                                          // max_num_partitions]
    scalar_t* __restrict__ tmp_out,       // [num_seqs, num_heads,
                                          // max_num_partitions, head_size]
    const scalar_t* __restrict__ q,       // [num_seqs, num_heads, head_size]
    const cache_t* __restrict__ k_cache,  // [num_blocks, num_kv_heads,
                                          // head_size/x, block_size, x]
    const cache_t* __restrict__ v_cache,  // [num_blocks, num_kv_heads,
                                          // head_size, block_size]
    const int num_kv_heads,               // [num_heads]
    const float scale,
    const int* __restrict__ block_tables,  // [num_seqs, max_num_blocks_per_seq]
    const int* __restrict__ seq_lens,      // [num_seqs]
    const int max_num_blocks_per_seq,
    const float* __restrict__ alibi_slopes,  // [num_heads]
    const int q_stride, const int kv_block_stride, const int kv_head_stride,
    const float* k_scale, const float* v_scale, const int tp_rank,
    const int blocksparse_local_blocks, const int blocksparse_vert_stride,
    const int blocksparse_block_size, const int blocksparse_head_sliding_step) {
  paged_attention_kernel<scalar_t, cache_t, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS,
                         KV_DTYPE, IS_BLOCK_SPARSE, PARTITION_SIZE>(
      exp_sums, max_logits, tmp_out, q, k_cache, v_cache, num_kv_heads, scale,
      block_tables, seq_lens, max_num_blocks_per_seq, alibi_slopes, q_stride,
      kv_block_stride, kv_head_stride, k_scale, v_scale, tp_rank,
      blocksparse_local_blocks, blocksparse_vert_stride, blocksparse_block_size,
      blocksparse_head_sliding_step);
}

template <typename scalar_t, typename cache_t, int HEAD_SIZE, int BLOCK_SIZE,
          int NUM_THREADS, vllm::Fp8KVCacheDataType KV_DTYPE,
          bool IS_BLOCK_SPARSE,
          int PARTITION_SIZE>
__global__ void paged_attention_v2_32N_kernel(
    float* __restrict__ exp_sums,  // [num_seqs, num_heads, max_num_partitions]
    float* __restrict__ max_logits,       // [num_seqs, num_heads,
                                          // max_num_partitions]
    scalar_t* __restrict__ tmp_out,       // [num_seqs, num_heads,
                                          // max_num_partitions, head_size]
    const scalar_t* __restrict__ q,       // [num_seqs, num_heads, head_size]
    const cache_t* __restrict__ k_cache,  // [num_blocks, num_kv_heads,
                                          // head_size/x, block_size, x]
    const cache_t* __restrict__ v_cache,  // [num_blocks, num_kv_heads,
                                          // head_size, block_size]
    const int num_kv_heads,               // [num_heads]
    const float scale,
    const int* __restrict__ block_tables,  // [num_seqs, max_num_blocks_per_seq]
    const int* __restrict__ seq_lens,      // [num_seqs]
    const int max_num_blocks_per_seq,
    const float* __restrict__ alibi_slopes,  // [num_heads]
    const int q_stride, const int kv_block_stride, const int kv_head_stride,
    const float* k_scale, const float* v_scale, const int tp_rank, const int blocksparse_local_blocks,
    const int blocksparse_vert_stride, const int blocksparse_block_size,
    const int blocksparse_head_sliding_step,const int max_num_partitions,const int num_heads) {
  paged_attention_kernel_32N<scalar_t, cache_t, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS,
                         KV_DTYPE, IS_BLOCK_SPARSE, PARTITION_SIZE>(
      exp_sums, max_logits, tmp_out, q, k_cache, v_cache, num_kv_heads, scale,
      block_tables, seq_lens, max_num_blocks_per_seq, alibi_slopes, q_stride,
      kv_block_stride, kv_head_stride, k_scale, v_scale, tp_rank,
      blocksparse_local_blocks, blocksparse_vert_stride, blocksparse_block_size,
      blocksparse_head_sliding_step,max_num_partitions,num_heads);
}

template <typename scalar_t, typename cache_t, int HEAD_SIZE, int BLOCK_SIZE,
          int NUM_THREADS, vllm::Fp8KVCacheDataType KV_DTYPE,
          bool IS_BLOCK_SPARSE,
          int PARTITION_SIZE>
__global__ void paged_attention_v2_kernel_final(
    float* __restrict__ exp_sums,  // [num_seqs, num_heads, max_num_partitions]
    float* __restrict__ max_logits,       // [num_seqs, num_heads,
                                          // max_num_partitions]
    int* __restrict__ block_count,          // [num_seqs, num_heads]
    scalar_t* __restrict__ tmp_out,       // [num_seqs, num_heads,
                                          // max_num_partitions, head_size]
    scalar_t* __restrict__ final_out,      // [num_seqs, num_heads, head_size]
    const scalar_t* __restrict__ q,       // [num_seqs, num_heads, head_size]
    const cache_t* __restrict__ k_cache,  // [num_blocks, num_kv_heads,
                                          // head_size/x, block_size, x]
    const cache_t* __restrict__ v_cache,  // [num_blocks, num_kv_heads,
                                          // head_size, block_size]
    const int num_kv_heads,               // [num_heads]
    const float scale,
    const int* __restrict__ block_tables,  // [num_seqs, max_num_blocks_per_seq]
    const int* __restrict__ seq_lens,      // [num_seqs]
    const int max_num_blocks_per_seq,
    const float* __restrict__ alibi_slopes,  // [num_heads]
    const int q_stride, const int kv_block_stride, const int kv_head_stride,
    const float* k_scale, const float* v_scale, const int tp_rank, const int blocksparse_local_blocks,
    const int blocksparse_vert_stride, const int blocksparse_block_size,
    const int blocksparse_head_sliding_step,const int max_num_partitions,
    const int blockDim_x,
    const int num_heads,
    const int grid_dim_y, const bool count_init_once) {
  paged_attention_kernel_32N_final<scalar_t, cache_t, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS,
                         KV_DTYPE, IS_BLOCK_SPARSE, PARTITION_SIZE>(
      exp_sums, max_logits, block_count, tmp_out, final_out, q, k_cache, v_cache, num_kv_heads, scale,
      block_tables, seq_lens, max_num_blocks_per_seq, alibi_slopes, q_stride,
      kv_block_stride, kv_head_stride, k_scale, v_scale, tp_rank,
      blocksparse_local_blocks, blocksparse_vert_stride, blocksparse_block_size,
      blocksparse_head_sliding_step,max_num_partitions,blockDim_x, num_heads,grid_dim_y,count_init_once);
}

// Grid: (num_heads, num_seqs).
template <typename scalar_t, int HEAD_SIZE, int NUM_THREADS,
          int PARTITION_SIZE>
__global__ void paged_attention_v2_reduce_kernel(
    scalar_t* __restrict__ out,            // [num_seqs, num_heads, head_size]
    const float* __restrict__ exp_sums,    // [num_seqs, num_heads,
                                           // max_num_partitions]
    const float* __restrict__ max_logits,  // [num_seqs, num_heads,
                                           // max_num_partitions]
    const scalar_t* __restrict__ tmp_out,  // [num_seqs, num_heads,
                                           // max_num_partitions, head_size]
    const int* __restrict__ seq_lens,      // [num_seqs]
    const int max_num_partitions) {
  const int num_heads = gridDim.x;
  const int head_idx = blockIdx.x;
  const int seq_idx = blockIdx.y;
  const int seq_len = seq_lens[seq_idx];
  const int num_partitions = DIVIDE_ROUND_UP(seq_len, PARTITION_SIZE);
  if (num_partitions == 1) {
    // No need to reduce. Only copy tmp_out to out.
    scalar_t* out_ptr = out + seq_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE;
    const scalar_t* tmp_out_ptr = tmp_out + seq_idx * num_heads * max_num_partitions * HEAD_SIZE
                                          + head_idx * max_num_partitions * HEAD_SIZE;
    for (int i = threadIdx.x; i < HEAD_SIZE; i += blockDim.x) {
      out_ptr[i] = tmp_out_ptr[i];
    }
    // Terminate the thread block.
    return;
  }

  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  const int warp_idx = threadIdx.x / WARP_SIZE;
  const int lane = threadIdx.x % WARP_SIZE;

  // Size: 2 * num_partitions.
  extern __shared__ char shared_mem[];
  // Workspace for reduction.
  __shared__ float red_smem[2 * NUM_WARPS];

  // Load max logits to shared memory.
  float* shared_max_logits = reinterpret_cast<float*>(shared_mem);
  const float* max_logits_ptr = max_logits + seq_idx * num_heads * max_num_partitions
                                           + head_idx * max_num_partitions;
  float max_logit = -FLT_MAX;
  for (int i = threadIdx.x; i < num_partitions; i += blockDim.x) {
    const float l = max_logits_ptr[i];
    shared_max_logits[i] = l;
    max_logit = fmaxf(max_logit, l);
  }
  __syncthreads();

  // Get the global max logit.
  // Reduce within the warp.
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
    max_logit = fmaxf(max_logit, VLLM_SHFL_XOR_SYNC(max_logit, mask));
  }
  if (lane == 0) {
    red_smem[warp_idx] = max_logit;
  }
  __syncthreads();
  // Reduce across warps.
  max_logit = lane < NUM_WARPS ? red_smem[lane] : -FLT_MAX;
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    max_logit = fmaxf(max_logit, VLLM_SHFL_XOR_SYNC(max_logit, mask));
  }
  // Broadcast the max value to all threads.
  max_logit = VLLM_SHFL_SYNC(max_logit, 0);

  // Load rescaled exp sums to shared memory.
  float* shared_exp_sums = reinterpret_cast<float*>(shared_mem + sizeof(float) * num_partitions);
  const float* exp_sums_ptr = exp_sums + seq_idx * num_heads * max_num_partitions
                                       + head_idx * max_num_partitions;
  float global_exp_sum = 0.0f;
  for (int i = threadIdx.x; i < num_partitions; i += blockDim.x) {
    float l = shared_max_logits[i];
    float rescaled_exp_sum = exp_sums_ptr[i] * expf(l - max_logit);
    global_exp_sum += rescaled_exp_sum;
    shared_exp_sums[i] = rescaled_exp_sum;
  }
  __syncthreads();
  global_exp_sum = block_sum<NUM_WARPS>(&red_smem[NUM_WARPS], global_exp_sum);
  const float inv_global_exp_sum = __fdividef(1.0f, global_exp_sum + 1e-6f);

  // Aggregate tmp_out to out.
  const scalar_t* tmp_out_ptr = tmp_out + seq_idx * num_heads * max_num_partitions * HEAD_SIZE
                                        + head_idx * max_num_partitions * HEAD_SIZE;
  scalar_t* out_ptr = out + seq_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE;
#pragma unroll
  for (int i = threadIdx.x; i < HEAD_SIZE; i += NUM_THREADS) {
    float acc = 0.0f;
    for (int j = 0; j < num_partitions; ++j) {
      acc += to_float(tmp_out_ptr[j * HEAD_SIZE + i]) * shared_exp_sums[j] * inv_global_exp_sum;
    }
    from_float(out_ptr[i], acc);
  }
}

}  // namespace vllm

#undef WARP_SIZE
#undef MAX
#undef MIN
#undef DIVIDE_ROUND_UP
