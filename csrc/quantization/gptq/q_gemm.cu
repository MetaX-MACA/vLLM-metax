// 2025 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved. 
/*
Adapted from https://github.com/turboderp/exllamav2 and
https://github.com/qwopqwop200/GPTQ-for-LLaMa
*/

#include <cstdint>
#include <cstdio>

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "matrix_view.cuh"
#include "qdq_2.cuh"
#include "qdq_3.cuh"
#include "qdq_4.cuh"
#include "qdq_8.cuh"

#include "hgemm_gptq.h"
#include "scalar_type.hpp"

#include "hgemv_nn_splitk_gptq.hpp"
#include "hgemv_nn_splitk_gptq_int8.hpp"
#include "hgemv_selector.hpp"
#include "Hgemm_nn_128x32x128_8m1n8k_gptq-4bits.hpp"
#include "Hgemm_nn_128x32x128_8m1n8k_gptq-8bits.hpp"

namespace vllm {
namespace gptq {

#define BLOCK_KN_SIZE 128
#define BLOCK_M_SIZE_MAX 8
#define MAX_GROUPS_IN_BLOCK (BLOCK_KN_SIZE / 32)
#define MAX_Q_GEMM_ROWS 50
#define MAX_Q_GEMM_ROWS_8BIT 24
#define MAX_ALT_GEMM_ROWS 8
#define THREADS_X 32
#define THREADS_Y 32
#define DIVIDE(x, size) (((x) + (size) - 1) / (size))
#define QUANT_GROUP 128

__forceinline__ __device__ half2 dot22_8(half2 (&dq)[4], const half* a_ptr,
                                         const half2 g_result) {
  half2 result = {};
  const half2* a2_ptr = (const half2*)a_ptr;
#pragma unroll
  for (int i = 0; i < 4; i++) result = __hfma2(dq[i], *a2_ptr++, result);
  return __hadd2(result, g_result);
}

__forceinline__ __device__ float dot22_8_f(half2 (&dq)[4], const half* a_ptr) {
  half2 result = {};
  const half2* a2_ptr = (const half2*)a_ptr;
#pragma unroll
  for (int i = 0; i < 4; i++) result = __hfma2(dq[i], *a2_ptr++, result);
  return __half2float(__low2half(result)) + __half2float(__high2half(result));
}

__forceinline__ __device__ half2 dot22_8(half2 (&dq)[4], const half* a_ptr,
                                         const half2 g_result,
                                         const half qs_h) {
  half2 result = {};
  const half2* a2_ptr = (const half2*)a_ptr;
#pragma unroll
  for (int i = 0; i < 4; i++) result = __hfma2(dq[i], *a2_ptr++, result);
  return __hfma2(result, __halves2half2(qs_h, qs_h), g_result);
}

__forceinline__ __device__ half2 dot22_16(half2 (&dq)[8], const half* a_ptr,
                                          const half2 g_result,
                                          const half qs_h) {
  half2 result = {};
  const half2* a2_ptr = (const half2*)a_ptr;
#pragma unroll
  for (int i = 0; i < 8; i++) result = __hfma2(dq[i], *a2_ptr++, result);
  return __hfma2(result, __halves2half2(qs_h, qs_h), g_result);
}

__forceinline__ __device__ half2 dot22_32(half2 (&dq)[16], const half* a_ptr,
                                          const half2 g_result,
                                          const half qs_h) {
  half2 result = {};
  const half2* a2_ptr = (const half2*)a_ptr;
#pragma unroll
  for (int i = 0; i < 16; i += 1) result = __hfma2(dq[i], *a2_ptr++, result);
  return __hfma2(result, __halves2half2(qs_h, qs_h), g_result);
}

__forceinline__ __device__ float dot22_8_f(half2 (&dq)[4], const half* a_ptr,
                                           const float g_result,
                                           const float qs_f) {
  half2 result = {};
  const half2* a2_ptr = (const half2*)a_ptr;
#pragma unroll
  for (int i = 0; i < 4; i++) result = __hfma2(dq[i], *a2_ptr++, result);
  float result_f =
      __half2float(__low2half(result)) + __half2float(__high2half(result));
  return fma(result_f, qs_f, g_result);
}

__forceinline__ __device__ float dot22_16_f(half2 (&dq)[8], const half* a_ptr,
                                            const float g_result,
                                            const float qs_f) {
  half2 result = {};
  const half2* a2_ptr = (const half2*)a_ptr;
#pragma unroll
  for (int i = 0; i < 8; i++) result = __hfma2(dq[i], *a2_ptr++, result);
  float result_f =
      __half2float(__low2half(result)) + __half2float(__high2half(result));
  return fma(result_f, qs_f, g_result);
}

__forceinline__ __device__ float dot22_32_f(half2 (&dq)[16], const half* a_ptr,
                                            const float g_result,
                                            const float qs_f) {
  half2 result = {};
  const half2* a2_ptr = (const half2*)a_ptr;
#pragma unroll
  for (int i = 0; i < 16; i += 1) result = __hfma2(dq[i], *a2_ptr++, result);
  float result_f =
      __half2float(__low2half(result)) + __half2float(__high2half(result));
  return fma(result_f, qs_f, g_result);
}

__forceinline__ __device__ half dot22_8_h(half2 (&dq)[4], const half* a_ptr,
                                          const half g_result,
                                          const half qs_h) {
  // Use FP32 accumulator to avoid potential overflow since unscaled weights are
  // in the range -128..127

  float result = {};
#pragma unroll
  for (int i = 0; i < 4; i++) {
    half2 w01 = dq[i];
    float w0 = __low2float(w01);
    float w1 = __high2float(w01);
    float x0 = __half2float(*a_ptr++);
    float x1 = __half2float(*a_ptr++);
    result = fma(w0, x0, result);
    result = fma(w1, x1, result);
  }
  float qs = __half2float(qs_h);
  result *= qs;
  half result_h = __float2half_rn(result);
  return __hadd(result_h, g_result);
}

__forceinline__ __device__ half dot22_16_h(half2 (&dq)[8], const half* a_ptr,
                                           const half g_result,
                                           const half qs_h) {
  half2 result = {};
  const half2* a2_ptr = (const half2*)a_ptr;
#pragma unroll
  for (int i = 0; i < 8; i++) result = __hfma2(dq[i], *a2_ptr++, result);
  half result_h = __hadd(__low2half(result), __high2half(result));
  return __hfma(result_h, qs_h, g_result);
}

__forceinline__ __device__ half dot22_32_h(half2 (&dq)[16], const half* a_ptr,
                                           const half g_result,
                                           const half qs_h) {
  half2 result = {};
  const half2* a2_ptr = (const half2*)a_ptr;
#pragma unroll
  for (int i = 0; i < 16; i += 1) result = __hfma2(dq[i], *a2_ptr++, result);
  half result_h = __hadd(__low2half(result), __high2half(result));
  return __hfma(result_h, qs_h, g_result);
}

typedef void (*fp_gemm_half_q_half_gptq_kernel)(const half*, const uint32_t*,
                                                const uint32_t*, const half*,
                                                half*, const int, const int,
                                                const int, const int,
                                                const int*);

template <bool first_block, int m_count>
__global__ void gemm_half_q_half_gptq_4bit_kernel(
    const half* __restrict__ a, const uint32_t* __restrict__ b_q_weight,
    const uint32_t* __restrict__ b_gptq_qzeros,
    const half* __restrict__ b_gptq_scales, half* __restrict__ c,
    const int size_m, const int size_n, const int size_k, const int groups,
    const int* __restrict__ b_q_perm) {
  MatrixView_half a_(a, size_m, size_k);
  MatrixView_half_rw c_(c, size_m, size_n);
  MatrixView_q4_row b_gptq_qzeros_(b_gptq_qzeros, groups, size_n);
  MatrixView_half b_gptq_scales_(b_gptq_scales, groups, size_n);

  auto t = threadIdx.x;

  // Block
  auto offset_n = blockIdx.x * BLOCK_KN_SIZE * 4;
  auto offset_m = blockIdx.y * m_count;
  auto offset_k = blockIdx.z * BLOCK_KN_SIZE;

  int end_k = min(offset_k + BLOCK_KN_SIZE, size_k);

  int n = offset_n + t * 4;

  // Preload block_a
  __shared__ half block_a[m_count][BLOCK_KN_SIZE];

  if (offset_k + t < end_k) {
    for (int m = 0; m < m_count; ++m) {
      const half* a_ptr = a_.item_ptr(offset_m + m, 0);
      half* block_a_ptr = block_a[m];

      half a0;
      if (b_q_perm)
        a0 = a_ptr[b_q_perm[offset_k + t]];
      else
        a0 = a_ptr[offset_k + t];
      block_a_ptr[t] = a0;
    }
  }

  // Zero output
  if (n >= size_n) return;

  if (blockIdx.z == 0) {
    for (int m = 0; m < m_count; m++)
      *((uint64_t*)c_.item_ptr(offset_m + m, n)) = 0;
  }

  __syncthreads();

  // Find initial group
  int groupsize = size_k / groups;
  int group = offset_k / groupsize;
  int nextgroup = offset_k + groupsize;

  // a, b offset
  int qk = offset_k / (32 / 4);

  const uint32_t* b_ptr = b_q_weight + qk * size_n + n;
  const half* a_ptr = &block_a[0][0];
  int a_stride = BLOCK_KN_SIZE;

  // Initial group
  int zeros[4];
  float scales[4];
  half2 z1z16[4][2];
  half2 y1y16[4][2];
  b_gptq_qzeros_.item4(zeros, group, n);
  b_gptq_scales_.item4_f(scales, group, n);
  dequant_4bit_8_prep_zero(zeros[0] + 1, z1z16[0], y1y16[0]);
  dequant_4bit_8_prep_zero(zeros[1] + 1, z1z16[1], y1y16[1]);
  dequant_4bit_8_prep_zero(zeros[2] + 1, z1z16[2], y1y16[2]);
  dequant_4bit_8_prep_zero(zeros[3] + 1, z1z16[3], y1y16[3]);

  // Column result
  float block_c[m_count][4] = {};

  // Dequantize and multiply
  int k = offset_k;
  while (k < end_k) {
    if (k == nextgroup) {
      group++;
      nextgroup += groupsize;
      b_gptq_qzeros_.item4(zeros, group, n);
      b_gptq_scales_.item4_f(scales, group, n);
      dequant_4bit_8_prep_zero(zeros[0] + 1, z1z16[0], y1y16[0]);
      dequant_4bit_8_prep_zero(zeros[1] + 1, z1z16[1], y1y16[1]);
      dequant_4bit_8_prep_zero(zeros[2] + 1, z1z16[2], y1y16[2]);
      dequant_4bit_8_prep_zero(zeros[3] + 1, z1z16[3], y1y16[3]);
    }

#pragma unroll
    for (int j = 0; j < 4; j++) {
      const int4* b_ptr4 = (int4*)b_ptr;
      int4 load_int4 = *b_ptr4;

      half2 dq[4][4];
      dequant_4bit_8_gptq(load_int4.x, dq[0], z1z16[0], y1y16[0], size_n,
                          false);
      dequant_4bit_8_gptq(load_int4.y, dq[1], z1z16[1], y1y16[1], size_n,
                          false);
      dequant_4bit_8_gptq(load_int4.z, dq[2], z1z16[2], y1y16[2], size_n,
                          false);
      dequant_4bit_8_gptq(load_int4.w, dq[3], z1z16[3], y1y16[3], size_n,
                          false);

#pragma unroll
      for (int m = 0; m < m_count; m++) {
        block_c[m][0] = fma(dot22_8_f(dq[0], a_ptr + m * a_stride), scales[0],
                            block_c[m][0]);
        block_c[m][1] = fma(dot22_8_f(dq[1], a_ptr + m * a_stride), scales[1],
                            block_c[m][1]);
        block_c[m][2] = fma(dot22_8_f(dq[2], a_ptr + m * a_stride), scales[2],
                            block_c[m][2]);
        block_c[m][3] = fma(dot22_8_f(dq[3], a_ptr + m * a_stride), scales[3],
                            block_c[m][3]);
      }

      b_ptr += size_n;
      a_ptr += 8;
    }

    k += 32;
  }

  for (int m = 0; m < m_count; m++) {
    half2* out = (half2*)c_.item_ptr(offset_m + m, n);
    half2 result01 = __halves2half2(__float2half_rn(block_c[m][0]),
                                    __float2half_rn(block_c[m][1]));
    half2 result23 = __halves2half2(__float2half_rn(block_c[m][2]),
                                    __float2half_rn(block_c[m][3]));
    atomicAdd(out, result01);
    atomicAdd(out + 1, result23);
  }
}

template <bool first_block, int m_count>
__global__ void gemm_half_q_half_gptq_2bit_kernel(
    const half* __restrict__ a, const uint32_t* __restrict__ b_q_weight,
    const uint32_t* __restrict__ b_gptq_qzeros,
    const half* __restrict__ b_gptq_scales, half* __restrict__ c,
    const int size_m, const int size_n, const int size_k, const int groups,
    const int* __restrict__ b_q_perm) {
  MatrixView_half a_(a, size_m, size_k);
  MatrixView_half_rw c_(c, size_m, size_n);
  MatrixView_q2_row b_gptq_qzeros_(b_gptq_qzeros, groups, size_n);
  MatrixView_half b_gptq_scales_(b_gptq_scales, groups, size_n);

  auto t = threadIdx.x;

  // Block
  auto offset_n = blockIdx.x * BLOCK_KN_SIZE * 4;
  auto offset_m = blockIdx.y * m_count;
  auto offset_k = blockIdx.z * BLOCK_KN_SIZE;

  int end_k = min(offset_k + BLOCK_KN_SIZE, size_k);

  int n = offset_n + t * 4;

  // Preload block_a
  __shared__ half block_a[m_count][BLOCK_KN_SIZE];

  if (offset_k + t < end_k) {
    for (int m = 0; m < m_count; ++m) {
      const half* a_ptr = a_.item_ptr(offset_m + m, 0);
      half* block_a_ptr = block_a[m];

      half a0;
      if (b_q_perm)
        a0 = a_ptr[b_q_perm[offset_k + t]];
      else
        a0 = a_ptr[offset_k + t];
      block_a_ptr[t] = a0;
    }
  }

  // Zero output
  if (n >= size_n) return;

  if (blockIdx.z == 0) {
    for (int m = 0; m < m_count; m++)
      *((uint64_t*)c_.item_ptr(offset_m + m, n)) = 0;
  }

  __syncthreads();

  // Find initial group
  int groupsize = size_k / groups;
  int group = offset_k / groupsize;
  int nextgroup = offset_k + groupsize;

  // a, b offset
  int qk = offset_k / (32 / 2);

  const uint32_t* b_ptr = b_q_weight + qk * size_n + n;
  const half* a_ptr = &block_a[0][0];
  int a_stride = BLOCK_KN_SIZE;

  // Initial group
  int zeros[4];
  half scales[4];
  b_gptq_qzeros_.item4(zeros, group, n);
  b_gptq_scales_.item4(scales, group, n);
  // Column result
  half block_c[m_count][4] = {};

  // Dequantize and multiply
  int k = offset_k;
  while (k < end_k) {
    if (k == nextgroup) {
      group++;
      nextgroup += groupsize;
      b_gptq_qzeros_.item4(zeros, group, n);
      b_gptq_scales_.item4(scales, group, n);
    }

#pragma unroll
    for (int j = 0; j < 1; j++) {
      const int4* b_ptr4 = (int4*)b_ptr;
      int4 load_int4 = *b_ptr4;

      half2 dq[4][8];
      dequant_2bit_16(load_int4.x, dq[0], size_n, zeros[0] + 1);
      dequant_2bit_16(load_int4.y, dq[1], size_n, zeros[1] + 1);
      dequant_2bit_16(load_int4.z, dq[2], size_n, zeros[2] + 1);
      dequant_2bit_16(load_int4.w, dq[3], size_n, zeros[3] + 1);

#pragma unroll
      for (int m = 0; m < m_count; m++) {
        block_c[m][0] =
            dot22_16_h(dq[0], a_ptr + m * a_stride, block_c[m][0], scales[0]);
        block_c[m][1] =
            dot22_16_h(dq[1], a_ptr + m * a_stride, block_c[m][1], scales[1]);
        block_c[m][2] =
            dot22_16_h(dq[2], a_ptr + m * a_stride, block_c[m][2], scales[2]);
        block_c[m][3] =
            dot22_16_h(dq[3], a_ptr + m * a_stride, block_c[m][3], scales[3]);
      }

      b_ptr += size_n;
      a_ptr += 16;
    }

    k += 16;
  }

  for (int m = 0; m < m_count; m++) {
    half2* out = (half2*)c_.item_ptr(offset_m + m, n);
    half2 result01 = __halves2half2(block_c[m][0], block_c[m][1]);
    half2 result23 = __halves2half2(block_c[m][2], block_c[m][3]);
    atomicAdd(out, result01);
    atomicAdd(out + 1, result23);
  }
}

template <bool first_block, int m_count>
__global__ void gemm_half_q_half_gptq_3bit_kernel(
    const half* __restrict__ a, const uint32_t* __restrict__ b_q_weight,
    const uint32_t* __restrict__ b_gptq_qzeros,
    const half* __restrict__ b_gptq_scales, half* __restrict__ c,
    const int size_m, const int size_n, const int size_k, const int groups,
    const int* __restrict__ b_q_perm) {
  MatrixView_half a_(a, size_m, size_k);
  MatrixView_half_rw c_(c, size_m, size_n);
  MatrixView_q3_row b_gptq_qzeros_(b_gptq_qzeros, groups, size_n);
  MatrixView_half b_gptq_scales_(b_gptq_scales, groups, size_n);

  auto t = threadIdx.x;

  // Block
  auto offset_n = blockIdx.x * BLOCK_KN_SIZE * 4;
  auto offset_m = blockIdx.y * m_count;
  auto offset_k = blockIdx.z * BLOCK_KN_SIZE;

  int end_k = min(offset_k + BLOCK_KN_SIZE, size_k);

  int n = offset_n + t * 4;

  // Preload block_a
  __shared__ half block_a[m_count][BLOCK_KN_SIZE];

  if (offset_k + t < end_k) {
    for (int m = 0; m < m_count; ++m) {
      const half* a_ptr = a_.item_ptr(offset_m + m, 0);
      half* block_a_ptr = block_a[m];

      half a0;
      if (b_q_perm)
        a0 = a_ptr[b_q_perm[offset_k + t]];
      else
        a0 = a_ptr[offset_k + t];
      block_a_ptr[t] = a0;
    }
  }

  // Zero output
  if (n >= size_n) return;

  if (blockIdx.z == 0) {
    for (int m = 0; m < m_count; m++)
      *((uint64_t*)c_.item_ptr(offset_m + m, n)) = 0;
  }

  __syncthreads();

  // Find initial group
  int groupsize = size_k / groups;
  int group = offset_k / groupsize;
  int nextgroup = offset_k + groupsize;

  // a, b offset
  int qk = offset_k / 32 * 3;

  const uint32_t* b_ptr = b_q_weight + qk * size_n + n;
  const half* a_ptr = &block_a[0][0];
  int a_stride = BLOCK_KN_SIZE;

  // Initial group
  int zeros[4];
  half scales[4];
  b_gptq_qzeros_.item4(zeros, group, n);
  b_gptq_scales_.item4(scales, group, n);
  // Column result
  half block_c[m_count][4] = {};

  // Dequantize and multiply
  int k = offset_k;
  while (k < end_k) {
    if (k == nextgroup) {
      group++;
      nextgroup += groupsize;
      b_gptq_qzeros_.item4(zeros, group, n);
      b_gptq_scales_.item4(scales, group, n);
    }

#pragma unroll
    for (int j = 0; j < 1; j++) {
      int4 load_int4[3];
      load_int4[0] = *((int4*)b_ptr);
      b_ptr += size_n;
      load_int4[1] = *((int4*)b_ptr);
      b_ptr += size_n;
      load_int4[2] = *((int4*)b_ptr);
      b_ptr += size_n;

      half2 dq[4][16];
      dequant_3bit_32(load_int4[0].x, load_int4[1].x, load_int4[2].x, dq[0],
                      size_n, zeros[0] + 1);
      dequant_3bit_32(load_int4[0].y, load_int4[1].y, load_int4[2].y, dq[1],
                      size_n, zeros[1] + 1);
      dequant_3bit_32(load_int4[0].z, load_int4[1].z, load_int4[2].z, dq[2],
                      size_n, zeros[2] + 1);
      dequant_3bit_32(load_int4[0].w, load_int4[1].w, load_int4[2].w, dq[3],
                      size_n, zeros[3] + 1);

#pragma unroll
      for (int m = 0; m < m_count; m++) {
        block_c[m][0] =
            dot22_32_h(dq[0], a_ptr + m * a_stride, block_c[m][0], scales[0]);
        block_c[m][1] =
            dot22_32_h(dq[1], a_ptr + m * a_stride, block_c[m][1], scales[1]);
        block_c[m][2] =
            dot22_32_h(dq[2], a_ptr + m * a_stride, block_c[m][2], scales[2]);
        block_c[m][3] =
            dot22_32_h(dq[3], a_ptr + m * a_stride, block_c[m][3], scales[3]);
      }
      a_ptr += 32;
    }

    k += 32;
  }

  for (int m = 0; m < m_count; m++) {
    half2* out = (half2*)c_.item_ptr(offset_m + m, n);
    half2 result01 = __halves2half2(block_c[m][0], block_c[m][1]);
    half2 result23 = __halves2half2(block_c[m][2], block_c[m][3]);
    atomicAdd(out, result01);
    atomicAdd(out + 1, result23);
  }
}

template <bool first_block, int m_count>
__global__ void gemm_half_q_half_gptq_8bit_kernel(
    const half* __restrict__ a, const uint32_t* __restrict__ b_q_weight,
    const uint32_t* __restrict__ b_gptq_qzeros,
    const half* __restrict__ b_gptq_scales, half* __restrict__ c,
    const int size_m, const int size_n, const int size_k, const int groups,
    const int* __restrict__ b_q_perm) {
  MatrixView_half a_(a, size_m, size_k);
  MatrixView_half_rw c_(c, size_m, size_n);
  MatrixView_q8_row b_gptq_qzeros_(b_gptq_qzeros, groups, size_n);
  MatrixView_half b_gptq_scales_(b_gptq_scales, groups, size_n);

  auto t = threadIdx.x;

  // Block
  auto offset_n = blockIdx.x * BLOCK_KN_SIZE * 4;
  auto offset_m = blockIdx.y * m_count;
  auto offset_k = blockIdx.z * BLOCK_KN_SIZE;

  int end_k = min(offset_k + BLOCK_KN_SIZE, size_k);

  int n = offset_n + t * 4;

  // Preload block_a
  __shared__ half block_a[m_count][BLOCK_KN_SIZE];

  if (offset_k + t < end_k) {
    for (int m = 0; m < m_count; ++m) {
      const half* a_ptr = a_.item_ptr(offset_m + m, 0);
      half* block_a_ptr = block_a[m];

      half a0;
      if (b_q_perm)
        a0 = a_ptr[b_q_perm[offset_k + t]];
      else
        a0 = a_ptr[offset_k + t];
      block_a_ptr[t] = a0;
    }
  }

  // Zero output
  if (n >= size_n) return;

  if (blockIdx.z == 0) {
    for (int m = 0; m < m_count; m++)
      *((uint64_t*)c_.item_ptr(offset_m + m, n)) = 0;
  }

  __syncthreads();

  // Find initial group
  int groupsize = size_k / groups;
  int group = offset_k / groupsize;
  int nextgroup = offset_k + groupsize;

  // a, b offset
  int qk = offset_k / (32 / 8);

  const uint32_t* b_ptr = b_q_weight + qk * size_n + n;
  const half* a_ptr = &block_a[0][0];
  int a_stride = BLOCK_KN_SIZE;

  // Initial group
  int zeros[4];
  half scales[4];
  b_gptq_qzeros_.item4(zeros, group, n);
  b_gptq_scales_.item4(scales, group, n);
  // Column result
  half block_c[m_count][4] = {};

  // Dequantize and multiply
  int k = offset_k;
  while (k < end_k) {
    if (k == nextgroup) {
      group++;
      nextgroup += groupsize;
      b_gptq_qzeros_.item4(zeros, group, n);
      b_gptq_scales_.item4(scales, group, n);
    }

#pragma unroll
    for (int j = 0; j < 4; j++) {
      int4 load_int4[2];
      load_int4[0] = *((int4*)b_ptr);
      b_ptr += size_n;
      load_int4[1] = *((int4*)b_ptr);
      b_ptr += size_n;

      half2 dq[4][4];
      dequant_8bit_8(load_int4[0].x, load_int4[1].x, dq[0], size_n,
                     zeros[0] + 1);
      dequant_8bit_8(load_int4[0].y, load_int4[1].y, dq[1], size_n,
                     zeros[1] + 1);
      dequant_8bit_8(load_int4[0].z, load_int4[1].z, dq[2], size_n,
                     zeros[2] + 1);
      dequant_8bit_8(load_int4[0].w, load_int4[1].w, dq[3], size_n,
                     zeros[3] + 1);

      for (int m = 0; m < m_count; m++) {
        block_c[m][0] =
            dot22_8_h(dq[0], a_ptr + m * a_stride, block_c[m][0], scales[0]);
        block_c[m][1] =
            dot22_8_h(dq[1], a_ptr + m * a_stride, block_c[m][1], scales[1]);
        block_c[m][2] =
            dot22_8_h(dq[2], a_ptr + m * a_stride, block_c[m][2], scales[2]);
        block_c[m][3] =
            dot22_8_h(dq[3], a_ptr + m * a_stride, block_c[m][3], scales[3]);
      }
      a_ptr += 8;
    }
    k += 32;
  }

  for (int m = 0; m < m_count; m++) {
    half2* out = (half2*)c_.item_ptr(offset_m + m, n);
    half2 result01 = __halves2half2(block_c[m][0], block_c[m][1]);
    half2 result23 = __halves2half2(block_c[m][2], block_c[m][3]);
    atomicAdd(out, result01);
    atomicAdd(out + 1, result23);
  }
}

fp_gemm_half_q_half_gptq_kernel pick_gemm_half_q_half_gptq_kernel(
    bool first_block, const int m_count, const int bit) {
#define SELECT_KERNEL(M_COUNT)                                             \
  if (m_count == M_COUNT) {                                                \
    if (bit == 2) return gemm_half_q_half_gptq_2bit_kernel<true, M_COUNT>; \
    if (bit == 3) return gemm_half_q_half_gptq_3bit_kernel<true, M_COUNT>; \
    if (bit == 4) return gemm_half_q_half_gptq_4bit_kernel<true, M_COUNT>; \
    if (bit == 8) return gemm_half_q_half_gptq_8bit_kernel<true, M_COUNT>; \
  }
#if BLOCK_M_SIZE_MAX >= 1
  SELECT_KERNEL(1);
#endif
#if BLOCK_M_SIZE_MAX >= 2
  SELECT_KERNEL(2);
#endif
#if BLOCK_M_SIZE_MAX >= 3
  SELECT_KERNEL(3);
#endif
#if BLOCK_M_SIZE_MAX >= 4
  SELECT_KERNEL(4);
#endif
#if BLOCK_M_SIZE_MAX >= 5
  SELECT_KERNEL(5);
#endif
#if BLOCK_M_SIZE_MAX >= 6
  SELECT_KERNEL(6);
#endif
#if BLOCK_M_SIZE_MAX >= 7
  SELECT_KERNEL(7);
#endif
#if BLOCK_M_SIZE_MAX >= 8
  SELECT_KERNEL(8);
#endif
  return NULL;
}

template <typename T>
__global__ void blasMemset(T* data, size_t cnt, T init) {
  size_t threads = gridDim.x * blockDim.x;
  size_t itemsPerThread = (cnt + threads - 1) / threads;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (size_t loop = 0; loop < itemsPerThread && loop * threads + tid < cnt;
       ++loop) {
    data[loop * threads + tid] = init;
  }
}

template <typename dstT, typename srcT, typename scalarT>
__global__ void blasMemcpy(dstT* dst, const srcT* src, size_t cnt,
                           scalarT beta) {
  size_t threads = gridDim.x * blockDim.x;
  size_t itemsPerThread = (cnt + threads - 1) / threads;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (size_t loop = 0; loop < itemsPerThread && loop * threads + tid < cnt;
       ++loop) {
    dst[loop * threads + tid] = static_cast<double>(beta) *
                                static_cast<double>(src[loop * threads + tid]);
  }
}

template <typename reducT, typename outputT, typename scalarT>
__global__ void blasReduc(outputT* dC_out, outputT* dC_in, reducT* d_acc,
                          int count, int segs, scalarT beta) {
  using accT = float;
  size_t threads = gridDim.x * blockDim.x;
  size_t itemsPerThread = (count + threads - 1) / threads;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (size_t loop = 0; loop < itemsPerThread && (loop * threads + tid) < count;
       ++loop) {
    accT acc = static_cast<accT>(beta) *
               static_cast<accT>(dC_in[loop * threads + tid]);
    for (size_t SEG = 0; SEG < segs; ++SEG) {
      acc += static_cast<accT>(d_acc[SEG * count + loop * threads + tid]);
    }
    dC_out[loop * threads + tid] = static_cast<outputT>(acc);
  }
}

template <typename T_ACC, typename T_ACC_PACK, typename T, typename T_PACK>
__global__ void split_reduce(const T_ACC* src, const int row, const int splitk,
                             T* dest) {
  constexpr int ELEMS = sizeof(T_ACC_PACK) / sizeof(T_ACC);
  for (int i = threadIdx.x + blockDim.x * blockIdx.x;
       i < row * sizeof(T_ACC) / sizeof(T_ACC_PACK);
       i += blockDim.x * gridDim.x) {
    T_ACC_PACK p0 = ((T_ACC_PACK*)src)[i];
    T_ACC p0_a[ELEMS];
    for (int j = 0; j < ELEMS; j++) p0_a[j] = ((T_ACC*)&p0)[j];
    for (int k = 1; k < splitk; k++) {
      p0 = ((T_ACC_PACK*)src)[i + row / ELEMS * k];
      for (int j = 0; j < ELEMS; j++) {
        p0_a[j] += ((T_ACC*)&p0)[j];
      }
    }
    T dest_pack[ELEMS];
    for (int j = 0; j < ELEMS; j++) dest_pack[j] = p0_a[j];
    ((T_PACK*)dest)[i] = *(T_PACK*)dest_pack;
  }
}

template <int tileK, int tileN>
__global__ void perm_b(half* output, const half* input, const int* idx, int k,
                       int n, int ldb) {
  int tid = threadIdx.x;
  int row = blockIdx.x * tileK + tid;
  if (row < k) {
    int index = idx[row];
    int col_offset = blockIdx.y * tileN;
#pragma unroll 1
    for (int i = 0; (i < tileN) && ((col_offset + i) < n); ++i) {
      int col = col_offset + i;
      output[row + ldb * col] = input[index + ldb * col];
    }
  }
}

#define SWITCH_CASE_BATCH(BlockDimX, SplitK, BATCH) \
  case BATCH: {                                     \
    CALL_GEMM(BlockDimX, SplitK, BATCH)             \
    break;                                          \
  }

#define APPLY_HGEMM_BATCH(BlockDimX, SplitK, BATCH)   \
  switch (BATCH) {                                    \
    SWITCH_CASE_BATCH(BlockDimX, SplitK, 1)           \
    SWITCH_CASE_BATCH(BlockDimX, SplitK, 2)           \
    SWITCH_CASE_BATCH(BlockDimX, SplitK, 3)           \
    SWITCH_CASE_BATCH(BlockDimX, SplitK, 4)           \
    default: {                                        \
      launched = false;                               \
      printf("ERROR: Unsupported BATCH %d\n", BATCH); \
      break;                                          \
    }                                                 \
  }

#define SWITCH_CASE_BlockDimX(BlockDimX, SplitK, BATCH) \
  case BlockDimX: {                                     \
    APPLY_HGEMM_BATCH(BlockDimX, SplitK, BATCH)         \
    break;                                              \
  }

#define APPLY_HGEMM(BlockDimX, SplitK, BATCH)                             \
  switch (BlockDimX) {                                                    \
    SWITCH_CASE_BlockDimX(16, SplitK, BATCH)                              \
        SWITCH_CASE_BlockDimX(32, SplitK, BATCH)                          \
            SWITCH_CASE_BlockDimX(64, SplitK, BATCH)                      \
                SWITCH_CASE_BlockDimX(128, SplitK, BATCH)                 \
                    SWITCH_CASE_BlockDimX(256, SplitK, BATCH) default : { \
      launched = false;                                                   \
      printf("ERROR: Unsupported BlockDimX %d\n", BlockDimX);             \
      break;                                                              \
    }                                                                     \
  }

bool call_kernel(const half* srcB, const quant_packed_type* srcA,
                 quant_packed_type* zeros, half* scales, half* dst_D, int m,
                 int n, int k, int srcStride, int dstStride, int block_x,
                 int split_k, int bit, const int* b_perm_D = nullptr) {
  constexpr int ThreadBlock = 256;
  const dim3 threadBlock = {static_cast<unsigned int>(ThreadBlock)};
  const dim3 gridBlock = {
      static_cast<unsigned int>(
          m / (block_x * sizeof(float4) / sizeof(quant_packed_type))),
      static_cast<unsigned int>(split_k)};
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (split_k * QUANT_GROUP > k || k % QUANT_GROUP != 0) return false;
  if (block_x < 16 || n > 4) return false;
  bool launched = true;
#define CALL_GEMM(BX, SK, N)                                                   \
  if (bit == 4) {                                                              \
    if (QUANT_GROUP * SK == k && BX == 256) {                                  \
      hgemv_nn_splitk_gptq_tb256_bx256_kb128<N>                                \
          <<<gridBlock, threadBlock, 0, stream>>>(                             \
              srcB, srcA, zeros, scales, dst_D, m, n, k, srcStride, dstStride, \
              k / SK, b_perm_D);                                               \
    } else {                                                                   \
      hgemv_nn_splitk_gptq<256, BX, N><<<gridBlock, threadBlock, 0, stream>>>( \
          srcB, srcA, zeros, scales, dst_D, m, n, k, srcStride, dstStride,     \
          k / SK, b_perm_D);                                                   \
    }                                                                          \
  }                                                                            \
  if (bit == 8) {                                                              \
    hgemv_nn_splitk_gptq_int8<256, BX, N>                                      \
        <<<gridBlock, threadBlock, 0, stream>>>(srcB, srcA, zeros, scales,     \
                                                dst_D, m, n, k, srcStride,     \
                                                dstStride, k / SK, b_perm_D);  \
  }
  APPLY_HGEMM(block_x, split_k, n);
  return launched;
}

void gemm_half_q_half_cuda_part(const half* a, const uint32_t* b_q_weight,
                                const uint32_t* b_gptq_qzeros,
                                const half* b_gptq_scales, const int* b_q_perm,
                                half* c, int size_m, int size_n, int size_k,
                                int m_count, int groups, int bit, bool m_sign,
                                bool v_sign) {
  if ((bit == 4 || bit == 8) && m_sign && !v_sign) {
    const int threads_n = 256;
    const int tileM = 128;
    const int tileN = 32;
    const int tileK = 128;
    int lda = size_n;
    int ldb = size_k;
    int ldc = size_n;

    int splitk_iters = 3;
    bool isSplitk = splitk_iters > 1;

    uint32_t gridx = (size_n - 1) / tileM + 1;
    uint32_t gridy = (size_m - 1) / tileN + 1;
    uint32_t gridz = splitk_iters;

    uint32_t* zeros = const_cast<uint32_t*>(b_gptq_qzeros);
    half* scales = const_cast<half*>(b_gptq_scales);

    dim3 dimBlock(threads_n, 1, 1);
    dim3 dimGrid(gridx, gridy, gridz);
    float alpha = 1.0, beta = 0.0;
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    bool isBetaZero = (beta == 0.0);
    Operation_t trans_a = Operation_t(0);
    Operation_t trans_b = Operation_t(0);

    if (trans_a == OP_N && trans_b == OP_N && size_n % 8 == 0 &&
        size_k % 8 == 0) {
      half* dB_perm;
      if (b_q_perm != nullptr) {
        mcMallocAsync((void**)&dB_perm, ldb * size_m * sizeof(input_type),
                      stream);
        const int threads_n1 = 128;
        const int tileK1 = 128;
        const int tileN1 = 8;
        uint32_t gridx1 = (size_k - 1) / tileK1 + 1;
        uint32_t gridy1 = (size_m - 1) / tileN1 + 1;
        dim3 dimBlock1(threads_n1, 1, 1);
        dim3 dimGrid1(gridx1, gridy1, 1);
        perm_b<tileK1, tileN1><<<dimGrid1, dimBlock1, 0, stream>>>(
            dB_perm, a, b_q_perm, size_k, size_m, ldb);
      }
      const half* dB_actual = (b_q_perm != nullptr ? dB_perm : a);
      if (bit == 4) {
        if (!isSplitk) {
          if (isBetaZero) {
            Hgemm_nn_128x32x128_8m1n8k_gptq_4bit<OP_N, OP_N, true, tileM, tileN,
                                                 tileK, true, false>
                <<<dimGrid, dimBlock, 0, stream>>>(
                    size_n, size_m, size_k, alpha, beta, b_q_weight, lda,
                    dB_actual, ldb, c, c, ldc, zeros, scales);
          } else {
            Hgemm_nn_128x32x128_8m1n8k_gptq_4bit<OP_N, OP_N, false, tileM,
                                                 tileN, tileK, true, false>
                <<<dimGrid, dimBlock, 0, stream>>>(
                    size_n, size_m, size_k, alpha, beta, b_q_weight, lda,
                    dB_actual, ldb, c, c, ldc, zeros, scales);
          }
        } else {
          if (isBetaZero) {
            Hgemm_nn_128x32x128_8m1n8k_gptq_4bit<OP_N, OP_N, true, tileM, tileN,
                                                 tileK, true, true>
                <<<dimGrid, dimBlock, 0, stream>>>(
                    size_n, size_m, size_k, alpha, beta, b_q_weight, lda,
                    dB_actual, ldb, c, c, ldc, zeros, scales, splitk_iters, c);
          } else {
            acc_type* d_acc;
            mcMalloc(reinterpret_cast<void**>(&d_acc),
                     size_n * size_m * sizeof(acc_type));
            blasMemcpy<<<104, 512, 0, stream>>>(d_acc, c, size_n * size_m,
                                                beta);
            Hgemm_nn_128x32x128_8m1n8k_gptq_4bit<OP_N, OP_N, false, tileM,
                                                 tileN, tileK, true, true>
                <<<dimGrid, dimBlock, 0, stream>>>(
                    size_n, size_m, size_k, alpha, beta, b_q_weight, lda,
                    dB_actual, ldb, c, c, ldc, zeros, scales, splitk_iters,
                    d_acc);
            blasMemcpy<<<104, 512, 0, stream>>>(c, d_acc, size_n * size_m, 1);
            mcFree(d_acc);
          }
        }
      } else if (bit == 8) {
        if (!isSplitk) {
          if (isBetaZero) {
            Hgemm_nn_128x32x128_8m1n8k_gptq_8bit<OP_N, OP_N, true, tileM, tileN,
                                                 tileK, true, false>
                <<<dimGrid, dimBlock, 0, stream>>>(
                    size_n, size_m, size_k, alpha, beta, b_q_weight, lda,
                    dB_actual, ldb, c, c, ldc, zeros, scales, 1, nullptr,
                    nullptr);
          } else {
            Hgemm_nn_128x32x128_8m1n8k_gptq_8bit<OP_N, OP_N, false, tileM,
                                                 tileN, tileK, true, false>
                <<<dimGrid, dimBlock, 0, stream>>>(
                    size_n, size_m, size_k, alpha, beta, b_q_weight, lda,
                    dB_actual, ldb, c, c, ldc, zeros, scales);
          }
        } else {
          if (isBetaZero) {
            Hgemm_nn_128x32x128_8m1n8k_gptq_8bit<OP_N, OP_N, true, tileM, tileN,
                                                 tileK, true, true>
                <<<dimGrid, dimBlock, 0, stream>>>(
                    size_n, size_m, size_k, alpha, beta, b_q_weight, lda,
                    dB_actual, ldb, c, c, ldc, zeros, scales, splitk_iters, c);
          } else {
            acc_type* d_acc;
            mcMallocAsync(reinterpret_cast<void**>(&d_acc),
                          size_n * size_m * sizeof(acc_type), stream);
            blasMemcpy<<<104, 512, 0, stream>>>(d_acc, c, size_n * size_m,
                                                beta);
            Hgemm_nn_128x32x128_8m1n8k_gptq_8bit<OP_N, OP_N, false, tileM,
                                                 tileN, tileK, true, true>
                <<<dimGrid, dimBlock, 0, stream>>>(
                    size_n, size_m, size_k, alpha, beta, b_q_weight, lda,
                    dB_actual, ldb, c, c, ldc, zeros, scales, splitk_iters,
                    d_acc);
            blasMemcpy<<<104, 512, 0, stream>>>(c, d_acc, size_n * size_m, 1);
            mcFreeAsync(d_acc, stream);
          }
        }
      }
      if (b_q_perm != nullptr) {
        mcFreeAsync(dB_perm, stream);
      }
    } else {
      printf("Parameters not supported!\n");
      return;
    }
  } else if ((bit == 4 || bit == 8) && v_sign) {
    constexpr int m_per_thread = 4;
    uint32_t* zeros = const_cast<uint32_t*>(b_gptq_qzeros);
    half* scales = const_cast<half*>(b_gptq_scales);
    auto kernel_testing = [&](int bx, int sk) -> bool {
      return call_kernel(a, b_q_weight, zeros, scales, c, size_n, size_m,
                         size_k, size_n, size_n, bx, sk, bit, b_q_perm);
    };
    if (bit == 4) {
      auto& sl_warmup = hgemv_selector::GemvSelectorHolder<
          QUANT_GROUP, 8, m_per_thread>::selector(size_n, size_m, size_k);
      if (sl_warmup.valid()) {
        if (!sl_warmup.selected()) {
          sl_warmup.select_in_warmup(kernel_testing);
          mcMemset(c, 0, size_n * size_m * sizeof(half));
          sl_warmup.run(kernel_testing);
        } else {
          sl_warmup.run(kernel_testing);
        }
      }
    } else {
      auto& sl_warmup = hgemv_selector::GemvSelectorHolder<
          QUANT_GROUP, 4, m_per_thread>::selector(size_n, size_m, size_k);
      if (sl_warmup.valid()) {
        if (!sl_warmup.selected()) {
          sl_warmup.select_in_warmup(kernel_testing);
          mcMemset(c, 0, size_n * size_m * sizeof(half));
          sl_warmup.run(kernel_testing);
        } else {
          sl_warmup.run(kernel_testing);
        }
      }
    }
  } else {
    dim3 blockDim, gridDim;
    blockDim.x = BLOCK_KN_SIZE;
    blockDim.y = 1;
    blockDim.z = 1;
    gridDim.x = DIVIDE(size_n, BLOCK_KN_SIZE * 4);
    gridDim.y = DIVIDE(size_m, m_count);
    gridDim.z = DIVIDE(size_k, BLOCK_KN_SIZE);

    fp_gemm_half_q_half_gptq_kernel kernel =
        pick_gemm_half_q_half_gptq_kernel(true, m_count, bit);

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    kernel<<<gridDim, blockDim, 0, stream>>>(a, b_q_weight, b_gptq_qzeros,
                                             b_gptq_scales, c, size_m, size_n,
                                             size_k, groups, b_q_perm);
  }
}

__global__ void reconstruct_exllama_8bit_kernel(
    const uint32_t* __restrict__ b_q_weight, const int* __restrict__ b_q_perm,
    const uint32_t* __restrict__ b_gptq_qzeros,
    const half* __restrict__ b_gptq_scales, const int size_k, const int size_n,
    const int groups, half* __restrict__ b) {
  MatrixView_half_rw b_(b, size_k, size_n);
  MatrixView_q8_row b_gptq_qzeros_(b_gptq_qzeros, groups, size_n);
  MatrixView_half b_gptq_scales_(b_gptq_scales, groups, size_n);

  auto offset_k = BLOCK_KN_SIZE * blockIdx.y;
  auto offset_n = BLOCK_KN_SIZE * blockIdx.x * 4;

  int end_k = min(offset_k + BLOCK_KN_SIZE, size_k);

  // Preload remapping table
  __shared__ int perm[BLOCK_KN_SIZE];
  auto t = threadIdx.x;

  if (b_q_perm) {
    if (offset_k + t < size_k) perm[t] = b_q_perm[offset_k + t];
  }

  // Column
  int n = offset_n + t * 4;
  if (n >= size_n) return;

  // Find initial group
  int groupsize = size_k / groups;
  int group = offset_k / groupsize;
  int nextgroup = offset_k + groupsize;

  // b offset
  int qk = offset_k / (32 / 8);

  const uint32_t* b_ptr = b_q_weight + qk * size_n + n;

  // Initial zeros/scale
  int zeros[4];
  half2 scales[4];
  b_gptq_qzeros_.item4(zeros, group, n);
  b_gptq_scales_.item4_h2(scales, group, n);

  __syncthreads();

  int k = offset_k;
  int lk = 0;

  while (k < end_k) {
    if (k == nextgroup) {
      group++;
      nextgroup += groupsize;
      b_gptq_qzeros_.item4(zeros, group, n);
      b_gptq_scales_.item4_h2(scales, group, n);
    }

    for (int p = 0; p < 4; p++) {
      int4 load_int4[2];
      load_int4[0] = *((int4*)b_ptr);
      b_ptr += size_n;
      load_int4[1] = *((int4*)b_ptr);
      b_ptr += size_n;

      half2 dq[4][4];
      dequant_8bit_8(load_int4[0].x, load_int4[1].x, dq[0], size_n,
                     zeros[0] + 1);
      dequant_8bit_8(load_int4[0].y, load_int4[1].y, dq[1], size_n,
                     zeros[1] + 1);
      dequant_8bit_8(load_int4[0].z, load_int4[1].z, dq[2], size_n,
                     zeros[2] + 1);
      dequant_8bit_8(load_int4[0].w, load_int4[1].w, dq[3], size_n,
                     zeros[3] + 1);

      // half* dqh = (half*)dq;
      if (b_q_perm) {
        for (int j = 0; j < 4; j++) {
          for (int v = 0; v < 4; v++) dq[v][j] = __hmul2(scales[v], dq[v][j]);
          b_.set4(perm[lk++], n, __low2half(dq[0][j]), __low2half(dq[1][j]),
                  __low2half(dq[2][j]), __low2half(dq[3][j]));
          b_.set4(perm[lk++], n, __high2half(dq[0][j]), __high2half(dq[1][j]),
                  __high2half(dq[2][j]), __high2half(dq[3][j]));
        }
      } else {
        for (int j = 0; j < 4; j++) {
          for (int v = 0; v < 4; v++) dq[v][j] = __hmul2(scales[v], dq[v][j]);
          b_.set4(offset_k + lk++, n, __low2half(dq[0][j]),
                  __low2half(dq[1][j]), __low2half(dq[2][j]),
                  __low2half(dq[3][j]));
          b_.set4(offset_k + lk++, n, __high2half(dq[0][j]),
                  __high2half(dq[1][j]), __high2half(dq[2][j]),
                  __high2half(dq[3][j]));
        }
      }
    }
    k += 32;
  }
}

__global__ void reconstruct_exllama_4bit_kernel(
    const uint32_t* __restrict__ b_q_weight, const int* __restrict__ b_q_perm,
    const uint32_t* __restrict__ b_gptq_qzeros,
    const half* __restrict__ b_gptq_scales, const int size_k, const int size_n,
    const int groups, half* __restrict__ b) {
  MatrixView_half_rw b_(b, size_k, size_n);
  MatrixView_q4_row b_gptq_qzeros_(b_gptq_qzeros, groups, size_n);
  MatrixView_half b_gptq_scales_(b_gptq_scales, groups, size_n);

  auto offset_k = BLOCK_KN_SIZE * blockIdx.y;
  auto offset_n = BLOCK_KN_SIZE * blockIdx.x * 4;

  int end_k = min(offset_k + BLOCK_KN_SIZE, size_k);

  // Preload remapping table
  __shared__ int perm[BLOCK_KN_SIZE];
  auto t = threadIdx.x;

  if (b_q_perm) {
    if (offset_k + t < size_k) perm[t] = b_q_perm[offset_k + t];
  }

  // Column
  int n = offset_n + t * 4;
  if (n >= size_n) return;

  // Find initial group
  int groupsize = size_k / groups;
  int group = offset_k / groupsize;
  int nextgroup = offset_k + groupsize;

  // b offset
  int qk = offset_k / (32 / 4);

  const uint32_t* b_ptr = b_q_weight + qk * size_n + n;

  // Initial zeros/scale
  int zeros[4];
  half2 scales[4];
  half2 z1z16[4][2];
  half2 y1y16[4][2];
  b_gptq_qzeros_.item4(zeros, group, n);
  b_gptq_scales_.item4_h2(scales, group, n);
  dequant_4bit_8_prep_zero(zeros[0] + 1, z1z16[0], y1y16[0]);
  dequant_4bit_8_prep_zero(zeros[1] + 1, z1z16[1], y1y16[1]);
  dequant_4bit_8_prep_zero(zeros[2] + 1, z1z16[2], y1y16[2]);
  dequant_4bit_8_prep_zero(zeros[3] + 1, z1z16[3], y1y16[3]);

  __syncthreads();

  int k = offset_k;
  int lk = 0;

  while (k < end_k) {
    if (k == nextgroup) {
      group++;
      nextgroup += groupsize;
      b_gptq_qzeros_.item4(zeros, group, n);
      b_gptq_scales_.item4_h2(scales, group, n);
      dequant_4bit_8_prep_zero(zeros[0] + 1, z1z16[0], y1y16[0]);
      dequant_4bit_8_prep_zero(zeros[1] + 1, z1z16[1], y1y16[1]);
      dequant_4bit_8_prep_zero(zeros[2] + 1, z1z16[2], y1y16[2]);
      dequant_4bit_8_prep_zero(zeros[3] + 1, z1z16[3], y1y16[3]);
    }

    for (int p = 0; p < 4; p++) {
      half2 dq[4][4];
      const int4* b_ptr4 = (int4*)b_ptr;
      int4 load_int4 = *b_ptr4;

      dequant_4bit_8_gptq(load_int4.x, dq[0], z1z16[0], y1y16[0], size_n,
                          false);
      dequant_4bit_8_gptq(load_int4.y, dq[1], z1z16[1], y1y16[1], size_n,
                          false);
      dequant_4bit_8_gptq(load_int4.z, dq[2], z1z16[2], y1y16[2], size_n,
                          false);
      dequant_4bit_8_gptq(load_int4.w, dq[3], z1z16[3], y1y16[3], size_n,
                          false);

      b_ptr += size_n;
      // half* dqh = (half*)dq;
      if (b_q_perm) {
        for (int j = 0; j < 4; j++) {
          for (int v = 0; v < 4; v++) dq[v][j] = __hmul2(scales[v], dq[v][j]);
          b_.set4(perm[lk++], n, __low2half(dq[0][j]), __low2half(dq[1][j]),
                  __low2half(dq[2][j]), __low2half(dq[3][j]));
          b_.set4(perm[lk++], n, __high2half(dq[0][j]), __high2half(dq[1][j]),
                  __high2half(dq[2][j]), __high2half(dq[3][j]));
        }
      } else {
        for (int j = 0; j < 4; j++) {
          for (int v = 0; v < 4; v++) dq[v][j] = __hmul2(scales[v], dq[v][j]);
          b_.set4(offset_k + lk++, n, __low2half(dq[0][j]),
                  __low2half(dq[1][j]), __low2half(dq[2][j]),
                  __low2half(dq[3][j]));
          b_.set4(offset_k + lk++, n, __high2half(dq[0][j]),
                  __high2half(dq[1][j]), __high2half(dq[2][j]),
                  __high2half(dq[3][j]));
        }
      }
    }
    k += 32;
  }
}

__global__ void reconstruct_exllama_3bit_kernel(
    const uint32_t* __restrict__ b_q_weight, const int* __restrict__ b_q_perm,
    const uint32_t* __restrict__ b_gptq_qzeros,
    const half* __restrict__ b_gptq_scales, const int size_k, const int size_n,
    const int groups, half* __restrict__ b) {
  MatrixView_half_rw b_(b, size_k, size_n);
  MatrixView_q3_row b_gptq_qzeros_(b_gptq_qzeros, groups, size_n);
  MatrixView_half b_gptq_scales_(b_gptq_scales, groups, size_n);

  auto offset_k = BLOCK_KN_SIZE * blockIdx.y;
  auto offset_n = BLOCK_KN_SIZE * blockIdx.x * 4;

  int end_k = min(offset_k + BLOCK_KN_SIZE, size_k);

  // Preload remapping table
  __shared__ int perm[BLOCK_KN_SIZE];
  auto t = threadIdx.x;

  if (b_q_perm) {
    if (offset_k + t < size_k) perm[t] = b_q_perm[offset_k + t];
  }

  // Column
  int n = offset_n + t * 4;
  if (n >= size_n) return;

  // Find initial group
  int groupsize = size_k / groups;
  int group = offset_k / groupsize;
  int nextgroup = offset_k + groupsize;

  // b offset
  int qk = offset_k / 32 * 3;

  const uint32_t* b_ptr = b_q_weight + qk * size_n + n;

  // Initial zeros/scale
  int zeros[4];
  half2 scales[4];
  b_gptq_qzeros_.item4(zeros, group, n);
  b_gptq_scales_.item4_h2(scales, group, n);

  __syncthreads();

  int k = offset_k;
  int lk = 0;

  while (k < end_k) {
    if (k == nextgroup) {
      group++;
      nextgroup += groupsize;
      b_gptq_qzeros_.item4(zeros, group, n);
      b_gptq_scales_.item4_h2(scales, group, n);
    }

    for (int p = 0; p < 1; p++) {
      int4 load_int4[3];
      load_int4[0] = *((int4*)b_ptr);
      b_ptr += size_n;
      load_int4[1] = *((int4*)b_ptr);
      b_ptr += size_n;
      load_int4[2] = *((int4*)b_ptr);
      b_ptr += size_n;

      half2 dq[4][16];
      dequant_3bit_32(load_int4[0].x, load_int4[1].x, load_int4[2].x, dq[0],
                      size_n, zeros[0] + 1);
      dequant_3bit_32(load_int4[0].y, load_int4[1].y, load_int4[2].y, dq[1],
                      size_n, zeros[1] + 1);
      dequant_3bit_32(load_int4[0].z, load_int4[1].z, load_int4[2].z, dq[2],
                      size_n, zeros[2] + 1);
      dequant_3bit_32(load_int4[0].w, load_int4[1].w, load_int4[2].w, dq[3],
                      size_n, zeros[3] + 1);

      if (b_q_perm) {
        for (int j = 0; j < 16; j++) {
          for (int v = 0; v < 4; v++) dq[v][j] = __hmul2(scales[v], dq[v][j]);
          b_.set4(perm[lk++], n, __low2half(dq[0][j]), __low2half(dq[1][j]),
                  __low2half(dq[2][j]), __low2half(dq[3][j]));
          b_.set4(perm[lk++], n, __high2half(dq[0][j]), __high2half(dq[1][j]),
                  __high2half(dq[2][j]), __high2half(dq[3][j]));
        }
      } else {
        for (int j = 0; j < 16; j++) {
          for (int v = 0; v < 4; v++) dq[v][j] = __hmul2(scales[v], dq[v][j]);
          b_.set4(offset_k + lk++, n, __low2half(dq[0][j]),
                  __low2half(dq[1][j]), __low2half(dq[2][j]),
                  __low2half(dq[3][j]));
          b_.set4(offset_k + lk++, n, __high2half(dq[0][j]),
                  __high2half(dq[1][j]), __high2half(dq[2][j]),
                  __high2half(dq[3][j]));
        }
      }
    }
    k += 32;
  }
}

__global__ void reconstruct_exllama_2bit_kernel(
    const uint32_t* __restrict__ b_q_weight, const int* __restrict__ b_q_perm,
    const uint32_t* __restrict__ b_gptq_qzeros,
    const half* __restrict__ b_gptq_scales, const int size_k, const int size_n,
    const int groups, half* __restrict__ b) {
  MatrixView_half_rw b_(b, size_k, size_n);
  MatrixView_q2_row b_gptq_qzeros_(b_gptq_qzeros, groups, size_n);
  MatrixView_half b_gptq_scales_(b_gptq_scales, groups, size_n);

  auto offset_k = BLOCK_KN_SIZE * blockIdx.y;
  auto offset_n = BLOCK_KN_SIZE * blockIdx.x * 4;

  int end_k = min(offset_k + BLOCK_KN_SIZE, size_k);

  // Preload remapping table
  __shared__ int perm[BLOCK_KN_SIZE];
  auto t = threadIdx.x;

  if (b_q_perm) {
    if (offset_k + t < size_k) perm[t] = b_q_perm[offset_k + t];
  }

  // Column
  int n = offset_n + t * 4;
  if (n >= size_n) return;

  // Find initial group
  int groupsize = size_k / groups;
  int group = offset_k / groupsize;
  int nextgroup = offset_k + groupsize;

  // b offset
  int qk = offset_k / (32 / 2);

  const uint32_t* b_ptr = b_q_weight + qk * size_n + n;

  // Initial zeros/scale
  int zeros[4];
  half2 scales[4];
  b_gptq_qzeros_.item4(zeros, group, n);
  b_gptq_scales_.item4_h2(scales, group, n);

  __syncthreads();

  int k = offset_k;
  int lk = 0;

  while (k < end_k) {
    if (k == nextgroup) {
      group++;
      nextgroup += groupsize;
      b_gptq_qzeros_.item4(zeros, group, n);
      b_gptq_scales_.item4_h2(scales, group, n);
    }

    for (int p = 0; p < 2; p++) {
      const int4* b_ptr4 = (int4*)b_ptr;
      int4 load_int4 = *b_ptr4;

      half2 dq[4][8];
      dequant_2bit_16(load_int4.x, dq[0], size_n, zeros[0] + 1);
      dequant_2bit_16(load_int4.y, dq[1], size_n, zeros[1] + 1);
      dequant_2bit_16(load_int4.z, dq[2], size_n, zeros[2] + 1);
      dequant_2bit_16(load_int4.w, dq[3], size_n, zeros[3] + 1);

      b_ptr += size_n;
      // half* dqh = (half*)dq;
      if (b_q_perm) {
        for (int j = 0; j < 8; j++) {
          for (int v = 0; v < 4; v++) dq[v][j] = __hmul2(scales[v], dq[v][j]);
          b_.set4(perm[lk++], n, __low2half(dq[0][j]), __low2half(dq[1][j]),
                  __low2half(dq[2][j]), __low2half(dq[3][j]));
          b_.set4(perm[lk++], n, __high2half(dq[0][j]), __high2half(dq[1][j]),
                  __high2half(dq[2][j]), __high2half(dq[3][j]));
        }
      } else {
        for (int j = 0; j < 8; j++) {
          for (int v = 0; v < 4; v++) dq[v][j] = __hmul2(scales[v], dq[v][j]);
          b_.set4(offset_k + lk++, n, __low2half(dq[0][j]),
                  __low2half(dq[1][j]), __low2half(dq[2][j]),
                  __low2half(dq[3][j]));
          b_.set4(offset_k + lk++, n, __high2half(dq[0][j]),
                  __high2half(dq[1][j]), __high2half(dq[2][j]),
                  __high2half(dq[3][j]));
        }
      }
    }
    k += 32;
  }
}

void reconstruct_exllama(const uint32_t* b_q_weight,
                         const uint32_t* b_gptq_qzeros,
                         const half* b_gptq_scales, const int* b_q_perm,
                         half* out, int height, int width, int groups,
                         int bit) {
  dim3 blockDim, gridDim;
  blockDim.x = BLOCK_KN_SIZE;
  blockDim.y = 1;
  gridDim.y = DIVIDE(height, BLOCK_KN_SIZE);
  gridDim.x = DIVIDE(width, BLOCK_KN_SIZE);

  auto reconstruct_exllama_kernel = reconstruct_exllama_4bit_kernel;
  if (bit == 2) {
    reconstruct_exllama_kernel = reconstruct_exllama_2bit_kernel;
  } else if (bit == 3) {
    reconstruct_exllama_kernel = reconstruct_exllama_3bit_kernel;
  } else if (bit == 8) {
    reconstruct_exllama_kernel = reconstruct_exllama_8bit_kernel;
  }

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  reconstruct_exllama_kernel<<<gridDim, blockDim, 0, stream>>>(
      b_q_weight, b_q_perm, b_gptq_qzeros, b_gptq_scales, height, width, groups,
      out);
}

__global__ void gemm_half_q_half_alt_4bit_kernel(
    const half2* __restrict__ vec, const uint32_t* __restrict__ mat,
    half* __restrict__ mul, const half* __restrict__ scales,
    const uint32_t* __restrict__ zeros, const int* __restrict__ g_idx,
    int batch, int height, int width) {
  int zero_width = width / 8;
  int vec_height = height * 4;
  const int blockwidth2 = BLOCK_KN_SIZE / 2;
  auto b = blockIdx.y * BLOCK_M_SIZE_MAX;
  int b_end = min(BLOCK_M_SIZE_MAX, batch - b);
  auto h = BLOCK_KN_SIZE * blockIdx.z / 8;
  int h_end = min(BLOCK_KN_SIZE / 8, height - h) * 4;
  auto w = BLOCK_KN_SIZE * blockIdx.x + threadIdx.x;

  __shared__ half2 blockvec[BLOCK_M_SIZE_MAX][blockwidth2];
  if (threadIdx.x < h_end) {
    for (int m = 0; m < b_end; ++m) {
      blockvec[m][threadIdx.x] =
          vec[(m + b) * vec_height + blockIdx.z * BLOCK_KN_SIZE / 2 +
              threadIdx.x];
    }
  }

  __shared__ half2 deq2[256][8];
  auto val = threadIdx.x / 8;
  auto off = threadIdx.x % 8;
  for (; val < 256; val += BLOCK_KN_SIZE / 8) {
    deq2[val][off] =
        __halves2half2(__int2half_rn(val & 0xF), __int2half_rn(val >> 4));
  }

  if (blockIdx.z == 0) {
    for (int m = 0; m < b_end; m++) mul[(b + m) * width + w] = __int2half_rn(0);
  }
  __syncthreads();

  int i = width * h + w;
  int g_h = h * 8;
  int k = 0;
  int z_w = w / 8;
  int z_mod = (w % 8) * 4;
  half2 res2;
  half res[BLOCK_M_SIZE_MAX] = {};

  unsigned int tmp;
  while (k < h_end) {
    tmp = mat[i];
    half2 scales_tmp[4];
    half2 zeros_tmp[4];
    for (int tmp_k = 0; tmp_k < 4; tmp_k++) {
      int g = g_idx[g_h + (k + tmp_k) * 2];
      int g2 = g_idx[g_h + (k + tmp_k) * 2 + 1];
      half scale_f = scales[g * width + w];
      half scale_f2 = scales[g2 * width + w];
      half2 scale = __halves2half2(scale_f, scale_f2);
      half2 zero = __halves2half2(
          __hmul(scale_f,
                 __int2half_rn(-((zeros[g * zero_width + z_w] >> z_mod) & 0xF) -
                               1)),
          __hmul(scale_f2,
                 __int2half_rn(
                     -((zeros[g2 * zero_width + z_w] >> z_mod) & 0xF) - 1)));
      scales_tmp[tmp_k] = scale;
      zeros_tmp[tmp_k] = zero;
    }
    for (int m = 0; m < b_end; m++) {
      res2 = {};
      res2 = __hfma2(
          __hfma2(deq2[(tmp >> 0) & 0xff][off], scales_tmp[0], zeros_tmp[0]),
          blockvec[m][k + 0], res2);
      res2 = __hfma2(
          __hfma2(deq2[(tmp >> 8) & 0xff][off], scales_tmp[1], zeros_tmp[1]),
          blockvec[m][k + 1], res2);
      res2 = __hfma2(
          __hfma2(deq2[(tmp >> 16) & 0xff][off], scales_tmp[2], zeros_tmp[2]),
          blockvec[m][k + 2], res2);
      res2 = __hfma2(
          __hfma2(deq2[(tmp >> 24) & 0xff][off], scales_tmp[3], zeros_tmp[3]),
          blockvec[m][k + 3], res2);
      res[m] = __hadd(res[m], __hadd(res2.x, res2.y));
    }
    i += width;
    k += 4;
  }
  for (int m = 0; m < b_end; m++) {
    atomicAdd(&mul[(b + m) * width + w], res[m]);
  }
}

__global__ void gemm_half_q_half_alt_8bit_kernel(
    const half2* __restrict__ vec, const uint32_t* __restrict__ mat,
    half* __restrict__ mul, const half* __restrict__ scales,
    const uint32_t* __restrict__ zeros, const int* __restrict__ g_idx,
    int batch, int height, int width) {
  int zero_width = width / 4;
  int vec_height = height * 2;
  const int blockwidth2 = BLOCK_KN_SIZE / 2;
  auto b = blockIdx.y * BLOCK_M_SIZE_MAX;
  int b_end = min(BLOCK_M_SIZE_MAX, batch - b);
  auto h = BLOCK_KN_SIZE * blockIdx.z / 4;
  int h_end = min(BLOCK_KN_SIZE / 4, height - h) * 2;
  auto w = BLOCK_KN_SIZE * blockIdx.x + threadIdx.x;

  __shared__ half2 blockvec[BLOCK_M_SIZE_MAX][blockwidth2];
  if (threadIdx.x < h_end) {
    for (int m = 0; m < b_end; ++m) {
      blockvec[m][threadIdx.x] =
          vec[(m + b) * vec_height + blockIdx.z * BLOCK_KN_SIZE / 2 +
              threadIdx.x];
    }
  }

  if (blockIdx.z == 0) {
    for (int m = 0; m < b_end; m++) mul[(b + m) * width + w] = __int2half_rn(0);
  }
  __syncthreads();

  int i = width * h + w;
  int g_h = h * 4;
  int k = 0;
  int z_w = w / 4;
  int z_mod = (w % 4) * 8;
  half2 res2;
  half res[BLOCK_M_SIZE_MAX] = {};

  unsigned int tmp;
  while (k < h_end) {
    tmp = mat[i];
    half2 scales_tmp[2];
    half2 zeros_tmp[2];
    for (int tmp_k = 0; tmp_k < 2; tmp_k++) {
      int g = g_idx[g_h + (k + tmp_k) * 2];
      int g2 = g_idx[g_h + (k + tmp_k) * 2 + 1];
      half scale_f = scales[g * width + w];
      half scale_f2 = scales[g2 * width + w];
      half2 scale = __halves2half2(scale_f, scale_f2);
      half2 zero = __halves2half2(
          __hmul(scale_f,
                 __int2half_rn(
                     -((zeros[g * zero_width + z_w] >> z_mod) & 0xff) - 1)),
          __hmul(scale_f2,
                 __int2half_rn(
                     -((zeros[g2 * zero_width + z_w] >> z_mod) & 0xff) - 1)));
      scales_tmp[tmp_k] = scale;
      zeros_tmp[tmp_k] = zero;
    }
    for (int m = 0; m < b_end; m++) {
      res2 = {};
      half2 v12 = __halves2half2(__int2half_rn(tmp & 0xFF),
                                 __int2half_rn((tmp >> 8) & 0xFF));
      res2 = __hfma2(__hfma2(v12, scales_tmp[0], zeros_tmp[0]),
                     blockvec[m][k + 0], res2);
      half2 v34 = __halves2half2(__int2half_rn((tmp >> 16) & 0xFF),
                                 __int2half_rn((tmp >> 24) & 0xFF));
      res2 = __hfma2(__hfma2(v34, scales_tmp[1], zeros_tmp[1]),
                     blockvec[m][k + 1], res2);
      res[m] = __hadd(res[m], __hadd(res2.x, res2.y));
    }
    i += width;
    k += 2;
  }
  for (int m = 0; m < b_end; m++) {
    atomicAdd(&mul[(b + m) * width + w], res[m]);
  }
}

void gemm_half_q_half_alt(const half* a, const uint32_t* b_q_weight,
                          const uint32_t* b_gptq_qzeros,
                          const half* b_gptq_scales, const int* b_g_idx,
                          half* c, int size_m, int size_n, int size_k,
                          int bit) {
  dim3 blockDim, gridDim;
  blockDim.x = BLOCK_KN_SIZE;
  blockDim.y = 1;
  blockDim.z = 1;
  gridDim.x = DIVIDE(size_n, BLOCK_KN_SIZE);
  gridDim.y = DIVIDE(size_m, BLOCK_M_SIZE_MAX);
  gridDim.z = DIVIDE(size_k, BLOCK_KN_SIZE);

  auto kernel = gemm_half_q_half_alt_4bit_kernel;
  if (bit == 8) {
    kernel = gemm_half_q_half_alt_8bit_kernel;
  }

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  kernel<<<gridDim, blockDim, 0, stream>>>(
      (const half2*)a, b_q_weight, c, b_gptq_scales, b_gptq_qzeros, b_g_idx,
      size_m, size_k / 32 * bit, size_n);
}

template <class T, int bit>
__global__ void reconstruct_gptq_kernel(const uint32_t* __restrict__ w,
                                        const half* __restrict__ w_scales,
                                        const uint32_t* __restrict__ w_zeros,
                                        const int* __restrict__ g_idx,
                                        const int height, const int width,
                                        const int group,
                                        half* __restrict__ out) {
  // Start of block

  auto column = BLOCK_KN_SIZE * blockIdx.x + threadIdx.x;
  auto row = blockIdx.y * 32 / bit;
  if (column >= width) return;

  // Views

  MatrixView_half_rw out_(out, height, width);
  MatrixView_half w_scales_(w_scales, group, width);
  T w_zeros_(w_zeros, group, width);

  uint32_t w_read = w[blockIdx.y * width + column];
  half* out_ptr = out_.item_ptr(row, column);

#pragma unroll
  for (int s = 0; s < 32; s += bit) {
    int group = g_idx[row + s / bit];
    half w_scale = w_scales_.item(group, column);
    uint32_t w_zero = w_zeros_.item(group, column) + 1;
    half w_item =
        __hmul(__int2half_rn((int)((w_read >> s) & ((1 << bit) - 1)) - w_zero),
               w_scale);
    *out_ptr = w_item;
    out_ptr += out_.width;
  }
}

__global__ void reconstruct_gptq_3bit_kernel(
    const uint32_t* __restrict__ w, const half* __restrict__ w_scales,
    const uint32_t* __restrict__ w_zeros, const int* __restrict__ g_idx,
    const int height, const int width, const int group,
    half* __restrict__ out) {
  // Start of block
  auto column = BLOCK_KN_SIZE * blockIdx.x + threadIdx.x;
  auto row = blockIdx.y * 32;
  if (column >= width) return;

  // Views

  MatrixView_half_rw out_(out, height, width);
  MatrixView_half w_scales_(w_scales, group, width);
  MatrixView_q3_row w_zeros_(w_zeros, group, width);

  uint32_t w1 = w[(blockIdx.y * 3) * width + column];
  uint32_t w2 = w[(blockIdx.y * 3 + 1) * width + column];
  uint32_t w3 = w[(blockIdx.y * 3 + 2) * width + column];
  half* out_ptr = out_.item_ptr(row, column);

#pragma unroll
  for (int i = 0; i < 32; i += 1) {
    int group = g_idx[row + i];
    half w_scale = w_scales_.item(group, column);
    uint32_t w_zero = w_zeros_.item(group, column) + 1;
    int w_item;
    if (i == 10) {
      w_item = (w1 >> 30) | ((w2 << 2) & 0x4);
    } else if (i == 21) {
      w_item = (w2 >> 31) | ((w3 << 1) & 0x6);
    } else if (i < 10) {
      w_item = ((w1 >> (i * 3)) & 0x7);
    } else if (i < 21) {
      w_item = ((w2 >> (i * 3 - 32)) & 0x7);
    } else {
      w_item = ((w3 >> (i * 3 - 64)) & 0x7);
    }
    *out_ptr = __hmul(__int2half_rn(w_item - w_zero), w_scale);
    out_ptr += out_.width;
  }
}

void reconstruct_gptq(const uint32_t* b_q_weight, const uint32_t* b_gptq_qzeros,
                      const half* b_gptq_scales, const int* b_g_idx, half* out,
                      int height, int width, int groups, int bit) {
  dim3 blockDim, gridDim;
  blockDim.x = BLOCK_KN_SIZE;
  blockDim.y = 1;
  gridDim.y = DIVIDE(height, 32 / bit);
  gridDim.x = DIVIDE(width, BLOCK_KN_SIZE);

  auto kernel = reconstruct_gptq_kernel<MatrixView_q4_row, 4>;
  if (bit == 2) {
    kernel = reconstruct_gptq_kernel<MatrixView_q2_row, 2>;
  } else if (bit == 8) {
    kernel = reconstruct_gptq_kernel<MatrixView_q8_row, 8>;
  } else if (bit == 3) {
    kernel = reconstruct_gptq_3bit_kernel;
    gridDim.y = DIVIDE(height, 32);
  }

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  kernel<<<gridDim, blockDim, 0, stream>>>(b_q_weight, b_gptq_scales,
                                           b_gptq_qzeros, b_g_idx, height,
                                           width, groups, out);
}

template <int tileK, int tileM, typename dtype>
__global__ void perm_a(dtype* output, const dtype* input, const int* idx, int k,
                       int m, int lda) {
  int tid = threadIdx.x;
  int row = blockIdx.x * tileK + tid;
  int col_st = blockIdx.y * tileM;
  if (row < k) {
    int index = idx[row];
#pragma unroll tileM
    for (int i = 0; i < tileM; ++i) {
      int col = col_st + i;
      if (col < m) {
        output[row + lda * col] = input[index + lda * col];
      }
    }
  }
}

template <typename input_tp, const vllm::ScalarTypeId w_type_id,
          typename output_tp, typename quant_packed_tp>
bool launch_gemm_gptq(int m, int n, int k, int quant_group, const input_tp* dA,
                      int lda, const quant_packed_tp* dB, int ldb,
                      output_tp* dC, float* dC_temp, int ldc,
                      quant_packed_tp* d_zeros, input_tp* d_scales,
                      const cudaStream_t stream, int chunks = 1) {
  using namespace hgemm_marlin_gptq;
  if (n % 16 != 0) {
    printf("n %% 16 != 0, n = %d\n", n);
    return false;
  }
  if (k % 32 != 0) {
    printf("k %% 32 != 0, k = %d\n", k);
    return false;
  }
  // const vllm::ScalarTypeId w_type_id = vllm::kU4B8.id();
  const int THREADS = 256;
  int BLOCKS_M = div_ceil(m, SLICE_M);
  if (BLOCKS_M >= MAX_BLOCKS_M && BLOCKS_M % MAX_BLOCKS_M != 0) {
    printf("Error: input m is error, m = %d, blocks_m = %d\n", m, BLOCKS_M);
    return false;
  }
  if (BLOCKS_M > MAX_BLOCKS_M) BLOCKS_M = MAX_BLOCKS_M;
  int BLOCKS_N = 8;
  // int BLOCKS_K = 4;
  // It is better let TILE_K = quant_group
  // But if quant_group is too large, a quant_group can be divided into two
  // parts
  int BLOCKS_K = quant_group / SLICE_K;
  if (quant_group > 128) BLOCKS_K = 128 / SLICE_K;

  if (BLOCKS_M == 1 || BLOCKS_M == 2) {
    BLOCKS_N = 16;
  }
  const bool HAS_ACT_ORDER = false;
  // const bool HAS_ZP = false;
  const bool HAS_ZP =
      (w_type_id == vllm::kU4.id()) || (w_type_id == vllm::kU8.id());
  int* g_idx = nullptr;
  bool HAS_NK_PRED = true;
  bool HAS_M_PRED = true;
  // int TILE_N = BLOCKS_N * SLICE_N;
  // int TILE_K = BLOCKS_K * SLICE_K;
  // int TILE_M = BLOCKS_M * SLICE_M;
  if (n % TILE_N == 0 && k % TILE_K == 0) {
    HAS_NK_PRED = false;
  }
  if (m % TILE_M == 0) {
    HAS_M_PRED = false;
  }

#define LAUNCH_GPTQ(threads, bm, bn, bk, has_act_order, has_zp, has_nk_pred, \
                    has_m_pred)                                              \
  else if (THREADS == threads && BLOCKS_M == bm && BLOCKS_N == bn &&         \
           BLOCKS_K == bk && HAS_ACT_ORDER == has_act_order &&               \
           HAS_ZP == has_zp && HAS_M_PRED == has_m_pred &&                   \
           HAS_NK_PRED == has_nk_pred) {                                     \
    launch_gemm_gptq_kernel<input_tp, w_type_id, threads, bm, bn, bk,        \
                            has_act_order, has_zp, has_m_pred, has_nk_pred>( \
        (const PackTypeInt4*)dA, (const PackTypeInt4*)dB, (PackTypeInt4*)dC, \
        (PackTypeInt4*)dC_temp, (const PackTypeInt4*)d_scales,               \
        (const PackTypeInt4*)d_zeros, nullptr, m, n, k, quant_group, chunks, \
        stream);                                                             \
  }

#define LAUNCH_GPTQ_K(bk, has_act_order, has_zp, has_nk_pred, has_m_pred)     \
  LAUNCH_GPTQ(256, 1, 16, bk, has_act_order, has_zp, has_nk_pred, has_m_pred) \
  LAUNCH_GPTQ(256, 2, 16, bk, has_act_order, has_zp, has_nk_pred, has_m_pred) \
  LAUNCH_GPTQ(256, 3, 8, bk, has_act_order, has_zp, has_nk_pred, has_m_pred)  \
  LAUNCH_GPTQ(256, 4, 8, bk, has_act_order, has_zp, has_nk_pred, has_m_pred)

#define LAUNCH_GPTQ_ZP(has_zp, has_nk_pred, has_m_pred)    \
  LAUNCH_GPTQ_K(1, false, has_zp, has_nk_pred, has_m_pred) \
  LAUNCH_GPTQ_K(2, false, has_zp, has_nk_pred, has_m_pred) \
  LAUNCH_GPTQ_K(4, false, has_zp, has_nk_pred, has_m_pred) \
  LAUNCH_GPTQ_K(8, false, has_zp, has_nk_pred, has_m_pred)

#define LAUNCH_GPTQ_PRED(has_nk_pred, has_m_pred) \
  LAUNCH_GPTQ_ZP(false, has_nk_pred, has_m_pred)
  // LAUNCH_GPTQ_ZP(true, has_nk_pred, has_m_pred)

  if (false) {
  }
  LAUNCH_GPTQ_PRED(true, true)
  LAUNCH_GPTQ_PRED(true, false)
  LAUNCH_GPTQ_PRED(false, true)
  LAUNCH_GPTQ_PRED(false, false)
  else {
    printf(
        "BLOCKS_M=%d, BLOCKS_N=%d, BLOCKS_k=%d, THREADS=%d, HAS_ACT_ORDER=%d, "
        "HAS_ZP=%d, quant_group=%d, HAS_M_PRED=%d, HAS_NK_PRED=%d is not "
        "supported\n",
        BLOCKS_M, BLOCKS_N, BLOCKS_K, THREADS, HAS_ACT_ORDER, HAS_ZP,
        quant_group, HAS_M_PRED, HAS_NK_PRED);
    return false;
  }

  return true;
}

#ifdef BF16_HIGH_PRECISION
__global__ void vectorized_elementwise_fp32tobf16(float* input,
                                                  __maca_bfloat16* output,
                                                  int N) {
  uint64_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < N) {
    // printf("tid = %d, input = %f, output = %f\n", tid, input[tid],
    // (float)(__maca_bfloat16)input[tid]);
    *(__maca_bfloat16*)(output + tid) = (__maca_bfloat16)input[tid];
  }
}
#else
__global__ void vectorized_elementwise_fp16tobf16(__maca_bfloat16* input,
                                                  int N) {
  uint64_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < N) {
    input[tid] = (float)(*(half*)(input + tid));
  }
}
#endif

template <typename input_tp, const vllm::ScalarTypeId w_type_id,
          typename output_tp, typename quant_packed_tp>
bool launch_gemm(int m, int n, int k, int quant_group, const input_tp* dA,
                 int lda, const quant_packed_tp* dB, int ldb, output_tp* dC,
                 float* dC_temp, int ldc, quant_packed_tp* d_zeros,
                 input_tp* d_scales, const int* g_idx, input_tp* perm_space,
                 bool is_gptq = true) {
  using namespace hgemm_marlin_gptq;
  // constexpr int max_blocks_m = 4;
  int total_m_blocks = div_ceil(m, SLICE_M);
  int chunks = total_m_blocks / MAX_BLOCKS_M;
  int rest_blocks_m = total_m_blocks % MAX_BLOCKS_M;
  // printf("m=%d,n=%d,k=%d,lda=%d,ldb=%d,ldc=%d,total_m_blocks=%d,chunks=%d,rest_blocks_m=%d\n",
  //     m, n, k, lda, ldb, ldc, total_m_blocks, chunks, rest_blocks_m
  // );
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  // input_tp *dA_perm;
  if (g_idx != nullptr) {
    // mcMalloc(reinterpret_cast<void **>(&dA_perm), k * m * sizeof(input_tp));
    // mcMallocAsync(reinterpret_cast<void **>(&dA_perm), k * m *
    // sizeof(input_tp), stream);
    const int threads = 256;
    const int tileK1 = 256;
    const int tileM1 = 16;
    uint32_t gridx1 = (k + tileK1 - 1) / tileK1;
    uint32_t gridy1 = (m + tileM1 - 1) / tileM1;
    dim3 dimBlock1(threads, 1, 1);
    dim3 dimGrid1(gridx1, gridy1, 1);
    perm_a<tileK1, tileM1, input_tp>
        <<<dimGrid1, dimBlock1, 0, stream>>>(perm_space, dA, g_idx, k, m, k);
  }
  const input_tp* dA_actual = (g_idx != nullptr ? perm_space : dA);
  bool ret = true;
  if (chunks > 0) {
    int real_m = m > chunks * MAX_BLOCKS_M * SLICE_M
                     ? chunks * MAX_BLOCKS_M * SLICE_M
                     : m;
    if (is_gptq) {
      ret = launch_gemm_gptq<input_tp, w_type_id, output_tp, quant_packed_tp>(
          real_m, n, k, quant_group, dA_actual, lda, dB, ldb, dC, dC_temp, ldc,
          d_zeros, d_scales, stream, chunks);
    }
  }
  if (rest_blocks_m > 0) {
    int m_offset = chunks * MAX_BLOCKS_M * SLICE_M;
    if (is_gptq) {
      ret = ret &&
            launch_gemm_gptq<input_tp, w_type_id, output_tp, quant_packed_tp>(
                m - m_offset, n, k, quant_group, dA_actual + lda * m_offset,
                lda, dB, ldb, dC + ldc * m_offset, dC_temp, ldc, d_zeros,
                d_scales, stream, 1);
    }
  }

  // #if 0
  if constexpr (std::is_same_v<input_tp, __maca_bfloat16>) {
    uint64_t size = m * n;
    uint64_t block = 512;
    uint64_t grid = div_ceil(size, block);
    vectorized_elementwise_fp32tobf16<<<grid, block, 0, stream>>>(
        (float*)dC_temp, (input_tp*)dC, size);
  }
#if 0
  #ifdef BF16_HIGH_PRECISION
	  vectorized_elementwise_fp32tobf16<<<grid, block, 0, stream>>>((float*)dC_temp, (input_tp*)dC, size);
  #else
          vectorized_elementwise_fp16tobf16<<<grid, block, 0, stream>>>((input_tp*)dC, size);
  #endif
#endif

  return ret;
}

void gemm_bf16_q_bf16_cuda(const __maca_bfloat16* a, const uint32_t* b_q_weight,
                           const uint32_t* b_gptq_qzeros,
                           const __maca_bfloat16* b_gptq_scales,
                           const int* b_g_idx, __maca_bfloat16* c,
                           float* temp_space, int size_m, int size_n,
                           int size_k, int bit, int group_size,
                           __maca_bfloat16* perm_space) {
  bool opt = ((group_size == 128) || (group_size == 64));
  using scalar_t = __maca_bfloat16;
  if ((bit == 4) && opt) {
    uint32_t* zeros = const_cast<uint32_t*>(b_gptq_qzeros);
    scalar_t* scales = const_cast<scalar_t*>(b_gptq_scales);
    launch_gemm<scalar_t, vllm::kU4B8.id(), scalar_t, quant_packed_type>(
        size_m, size_n, size_k, group_size, a, size_k, b_q_weight, size_n, c,
        temp_space, size_n, zeros, scales, b_g_idx, perm_space, true);
  } else if ((bit == 8) && opt) {
    uint32_t* zeros = const_cast<uint32_t*>(b_gptq_qzeros);
    scalar_t* scales = const_cast<scalar_t*>(b_gptq_scales);
    launch_gemm<scalar_t, vllm::kU8B128.id(), scalar_t, quant_packed_type>(
        size_m, size_n, size_k, group_size, a, size_k, b_q_weight, size_n, c,
        temp_space, size_n, zeros, scales, b_g_idx, perm_space, true);
  } else {
    printf("Only supported bit-4 , bit-8 of block_size 128 or 64 now!\n");
  }
}
void gemm_half_q_half_cuda(cublasHandle_t cublas_handle, const half* a,
                           const uint32_t* b_q_weight,
                           const uint32_t* b_gptq_qzeros,
                           const half* b_gptq_scales, const int* b_g_idx,
                           half* c, half* temp_dq, int size_m, int size_n,
                           int size_k, int groups, bool use_exllama, int bit,
                           int group_size, half* perm_space) {
  bool use_reconstruct;
  bool opt = ((group_size == 128) || (group_size == 64));
  if ((bit == 4) && opt) {
    if ((size_m <= 2) && (group_size == 128)) {
      gemm_half_q_half_cuda_part(a, b_q_weight, b_gptq_qzeros, b_gptq_scales,
                                 b_g_idx, c, size_m, size_n, size_k,
                                 BLOCK_M_SIZE_MAX, groups, bit, true, true);
    } else {
      uint32_t* zeros = const_cast<uint32_t*>(b_gptq_qzeros);
      half* scales = const_cast<half*>(b_gptq_scales);
      launch_gemm<input_type, vllm::kU4B8.id(), output_type, quant_packed_type>(
          size_m, size_n, size_k, group_size, a, size_k, b_q_weight, size_n, c,
          nullptr, size_n, zeros, scales, b_g_idx, perm_space, true);
    }
  } else if ((bit == 8) && opt) {
    uint32_t* zeros = const_cast<uint32_t*>(b_gptq_qzeros);
    half* scales = const_cast<half*>(b_gptq_scales);
    launch_gemm<input_type, vllm::kU8B128.id(), output_type, quant_packed_type>(
        size_m, size_n, size_k, group_size, a, size_k, b_q_weight, size_n, c,
        nullptr, size_n, zeros, scales, b_g_idx, perm_space, true);
  } else {
    if (use_exllama) {
      use_reconstruct = ((bit == 8 && size_m > MAX_Q_GEMM_ROWS_8BIT) ||
                         (bit != 8 && size_m > MAX_Q_GEMM_ROWS));
    } else {
      // The 2/3-bit kernels are somehow slower than dequant + gemm baseline, so
      // we disabled them for now.
      use_reconstruct = (bit < 4 || size_m > 0);
    }
    if (use_reconstruct) {
      // Reconstruct FP16 matrix, then cuBLAS
      if (use_exllama) {
        reconstruct_exllama(b_q_weight, b_gptq_qzeros, b_gptq_scales, b_g_idx,
                            temp_dq, size_k, size_n, groups, bit);
      } else {
        reconstruct_gptq(b_q_weight, b_gptq_qzeros, b_gptq_scales, b_g_idx,
                         temp_dq, size_k, size_n, groups, bit);
      }

      const half alpha = __float2half(1.0f);
      const half beta = __float2half(0.0f);
      cublasHgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, size_n, size_m,
                  size_k, &alpha, temp_dq, size_n, a, size_k, &beta, c, size_n);
    } else if (use_exllama) {
      // Quantized matmul
      int max_chunks = size_m / BLOCK_M_SIZE_MAX;
      int last_chunk = max_chunks * BLOCK_M_SIZE_MAX;
      int last_chunk_size = size_m - last_chunk;

      bool m_sign;
      bool v_sign;
      if (group_size == 128) {
        m_sign = size_m <= 50;
        v_sign = size_m <= 4;
      } else {
        m_sign = false;
        v_sign = false;
      }

      if (max_chunks) {
        gemm_half_q_half_cuda_part(
            a, b_q_weight, b_gptq_qzeros, b_gptq_scales, b_g_idx, c, last_chunk,
            size_n, size_k, BLOCK_M_SIZE_MAX, groups, bit, m_sign, v_sign);
      }

      if (last_chunk_size) {
        gemm_half_q_half_cuda_part(
            a + last_chunk * size_k, b_q_weight, b_gptq_qzeros, b_gptq_scales,
            b_g_idx, c + last_chunk * size_n, last_chunk_size, size_n, size_k,
            last_chunk_size, groups, bit, m_sign, v_sign);
      }
    } else {
      gemm_half_q_half_alt(a, b_q_weight, b_gptq_qzeros, b_gptq_scales, b_g_idx,
                           c, size_m, size_n, size_k, bit);
    }
  }
}

__global__ void shuffle_4bit_kernel(uint32_t* __restrict__ b_q_weight,
                                    const int size_k, const int size_n) {
  auto n = blockIdx.x * THREADS_X + threadIdx.x;
  if (n >= size_n) return;
  int k = 0;
  uint32_t* b_ptr = b_q_weight + n;
  while (k < size_k) {
    shuffle_4bit_8(b_ptr, size_n);
    b_ptr += 1 * size_n;
    k += 8;
  }
}

__global__ void shuffle_8bit_kernel(uint32_t* __restrict__ b_q_weight,
                                    const int size_k, const int size_n) {
  auto n = blockIdx.x * THREADS_X + threadIdx.x;
  if (n >= size_n) return;
  int k = 0;
  uint32_t* b_ptr = b_q_weight + n;
  while (k < size_k) {
    shuffle_8bit_4(b_ptr, size_n);
    b_ptr += 1 * size_n;
    k += 4;
  }
}

__global__ void shuffle_2bit_kernel(uint32_t* __restrict__ b_q_weight,
                                    const int size_k, const int size_n) {
  auto n = blockIdx.x * THREADS_X + threadIdx.x;
  if (n >= size_n) return;
  int k = 0;
  uint32_t* b_ptr = b_q_weight + n;
  while (k < size_k) {
    shuffle_2bit_16(b_ptr, size_n);
    b_ptr += 1 * size_n;
    k += 16;
  }
}

__global__ void shuffle_3bit_kernel(uint32_t* __restrict__ b_q_weight,
                                    const int size_k, const int size_n) {
  auto n = blockIdx.x * THREADS_X + threadIdx.x;
  if (n >= size_n) return;
  int k = 0;
  uint32_t* b_ptr = b_q_weight + n;
  while (k < size_k) {
    shuffle_3bit_32(b_ptr, size_n);
    b_ptr += 3 * size_n;
    k += 32;
  }
}

__global__ void make_sequential_4bit_kernel(const uint32_t* __restrict__ w,
                                            uint32_t* __restrict__ w_new,
                                            const int* __restrict__ q_perm,
                                            const int w_width) {
  const uint64_t* w2 = (uint64_t*)w;
  uint64_t* w_new2 = (uint64_t*)w_new;
  int w2_stride = w_width >> 1;
  auto w2_column = THREADS_X * blockIdx.x + threadIdx.x;
  if (w2_column >= w2_stride) return;
  auto w_new2_row = blockIdx.y;
  int q_perm_idx = w_new2_row << 3;
  uint64_t dst = 0;

#pragma unroll
  for (int i = 0; i < 8; i++) {
    int source_row = q_perm[q_perm_idx++];

    int w2_row = source_row >> 3;
    int w2_subrow = source_row & 0x07;
    int w2_row_shift = w2_subrow << 2;
    int wnew2_row_shift = i << 2;

    uint64_t src = w2[w2_row * w2_stride + w2_column];
    src >>= w2_row_shift;
    src &= 0x0000000f0000000f;
    src <<= wnew2_row_shift;
    dst |= src;
  }
  w_new2[w_new2_row * w2_stride + w2_column] = dst;
}

__global__ void make_sequential_2bit_kernel(const uint32_t* __restrict__ w,
                                            uint32_t* __restrict__ w_new,
                                            const int* __restrict__ q_perm,
                                            const int w_width) {
  const uint64_t* w2 = (uint64_t*)w;
  uint64_t* w_new2 = (uint64_t*)w_new;
  int w2_stride = w_width >> 1;
  auto w2_column = THREADS_X * blockIdx.x + threadIdx.x;
  if (w2_column >= w2_stride) return;
  auto w_new2_row = blockIdx.y;
  int q_perm_idx = w_new2_row << 4;
  uint64_t dst = 0;

#pragma unroll
  for (int i = 0; i < 16; i++) {
    int source_row = q_perm[q_perm_idx++];

    int w2_row = source_row >> 4;
    int w2_subrow = source_row & 0x0f;
    int w2_row_shift = w2_subrow << 1;
    int wnew2_row_shift = i << 1;

    uint64_t src = w2[w2_row * w2_stride + w2_column];
    src >>= w2_row_shift;
    src &= 0x0000000300000003;
    src <<= wnew2_row_shift;
    dst |= src;
  }
  w_new2[w_new2_row * w2_stride + w2_column] = dst;
}

__global__ void make_sequential_3bit_kernel(const uint32_t* __restrict__ w,
                                            uint32_t* __restrict__ w_new,
                                            const int* __restrict__ q_perm,
                                            const int w_width) {
  auto w_column = THREADS_X * blockIdx.x + threadIdx.x;
  if (w_column >= w_width) return;
  auto w_new_row = blockIdx.y * 3;
  auto q_perm_idx = blockIdx.y << 5;
  uint32_t dst[3] = {0, 0, 0};

#pragma unroll
  for (int i = 0; i < 32; i++) {
    int source_row = q_perm[q_perm_idx++];
    int z_w = (source_row / 32) * 3;
    int z_mod = source_row % 32;
    int z_bit;

    if (z_mod != 10) {
      if (z_mod != 21) {
        z_bit = z_mod;
        if (z_bit > 21) {
          z_bit *= 3;
          z_bit -= 64;
          z_w += 2;
        } else if (z_bit > 10) {
          z_bit *= 3;
          z_bit -= 32;
          z_w += 1;
        } else {
          z_bit *= 3;
        }
      } else {
        z_w += 1;
      }
    }

    uint64_t src;
    if (z_mod == 10) {
      src = (w[z_w * w_width + w_column] >> 30) |
            ((w[(z_w + 1) * w_width + w_column] << 2) & 0x4);
    } else if (z_mod == 21) {
      src = (w[z_w * w_width + w_column] >> 31) |
            ((w[(z_w + 1) * w_width + w_column] << 1) & 0x6);
    } else {
      src = w[z_w * w_width + w_column];
      src >>= z_bit;
      src &= 0x07;
    }

    z_w = 0;
    if (i != 10) {
      if (i != 21) {
        z_bit = i;
        if (z_bit > 21) {
          z_bit *= 3;
          z_bit -= 64;
          z_w += 2;
        } else if (z_bit > 10) {
          z_bit *= 3;
          z_bit -= 32;
          z_w += 1;
        } else {
          z_bit *= 3;
        }
      } else {
        z_w += 1;
      }
    }
    if (i == 10) {
      dst[z_w] |= (src & 0x03) << 30;
      dst[z_w + 1] |= ((src & 0x4) >> 2);
    } else if (i == 21) {
      dst[z_w] |= (src & 0x01) << 31;
      dst[z_w + 1] |= ((src & 0x6) >> 1);
    } else {
      dst[z_w] |= (src << z_bit);
    }
  }
  w_new[w_new_row * w_width + w_column] = dst[0];
  w_new[(w_new_row + 1) * w_width + w_column] = dst[1];
  w_new[(w_new_row + 2) * w_width + w_column] = dst[2];
}

__global__ void make_sequential_8bit_kernel(const uint32_t* __restrict__ w,
                                            uint32_t* __restrict__ w_new,
                                            const int* __restrict__ q_perm,
                                            const int w_width) {
  const uint64_t* w2 = (uint64_t*)w;
  uint64_t* w_new2 = (uint64_t*)w_new;
  int w2_stride = w_width >> 1;
  auto w2_column = THREADS_X * blockIdx.x + threadIdx.x;
  if (w2_column >= w2_stride) return;
  auto w_new2_row = blockIdx.y;
  int q_perm_idx = w_new2_row << 2;
  uint64_t dst = 0;

#pragma unroll
  for (int i = 0; i < 4; i++) {
    int source_row = q_perm[q_perm_idx++];

    int w2_row = source_row >> 2;
    int w2_subrow = source_row & 0x03;
    int w2_row_shift = w2_subrow << 3;
    int wnew2_row_shift = i << 3;

    uint64_t src = w2[w2_row * w2_stride + w2_column];
    src >>= w2_row_shift;
    src &= 0x000000ff000000ff;
    src <<= wnew2_row_shift;
    dst |= src;
  }
  w_new2[w_new2_row * w2_stride + w2_column] = dst;
}

void shuffle_exllama_weight(uint32_t* q_weight, int* q_perm, int height,
                            int width, int bit) {
  if (q_perm) {
    uint32_t* new_qweight = NULL;
    cudaMalloc(&new_qweight, height / 32 * bit * width * sizeof(uint32_t));

    dim3 blockDim, gridDim;
    blockDim.x = THREADS_X;
    blockDim.y = 1;
    gridDim.x = DIVIDE(width, THREADS_X);
    gridDim.y = height / 32 * bit;

    auto kernel = make_sequential_4bit_kernel;
    if (bit == 2) {
      kernel = make_sequential_2bit_kernel;
    } else if (bit == 3) {
      kernel = make_sequential_3bit_kernel;
      gridDim.y = height / 32;
    } else if (bit == 8) {
      kernel = make_sequential_8bit_kernel;
    }
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    kernel<<<gridDim, blockDim, 0, stream>>>(q_weight, new_qweight, q_perm,
                                             width);
    // Replace qweights
    cudaMemcpyAsync(q_weight, new_qweight,
                    height / 32 * bit * width * sizeof(uint32_t),
                    cudaMemcpyDeviceToDevice);
    // Cleanup
    cudaDeviceSynchronize();
    cudaFree(new_qweight);
  }
  dim3 blockDim, gridDim;
  blockDim.x = THREADS_X;
  blockDim.y = 1;
  gridDim.x = DIVIDE(width, THREADS_X);
  gridDim.y = 1;
  auto shuffle_kernel = shuffle_4bit_kernel;
  if (bit == 2) {
    shuffle_kernel = shuffle_2bit_kernel;
  } else if (bit == 3) {
    shuffle_kernel = shuffle_3bit_kernel;
  } else if (bit == 8) {
    shuffle_kernel = shuffle_8bit_kernel;
  }
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  shuffle_kernel<<<gridDim, blockDim, 0, stream>>>(q_weight, height, width);
}

}  // namespace gptq
}  // namespace vllm

torch::Tensor gptq_gemm(torch::Tensor a, torch::Tensor b_q_weight,
                        torch::Tensor b_gptq_qzeros,
                        torch::Tensor b_gptq_scales, torch::Tensor b_g_idx,
                        bool use_exllama, int64_t bit, int64_t group_size,
                        torch::Tensor perm_space, torch::Tensor temp_space,
                        bool dtype_bf16) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  auto options = torch::TensorOptions().dtype(a.dtype()).device(a.device());
  at::Tensor c = torch::zeros({a.size(0), b_q_weight.size(1)}, options);
  at::Tensor temp_dq = torch::empty(
      {b_q_weight.size(0) * 32 / bit, b_q_weight.size(1)}, options);

  if (dtype_bf16) {
    vllm::gptq::gemm_bf16_q_bf16_cuda(
        (const __maca_bfloat16*)a.data_ptr(),
        (const uint32_t*)b_q_weight.data_ptr(),
        (const uint32_t*)b_gptq_qzeros.data_ptr(),
        (const __maca_bfloat16*)b_gptq_scales.data_ptr(),
        b_g_idx.device().is_meta() ? NULL : (const int*)b_g_idx.data_ptr(),
        (__maca_bfloat16*)c.data_ptr(), (float*)temp_space.data_ptr(),
        c.size(0),  // m
        c.size(1),  // n
        a.size(1),  // k
        bit, group_size, (__maca_bfloat16*)perm_space.data_ptr());
  } else {
    vllm::gptq::gemm_half_q_half_cuda(
        at::cuda::getCurrentCUDABlasHandle(), (const half*)a.data_ptr(),
        (const uint32_t*)b_q_weight.data_ptr(),
        (const uint32_t*)b_gptq_qzeros.data_ptr(),
        (const half*)b_gptq_scales.data_ptr(),
        b_g_idx.device().is_meta() ? NULL : (const int*)b_g_idx.data_ptr(),
        (half*)c.data_ptr(), (half*)temp_dq.data_ptr(),
        c.size(0),              // m
        c.size(1),              // n
        a.size(1),              // k
        b_gptq_qzeros.size(0),  // group number
        use_exllama, bit, group_size, (half*)perm_space.data_ptr());
  }

  return c;
}

void gptq_shuffle(torch::Tensor q_weight, torch::Tensor q_perm, int64_t bit) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(q_weight));
  vllm::gptq::shuffle_exllama_weight(
      (uint32_t*)q_weight.data_ptr(),
      q_perm.device().is_meta() || q_perm.numel() == 0
          ? NULL
          : (int*)q_perm.data_ptr(),
      q_weight.size(0) * 32 / bit, q_weight.size(1), bit);
}
