// 2024 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights
// Reserved.
#pragma once

#include <mc_runtime.h>
#include <maca_fp16.h>
#include "Hgemm_common.cuh"
#include "dequant.cuh"
#define quant_packed_type uint32_t
typedef __NATIVE_VECTOR__(2, float) v2f;
template <int ThreadBlock, int BlockDimX, int BATCH>
__global__ __launch_bounds__(256) void hgemv_nn_splitk_gptq(
    const half* __restrict__ srcB, const quant_packed_type* __restrict__ srcA,
    const quant_packed_type* __restrict__ zeros,
    const half* __restrict__ scales, half* dst, int m, int n, int k,
    int srcAStride, int dstStride, int k_div_sk,
    const int* __restrict__ b_perm = nullptr) {
  constexpr int QBITS = 4;
  constexpr int PACK_RATIO = (32 / QBITS);
  constexpr int QUANT_GROUP = 128;
  constexpr int LoopNum = QUANT_GROUP / (ThreadBlock / BlockDimX);
  constexpr int N = BATCH;
  const int k_stride = k;
  const int k_block = (k_div_sk + QUANT_GROUP - 1) / QUANT_GROUP * QUANT_GROUP;
  const int splitKOffset = blockIdx.y * k_block;
  if (splitKOffset + k_block > k)
    k = k - splitKOffset;
  else
    k = k_block;
  srcA += splitKOffset * srcAStride / PACK_RATIO;
  // srcB += splitKOffset;

  constexpr int quant_groups = 1;
  constexpr int m_per_thread = sizeof(float4) / sizeof(quant_packed_type);
  constexpr int thread_groups = ThreadBlock / BlockDimX;
  constexpr int group_elements = BlockDimX * m_per_thread;
  constexpr int reduce_size = ThreadBlock * m_per_thread;
  constexpr int data_cache_size =
      LoopNum * thread_groups * N + group_elements * 2;
  float *bsm_b_ptr, *bsm_zeros_ptr, *bsm_scales_ptr, *smem;
  if constexpr (reduce_size > data_cache_size) {
    __shared__ float bsm_ptr[reduce_size];
    bsm_b_ptr = bsm_ptr;
    bsm_zeros_ptr = bsm_ptr + LoopNum * thread_groups * N;
    bsm_scales_ptr = bsm_zeros_ptr + group_elements;
    smem = bsm_ptr;
  } else {
    __shared__ float bsm_ptr[data_cache_size];
    bsm_b_ptr = bsm_ptr;
    bsm_zeros_ptr = bsm_ptr + LoopNum * thread_groups * N;
    bsm_scales_ptr = bsm_zeros_ptr + group_elements;
    smem = bsm_ptr;
  }

  const int zeros_stride = srcAStride / PACK_RATIO;
  const int scales_stride = srcAStride;
  zeros += splitKOffset * zeros_stride / QUANT_GROUP;
  scales += splitKOffset * scales_stride / QUANT_GROUP;

  dst += group_elements * blockIdx.x;
  const int m_offset_a = group_elements * blockIdx.x;
  const int m_offset_zeros = group_elements / PACK_RATIO * blockIdx.x;
  const int m_offset_scales = group_elements * blockIdx.x;

  // store splited fma results
  v2f c_splited[N][m_per_thread];
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < m_per_thread; j++) c_splited[i][j] = {0, 0};
  }

  int tid = threadIdx.x;
  int tidCol = tid / BlockDimX;
  int tidRow = tid % BlockDimX;

  for (int i = 0; i < k; i += LoopNum * thread_groups) {
    int quant_group = i / QUANT_GROUP;
    constexpr int loading_pack = 2;
    constexpr int loading_count = group_elements / loading_pack;
    // Load needed zeros, scales
    for (int x = tid; x < loading_count; x += BlockDimX) {
      uint8_t temp_zeros = 0;
      temp_zeros =
          ((uint8_t*)(zeros + quant_group * zeros_stride + m_offset_zeros))[x];
      half temp_scales[2];
      *((float*)temp_scales) =
          ((float*)(scales + quant_group * scales_stride + m_offset_scales))[x];
      uint32_t z = temp_zeros;
      uint32_t z0 = __builtin_mxc_ubfe(z, 0, QBITS) + 1;
      uint32_t z1 = __builtin_mxc_ubfe(z, 4, QBITS) + 1;
      float s1 = (float)(temp_scales[0]);
      float s2 = (float)(temp_scales[1]);
      // Store to shared memory
      bsm_zeros_ptr[x * 2] = (float)z0 * s1 * -1.0f;
      bsm_zeros_ptr[x * 2 + 1] = (float)z1 * s2 * -1.0f;
      bsm_scales_ptr[x * 2] = s1;
      bsm_scales_ptr[x * 2 + 1] = s2;
    }

    int loop_index = 0;
    quant_packed_type A[m_per_thread];
    // Load A
    *((float4*)A) = *(float4*)(srcA +
                               (loop_index + i + tidCol * LoopNum) /
                                   PACK_RATIO * srcAStride +
                               m_offset_a + m_per_thread * tidRow);

    // Load B and transform to float
    if (b_perm != nullptr) {
      for (int y = 0; y < N; y++) {
        for (int x = tidRow; x < LoopNum; x += BlockDimX) {
          bsm_b_ptr[x + tidCol * LoopNum + y * LoopNum * thread_groups] =
              srcB[b_perm[splitKOffset + i + tidCol * LoopNum + x] +
                   y * k_stride];
        }
      }
    } else {
      for (int y = 0; y < N; y++) {
        for (int x = tidRow; x < LoopNum; x += BlockDimX) {
          bsm_b_ptr[x + tidCol * LoopNum + y * LoopNum * thread_groups] =
              srcB[splitKOffset + i + tidCol * LoopNum + x + y * k_stride];
        }
      }
    }

    __syncthreads();

    // Load zero and scale from bsm
    int m_index = tidRow * m_per_thread;

    v2f local_scales[m_per_thread];
    v2f local_scales_d16[m_per_thread];
    for (int c = 0; c < m_per_thread; c++) {
      float s = bsm_scales_ptr[m_index + c];
      float s_d16 = s / 16;
      local_scales[c] = {s, s};
      local_scales_d16[c] = {s_d16, s_d16};
    }
    v2f local_zeros[m_per_thread];
    for (int c = 0; c < m_per_thread; c++)
      local_zeros[c] = {bsm_zeros_ptr[m_index + c], bsm_zeros_ptr[m_index + c]};

    const int shuffled_dequant_index[PACK_RATIO] = {0, 4, 1, 5, 2, 6, 3, 7};
    for (; loop_index < LoopNum; loop_index += PACK_RATIO) {
      if constexpr (N <= 4) {
        v2f local_b[PACK_RATIO / 2 * N];
        for (int y = 0; y < N; y++) {
          for (int x = 0; x < PACK_RATIO / 2; x++) {
            local_b[x + PACK_RATIO / 2 * y].x =
                bsm_b_ptr[tidCol * LoopNum + loop_index + x * 2 +
                          y * thread_groups * LoopNum];
            local_b[x + PACK_RATIO / 2 * y].y =
                bsm_b_ptr[tidCol * LoopNum + loop_index + x * 2 + 1 +
                          y * thread_groups * LoopNum];
          }
        }
#pragma unroll m_per_thread
        for (int c = 0; c < m_per_thread; c++) {
          uint32_t p0 = A[c] & 0x0f0f0f0f;
          uint32_t p1 = A[c] & 0xf0f0f0f0;
          float o1, o2, o3, o4, o5, o6, o7, o8;
          asm volatile("cvt_b0tof32 %0,%1;\n" : "=r"(o1) : "r"(p0));
          asm volatile("cvt_b1tof32 %0,%1;\n" : "=r"(o2) : "r"(p0));
          asm volatile("cvt_b2tof32 %0,%1;\n" : "=r"(o3) : "r"(p0));
          asm volatile("cvt_b3tof32 %0,%1;\n" : "=r"(o4) : "r"(p0));
          asm volatile("cvt_b0tof32 %0,%1;\n" : "=r"(o5) : "r"(p1));
          asm volatile("cvt_b1tof32 %0,%1;\n" : "=r"(o6) : "r"(p1));
          asm volatile("cvt_b2tof32 %0,%1;\n" : "=r"(o7) : "r"(p1));
          asm volatile("cvt_b3tof32 %0,%1;\n" : "=r"(o8) : "r"(p1));
          v2f a0 = {o1, o3};
          v2f a1 = {o5, o7};
          v2f a2 = {o2, o4};
          v2f a3 = {o6, o8};

          a0 = __builtin_mxc_pk_fma_f32(a0, local_scales[c], local_zeros[c]);
          a1 =
              __builtin_mxc_pk_fma_f32(a1, local_scales_d16[c], local_zeros[c]);
          a2 = __builtin_mxc_pk_fma_f32(a2, local_scales[c], local_zeros[c]);
          a3 =
              __builtin_mxc_pk_fma_f32(a3, local_scales_d16[c], local_zeros[c]);

#pragma unroll N
          for (int y = 0; y < N; y++) {
            c_splited[y][c] = __builtin_mxc_pk_fma_f32(
                a0, local_b[0 + PACK_RATIO / 2 * y], c_splited[y][c]);
            c_splited[y][c] = __builtin_mxc_pk_fma_f32(
                a1, local_b[1 + PACK_RATIO / 2 * y], c_splited[y][c]);
            c_splited[y][c] = __builtin_mxc_pk_fma_f32(
                a2, local_b[2 + PACK_RATIO / 2 * y], c_splited[y][c]);
            c_splited[y][c] = __builtin_mxc_pk_fma_f32(
                a3, local_b[3 + PACK_RATIO / 2 * y], c_splited[y][c]);
          }
        }
      } else {
        v2f local_b;
        for (int x = 0; x < PACK_RATIO / 2; x++) {
          for (int y = 0; y < N; y++) {
            local_b.x = bsm_b_ptr[tidCol * LoopNum + loop_index + x * 2 +
                                  y * thread_groups * LoopNum];
            local_b.y = bsm_b_ptr[tidCol * LoopNum + loop_index + x * 2 + 1 +
                                  y * thread_groups * LoopNum];
            for (int c = 0; c < m_per_thread; c++) {
              float a1 = (float)__builtin_mxc_ubfe(
                  A[c], QBITS * shuffled_dequant_index[x * 2], QBITS);
              float a2 = (float)__builtin_mxc_ubfe(
                  A[c], QBITS * shuffled_dequant_index[x * 2 + 1], QBITS);
              v2f a = {a1, a2};
              a = __builtin_mxc_pk_fma_f32(a, local_scales[c], local_zeros[c]);
              c_splited[y][c] =
                  __builtin_mxc_pk_fma_f32(a, local_b, c_splited[y][c]);
            }
          }
        }
      }

      if constexpr (thread_groups != QUANT_GROUP / PACK_RATIO) {
        if (loop_index + PACK_RATIO < LoopNum) {
          *((float4*)A) =
              *(float4*)(srcA +
                         (loop_index + PACK_RATIO + i + tidCol * LoopNum) /
                             PACK_RATIO * srcAStride +
                         m_offset_a + m_per_thread * tidRow);
        }
      }
    }
    __syncthreads();
  }

#pragma unroll N
  for (int y = 0; y < N; y++) {
    for (int i = 0; i < m_per_thread; i++) {
      smem[tidCol + (tidRow * m_per_thread + i) * ThreadBlock / BlockDimX] =
          c_splited[y][i].x + c_splited[y][i].y;
    }
    __syncthreads();
    constexpr int stride = ThreadBlock / BlockDimX;
    int data_size = ThreadBlock * m_per_thread;
#pragma unroll
    for (int i = ThreadBlock / BlockDimX / 2; i > 0; i /= 2) {
      for (int j = tid; j < data_size / 2; j += ThreadBlock) {
        int reduce_group = j / i;
        int reduce_index = j % i;
        smem[reduce_index + reduce_group * stride] +=
            smem[reduce_index + reduce_group * stride + i];
      }
      __syncthreads();
      data_size /= 2;
    }
    for (int i = tid; i < m_per_thread * BlockDimX; i += ThreadBlock) {
      atomicAdd(dst + i + y * dstStride, (half)smem[i * stride]);
    }
    if constexpr (N > 1) {
      if (y + 1 < N) {
        __syncthreads();
      }
    }
  }
}

template <int BATCH>
__global__ __launch_bounds__(256) void hgemv_nn_splitk_gptq_tb256_bx256_kb128(
    const half* __restrict__ srcB, const quant_packed_type* __restrict__ srcA,
    const quant_packed_type* __restrict__ zeros,
    const half* __restrict__ scales, half* dst, int m, int n, int k,
    int srcAStride, int dstStride, int k_div_sk,
    const int* __restrict__ b_perm = nullptr) {
  constexpr int QBITS = 4;
  constexpr int PACK_RATIO = (32 / QBITS);
  constexpr int QUANT_GROUP = 128;
  constexpr int ThreadBlock = 256;
  constexpr int BlockDimX = 256;
  constexpr int LoopNum = QUANT_GROUP / (ThreadBlock / BlockDimX);
  constexpr int N = BATCH;
  const int k_stride = k;
  constexpr int k_block = QUANT_GROUP;
  const int splitKOffset = blockIdx.y * k_block;
  k = k_block;
  srcA += splitKOffset * srcAStride / PACK_RATIO;
  // srcB += splitKOffset;

  constexpr int quant_groups = 1;
  constexpr int m_per_thread = sizeof(float4) / sizeof(quant_packed_type);
  constexpr int thread_groups = ThreadBlock / BlockDimX;
  constexpr int group_elements = BlockDimX * m_per_thread;
  constexpr int reduce_size = ThreadBlock * m_per_thread;
  constexpr int data_cache_size =
      LoopNum * thread_groups * N + group_elements + group_elements / 2;
  float *bsm_b_ptr, *bsm_zeros_ptr;
  half* bsm_scales_ptr;

  __shared__ float
      bsm_ptr[2048];  // 128*N+256*4+256*4/2 = 最大8K，每个block
                      // 256线程，占据半个PEU，一个AP可以跑满8个block
  bsm_b_ptr = bsm_ptr;
  bsm_zeros_ptr = bsm_ptr + LoopNum * thread_groups * N;
  bsm_scales_ptr = (half*)(bsm_zeros_ptr + group_elements);
  // smem = bsm_ptr;

  const int zeros_stride = srcAStride / PACK_RATIO;
  const int scales_stride = srcAStride;
  zeros += splitKOffset * zeros_stride / QUANT_GROUP;
  scales += splitKOffset * scales_stride / QUANT_GROUP;

  dst += group_elements * blockIdx.x;
  const int m_offset_a = group_elements * blockIdx.x;
  const int m_offset_zeros = group_elements / PACK_RATIO * blockIdx.x;
  const int m_offset_scales = group_elements * blockIdx.x;

  // store splited fma results
  v2f c_splited[N][m_per_thread];
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < m_per_thread; j++) c_splited[i][j] = {0, 0};
  }

  int tid = threadIdx.x;
  int tidCol = tid / BlockDimX;
  int tidRow = tid % BlockDimX;

  for (int i = 0; i < k; i += LoopNum * thread_groups) {
    int quant_group = i / QUANT_GROUP;
    constexpr int loading_pack = 2;
    constexpr int loading_count = group_elements / loading_pack;
    // Load needed zeros, scales
    for (int x = tid; x < loading_count; x += ThreadBlock) {
      uint8_t temp_zeros = 0;
      temp_zeros =
          ((uint8_t*)(zeros + quant_group * zeros_stride + m_offset_zeros))[x];
      half temp_scales[2];
      *((float*)temp_scales) =
          ((float*)(scales + quant_group * scales_stride + m_offset_scales))[x];
      uint32_t z = temp_zeros;
      uint32_t z0 = __builtin_mxc_ubfe(z, 0, QBITS) + 1;
      uint32_t z1 = __builtin_mxc_ubfe(z, 4, QBITS) + 1;
      float s1 = (float)(temp_scales[0]);
      float s2 = (float)(temp_scales[1]);
      // Store to shared memory
      bsm_zeros_ptr[x * 2] = (float)z0 * s1 * -1.0f;
      bsm_zeros_ptr[x * 2 + 1] = (float)z1 * s2 * -1.0f;
      bsm_scales_ptr[x * 2] = temp_scales[0];
      bsm_scales_ptr[x * 2 + 1] = temp_scales[1];
    }

    int loop_index = 0;
    quant_packed_type A[m_per_thread];
    // Load A
    *((float4*)A) = *(float4*)(srcA +
                               (loop_index + i + tidCol * LoopNum) /
                                   PACK_RATIO * srcAStride +
                               m_offset_a + m_per_thread * tidRow);

    // Load B and transform to float
    if (b_perm != nullptr) {
      for (int y = 0; y < N; y++) {
        for (int x = tidRow; x < LoopNum; x += BlockDimX) {
          bsm_b_ptr[x + tidCol * LoopNum + y * LoopNum * thread_groups] =
              srcB[b_perm[splitKOffset + i + tidCol * LoopNum + x] +
                   y * k_stride];
        }
      }
    } else {
      for (int y = 0; y < N; y++) {
        for (int x = tidRow; x < LoopNum; x += BlockDimX) {
          bsm_b_ptr[x + tidCol * LoopNum + y * LoopNum * thread_groups] =
              srcB[splitKOffset + i + tidCol * LoopNum + x + y * k_stride];
        }
      }
    }

    __syncthreads();

    // Load zero and scale from bsm
    int m_index = tidRow * m_per_thread;

    v2f local_scales[m_per_thread];
    v2f local_scales_d16[m_per_thread];
    for (int c = 0; c < m_per_thread; c++) {
      float s = (float)bsm_scales_ptr[m_index + c];
      float s_d16 = s / 16;
      local_scales[c] = {s, s};
      local_scales_d16[c] = {s_d16, s_d16};
    }
    v2f local_zeros[m_per_thread];
    for (int c = 0; c < m_per_thread; c++)
      local_zeros[c] = {bsm_zeros_ptr[m_index + c], bsm_zeros_ptr[m_index + c]};

    const int shuffled_dequant_index[PACK_RATIO] = {0, 4, 1, 5, 2, 6, 3, 7};

#pragma unroll 16
    for (; loop_index < LoopNum; loop_index += PACK_RATIO) {
      v2f local_b[PACK_RATIO / 2 * N];
      for (int y = 0; y < N; y++) {
        for (int x = 0; x < PACK_RATIO / 2; x++) {
          local_b[x + PACK_RATIO / 2 * y].x =
              bsm_b_ptr[tidCol * LoopNum + loop_index + x * 2 +
                        y * thread_groups * LoopNum];
          local_b[x + PACK_RATIO / 2 * y].y =
              bsm_b_ptr[tidCol * LoopNum + loop_index + x * 2 + 1 +
                        y * thread_groups * LoopNum];
        }
      }
#pragma unroll m_per_thread
      for (int c = 0; c < m_per_thread; c++) {
        uint32_t p0 = A[c] & 0x0f0f0f0f;
        uint32_t p1 = A[c] & 0xf0f0f0f0;
        float o1, o2, o3, o4, o5, o6, o7, o8;
        asm volatile("cvt_b0tof32 %0,%1;\n" : "=r"(o1) : "r"(p0));
        asm volatile("cvt_b1tof32 %0,%1;\n" : "=r"(o2) : "r"(p0));
        asm volatile("cvt_b2tof32 %0,%1;\n" : "=r"(o3) : "r"(p0));
        asm volatile("cvt_b3tof32 %0,%1;\n" : "=r"(o4) : "r"(p0));
        asm volatile("cvt_b0tof32 %0,%1;\n" : "=r"(o5) : "r"(p1));
        asm volatile("cvt_b1tof32 %0,%1;\n" : "=r"(o6) : "r"(p1));
        asm volatile("cvt_b2tof32 %0,%1;\n" : "=r"(o7) : "r"(p1));
        asm volatile("cvt_b3tof32 %0,%1;\n" : "=r"(o8) : "r"(p1));
        v2f a0 = {o1, o3};
        v2f a1 = {o5, o7};
        v2f a2 = {o2, o4};
        v2f a3 = {o6, o8};

        a0 = __builtin_mxc_pk_fma_f32(a0, local_scales[c], local_zeros[c]);
        a1 = __builtin_mxc_pk_fma_f32(a1, local_scales_d16[c], local_zeros[c]);
        a2 = __builtin_mxc_pk_fma_f32(a2, local_scales[c], local_zeros[c]);
        a3 = __builtin_mxc_pk_fma_f32(a3, local_scales_d16[c], local_zeros[c]);

#pragma unroll N
        for (int y = 0; y < N; y++) {
          c_splited[y][c] = __builtin_mxc_pk_fma_f32(
              a0, local_b[0 + PACK_RATIO / 2 * y], c_splited[y][c]);
          c_splited[y][c] = __builtin_mxc_pk_fma_f32(
              a1, local_b[1 + PACK_RATIO / 2 * y], c_splited[y][c]);
          c_splited[y][c] = __builtin_mxc_pk_fma_f32(
              a2, local_b[2 + PACK_RATIO / 2 * y], c_splited[y][c]);
          c_splited[y][c] = __builtin_mxc_pk_fma_f32(
              a3, local_b[3 + PACK_RATIO / 2 * y], c_splited[y][c]);
        }
      }

      if constexpr (thread_groups != QUANT_GROUP / PACK_RATIO) {
        if (loop_index + PACK_RATIO < LoopNum) {
          *((float4*)A) =
              *(float4*)(srcA +
                         (loop_index + PACK_RATIO + i + tidCol * LoopNum) /
                             PACK_RATIO * srcAStride +
                         m_offset_a + m_per_thread * tidRow);
        }
      }
    }
    __syncthreads();
  }

  // directly do atomic add, may cause partial write?
  for (int y = 0; y < N; y++) {
    for (int c = 0; c < m_per_thread; c++) {
      atomicAdd(dst + tidRow * m_per_thread + c + y * dstStride,
                (half)(c_splited[y][c].x + c_splited[y][c].y));
    }
  }
}

template <>
__global__
__launch_bounds__(256) void hgemv_nn_splitk_gptq_tb256_bx256_kb128<4>(
    const half* __restrict__ srcB, const quant_packed_type* __restrict__ srcA,
    const quant_packed_type* __restrict__ zeros,
    const half* __restrict__ scales, half* dst, int m, int n, int k,
    int srcAStride, int dstStride, int k_div_sk,
    const int* __restrict__ b_perm) {
  constexpr int QBITS = 4;
  constexpr int PACK_RATIO = (32 / QBITS);
  constexpr int QUANT_GROUP = 128;
  constexpr int ThreadBlock = 256;
  constexpr int BlockDimX = 256;
  constexpr int LoopNum = QUANT_GROUP / (ThreadBlock / BlockDimX);
  constexpr int N = 4;
  const int k_stride = k;
  constexpr int k_block = QUANT_GROUP;
  const int splitKOffset = blockIdx.y * k_block;
  k = k_block;
  srcA += splitKOffset * srcAStride / PACK_RATIO;
  // srcB += splitKOffset;

  constexpr int quant_groups = 1;
  constexpr int m_per_thread = sizeof(float4) / sizeof(quant_packed_type);
  constexpr int thread_groups = ThreadBlock / BlockDimX;
  constexpr int group_elements = BlockDimX * m_per_thread;
  constexpr int reduce_size = ThreadBlock * m_per_thread;
  constexpr int data_cache_size =
      LoopNum * thread_groups * N + group_elements + group_elements / 2;
  float *bsm_b_ptr, *bsm_zeros_ptr;
  half* bsm_scales_ptr;

  __shared__ float
      bsm_ptr[2048];  // 128*N+256*4+256*4/2 = 最大8K，每个block
                      // 256线程，占据半个PEU，一个AP可以跑满8个block
  bsm_b_ptr = bsm_ptr;
  bsm_zeros_ptr = bsm_ptr + LoopNum * thread_groups * N;
  bsm_scales_ptr = (half*)(bsm_zeros_ptr + group_elements);
  // smem = bsm_ptr;

  const int zeros_stride = srcAStride / PACK_RATIO;
  const int scales_stride = srcAStride;
  zeros += splitKOffset * zeros_stride / QUANT_GROUP;
  scales += splitKOffset * scales_stride / QUANT_GROUP;

  dst += group_elements * blockIdx.x;
  const int m_offset_a = group_elements * blockIdx.x;
  const int m_offset_zeros = group_elements / PACK_RATIO * blockIdx.x;
  const int m_offset_scales = group_elements * blockIdx.x;

  // store splited fma results
  v2f c_splited[N][m_per_thread];
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < m_per_thread; j++) c_splited[i][j] = {0, 0};
  }

  int tid = threadIdx.x;
  int tidCol = tid / BlockDimX;
  int tidRow = tid % BlockDimX;

  for (int i = 0; i < k; i += LoopNum * thread_groups) {
    int quant_group = i / QUANT_GROUP;
    constexpr int loading_pack = 2;
    constexpr int loading_count = group_elements / loading_pack;
    // Load needed zeros, scales
    for (int x = tid; x < loading_count; x += ThreadBlock) {
      uint8_t temp_zeros = 0;
      temp_zeros =
          ((uint8_t*)(zeros + quant_group * zeros_stride + m_offset_zeros))[x];
      half temp_scales[2];
      *((float*)temp_scales) =
          ((float*)(scales + quant_group * scales_stride + m_offset_scales))[x];
      uint32_t z = temp_zeros;
      uint32_t z0 = __builtin_mxc_ubfe(z, 0, QBITS) + 1;
      uint32_t z1 = __builtin_mxc_ubfe(z, 4, QBITS) + 1;
      float s1 = (float)(temp_scales[0]);
      float s2 = (float)(temp_scales[1]);
      // Store to shared memory
      bsm_zeros_ptr[x * 2] = (float)z0 * s1 * -1.0f;
      bsm_zeros_ptr[x * 2 + 1] = (float)z1 * s2 * -1.0f;
      bsm_scales_ptr[x * 2] = temp_scales[0];
      bsm_scales_ptr[x * 2 + 1] = temp_scales[1];
    }

    int loop_index = 0;
    quant_packed_type A[m_per_thread];
    // Load A
    *((float4*)A) = *(float4*)(srcA +
                               (loop_index + i + tidCol * LoopNum) /
                                   PACK_RATIO * srcAStride +
                               m_offset_a + m_per_thread * tidRow);

    // Load B and transform to float
    if (b_perm != nullptr) {
      for (int y = 0; y < N; y++) {
        for (int x = tidRow; x < LoopNum; x += BlockDimX) {
          bsm_b_ptr[x + tidCol * LoopNum + y * LoopNum * thread_groups] =
              srcB[b_perm[splitKOffset + i + tidCol * LoopNum + x] +
                   y * k_stride];
        }
      }
    } else {
      for (int y = 0; y < N; y++) {
        for (int x = tidRow; x < LoopNum; x += BlockDimX) {
          bsm_b_ptr[x + tidCol * LoopNum + y * LoopNum * thread_groups] =
              srcB[splitKOffset + i + tidCol * LoopNum + x + y * k_stride];
        }
      }
    }

    __syncthreads();

    // Load zero and scale from bsm
    int m_index = tidRow * m_per_thread;

    v2f local_scales[m_per_thread];
    for (int c = 0; c < m_per_thread; c++) {
      float s = (float)bsm_scales_ptr[m_index + c];
      local_scales[c] = {s, s};
    }
    v2f local_zeros[m_per_thread];
    for (int c = 0; c < m_per_thread; c++)
      local_zeros[c] = {bsm_zeros_ptr[m_index + c], bsm_zeros_ptr[m_index + c]};

    // const int shuffled_dequant_index[PACK_RATIO] = {0, 4, 1, 5, 2, 6, 3, 7};

#pragma unroll 16
    for (; loop_index < LoopNum; loop_index += PACK_RATIO) {
      // Split dequant and w*b into 4 parts so we can reduce registers usage
      // from 114 to 76, and each peu will run 6 waves
      v2f local_b[N];
      uint32_t Aq[m_per_thread];
      for (int y = 0; y < N; y++) {
        int x = 0;
        local_b[y].x = bsm_b_ptr[tidCol * LoopNum + loop_index + x * 2 +
                                 y * thread_groups * LoopNum];
        local_b[y].y = bsm_b_ptr[tidCol * LoopNum + loop_index + x * 2 + 1 +
                                 y * thread_groups * LoopNum];
      }
#pragma unroll m_per_thread
      for (int c = 0; c < m_per_thread; c++) {
        Aq[c] = A[c] & 0x0f0f0f0f;
        float o1, o3;
        asm volatile("cvt_b0tof32 %0,%1;\n" : "=r"(o1) : "r"(Aq[c]));
        asm volatile("cvt_b2tof32 %0,%1;\n" : "=r"(o3) : "r"(Aq[c]));
        v2f a0 = {o1, o3};

        a0 = __builtin_mxc_pk_fma_f32(a0, local_scales[c], local_zeros[c]);

#pragma unroll N
        for (int y = 0; y < N; y++) {
          c_splited[y][c] =
              __builtin_mxc_pk_fma_f32(a0, local_b[y], c_splited[y][c]);
        }
      }

      for (int y = 0; y < N; y++) {
        int x = 2;
        local_b[y].x = bsm_b_ptr[tidCol * LoopNum + loop_index + x * 2 +
                                 y * thread_groups * LoopNum];
        local_b[y].y = bsm_b_ptr[tidCol * LoopNum + loop_index + x * 2 + 1 +
                                 y * thread_groups * LoopNum];
      }

#pragma unroll m_per_thread
      for (int c = 0; c < m_per_thread; c++) {
        float o2, o4;
        asm volatile("cvt_b1tof32 %0,%1;\n" : "=r"(o2) : "r"(Aq[c]));
        asm volatile("cvt_b3tof32 %0,%1;\n" : "=r"(o4) : "r"(Aq[c]));
        v2f a2 = {o2, o4};

        a2 = __builtin_mxc_pk_fma_f32(a2, local_scales[c], local_zeros[c]);

#pragma unroll N
        for (int y = 0; y < N; y++) {
          c_splited[y][c] =
              __builtin_mxc_pk_fma_f32(a2, local_b[y], c_splited[y][c]);
        }
      }

      for (int y = 0; y < N; y++) {
        int x = 1;
        local_b[y].x = bsm_b_ptr[tidCol * LoopNum + loop_index + x * 2 +
                                 y * thread_groups * LoopNum];
        local_b[y].y = bsm_b_ptr[tidCol * LoopNum + loop_index + x * 2 + 1 +
                                 y * thread_groups * LoopNum];
      }

      for (int c = 0; c < m_per_thread; c++) {
        Aq[c] = (A[c] >> 4) & 0x0f0f0f0f;
        float o5, o7;
        asm volatile("cvt_b0tof32 %0,%1;\n" : "=r"(o5) : "r"(Aq[c]));
        asm volatile("cvt_b2tof32 %0,%1;\n" : "=r"(o7) : "r"(Aq[c]));
        v2f a1 = {o5, o7};
        a1 = __builtin_mxc_pk_fma_f32(a1, local_scales[c], local_zeros[c]);

#pragma unroll N
        for (int y = 0; y < N; y++) {
          c_splited[y][c] =
              __builtin_mxc_pk_fma_f32(a1, local_b[y], c_splited[y][c]);
        }
      }

      for (int y = 0; y < N; y++) {
        int x = 3;
        local_b[y].x = bsm_b_ptr[tidCol * LoopNum + loop_index + x * 2 +
                                 y * thread_groups * LoopNum];
        local_b[y].y = bsm_b_ptr[tidCol * LoopNum + loop_index + x * 2 + 1 +
                                 y * thread_groups * LoopNum];
      }

      for (int c = 0; c < m_per_thread; c++) {
        float o6, o8;
        asm volatile("cvt_b1tof32 %0,%1;\n" : "=r"(o6) : "r"(Aq[c]));
        asm volatile("cvt_b3tof32 %0,%1;\n" : "=r"(o8) : "r"(Aq[c]));
        v2f a3 = {o6, o8};
        a3 = __builtin_mxc_pk_fma_f32(a3, local_scales[c], local_zeros[c]);

#pragma unroll N
        for (int y = 0; y < N; y++) {
          c_splited[y][c] =
              __builtin_mxc_pk_fma_f32(a3, local_b[y], c_splited[y][c]);
        }
      }

      if constexpr (thread_groups != QUANT_GROUP / PACK_RATIO) {
        if (loop_index + PACK_RATIO < LoopNum) {
          *((float4*)A) =
              *(float4*)(srcA +
                         (loop_index + PACK_RATIO + i + tidCol * LoopNum) /
                             PACK_RATIO * srcAStride +
                         m_offset_a + m_per_thread * tidRow);
        }
      }
    }
    __syncthreads();
  }

  // directly do atomic add, may cause partial write?
  for (int y = 0; y < N; y++) {
    for (int c = 0; c < m_per_thread; c++) {
      atomicAdd(dst + tidRow * m_per_thread + c + y * dstStride,
                (half)(c_splited[y][c].x + c_splited[y][c].y));
    }
  }
}
