#pragma once

#include <mc_runtime.h>
#include <maca_fp16.h>
#include "Hgemm_common.cuh"
#include "dequant.cuh"
#define quant_packed_type uint32_t
typedef __NATIVE_VECTOR__(2, float) v2f;
template <int ThreadBlock, int BlockDimX, int BATCH>
__global__ __launch_bounds__(256) void hgemv_nn_splitk_gptq_int8(
    const half* __restrict__ srcB, const quant_packed_type* __restrict__ srcA,
    const quant_packed_type* __restrict__ zeros,
    const half* __restrict__ scales, half* dst, int m, int n, int k,
    int srcAStride, int dstStride, int k_div_sk,
    const int* __restrict__ b_perm = nullptr) {
  constexpr int QBITS = 8;
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
    constexpr int loading_pack = 1;
    constexpr int loading_count = group_elements / loading_pack;
    // Load needed zeros, scales
    for (int x = tid; x < loading_count; x += ThreadBlock) {
      uint8_t temp_zeros = 0;
      temp_zeros =
          ((uint8_t*)(zeros + quant_group * zeros_stride + m_offset_zeros))[x];
      half temp_scales;
      temp_scales = (scales + quant_group * scales_stride + m_offset_scales)[x];
      uint32_t z = temp_zeros;
      float z0;
      asm volatile("cvt_b0tof32 %0,%1;\n" : "=r"(z0) : "r"(z));
      float s1 = (float)(temp_scales);
      // Store to shared memory
      // bsm_zeros_ptr[x] = (float)z0 * s1 * -1.0f;    //modify 11.19
      bsm_zeros_ptr[x] = (float)(z0 + 1) * s1 * -1.0f;
      bsm_scales_ptr[x] = s1;
      // if (i == 0 && blockIdx.x == 0) {
      //     printf("tid %d, x = %d, temp_zero=%u, temp_scale=%f,  z0 = %f, s1 =
      //     %f\n", tid, x, (uint32_t)temp_zeros, (float)temp_scales, z0, s1);
      // }
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
      float s = bsm_scales_ptr[m_index + c];
      local_scales[c] = {s, s};
      // if (tid == 0 && c == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
      //     printf("get scale %f from index %d\n", s, m_index+c);
      // }
    }
    v2f local_zeros[m_per_thread];
    for (int c = 0; c < m_per_thread; c++) {
      local_zeros[c] = {bsm_zeros_ptr[m_index + c], bsm_zeros_ptr[m_index + c]};
      // if (tid == 0 && c == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
      //     printf("get zero %f from index %d\n", local_zeros[c].x, m_index+c);
      // }
    }

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
        float o1, o2, o3, o4;
        asm volatile("cvt_b0tof32 %0,%1;\n" : "=r"(o1) : "r"(A[c]));
        asm volatile("cvt_b1tof32 %0,%1;\n" : "=r"(o2) : "r"(A[c]));
        asm volatile("cvt_b2tof32 %0,%1;\n" : "=r"(o3) : "r"(A[c]));
        asm volatile("cvt_b3tof32 %0,%1;\n" : "=r"(o4) : "r"(A[c]));
        v2f a0 = {o1, o2};
        v2f a1 = {o3, o4};

        // if (tid == 0 && c == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
        //     printf("GPU int8 a0=%f,%f, a1=%f,%f,scale=%f,zero=%f\n",
        //     a0.x, a0.y, a1.x, a1.y, local_scales[c].x, local_zeros[c].x);
        // }

        a0 = __builtin_mxc_pk_fma_f32(a0, local_scales[c], local_zeros[c]);
        a1 = __builtin_mxc_pk_fma_f32(a1, local_scales[c], local_zeros[c]);

        // if (tid == 0 && c == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
        //     printf("GPU a=%x, a0=%f,%f,a1=%f,%f,b=%f,%f,%f,%f\n",
        //         A[c], a0.x, a0.y, a1.x, a1.y, local_b[0].x, local_b[0].y,
        //         local_b[1].x, local_b[1].y
        //     );
        // }

#pragma unroll N
        for (int y = 0; y < N; y++) {
          c_splited[y][c] = __builtin_mxc_pk_fma_f32(
              a0, local_b[0 + PACK_RATIO / 2 * y], c_splited[y][c]);
          c_splited[y][c] = __builtin_mxc_pk_fma_f32(
              a1, local_b[1 + PACK_RATIO / 2 * y], c_splited[y][c]);
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
