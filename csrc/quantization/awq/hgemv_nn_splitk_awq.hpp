// 2025 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
/**
 * @file hgemv_nn_splitk.hpp
 * @author Jiawen Yang (jiawen.yang@metax-tech.com)
 * @brief *
 * @version 0.1
 * @date 2024-03-05
 * @copyright Copyright (c) 2024
 *
 *   fp16 gemv kernel template for some gemv cases
 *   Note:
 *    1. BlockDimX * BlockDimY = 64, and BlockDimX should be 8/16/32/64
 *    2. LoopNum % 2 == 0, so Load_B can use ldg_b32 or ldg_b64
 *    3. m % (BlockDimX * 8) == 0
 *    4. k % (ThreadBlock / BlockDimX * LoopNum * SplitKNum) = 0
 *
 *    A load layout:
 *
 *       **************************** Wave_0 ******************* | Wave_1  ...
 *       ********* Repeat LoopNum *********                      |
 *       tid_0(ldg_b128)   tid_0 ... tid_0 | tid_(BlockDimX) ... |
 *       tid_1                             |                     |
 *       tid_2                             |                     |
 *       ……                                |                     |
 *       tid_(BlockDimX-1)                 |                     |
 *
 */
#pragma once

#include <mc_runtime.h>
#include <maca_fp16.h>
#include "../gptq/Hgemm_common.cuh"
#include "dequant.cuh"
#define quant_packed_type uint32_t
typedef __NATIVE_VECTOR__(2, float) v2f;

template<int N, int m_per_thread>
__device__ __forceinline__ void dequant_fma_awq_int4(
                const quant_packed_type& a,
                const v2f (&scale)[m_per_thread/2],
                const v2f (&zero)[m_per_thread/2],
                const v2f (&b)[N],
                v2f (&out)[N][m_per_thread/2]) {
    uint32_t p0 = a & 0x0f0f0f0f;
    float o1,o3;
    asm volatile("cvt_b0tof32 %0,%1;\n":"=r"(o1):"r"(p0));
    asm volatile("cvt_b2tof32 %0,%1;\n":"=r"(o3):"r"(p0));
    v2f a0 = {o1, o3};
    a0 = __builtin_mxc_pk_fma_f32(a0, scale[0], zero[0]);

    #pragma unroll N
    for (int y = 0; y < N; y++) {
        out[y][0] = __builtin_mxc_pk_fma_f32(a0, b[y], out[y][0]);
    }

    asm volatile("cvt_b1tof32 %0,%1;\n":"=r"(o1):"r"(p0));
    asm volatile("cvt_b3tof32 %0,%1;\n":"=r"(o3):"r"(p0));
    a0 = {o1, o3};
    a0 = __builtin_mxc_pk_fma_f32(a0, scale[2], zero[2]);

    #pragma unroll N
    for (int y = 0; y < N; y++) {
        out[y][2] = __builtin_mxc_pk_fma_f32(a0, b[y], out[y][2]);
    }

    uint32_t p1 = (a >> 4) & 0x0f0f0f0f;
    asm volatile("cvt_b0tof32 %0,%1;\n":"=r"(o1):"r"(p1));
    asm volatile("cvt_b2tof32 %0,%1;\n":"=r"(o3):"r"(p1));
    a0 = {o1, o3};
    a0 = __builtin_mxc_pk_fma_f32(a0, scale[1], zero[1]);

    #pragma unroll N
    for (int y = 0; y < N; y++) {
        out[y][1] = __builtin_mxc_pk_fma_f32(a0, b[y], out[y][1]);
    }

    asm volatile("cvt_b1tof32 %0,%1;\n":"=r"(o1):"r"(p1));
    asm volatile("cvt_b3tof32 %0,%1;\n":"=r"(o3):"r"(p1));
    a0 = {o1, o3};
    a0 = __builtin_mxc_pk_fma_f32(a0, scale[3], zero[3]);

    #pragma unroll N
    for (int y = 0; y < N; y++) {
        out[y][3] = __builtin_mxc_pk_fma_f32(a0, b[y], out[y][3]);
    }
};

template <int ThreadBlock, int BlockDimX, int BATCH>
__global__ __launch_bounds__(256) void hgemv_nn_splitk_awq(const half* __restrict__ srcB,
                                                        const quant_packed_type* __restrict__ srcA,
                                                        const quant_packed_type* __restrict__ zeros,
                                                        const half* __restrict__ scales,
                                                        half *dst,
                                                        int m,
                                                        int n,
                                                        int k,
                                                        int srcAStride,
                                                        int dstStride,
                                                        int k_div_sk,
                                                        const int* __restrict__ b_perm = nullptr) {
    constexpr int QBITS = 4;
    constexpr int PACK_RATIO = (32 / QBITS);
    constexpr int QUANT_GROUP = 128;
    constexpr int LoopNum = QUANT_GROUP / (ThreadBlock / BlockDimX);
    constexpr int N = BATCH;
    const int k_stride = k;
    const int k_block = (k_div_sk + QUANT_GROUP - 1) / QUANT_GROUP * QUANT_GROUP;
    const int splitKOffset = blockIdx.y * k_block;
    if (splitKOffset + k_block > k) k = k - splitKOffset;
    else k = k_block;
    srcA += splitKOffset * srcAStride / PACK_RATIO;
    //srcB += splitKOffset;

    constexpr int quant_groups = 1;
    constexpr int m_per_thread = PACK_RATIO;
    constexpr int thread_groups = ThreadBlock / BlockDimX;
    constexpr int group_elements = BlockDimX * m_per_thread;
    constexpr int reduce_size = ThreadBlock * m_per_thread;
    constexpr int data_cache_size = LoopNum * thread_groups * N + group_elements + group_elements / 2;

    float *bsm_b_ptr, *bsm_zeros_ptr, *smem;
    half *bsm_scales_ptr;
    if constexpr(reduce_size > data_cache_size) {
        __shared__ float bsm_ptr[reduce_size];
        bsm_b_ptr = bsm_ptr;
        bsm_zeros_ptr = bsm_ptr + LoopNum * thread_groups * N;
        bsm_scales_ptr = (half*)(bsm_zeros_ptr + group_elements);
        smem = bsm_ptr;
    } else {
        __shared__ float bsm_ptr[data_cache_size];
        bsm_b_ptr = bsm_ptr;
        bsm_zeros_ptr = bsm_ptr + LoopNum * thread_groups * N;
        bsm_scales_ptr = (half*)(bsm_zeros_ptr + group_elements);
        smem = bsm_ptr;
    }

    const int zeros_stride = srcAStride / PACK_RATIO;
    const int scales_stride = srcAStride;
    zeros += splitKOffset * zeros_stride / QUANT_GROUP;
    scales += splitKOffset * scales_stride / QUANT_GROUP;

    dst += group_elements * blockIdx.x;
    const int m_offset_a = group_elements * blockIdx.x / PACK_RATIO;
    const int m_offset_zeros = group_elements / PACK_RATIO * blockIdx.x;
    const int m_offset_scales = group_elements * blockIdx.x;

    //store splited fma results
    v2f c_splited[N][m_per_thread/2];
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < m_per_thread/2; j++) c_splited[i][j] = {0, 0};
    }

    int tid = threadIdx.x;
    int tidCol = tid / BlockDimX;
    int tidRow = tid % BlockDimX;
    int this_group_elements = (blockIdx.x + 1) * group_elements <= m ? group_elements : m - blockIdx.x * group_elements;
    int m_index = tidRow * m_per_thread;
    for (int i = 0; i < k; i += LoopNum * thread_groups) {
        int quant_group = i / QUANT_GROUP;
        constexpr int loading_pack = 2;
        int loading_count = this_group_elements / loading_pack;
        //Load needed zeros, scales
        const int shuffle_rank[8] = {0, 2, 4, 6, 1, 3, 5, 7};
        for (int x = tid; x < loading_count; x += ThreadBlock) {
            uint8_t temp_zeros = 0;
            temp_zeros = ((uint8_t*)(zeros + quant_group * zeros_stride + m_offset_zeros))[x];
            half temp_scales[2];
            int packed_group_offset = (x >> 2) << 3;
            int packed_index = (x << 1) % 8;
            int dest_offset_0 = packed_group_offset + shuffle_rank[packed_index];
            int dest_offset_1 = packed_group_offset + shuffle_rank[packed_index + 1];
            //*((float*)temp_scales) = ((float*)(scales + quant_group * scales_stride + m_offset_scales))[x];
            temp_scales[0] = scales[quant_group * scales_stride + m_offset_scales + dest_offset_0];
            temp_scales[1] = scales[quant_group * scales_stride + m_offset_scales + dest_offset_1];
            uint32_t z = temp_zeros;
            uint32_t z0 = __builtin_mxc_ubfe(z, 0, QBITS);
            uint32_t z1 = __builtin_mxc_ubfe(z, 4, QBITS);
            float s1 = (float)(temp_scales[0]);
            float s2 = (float)(temp_scales[1]);
            //Store to shared memory
            bsm_zeros_ptr[dest_offset_0] = (float)z0 * s1 * -1.0f;
            bsm_zeros_ptr[dest_offset_1] = (float)z1 * s2 * -1.0f;
            bsm_scales_ptr[dest_offset_0] = temp_scales[0];
            bsm_scales_ptr[dest_offset_1] = temp_scales[1];
        }

        int loop_index = 0;

        //Load B and transform to float
        if (b_perm != nullptr) {
            for (int y = 0; y < N; y++) {
                for (int x = tidRow; x < LoopNum; x += BlockDimX) {
                    bsm_b_ptr[x + tidCol * LoopNum + y * LoopNum * thread_groups] = srcB[b_perm[splitKOffset + i + tidCol * LoopNum + x] + y * k_stride];
                }
            }
        } else {
            for (int y = 0; y < N; y++) {
                for (int x = tidRow; x < LoopNum; x += BlockDimX) {
                    bsm_b_ptr[x + tidCol * LoopNum + y * LoopNum * thread_groups] = srcB[splitKOffset + i + tidCol * LoopNum + x + y * k_stride];
                }
            }
        }

        __syncthreads();

        //Load zero and scale from bsm
        if (m_index < this_group_elements) {
            v2f local_scales[m_per_thread/2];
            for (int c = 0; c < m_per_thread / 2; c++) {
                float s0 = (float)bsm_scales_ptr[m_index + c*2];
                float s1 = (float)bsm_scales_ptr[m_index + c*2+1];
                local_scales[c] = {s0, s1};
            }
            v2f local_zeros[m_per_thread/2];
            for (int c = 0; c < m_per_thread/2; c++) local_zeros[c] = {bsm_zeros_ptr[m_index + c*2],bsm_zeros_ptr[m_index + c*2+1]};

    #define DEQUANT_FMA(a, b) \
            dequant_fma_awq_int4<N, m_per_thread>(a, local_scales, local_zeros, b, c_splited);

            quant_packed_type A[4];
            const int packed_a_stride = srcAStride / PACK_RATIO;
            int src_a_offset = (loop_index + i + tidCol * LoopNum) * srcAStride / PACK_RATIO + m_offset_a + m_per_thread * tidRow / PACK_RATIO;
            A[0] = srcA[src_a_offset];
            src_a_offset += packed_a_stride;
            A[1] = srcA[src_a_offset];

            v2f local_b[4][N];
            //#pragma unroll LoopNum / 4 - 1
            for (; loop_index < LoopNum - 4; loop_index += 4) {
                //Load A
                src_a_offset += packed_a_stride;
                A[2] = srcA[src_a_offset];
                src_a_offset += packed_a_stride;
                A[3] = srcA[src_a_offset];

                for (int y = 0; y < N; y++) {
                    float s[4];
                    *(float4*)s = *(float4*)(bsm_b_ptr+tidCol*LoopNum+loop_index+y*thread_groups*LoopNum);
                    local_b[0][y] = {s[0], s[0]};
                    local_b[1][y] = {s[1], s[1]};
                    local_b[2][y] = {s[2], s[2]};
                    local_b[3][y] = {s[3], s[3]};
                }
                DEQUANT_FMA(A[0], local_b[0])
                DEQUANT_FMA(A[1], local_b[1])
                src_a_offset += packed_a_stride;
                A[0] = srcA[src_a_offset];
                src_a_offset += packed_a_stride;
                A[1] = srcA[src_a_offset];
                DEQUANT_FMA(A[2], local_b[2])
                DEQUANT_FMA(A[3], local_b[3])
            }
            src_a_offset += packed_a_stride;
            A[2] = srcA[src_a_offset];
            src_a_offset += packed_a_stride;
            A[3] = srcA[src_a_offset];
            for (int y = 0; y < N; y++) {
                float s[4];
                *(float4*)s = *(float4*)(bsm_b_ptr+tidCol*LoopNum+loop_index+y*thread_groups*LoopNum);
                local_b[0][y] = {s[0], s[0]};
                local_b[1][y] = {s[1], s[1]};
                local_b[2][y] = {s[2], s[2]};
                local_b[3][y] = {s[3], s[3]};
            }
            DEQUANT_FMA(A[0], local_b[0])
            DEQUANT_FMA(A[1], local_b[1])
            DEQUANT_FMA(A[2], local_b[2])
            DEQUANT_FMA(A[3], local_b[3])
        }
        __syncthreads();
    }
#undef DEQUANT_FMA
    #pragma unroll N
    for (int y = 0; y < N; y++) {
        if (m_index < this_group_elements) {
            for (int i = 0; i < m_per_thread/2; i++) {
                smem[tidCol + (tidRow * m_per_thread + i*2) * ThreadBlock / BlockDimX] = c_splited[y][i].x;
                smem[tidCol + (tidRow * m_per_thread + i*2+1) * ThreadBlock / BlockDimX] = c_splited[y][i].y;
            }
        }
        __syncthreads();
        constexpr int stride = ThreadBlock / BlockDimX;
        int data_size = ThreadBlock * m_per_thread;
        #pragma unroll
        for (int i = ThreadBlock / BlockDimX / 2; i > 0; i /= 2) {
            for (int j = tid; j < data_size / 2; j += ThreadBlock) {
                int reduce_group = j / i;
                int reduce_index = j % i;
                smem[reduce_index + reduce_group * stride] += smem[reduce_index + reduce_group * stride + i];
            }
            __syncthreads();
            data_size /= 2;
        }
        for (int i = tid; i < this_group_elements; i += ThreadBlock) {
            atomicAdd(dst + i + y * dstStride, (half)smem[i*stride]);
        }
        if constexpr(N > 1) {
            if (y + 1 < N) {
                __syncthreads();
            }
        }
    }
}

template <int BX, int BATCH>
__global__ __launch_bounds__(256) void hgemv_nn_splitk_awq_kb128(const half* __restrict__ srcB,
                                                        const quant_packed_type* __restrict__ srcA,
                                                        const quant_packed_type* __restrict__ zeros,
                                                        const half* __restrict__ scales,
                                                        half *dst,
                                                        int m,
                                                        int n,
                                                        int k,
                                                        int srcAStride,
                                                        int dstStride,
                                                        int k_div_sk,
                                                        const int* __restrict__ b_perm = nullptr) {
    constexpr int QBITS = 4;
    constexpr int PACK_RATIO = (32 / QBITS);
    constexpr int QUANT_GROUP = 128;
    constexpr int ThreadBlock = 256;
    constexpr int BlockDimX = BX;
    constexpr int LoopNum = QUANT_GROUP / (ThreadBlock / BlockDimX);
    constexpr int N = BATCH;
    const int k_stride = k;
    const int k_block = 128;
    const int splitKOffset = blockIdx.y * k_block;

    k = k_block;
    srcA += splitKOffset * srcAStride / PACK_RATIO;
    //srcB += splitKOffset;

    constexpr int quant_groups = 1;
    constexpr int m_per_thread = PACK_RATIO;
    constexpr int thread_groups = ThreadBlock / BlockDimX;
    constexpr int group_elements = BlockDimX * m_per_thread;
    constexpr int reduce_size = ThreadBlock * m_per_thread;
    constexpr int data_cache_size = LoopNum * thread_groups * N + group_elements + group_elements / 2;
    float *bsm_b_ptr, *bsm_zeros_ptr, *smem;
    half *bsm_scales_ptr;
    if constexpr(reduce_size > data_cache_size) {
        __shared__ float bsm_ptr[reduce_size];
        bsm_b_ptr = bsm_ptr;
        bsm_zeros_ptr = bsm_ptr + LoopNum * thread_groups * N;
        bsm_scales_ptr = (half*)(bsm_zeros_ptr + group_elements);
        smem = bsm_ptr;
    } else {
        __shared__ float bsm_ptr[data_cache_size];
        bsm_b_ptr = bsm_ptr;
        bsm_zeros_ptr = bsm_ptr + LoopNum * thread_groups * N;
        bsm_scales_ptr = (half*)(bsm_zeros_ptr + group_elements);
        smem = bsm_ptr;
    }

    const int zeros_stride = srcAStride / PACK_RATIO;
    const int scales_stride = srcAStride;
    zeros += splitKOffset * zeros_stride / QUANT_GROUP;
    scales += splitKOffset * scales_stride / QUANT_GROUP;

    dst += group_elements * blockIdx.x;
    const int m_offset_a = group_elements * blockIdx.x / PACK_RATIO;
    const int m_offset_zeros = group_elements / PACK_RATIO * blockIdx.x;
    const int m_offset_scales = group_elements * blockIdx.x;

    //store splited fma results
    v2f c_splited[N][m_per_thread/2];
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < m_per_thread/2; j++) c_splited[i][j] = {0, 0};
    }

    int tid = threadIdx.x;
    int tidCol = tid / BlockDimX;
    int tidRow = tid % BlockDimX;
    int this_group_elements = (blockIdx.x + 1) * group_elements <= m ? group_elements : m - blockIdx.x * group_elements;

    constexpr int quant_group = 0;
    constexpr int loading_pack = 2;
    int loading_count = this_group_elements / loading_pack;
    //Load needed zeros, scales
    const int shuffle_rank[8] = {0, 2, 4, 6, 1, 3, 5, 7};
    for (int x = tid; x < loading_count; x += ThreadBlock) {
        uint8_t temp_zeros = 0;
        temp_zeros = ((uint8_t*)(zeros + quant_group * zeros_stride + m_offset_zeros))[x];
        half temp_scales[2];
        int packed_group_offset = (x >> 2) << 3;
        int packed_index = (x << 1) % 8;
        int dest_offset_0 = packed_group_offset + shuffle_rank[packed_index];
        int dest_offset_1 = packed_group_offset + shuffle_rank[packed_index + 1];
        //*((float*)temp_scales) = ((float*)(scales + quant_group * scales_stride + m_offset_scales))[x];
        temp_scales[0] = scales[quant_group * scales_stride + m_offset_scales + dest_offset_0];
        temp_scales[1] = scales[quant_group * scales_stride + m_offset_scales + dest_offset_1];
        uint32_t z = temp_zeros;
        uint32_t z0 = __builtin_mxc_ubfe(z, 0, QBITS);
        uint32_t z1 = __builtin_mxc_ubfe(z, 4, QBITS);
        float s1 = (float)(temp_scales[0]);
        float s2 = (float)(temp_scales[1]);
        //Store to shared memory
        bsm_zeros_ptr[dest_offset_0] = (float)z0 * s1 * -1.0f;
        bsm_zeros_ptr[dest_offset_1] = (float)z1 * s2 * -1.0f;
        bsm_scales_ptr[dest_offset_0] = temp_scales[0];
        bsm_scales_ptr[dest_offset_1] = temp_scales[1];
    }

    int loop_index = 0;

    //Load B and transform to float
    if (b_perm != nullptr) {
        for (int y = 0; y < N; y++) {
            for (int x = tidRow; x < LoopNum; x += BlockDimX) {
                bsm_b_ptr[x + tidCol * LoopNum + y * LoopNum * thread_groups] = srcB[b_perm[splitKOffset + tidCol * LoopNum + x] + y * k_stride];
            }
        }
    } else {
        for (int y = 0; y < N; y++) {
            for (int x = tidRow; x < LoopNum; x += BlockDimX) {
                bsm_b_ptr[x + tidCol * LoopNum + y * LoopNum * thread_groups] = srcB[splitKOffset + tidCol * LoopNum + x + y * k_stride];
            }
        }
    }

    __syncthreads();

    //Load zero and scale from bsm
    int m_index = tidRow * m_per_thread;
    if (m_index < this_group_elements) {
        v2f local_scales[m_per_thread/2];
        for (int c = 0; c < m_per_thread / 2; c++) {
            float s0 = (float)bsm_scales_ptr[m_index + c*2];
            float s1 = (float)bsm_scales_ptr[m_index + c*2+1];
            local_scales[c] = {s0, s1};
        }
        v2f local_zeros[m_per_thread/2];
        for (int c = 0; c < m_per_thread/2; c++) local_zeros[c] = {bsm_zeros_ptr[m_index + c*2],bsm_zeros_ptr[m_index + c*2+1]};

#define DEQUANT_FMA(a, b) \
        dequant_fma_awq_int4<N, m_per_thread>(a, local_scales, local_zeros, b, c_splited);

        quant_packed_type A[4];
        const int packed_a_stride = srcAStride / PACK_RATIO;
        int src_a_offset = (loop_index + tidCol * LoopNum) * srcAStride / PACK_RATIO + m_offset_a + m_per_thread * tidRow / PACK_RATIO;
        A[0] = srcA[src_a_offset];
        src_a_offset += packed_a_stride;
        A[1] = srcA[src_a_offset];

        v2f local_b[4][N];
        #pragma unroll LoopNum / 4 - 1
        for (; loop_index < LoopNum - 4; loop_index += 4) {
            //Load A
            src_a_offset += packed_a_stride;
            A[2] = srcA[src_a_offset];
            src_a_offset += packed_a_stride;
            A[3] = srcA[src_a_offset];

            for (int y = 0; y < N; y++) {
                float s[4];
                *(float4*)s = *(float4*)(bsm_b_ptr+tidCol*LoopNum+loop_index+y*thread_groups*LoopNum);
                local_b[0][y] = {s[0], s[0]};
                local_b[1][y] = {s[1], s[1]};
                local_b[2][y] = {s[2], s[2]};
                local_b[3][y] = {s[3], s[3]};
            }
            DEQUANT_FMA(A[0], local_b[0])
            DEQUANT_FMA(A[1], local_b[1])
            src_a_offset += packed_a_stride;
            A[0] = srcA[src_a_offset];
            src_a_offset += packed_a_stride;
            A[1] = srcA[src_a_offset];
            DEQUANT_FMA(A[2], local_b[2])
            DEQUANT_FMA(A[3], local_b[3])
        }
        src_a_offset += packed_a_stride;
        A[2] = srcA[src_a_offset];
        src_a_offset += packed_a_stride;
        A[3] = srcA[src_a_offset];
        for (int y = 0; y < N; y++) {
            float s[4];
            *(float4*)s = *(float4*)(bsm_b_ptr+tidCol*LoopNum+loop_index+y*thread_groups*LoopNum);
            local_b[0][y] = {s[0], s[0]};
            local_b[1][y] = {s[1], s[1]};
            local_b[2][y] = {s[2], s[2]};
            local_b[3][y] = {s[3], s[3]};
        }
        DEQUANT_FMA(A[0], local_b[0])
        DEQUANT_FMA(A[1], local_b[1])
        DEQUANT_FMA(A[2], local_b[2])
        DEQUANT_FMA(A[3], local_b[3])
    }
    __syncthreads();

#undef DEQUANT_FMA
    #pragma unroll N
    for (int y = 0; y < N; y++) {
        if (m_index < this_group_elements) {
            for (int i = 0; i < m_per_thread/2; i++) {
                smem[tidCol + (tidRow * m_per_thread + i*2) * ThreadBlock / BlockDimX] = c_splited[y][i].x;
                smem[tidCol + (tidRow * m_per_thread + i*2+1) * ThreadBlock / BlockDimX] = c_splited[y][i].y;
            }
        }
        __syncthreads();
        constexpr int stride = ThreadBlock / BlockDimX;
        int data_size = ThreadBlock * m_per_thread;
        #pragma unroll
        for (int i = ThreadBlock / BlockDimX / 2; i > 0; i /= 2) {
            for (int j = tid; j < data_size / 2; j += ThreadBlock) {
                int reduce_group = j / i;
                int reduce_index = j % i;
                smem[reduce_index + reduce_group * stride] += smem[reduce_index + reduce_group * stride + i];
            }
            __syncthreads();
            data_size /= 2;
        }
        for (int i = tid; i < this_group_elements; i += ThreadBlock) {
            atomicAdd(dst + i + y * dstStride, (half)smem[i*stride]);
        }
        if constexpr(N > 1) {
            if (y + 1 < N) {
                __syncthreads();
            }
        }
    }
}
