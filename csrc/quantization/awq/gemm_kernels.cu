// 2025 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
/*
Adapted from https://github.com/mit-han-lab/llm-awq
@article{lin2023awq,
  title={AWQ: Activation-aware Weight Quantization for LLM Compression and
Acceleration}, author={Lin, Ji and Tang, Jiaming and Tang, Haotian and Yang,
Shang and Dang, Xingyu and Han, Song}, journal={arXiv}, year={2023}
}
 */

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>

#include "dequantize.cuh"

#include <cuda_fp16.h>

#include "../gptq/hgemm_gptq.h"
#include "../gptq/scalar_type.hpp"

//#include "hgemv_nn_splitk_awq.hpp"
//#include "hgemv_selector.hpp"
//#include "Hgemm_nn_128x32x128_8m1n8k_awq.hpp"

namespace vllm {
namespace awq {
#define input_type __half
#define output_type __half
#define quant_packed_type uint32_t
#define QUANT_GROUP 128

struct DivModFast {
    DivModFast(int d = 1)
    {
        d_ = (d == 0) ? 1 : d;
        for (l_ = 0;; ++l_) {
            if ((1U << l_) >= d_)
                break;
        }
        uint64_t one = 1;
        uint64_t m   = ((one << 32) * ((one << l_) - d_)) / d_ + 1;
        m_           = static_cast<uint32_t>(m);
    }

    __device__ __inline__ int div(int idx) const
    {
        uint32_t tm = __umulhi(m_, idx); // get high 32-bit of the product
        return (tm + idx) >> l_;
    }

    __device__ __inline__ int mod(int idx) const
    {
        return idx - d_ * div(idx);
    }

    __device__ __inline__ void divmod(int idx, int &quo, int &rem)
    {
        quo = div(idx);
        rem = idx - quo * d_;
    }
    
    uint32_t d_; // divisor
    uint32_t l_; // ceil(log2(d_))
    uint32_t m_; // m' in the papaer
};

#if 0
template <int N>
__global__ void __launch_bounds__(64)
    gemm_forward_4bit_cuda_m16nXk32(int G, int split_k_iters,
                                    half* __restrict__ A, int* __restrict__ B,
                                    half* __restrict__ scaling_factors,
                                    int* __restrict__ zeros, int M, int IC,
                                    int OC, half* __restrict__ C) {
  // Only support matrix n = 64 or 128
  assert(N == 64 || N == 128);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 750
  assert(false);
#else
  static constexpr uint32_t ZERO = 0x0;
  float C_warp[32];
  __shared__ half A_shared[16 * (32 + 8)];
  __shared__ half B_shared[32 * (N + 8)];

  int j_factors1 = ((OC + N - 1) / N);
  int blockIdx_y = blockIdx.x % ((M + 16 - 1) / 16 * j_factors1);
  int blockIdx_z = blockIdx.x / ((M + 16 - 1) / 16 * j_factors1);

  half A_shared_warp[8];
  half B_shared_warp[N / 4];
  for (int j_0_4_init = 0; j_0_4_init < N / 32; ++j_0_4_init) {
    for (int i = 0; i < 8; ++i) {
      C_warp[(j_0_4_init * 8) + i] = 0.0;
    }
  }

  static constexpr int row_stride_warp = 32 * 8 / 32;
  static constexpr int row_stride = 2 * 32 * 8 / N;
  // TODO: Haotian: blockIdx_y / j_factors1 in A loading to support bsz > 16
  bool ld_A_flag =
      (blockIdx_y / j_factors1 * 16 + threadIdx.y * row_stride_warp +
       threadIdx.x * 8 / 32) < M;  // threadIdx.y is warp_id
  // bool wb_C_flag = (threadIdx.x / 4) < M;

  half* A_ptr =
      A +
      (((int)blockIdx_y) / j_factors1 * 16 +
       (((int)threadIdx.y) * row_stride_warp) + ((int)threadIdx.x) / (32 / 8)) *
          IC +
      (((int)threadIdx.x) % (32 / 8)) * 8;

  int* B_ptr = B + ((int)threadIdx.y) * (OC / 8) * (256 / N) +
               (((int)threadIdx.x) / (N / 8)) * (OC / 8) +
               (((int)blockIdx_y) % j_factors1) * (N / 8) +
               (((int)threadIdx.x) % (N / 8)) * 1;
  // Why * 1 in the above line?

  half* A_shared_ptr = A_shared +
                       ((int)threadIdx.y) * row_stride_warp * (32 + 8) +
                       (((int)threadIdx.x) / (32 / 8)) * (32 + 8) +
                       (((int)threadIdx.x) % (32 / 8)) * 8;

  half* B_shared_ptr = B_shared +
                       ((int)threadIdx.y) * (row_stride / 2) * (N + 8) +
                       (((int)threadIdx.x) / (N / 8)) * (N + 8) +
                       (((int)threadIdx.x) % (N / 8)) * 8;

  int* zeros_ptr = zeros + (((int)blockIdx_y) % j_factors1) * (N / 8) +
                   ((int)threadIdx.x) % (N / 8);

  half* scaling_factors_ptr = scaling_factors +
                              (((int)blockIdx_y) % j_factors1) * N +
                              (((int)threadIdx.x) % (N / 8)) * 8;

  half* C_ptr =
      C +
      static_cast<long long>(blockIdx_z) * M * OC  // blockIdz.x -> split_k dim
      + (((int)blockIdx_y) % j_factors1) * N + ((int)threadIdx.y) * (N / 2) +
      (((int)threadIdx.x) % 4) * 2;

  // preload s.f. and zeros
  int k_bound = (IC / 32 + split_k_iters - 1) / split_k_iters;
  if ((k_bound - 1) * split_k_iters * 32 + blockIdx_z * 32 >= IC) k_bound -= 1;
  for (int _k_0_0 = 0; _k_0_0 < k_bound; ++_k_0_0) {
    int k_0_0 = _k_0_0 * split_k_iters + blockIdx_z;
    __syncthreads();
    // TODO: Haotian: blockIdx_y / j_factors1 in A loading to support bsz > 16
    if (ld_A_flag) {
      *(uint4*)(A_shared_ptr) = *(uint4*)(A_ptr + (k_0_0 * 32));
    } else {
      *(uint4*)(A_shared_ptr) = make_uint4(0, 0, 0, 0);
    }

    // for (int ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < 2; ++ax0_ax1_fused_0) {
    uint32_t zeros_loaded = *(uint32_t*)(zeros_ptr + k_0_0 * 32 / G * (OC / 8));
    uint4 B_loaded_zero = dequantize_s4_to_fp16x2(zeros_loaded);
    uint4 B_loaded_scale =
        *(uint4*)(scaling_factors_ptr + k_0_0 * 32 / G * (OC));
    /*
    if (blockIdx_z == 0 && blockIdx_y == 0 && k_0_0 == 0 && threadIdx.x == 0 &&
    threadIdx.y == 0){ printf("%x %x %x %x %x %x %x %x\n", B_loaded_scale.x,
    B_loaded_scale.y, B_loaded_scale.z, B_loaded_scale.w, B_loaded_zero.x,
    B_loaded_zero.y, B_loaded_zero.z, B_loaded_zero.w);
    }
    */
    // uint4 B_loaded_scale = make_uint4(0, 0, 0, 0);
    int* B_ptr_local = B_ptr + k_0_0 * 32 * (OC / 8);

    for (int ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < N / 16; ++ax0_ax1_fused_0) {
      // B: 32 x 136 (128+8) float16
      // each warp: 32 x 4
      // each thr: read 32 bit -> convert to 8xFP16 (a UINT4) -> scale and minus
      // zero -> WB UINT4
      // *(uint4*)(B_shared + ((((ax0_ax1_fused_0 * 544) + (((int)threadIdx.y) *
      // 272)) + ((((int)threadIdx.x) >> 4) * 136)) + ((((int)threadIdx.x) & 15)
      // * 8))) = *(uint4*)(B + ((((((k_0_0 * 163840) + (ax0_ax1_fused_0 *
      // 20480)) + (((int)threadIdx.y) * 10240)) + ((((int)threadIdx.x) >> 4) *
      // 5120)) + (((int)blockIdx_y) * 128)) + ((((int)threadIdx.x) & 15) *
      // 8))); row stride in shared memory: (NWARPS * 32 * 8 / cta_N)
      uint32_t B_loaded =
          *(uint32_t*)(B_ptr_local + ax0_ax1_fused_0 * row_stride * (OC / 8));
      uint4 B_loaded_fp16 = dequantize_s4_to_fp16x2(B_loaded);

      // - zero and * scale
      // TODO (Haotian): can save 4 assembly instructions if sormulate as deq =
      // q * scale - zero * scale.
#ifdef MX_MACA
      asm volatile("sub.f16x2 %0, %1, %2;\n"
                   : "=r"(B_loaded_fp16.x)
                   : "r"(B_loaded_fp16.x), "r"(B_loaded_zero.x));
      asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n"
                   : "=r"(B_loaded_fp16.x)
                   : "r"(B_loaded_fp16.x), "r"(B_loaded_scale.x), "r"(ZERO));
      asm volatile("sub.f16x2 %0, %1, %2;\n"
                   : "=r"(B_loaded_fp16.y)
                   : "r"(B_loaded_fp16.y), "r"(B_loaded_zero.y));
      asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n"
                   : "=r"(B_loaded_fp16.y)
                   : "r"(B_loaded_fp16.y), "r"(B_loaded_scale.y), "r"(ZERO));
      asm volatile("sub.f16x2 %0, %1, %2;\n"
                   : "=r"(B_loaded_fp16.z)
                   : "r"(B_loaded_fp16.z), "r"(B_loaded_zero.z));
      asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n"
                   : "=r"(B_loaded_fp16.z)
                   : "r"(B_loaded_fp16.z), "r"(B_loaded_scale.z), "r"(ZERO));
      asm volatile("sub.f16x2 %0, %1, %2;\n"
                   : "=r"(B_loaded_fp16.w)
                   : "r"(B_loaded_fp16.w), "r"(B_loaded_zero.w));
      asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n"
                   : "=r"(B_loaded_fp16.w)
                   : "r"(B_loaded_fp16.w), "r"(B_loaded_scale.w), "r"(ZERO));
#endif
      /*
      if (ax0_ax1_fused_0 == 0 && blockIdx_z == 0 && blockIdx_y == 0 && k_0_0 ==
      0 && threadIdx.x == 17 && threadIdx.y == 0){ printf("[x] %X %X %X %X\n",
      B_loaded_fp16.x, B_loaded_fp16.y, B_loaded_fp16.z, B_loaded_fp16.w);
      }
      */

      // write back
      *(uint4*)(B_shared_ptr + ax0_ax1_fused_0 * row_stride * (N + 8)) =
          B_loaded_fp16;
    }
    __syncthreads();

    for (int k_0_1 = 0; k_0_1 < 2; ++k_0_1) {
      {
        unsigned int addr;
#ifdef MX_MACA
        __asm__ __volatile__(
            "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, "
            "addr; }\n"
            : "=r"(addr)
            : "l"((void*)((&(A_shared[(k_0_1 * 16)])) +
                          (((((int)threadIdx.x) & 15) * 40) +
                           ((((int)threadIdx.x) >> 4) * 8)))));

        __asm__ __volatile__(
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
            "{%0, %1, %2, %3}, [%4];\n"
            : "=r"(((unsigned*)(A_shared_warp + 0))[0]),
              "=r"(((unsigned*)(A_shared_warp + 0))[1]),
              "=r"(((unsigned*)(A_shared_warp + 0))[2]),
              "=r"(((unsigned*)(A_shared_warp + 0))[3])
            : "r"(addr));
#endif
      }

      for (int ax1_0 = 0; ax1_0 < N / 32; ++ax1_0) {
        {
          unsigned int addr;
#ifdef MX_MACA
          __asm__ __volatile__(
              "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, "
              "addr; }\n"
              : "=r"(addr)
              : "l"((void*)((&(B_shared[(((k_0_1 * (N * 16 + 128)) +
                                          (((int)threadIdx.y) * (N / 2))) +
                                         (ax1_0 * 16))])) +
                            (((((int)threadIdx.x) & 15) * (N + 8)) +
                             ((((int)threadIdx.x) >> 4) * 8)))));
          __asm__ __volatile__(
              "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16"
              "{%0, %1, %2, %3}, [%4];\n"
              : "=r"(((unsigned*)(B_shared_warp + (ax1_0 * 8)))[0]),
                "=r"(((unsigned*)(B_shared_warp + (ax1_0 * 8)))[1]),
                "=r"(((unsigned*)(B_shared_warp + (ax1_0 * 8)))[2]),
                "=r"(((unsigned*)(B_shared_warp + (ax1_0 * 8)))[3])
              : "r"(addr));
#endif
        }
      }
      for (int j_0_4 = 0; j_0_4 < N / 32; ++j_0_4) {
#ifdef MX_MACA
  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 750
        {
          __asm__ __volatile__(
              "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
              "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
              : "=f"(((float*)(C_warp + (j_0_4 * 8)))[0]),
                "=f"(((float*)(C_warp + (j_0_4 * 8)))[1]),
                "=f"(((float*)(C_warp + (j_0_4 * 8)))[2]),
                "=f"(((float*)(C_warp + (j_0_4 * 8)))[3])
              : "r"(((unsigned*)(A_shared_warp + 0))[0]),
                "r"(((unsigned*)(A_shared_warp + 0))[1]),
                "r"(((unsigned*)(B_shared_warp + (j_0_4 * 8)))[0]),
                "f"(((float*)(C_warp + (j_0_4 * 8)))[0]),
                "f"(((float*)(C_warp + (j_0_4 * 8)))[1]),
                "f"(((float*)(C_warp + (j_0_4 * 8)))[2]),
                "f"(((float*)(C_warp + (j_0_4 * 8)))[3]));
        }

        {
          __asm__ __volatile__(
              "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
              "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
              : "=f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[0]),
                "=f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[1]),
                "=f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[2]),
                "=f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[3])
              : "r"(((unsigned*)(A_shared_warp + 0))[0]),
                "r"(((unsigned*)(A_shared_warp + 0))[1]),
                "r"(((unsigned*)(B_shared_warp + ((j_0_4 * 8) + 4)))[0]),
                "f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[0]),
                "f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[1]),
                "f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[2]),
                "f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[3]));
        }

        {
          __asm__ __volatile__(
              "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
              "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
              : "=f"(((float*)(C_warp + (j_0_4 * 8)))[0]),
                "=f"(((float*)(C_warp + (j_0_4 * 8)))[1]),
                "=f"(((float*)(C_warp + (j_0_4 * 8)))[2]),
                "=f"(((float*)(C_warp + (j_0_4 * 8)))[3])
              : "r"(((unsigned*)(A_shared_warp + 0))[2]),
                "r"(((unsigned*)(A_shared_warp + 0))[3]),
                "r"(((unsigned*)(B_shared_warp + (j_0_4 * 8)))[1]),
                "f"(((float*)(C_warp + (j_0_4 * 8)))[0]),
                "f"(((float*)(C_warp + (j_0_4 * 8)))[1]),
                "f"(((float*)(C_warp + (j_0_4 * 8)))[2]),
                "f"(((float*)(C_warp + (j_0_4 * 8)))[3]));
        }

        {
          __asm__ __volatile__(
              "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
              "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
              : "=f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[0]),
                "=f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[1]),
                "=f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[2]),
                "=f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[3])
              : "r"(((unsigned*)(A_shared_warp + 0))[2]),
                "r"(((unsigned*)(A_shared_warp + 0))[3]),
                "r"(((unsigned*)(B_shared_warp + ((j_0_4 * 8) + 4)))[1]),
                "f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[0]),
                "f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[1]),
                "f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[2]),
                "f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[3]));
        }
  #else
        {
          __asm__ __volatile__(
              "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
              "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, "
              "%13};\n"
              : "=f"(((float*)(C_warp + (j_0_4 * 8)))[0]),
                "=f"(((float*)(C_warp + (j_0_4 * 8)))[1]),
                "=f"(((float*)(C_warp + (j_0_4 * 8)))[2]),
                "=f"(((float*)(C_warp + (j_0_4 * 8)))[3])
              : "r"(((unsigned*)(A_shared_warp + 0))[0]),
                "r"(((unsigned*)(A_shared_warp + 0))[1]),
                "r"(((unsigned*)(A_shared_warp + 0))[2]),
                "r"(((unsigned*)(A_shared_warp + 0))[3]),
                "r"(((unsigned*)(B_shared_warp + (j_0_4 * 8)))[0]),
                "r"(((unsigned*)(B_shared_warp + (j_0_4 * 8)))[1]),
                "f"(((float*)(C_warp + (j_0_4 * 8)))[0]),
                "f"(((float*)(C_warp + (j_0_4 * 8)))[1]),
                "f"(((float*)(C_warp + (j_0_4 * 8)))[2]),
                "f"(((float*)(C_warp + (j_0_4 * 8)))[3]));
        }

        {
          __asm__ __volatile__(
              "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
              "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, "
              "%13};\n"
              : "=f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[0]),
                "=f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[1]),
                "=f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[2]),
                "=f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[3])
              : "r"(((unsigned*)(A_shared_warp + 0))[0]),
                "r"(((unsigned*)(A_shared_warp + 0))[1]),
                "r"(((unsigned*)(A_shared_warp + 0))[2]),
                "r"(((unsigned*)(A_shared_warp + 0))[3]),
                "r"(((unsigned*)(B_shared_warp + ((j_0_4 * 8) + 4)))[0]),
                "r"(((unsigned*)(B_shared_warp + ((j_0_4 * 8) + 4)))[1]),
                "f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[0]),
                "f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[1]),
                "f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[2]),
                "f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[3]));
        }

  #endif
#endif
      }
    }
  }

  // TODO: Shang: Hoist loop invariance.
  for (int ax1_0_1 = 0; ax1_0_1 < 4; ++ax1_0_1) {
    for (int local_id = 0; local_id < 8; ++local_id) {
      int row_offset = (((int)blockIdx_y) / j_factors1) * 16 +
                       ((int)threadIdx.x) / 4 + (local_id % 4) / 2 * 8;
      if (row_offset < M) {
        *(C_ptr + ax1_0_1 * 16 + row_offset * OC + (local_id / 4) * 8 +
          local_id % 2) = __float2half(C_warp[(ax1_0_1 * 8) + local_id]);
      }
    }
  }
#endif
}
#endif

template<typename T, typename VT>
__global__ void __launch_bounds__(64)
    dequantize_weights(int* __restrict__ B, T* __restrict__ scaling_factors,
                       int* __restrict__ zeros, T* __restrict__ C, int G) {
  static constexpr uint32_t ZERO = 0x0;
  T B_shared[8];
  T B_loaded_scale[8];
  T* B_shared_ptr2 = B_shared;

  int N = blockDim.x * gridDim.x;  // 2
  int col = (blockIdx.x * blockDim.x + threadIdx.x);
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int index1 = 8 * col + 8 * row * N;
  T* C_ptr2 = C + index1;

  int index2 = col + row * N;
  int* B_ptr2 = B + index2;

  int index3 = col + (int)(row / G) * N;
  int* zeros_ptr2 = zeros + index3;
  int index4 = 8 * col + (int)(row / G) * N * 8;
  T* scaling_factors_ptr2 = scaling_factors + index4;

  uint32_t zeros_loaded = *(uint32_t*)(zeros_ptr2);
  *(uint4*)B_loaded_scale = *(uint4*)(scaling_factors_ptr2);
  uint32_t B_loaded = *(uint32_t*)B_ptr2;
#if 1
  //zero //4bit scale 8bit b 4bit
  hgemm_marlin_gptq::awq_dequant_4bits<T>(B_loaded,B_shared, B_loaded_scale, zeros_loaded);
#else
  uint4 B_loaded_zero = dequantize_s4_to_fp16x2<VT>(zeros_loaded);
  uint4 B_loaded_fp16 = dequantize_s4_to_fp16x2<VT>(B_loaded);
#ifdef MX_MACA
  asm volatile("sub.f16x2 %0, %1, %2;\n"
               : "=r"(B_loaded_fp16.x)
               : "r"(B_loaded_fp16.x), "r"(B_loaded_zero.x));
  asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n"
               : "=r"(B_loaded_fp16.x)
               : "r"(B_loaded_fp16.x), "r"(B_loaded_scale.x), "r"(ZERO));
  asm volatile("sub.f16x2 %0, %1, %2;\n"
               : "=r"(B_loaded_fp16.y)
               : "r"(B_loaded_fp16.y), "r"(B_loaded_zero.y));
  asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n"
               : "=r"(B_loaded_fp16.y)
               : "r"(B_loaded_fp16.y), "r"(B_loaded_scale.y), "r"(ZERO));
  asm volatile("sub.f16x2 %0, %1, %2;\n"
               : "=r"(B_loaded_fp16.z)
               : "r"(B_loaded_fp16.z), "r"(B_loaded_zero.z));
  asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n"
               : "=r"(B_loaded_fp16.z)
               : "r"(B_loaded_fp16.z), "r"(B_loaded_scale.z), "r"(ZERO));
  asm volatile("sub.f16x2 %0, %1, %2;\n"
               : "=r"(B_loaded_fp16.w)
               : "r"(B_loaded_fp16.w), "r"(B_loaded_zero.w));
  asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n"
               : "=r"(B_loaded_fp16.w)
               : "r"(B_loaded_fp16.w), "r"(B_loaded_scale.w), "r"(ZERO));
#else
     // >>>> PTX2CPP Success <<<<
{
{
unsigned int __a=(B_loaded_fp16.x);
unsigned int __b=(B_loaded_zero.x);
VT __d=__hsub2(*(VT*)&__a,*(VT*)&__b);
(B_loaded_fp16.x)=*(unsigned int*)&__d;
}
}


// >>>> PTX2CPP Success <<<<
{
{
unsigned int __a=(B_loaded_fp16.x);
unsigned int __b=(B_loaded_scale.x);
unsigned int __c=(ZERO);
VT __d=__hfma2(*(VT*)&__a,*(VT*)&__b,*(VT*)&__c);
(B_loaded_fp16.x)=*(unsigned int*)&__d;
}
}

// >>>> PTX2CPP Success <<<<
{
{
unsigned int __a=(B_loaded_fp16.y);
unsigned int __b=(B_loaded_zero.y);
VT __d=__hsub2(*(VT*)&__a,*(VT*)&__b);
(B_loaded_fp16.y)=*(unsigned int*)&__d;
}
}
// >>>> PTX2CPP Success <<<<
{
{
unsigned int __a=(B_loaded_fp16.y);
unsigned int __b=(B_loaded_scale.y);
unsigned int __c=(ZERO);
VT __d=__hfma2(*(VT*)&__a,*(VT*)&__b,*(VT*)&__c);
(B_loaded_fp16.y)=*(unsigned int*)&__d;
}
}
// >>>> PTX2CPP Success <<<<
{
{
unsigned int __a=(B_loaded_fp16.z);
unsigned int __b=(B_loaded_zero.z);
VT __d=__hsub2(*(VT*)&__a,*(VT*)&__b);
(B_loaded_fp16.z)=*(unsigned int*)&__d;
}
}

// >>>> PTX2CPP Success <<<<
{
{
unsigned int __a=(B_loaded_fp16.z);
unsigned int __b=(B_loaded_scale.z);
unsigned int __c=(ZERO);
VT __d=__hfma2(*(VT*)&__a,*(VT*)&__b,*(VT*)&__c);
(B_loaded_fp16.z)=*(unsigned int*)&__d;
}
}
// >>>> PTX2CPP Success <<<<
{
{
unsigned int __a=(B_loaded_fp16.w);
unsigned int __b=(B_loaded_zero.w);
VT __d=__hsub2(*(VT*)&__a,*(VT*)&__b);
(B_loaded_fp16.w)=*(unsigned int*)&__d;
}
}


// >>>> PTX2CPP Success <<<<
{
{
unsigned int __a=(B_loaded_fp16.w);
unsigned int __b=(B_loaded_scale.w);
unsigned int __c=(ZERO);
VT __d=__hfma2(*(VT*)&__a,*(VT*)&__b,*(VT*)&__c);
(B_loaded_fp16.w)=*(unsigned int*)&__d;
}
}
*(uint4*)B_shared_ptr2 = B_loaded_fp16;
#endif
#endif
  
  for (int i = 0; i < 8; ++i) {
    *(C_ptr2 + i) = B_shared[i];
  }
}

template<typename T>
__global__ void dequantize_weights_opt(int* __restrict__ B, T* __restrict__ scaling_factors,
                       int* __restrict__ zeros, T* __restrict__ C, int G, int length, int blocksize, int num_elems, DivModFast length_fast) {
    constexpr int N = 8;
    T B_loaded_scale[8];
    T B_shared[8];
    int tid = blockIdx.x * blocksize + threadIdx.x;
    if(tid >= num_elems) return;
    // int row = tid / length;
    // int col = tid % length;
    int row, col;
    length_fast.divmod(tid, row, col);
    int group_row = row / G;
    int group_offset = group_row * length + col;
    int offset = row * length + col;
    uint32_t* ptr_zeros = (uint32_t*)(zeros + group_offset);
    uint32_t* ptr_B = (uint32_t*)(B + offset);
    T* ptr_scale = scaling_factors + group_offset * N;
    T* ptr_C = C + offset * N;
    uint32_t zeros_loaded = *(uint32_t*)ptr_zeros;
    uint32_t B_loaded = *(uint32_t*)ptr_B;
    *(uint4*)(B_loaded_scale) = *(uint4*)(ptr_scale);
    hgemm_marlin_gptq::awq_dequant_4bits<T>(B_loaded,B_shared, B_loaded_scale, zeros_loaded);
    *(float4*)(ptr_C) = *(float4*)(B_shared);
}


#if 0
template <typename dstT, typename srcT, typename scalarT>
__global__ void blasMemcpy(dstT *dst, const srcT *src, size_t cnt, scalarT beta) {
    size_t threads = gridDim.x * blockDim.x;
    size_t itemsPerThread = (cnt + threads - 1) / threads;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t loop = 0; loop < itemsPerThread && loop * threads + tid < cnt; ++loop) {
        dst[loop * threads + tid] =
            static_cast<double>(beta) * static_cast<double>(src[loop * threads + tid]);
    }
}

#define SWITCH_CASE_BATCH(BlockDimX, SplitK, BATCH) \
    case BATCH: {                                   \
        CALL_GEMM(BlockDimX, SplitK, BATCH)         \
        break;                                      \
    }

#define APPLY_HGEMM_BATCH(BlockDimX, SplitK, BATCH) \
    switch(BATCH) {                                 \
        SWITCH_CASE_BATCH(BlockDimX, SplitK, 1)           \
        SWITCH_CASE_BATCH(BlockDimX, SplitK, 2)           \
        SWITCH_CASE_BATCH(BlockDimX, SplitK, 3)           \
        SWITCH_CASE_BATCH(BlockDimX, SplitK, 4)           \
        default: {                                          \
            launched = false;                               \
            printf("ERROR: Unsupported BATCH %d\n", BATCH); \
            break;                                          \
        }                                                   \
    }

#define SWITCH_CASE_BlockDimX(BlockDimX, SplitK, BATCH) \
    case BlockDimX: {                                   \
        APPLY_HGEMM_BATCH(BlockDimX, SplitK, BATCH)    \
        break;                                          \
    }

#define APPLY_HGEMM(BlockDimX, SplitK, BATCH)           \
    switch (BlockDimX) {                                \
        SWITCH_CASE_BlockDimX(16, SplitK, BATCH)        \
        SWITCH_CASE_BlockDimX(32, SplitK, BATCH)        \
        SWITCH_CASE_BlockDimX(64, SplitK, BATCH)        \
        SWITCH_CASE_BlockDimX(128, SplitK, BATCH)       \
        SWITCH_CASE_BlockDimX(256, SplitK, BATCH)       \
        default: {                                                  \
            launched = false;                                       \
            printf("ERROR: Unsupported BlockDimX %d\n", BlockDimX); \
            break;                                                  \
        }                                                           \
    }

bool call_kernel(const half *srcB,
    const quant_packed_type *srcA,
    quant_packed_type *zeros, half *scales,
    half* dst_D,
    int m, int n, int k, int srcStride, int dstStride,
    int block_x, int split_k,
    const int* b_perm_D = nullptr) {
    //constexpr int PACK_RATIO = 8;
    constexpr int ThreadBlock = 256;
    const dim3 threadBlock = {static_cast<unsigned int>(ThreadBlock)};
    const dim3 gridBlock = {static_cast<unsigned int>((m + 8*block_x-1) / 8 / block_x), static_cast<unsigned int>(split_k)};
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (split_k * QUANT_GROUP > k || k % QUANT_GROUP != 0) return false;
    if (block_x < 16 || n > 4) return false;
    bool launched = true;
    #define CALL_GEMM(BX, SK, N) \
        if (SK * 128 == k) {   \
            hgemv_nn_splitk_awq_kb128<BX, N><<<gridBlock, threadBlock, 0, stream>>>( \
                srcB, srcA, zeros, scales, dst_D, m, n, k, srcStride, dstStride, k/SK, b_perm_D); \
        } else { \
            hgemv_nn_splitk_awq<256, BX, N><<<gridBlock, threadBlock, 0, stream>>>( \
                srcB, srcA, zeros, scales, dst_D, m, n, k, srcStride, dstStride, k/SK, b_perm_D); \
        }
    APPLY_HGEMM(block_x, split_k, n);
    return launched;
}

void launch_gemm_awq(Operation_t trans_a,
                      Operation_t trans_b,
                      int m,
                      int n,
                      int k,
                      const float alpha,
                      const float beta,
                      const uint32_t* dA,
                      int lda,
                      const half* dB,
                      int ldb,
                      half* dC,
                      int ldc,
                      uint32_t* d_zeros,
                      half* d_scales,
                      float* space_mid,
                      cudaStream_t stream,
                      int splitk_iters = 1) {
    if (n <= 4) {
            constexpr int thread_block = 256;
            constexpr int m_per_thread = 8;
            auto kernel_testing = [&](int bx, int sk) -> bool {
                return call_kernel(dB, dA, d_zeros, d_scales, dC, m, n, k, m, m, bx, sk);
            };
            //Select parameters when warmup
            auto& sl_warmup = hgemv_selector::GemvSelectorHolder<QUANT_GROUP,8,m_per_thread>::selector(m, n, k, true);
            if (sl_warmup.valid()) {
                sl_warmup.run(kernel_testing);
            }
    }
    else {
            const int threads_n = 256;
            const int tileM = 128;
            const int tileN = 32;
            const int tileK = 128;

            bool isSplitk = splitk_iters > 1;

            uint32_t gridx = (m - 1) / tileM + 1;
            uint32_t gridy = (n - 1) / tileN + 1;
            uint32_t gridz = splitk_iters;

            dim3 dimBlock(threads_n, 1, 1);
            dim3 dimGrid(gridx, gridy, gridz);
            bool isBetaZero = (beta == 0.0);

            if (trans_a == OP_N && trans_b == OP_N && m % 8 == 0 && k % 8 == 0) {
                if (!isSplitk) {
                    if (isBetaZero)
                        Hgemm_nn_128x32x128_8m1n8k_awq_4bit<OP_N, OP_N, true, tileM, tileN, tileK>
                            <<<dimGrid, dimBlock, 0, stream>>>(m, n, k, alpha, beta, dA, lda, dB, ldb, dC,
                                                               dC, ldc, d_zeros, d_scales);
                    else
                        Hgemm_nn_128x32x128_8m1n8k_awq_4bit<OP_N, OP_N, false, tileM, tileN, tileK>
                            <<<dimGrid, dimBlock, 0, stream>>>(m, n, k, alpha, beta, dA, lda, dB, ldb, dC,
                                                               dC, ldc, d_zeros, d_scales);
                } else {
                    if (!isBetaZero)
                        blasMemcpy<<<104, 512, 0, stream>>>(space_mid, dC, m * n, beta);
                    Hgemm_nn_128x32x128_8m1n8k_awq_4bit<OP_N, OP_N, true, tileM, tileN, tileK, true>
                        <<<dimGrid, dimBlock, 0, stream>>>(m, n, k, alpha, beta, dA, lda, dB, ldb, dC, dC,
                                                           ldc, d_zeros, d_scales, splitk_iters, space_mid);
                    blasMemcpy<<<104, 512, 0, stream>>>(dC, space_mid, m * n, 1);
                }
            } else {
                printf("Parameters not supported!\n");
                return;
            }
    }
}
#endif

template <int BLOCK_SIZE>
__global__ void awq_to_gptq_4bit(uint32_t *output, const uint32_t *input, int k, int n) {
    constexpr int COMPACT_FACTOR = 8;
    constexpr int QBIT = 4;
    int tid = threadIdx.x;
    int tile_idx = blockIdx.x * BLOCK_SIZE + tid;
    int N_COMPACT = (n + COMPACT_FACTOR - 1) / COMPACT_FACTOR;
    int K_COMPACT = (k + COMPACT_FACTOR - 1) / COMPACT_FACTOR;
    int tile_n_idx = tile_idx / K_COMPACT;
    int tile_k_idx = tile_idx % K_COMPACT;

    uint32_t awq_data[COMPACT_FACTOR];
    uint32_t temp_data[COMPACT_FACTOR];
    uint32_t gptq_data[COMPACT_FACTOR];

    int gptq_shift[COMPACT_FACTOR] = {0, 4, 1, 5, 2, 6, 3, 7};
    int awq_shift[COMPACT_FACTOR] = {0, 4, 1, 5, 2, 6, 3, 7};

    // load k8xn8
    #pragma unroll
    for (int i = 0; i < COMPACT_FACTOR; i++) {
        int gvm_addr_offset = (tile_k_idx * COMPACT_FACTOR + i) * N_COMPACT + tile_n_idx;
        int pred_k = tile_k_idx * COMPACT_FACTOR + i < k;
        int pred_n = tile_n_idx * COMPACT_FACTOR < n;
        if (pred_k && pred_n) {
            awq_data[i] = *(input + gvm_addr_offset);
        }
    }

    // decompress awq_data and recompress to gptq_data
    #pragma unroll
    for (int i = 0; i < COMPACT_FACTOR; i++) {
        #pragma unroll
        for(int j = 0; j < COMPACT_FACTOR; j++) {
            temp_data[j] = ((awq_data[j] >> (awq_shift[i] * QBIT)) & 0xf);
        }
        #pragma unroll
        for(int j = 0; j < COMPACT_FACTOR; j++) {
            gptq_data[i] &= (~(0xf << (gptq_shift[j] * QBIT)));
            gptq_data[i] |= temp_data[j] << (gptq_shift[j] * QBIT);

        }
    }

    // store k8xn8
    #pragma unroll
    for (int i = 0; i < COMPACT_FACTOR; i++) {
        int gvm_addr_offset = tile_k_idx * n + tile_n_idx * COMPACT_FACTOR + i;
        int pred_k = tile_k_idx * COMPACT_FACTOR < k;
        int pred_n = tile_n_idx * COMPACT_FACTOR + i < n;
        if (pred_k && pred_n) {
            *(output + gvm_addr_offset) = gptq_data[i];
        } else {
            *(output + gvm_addr_offset) = 0x00000000;
        }
    }
}


template <typename input_tp, const vllm::ScalarTypeId w_type_id, typename output_tp, typename quant_packed_tp>
bool launch_gemm_gptq(int m,
                      int n,
                      int k,
                      int quant_group,
                      const input_tp *dA,
                      int lda,
                      const quant_packed_tp *dB,
                      int ldb,
                      output_tp *dC,
		      float *dC_temp,
                      int ldc,
                      quant_packed_tp *d_zeros,
                      input_tp *d_scales,
                      const cudaStream_t stream,
                      int chunks = 1) {
    using namespace hgemm_marlin_gptq;
    if(n % 16 != 0) {
        printf("n %% 16 != 0, n = %d\n", n);
        return false;
    }
    if(k % 32 != 0) {
        printf("k %% 32 != 0, k = %d\n", k);
        return false;
    }
    //const vllm::ScalarTypeId w_type_id = vllm::kU4B8.id();
    const int THREADS = 256;
    int BLOCKS_M = div_ceil(m, SLICE_M);
    if(BLOCKS_M >= MAX_BLOCKS_M && BLOCKS_M % MAX_BLOCKS_M != 0) {
        printf("Error: input m is error, m = %d, blocks_m = %d\n", m, BLOCKS_M);
        return false;
    }
    if (BLOCKS_M > MAX_BLOCKS_M) BLOCKS_M = MAX_BLOCKS_M;
    int BLOCKS_N = 8;
    //It is better let TILE_K = quant_group
    //But if quant_group is too large, a quant_group can be divided into two parts
    int BLOCKS_K = quant_group / SLICE_K;
    if (quant_group > 128) BLOCKS_K = 128 / SLICE_K;
    //if (BLOCKS_M == 1 || BLOCKS_M == 2) {
    //    BLOCKS_N = 16;
    //}
    const bool HAS_ACT_ORDER = false;
    const bool HAS_ZP = (w_type_id == vllm::kU4.id()) || (w_type_id == vllm::kU8.id());
    int *g_idx = nullptr;
    bool HAS_NK_PRED = true;
    bool HAS_M_PRED = true;
    if (n % TILE_N == 0 && k % TILE_K == 0) {
        HAS_NK_PRED = false;
    }
    if (m % TILE_M == 0) {
        HAS_M_PRED = false;
    }

#define LAUNCH_AWQ(threads, bm, bn, bk, has_act_order, has_zp, has_nk_pred, has_m_pred) \
    else if (THREADS == threads && BLOCKS_M == bm && BLOCKS_N == bn \
        && BLOCKS_K == bk  && HAS_ACT_ORDER == has_act_order \
        && HAS_ZP == has_zp \
        && HAS_M_PRED == has_m_pred && HAS_NK_PRED == has_nk_pred) { \
            launch_gemm_gptq_kernel<input_tp, w_type_id, \
                    threads, bm, bn, bk, has_act_order, has_zp, has_m_pred, has_nk_pred>( \
                    (const PackTypeInt4*)dA, \
                    (const PackTypeInt4*)dB, \
                    (PackTypeInt4*)dC, \
                    (PackTypeInt4*)dC_temp, \
                    (const PackTypeInt4*)d_scales, \
                    (const PackTypeInt4*)d_zeros, \
                    nullptr, m, n, k, quant_group, chunks,\
                    stream); \
    }

#define LAUNCH_AWQ_K(bk, has_act_order, has_zp, has_nk_pred, has_m_pred) \
    LAUNCH_AWQ(256, 1, 8, bk, has_act_order, has_zp, has_nk_pred, has_m_pred) \
    LAUNCH_AWQ(256, 2, 8, bk, has_act_order, has_zp, has_nk_pred, has_m_pred) \
    LAUNCH_AWQ(256, 3, 8, bk, has_act_order, has_zp, has_nk_pred, has_m_pred) \
    LAUNCH_AWQ(256, 4, 8, bk, has_act_order, has_zp, has_nk_pred, has_m_pred) \
    LAUNCH_AWQ(256, 1, 16, bk, has_act_order, has_zp, has_nk_pred, has_m_pred) \
    LAUNCH_AWQ(256, 2, 16, bk, has_act_order, has_zp, has_nk_pred, has_m_pred)


#define LAUNCH_AWQ_ZP(has_zp, has_nk_pred, has_m_pred) \
    LAUNCH_AWQ_K(1, false, has_zp, has_nk_pred, has_m_pred) \
    LAUNCH_AWQ_K(2, false, has_zp, has_nk_pred, has_m_pred) \
    LAUNCH_AWQ_K(4, false, has_zp, has_nk_pred, has_m_pred) \
    LAUNCH_AWQ_K(8, false, has_zp, has_nk_pred, has_m_pred)

#define LAUNCH_AWQ_PRED(has_nk_pred, has_m_pred) \
    LAUNCH_AWQ_ZP(false, has_nk_pred, has_m_pred) \
    LAUNCH_AWQ_ZP(true, has_nk_pred, has_m_pred)

    if (false) {

    }
    LAUNCH_AWQ_PRED(true, true)
    LAUNCH_AWQ_PRED(true, false)
    LAUNCH_AWQ_PRED(false, true)
    LAUNCH_AWQ_PRED(false, false)
    else {
        printf("BLOCKS_M=%d, BLOCKS_N=%d, BLOCKS_k=%d, THREADS=%d, HAS_ACT_ORDER=%d, HAS_ZP=%d, quant_group=%d, HAS_M_PRED=%d, HAS_NK_PRED=%d is not supported\n",
        BLOCKS_M, BLOCKS_N, BLOCKS_K, THREADS, HAS_ACT_ORDER, HAS_ZP, quant_group, HAS_M_PRED, HAS_NK_PRED);
        return false;
    }

    return true;
}


template <typename input_tp, const vllm::ScalarTypeId w_type_id, typename output_tp, typename quant_packed_tp>
bool launch_gemm(int quant_group,
                int m,
                int n,
                int k,
                const input_tp *dA,
                int lda,
                const quant_packed_tp *dB,
                int ldb,
                output_tp *dC,
		            float* dC_temp,
                int ldc,
                quant_packed_tp *d_zeros,
                input_tp *d_scales,
		            const cudaStream_t stream) {
    using namespace hgemm_marlin_gptq;
    //constexpr int max_blocks_m = 4;
    int total_m_blocks = div_ceil(m, SLICE_M);
    int chunks = total_m_blocks / MAX_BLOCKS_M;
    int rest_blocks_m = total_m_blocks % MAX_BLOCKS_M;
    // printf("m=%d,n=%d,k=%d,lda=%d,ldb=%d,ldc=%d,total_m_blocks=%d,chunks=%d,rest_blocks_m=%d\n",
    //     m, n, k, lda, ldb, ldc, total_m_blocks, chunks, rest_blocks_m
    // );
    //const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    // const int quant_group = 128;
    bool ret = true;
    if (chunks > 0) {
        int real_m = m > chunks * MAX_BLOCKS_M * SLICE_M ? chunks * MAX_BLOCKS_M * SLICE_M : m;
        ret = launch_gemm_gptq<input_tp, w_type_id, output_tp, quant_packed_tp>(real_m, n, k, quant_group, dA, lda, dB, ldb, dC, dC_temp, ldc, d_zeros, d_scales, stream, chunks);
    }
    if (rest_blocks_m > 0) {
        int m_offset = chunks * MAX_BLOCKS_M * SLICE_M;
        ret = ret && launch_gemm_gptq<input_tp, w_type_id, output_tp, quant_packed_tp>(m - m_offset, n, k, quant_group, dA + lda * m_offset, lda, dB, ldb, dC + ldc * m_offset, dC_temp + ldc * m_offset, ldc, d_zeros, d_scales, stream, 1);
    }

    return ret;
}



}  // namespace awq
}  // namespace vllm

torch::Tensor awq_to_gptq_4bit(torch::Tensor qweight) {

  const at::cuda::OptionalCUDAGuard device_guard(device_of(qweight));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const uint32_t* qweight_ptr = reinterpret_cast<const uint32_t*>(qweight.data_ptr<int>());

  int num_in_channels = qweight.size(0);
  int num_out_channels = qweight.size(1) * 8;

  int compact_n = (num_out_channels + hgemm_marlin_gptq::PACK_RATIO_4BITS - 1) / hgemm_marlin_gptq::PACK_RATIO_4BITS;
  int compact_output_k = (num_in_channels + hgemm_marlin_gptq::PACK_RATIO_4BITS - 1) / hgemm_marlin_gptq::PACK_RATIO_4BITS;;

  int block_size = 256;
  int tile_all_num = compact_n * compact_output_k;
  int grid_size = (tile_all_num + 255) / 256;

  auto options = torch::TensorOptions()
                     .dtype(qweight.dtype())
                     .device(qweight.device());

  torch::Tensor out = torch::zeros({num_out_channels,  compact_output_k}, options);
  uint32_t* out_ptr = reinterpret_cast<uint32_t*>(out.data_ptr<int>());

  vllm::awq::awq_to_gptq_4bit<256><<<grid_size, block_size, 0, stream>>>((uint32_t*)out_ptr, (const uint32_t*)qweight_ptr, num_in_channels, num_out_channels);

  return out;
}

torch::Tensor awq_dequantize(torch::Tensor _kernel,
                             torch::Tensor _scaling_factors,
                             torch::Tensor _zeros, int64_t split_k_iters,
                             int64_t thx, int64_t thy) {
  int in_c = _kernel.size(0);
  int qout_c = _kernel.size(1);
  int out_c = qout_c * 8;
  int G = in_c / _scaling_factors.size(0);
#if 0
  int x_thread = thx;
  int y_thread = thy;

  int x_blocks = 1;
  int y_blocks = 1;
  if (thx == 0) {
    x_thread = qout_c;
  }
  if (thy == 0) {
    y_thread = in_c;
  }
  if (thx == 0 && thy == 0) {
    x_thread = 8;
    y_thread = 8;
    x_blocks = (int)(qout_c / 8);
    y_blocks = (int)(in_c / 8);
  }
#endif
  const at::cuda::OptionalCUDAGuard device_guard(device_of(_scaling_factors));

  auto options = torch::TensorOptions()
                     .dtype(_scaling_factors.dtype())
                     .device(_scaling_factors.device());
  at::Tensor _de_kernel = torch::empty({in_c, out_c}, options);
  
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto kernel = reinterpret_cast<int*>(_kernel.data_ptr<int>());
  auto zeros = reinterpret_cast<int*>(_zeros.data_ptr<int>());
#if 1
  int blocksize = 512;
  int num_elems = in_c * qout_c;
  int gridsize = (num_elems + blocksize - 1) / blocksize;
  if(_scaling_factors.dtype() == at::ScalarType::Half) {
    auto de_kernel = reinterpret_cast<half*>(_de_kernel.data_ptr<at::Half>());
    auto scaling_factors =
        reinterpret_cast<half*>(_scaling_factors.data_ptr<at::Half>());
    vllm::awq::dequantize_weights_opt<__half><<<gridsize, blocksize, 0 , stream>>>(kernel, scaling_factors, zeros, de_kernel, G, qout_c, blocksize, num_elems, vllm::awq::DivModFast(qout_c));
  } else if(_scaling_factors.dtype() == at::ScalarType::BFloat16) {
    auto de_kernel = reinterpret_cast<maca_bfloat16*>(_de_kernel.data_ptr<at::BFloat16>());
    auto scaling_factors =
        reinterpret_cast<maca_bfloat16*>(_scaling_factors.data_ptr<at::BFloat16>());
    vllm::awq::dequantize_weights_opt<maca_bfloat16><<<gridsize, blocksize, 0, stream>>>(kernel, scaling_factors, zeros, de_kernel, G, qout_c, blocksize, num_elems, vllm::awq::DivModFast(qout_c));
  } else {
    printf("not support this type\n");
    assert(0);
  }
#else
  dim3 num_blocks(x_blocks, y_blocks);
  dim3 threads_per_block(x_thread, y_thread);

  if(_scaling_factors.dtype() == at::ScalarType::Half) {
    auto de_kernel = reinterpret_cast<half*>(_de_kernel.data_ptr<at::Half>());
    auto scaling_factors =
        reinterpret_cast<half*>(_scaling_factors.data_ptr<at::Half>());
    vllm::awq::dequantize_weights<__half,__half2><<<num_blocks, threads_per_block, 0, stream>>>(
        kernel, scaling_factors, zeros, de_kernel, G);
  } else if(_scaling_factors.dtype() == at::ScalarType::BFloat16) {
    auto de_kernel = reinterpret_cast<maca_bfloat16*>(_de_kernel.data_ptr<at::BFloat16>());
    auto scaling_factors =
        reinterpret_cast<maca_bfloat16*>(_scaling_factors.data_ptr<at::BFloat16>());
    vllm::awq::dequantize_weights<maca_bfloat16,maca_bfloat162><<<num_blocks, threads_per_block, 0, stream>>>(
        kernel, scaling_factors, zeros, de_kernel, G);
  } else {
    printf("not support this type\n");
    assert(0);
  }
#endif
  return _de_kernel;
}

// in_feats: M, IC [float16]
// kernel: IC, OC // 8 [int32] -> cast to IC, OC [uint4b]
// scaling_factors: IC // G, OC [float16]
// zeros: IC // G, OC // 8 [int32] -> cast to IC // G, OC [uint4b]
// assume that batch_size < 16 for now

torch::Tensor awq_gemm(torch::Tensor _in_feats, torch::Tensor _kernel,
                       torch::Tensor _scaling_factors, torch::Tensor _zeros,
                       int64_t split_k_iters,
                       torch::Tensor _temp_space,
                       bool dtype_bf16) {
  int num_in_feats = _in_feats.size(0);
  int num_in_channels = _in_feats.size(1);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(_in_feats));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  auto options = torch::TensorOptions()
                     .dtype(_in_feats.dtype())
                     .device(_in_feats.device());
                     
  // int num_out_channels = _kernel.size(1) * 8;
  int num_out_channels = _kernel.size(0); 
  at::Tensor _out_feats =
      torch::zeros({num_in_feats, num_out_channels}, options);

  //auto in_feats = reinterpret_cast<half*>(_in_feats.data_ptr<at::Half>());
  auto kernel = reinterpret_cast<int*>(_kernel.data_ptr<int>());
  //auto out_feats = reinterpret_cast<half*>(_out_feats.data_ptr<at::Half>());
  //auto scaling_factors =
  //    reinterpret_cast<half*>(_scaling_factors.data_ptr<at::Half>());
  auto zeros = reinterpret_cast<int*>(_zeros.data_ptr<int>());
  auto temp_space = reinterpret_cast<float*>(_temp_space.data_ptr<float>());
  int group_size = num_in_channels / _scaling_factors.size(0);

#if 0
  int lda = num_out_channels;
  int ldb = num_in_channels;
  int ldc = num_out_channels;

  float alpha = 1.0, beta = 0.0;
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  vllm::awq::launch_gemm_awq(Operation_t(0), Operation_t(0), num_out_channels, num_in_feats, num_in_channels, alpha, beta, (const uint32_t*)kernel, lda,
                             (const half*)in_feats, ldb,
                             (half*)out_feats, ldc, (uint32_t*)zeros, (half*)scaling_factors, space_mid, stream, 3);
#endif

  int lda = num_in_channels;
  int ldb = num_out_channels;
  int ldc = num_out_channels;

  if (dtype_bf16) {
	  using scalar_t = __maca_bfloat16;
	  vllm::awq::launch_gemm<scalar_t, vllm::kU4.id(), scalar_t, quant_packed_type>(group_size, num_in_feats, num_out_channels, num_in_channels,
                                     (const scalar_t*)_in_feats.data_ptr(), lda, (const uint32_t*)kernel, ldb, (scalar_t*)_out_feats.data_ptr(), temp_space, ldc,
                                     (uint32_t*)zeros, (scalar_t*)_scaling_factors.data_ptr(), stream);
  } else {
	  auto in_feats = reinterpret_cast<half*>(_in_feats.data_ptr<at::Half>());
	  auto scaling_factors = reinterpret_cast<half*>(_scaling_factors.data_ptr<at::Half>());
	  auto out_feats = reinterpret_cast<half*>(_out_feats.data_ptr<at::Half>());
	  vllm::awq::launch_gemm<input_type, vllm::kU4.id(), output_type, quant_packed_type>(group_size, num_in_feats, num_out_channels, num_in_channels,
                                     (const half*)in_feats, lda, (const uint32_t*)kernel, ldb, (half*)out_feats, nullptr, ldc,
                                     (uint32_t*)zeros, (half*)scaling_factors, stream);
  }


  return _out_feats;
}
