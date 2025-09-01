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
  //zero //4bit scale 8bit b 4bit
  hgemm_marlin_gptq::awq_dequant_4bits<T>(B_loaded,B_shared, B_loaded_scale, zeros_loaded);

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

  const at::cuda::OptionalCUDAGuard device_guard(device_of(_scaling_factors));

  auto options = torch::TensorOptions()
                     .dtype(_scaling_factors.dtype())
                     .device(_scaling_factors.device());
  at::Tensor _de_kernel = torch::empty({in_c, out_c}, options);

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto kernel = reinterpret_cast<int*>(_kernel.data_ptr<int>());
  auto zeros = reinterpret_cast<int*>(_zeros.data_ptr<int>());
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
