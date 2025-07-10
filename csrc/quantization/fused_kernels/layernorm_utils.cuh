#pragma once

/**
 * __device__ layernorm utilities.
 */

#include "quantization/vectorization.cuh"
#include "quantization/utils.cuh"
#include "quant_conversions.cuh"

#ifndef USE_ROCM
  #include <cub/cub.cuh>
#else
  #include <hipcub/hipcub.hpp>
#endif

namespace vllm {

// has_residual must be true, if residual is not a nullptr
template <typename scalar_t, bool has_residual = false>
__device__ void compute_rms(float* rms, scalar_t const* __restrict__ input,
                            int32_t const hidden_size, float const epsilon,
                            scalar_t const* __restrict__ residual = nullptr) {
  int64_t const token_offset = blockIdx.x * static_cast<int64_t>(hidden_size);
  // sum of squares
  float ss = 0.0f;

  for (auto i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    float x = static_cast<float>(input[token_offset + i]);
    if constexpr (has_residual) {
      x += static_cast<float>(residual[token_offset + i]);
    }

    ss += x * x;
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  ss = BlockReduce(reduceStore).Reduce(ss, cub::Sum{}, blockDim.x);

  __shared__ float s_rms;
  if (threadIdx.x == 0) {
    s_rms = rsqrtf(ss / hidden_size + epsilon);
  }
  __syncthreads();

  *rms = s_rms;
}

template <typename scalar_t, typename scalar_out_t, bool has_residual = false>
__device__ void compute_dynamic_per_token_scales(
    float* __restrict__ token_scale, float* __restrict__ all_token_scales,
    scalar_t const* __restrict__ input, scalar_t const* __restrict__ weight,
    float const rms, float const* __restrict__ scale_ub,
    int32_t const hidden_size,
    scalar_t const* __restrict__ residual = nullptr) {
  int64_t const token_offset = blockIdx.x * static_cast<int64_t>(hidden_size);
  ;
  constexpr scalar_out_t qmax{quant_type_max_v<scalar_out_t>};

  float block_absmax_val_maybe = 0.0f;
  for (auto i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    float x = static_cast<float>(input[token_offset + i]);
    if constexpr (has_residual) {
      x += static_cast<float>(residual[token_offset + i]);
    }

    x = static_cast<float>(static_cast<scalar_t>(x * rms) * weight[i]);
    block_absmax_val_maybe = fmaxf(block_absmax_val_maybe, fabsf(x));
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  block_absmax_val_maybe =
      BlockReduce(reduceStore)
          .Reduce(block_absmax_val_maybe, cub::Max{}, blockDim.x);

  __shared__ float s_token_scale;
  if (threadIdx.x == 0) {
    float scale = 0.0f;
    if (scale_ub) {
      scale = min(block_absmax_val_maybe, *scale_ub);
    } else {
      scale = block_absmax_val_maybe;
    }
    // token scale computation
    scale = max(scale / qmax, min_scaling_factor<scalar_out_t>::val());
    s_token_scale = scale;                 // Shared memory store
    all_token_scales[blockIdx.x] = scale;  // Global output store
  }
  __syncthreads();

  *token_scale = s_token_scale;
}

template <typename scalar_t, typename scalar_out_t, bool is_scale_inverted,
          bool has_residual = false>
__device__ void norm_and_quant(scalar_out_t* __restrict__ output,
                               scalar_t const* __restrict__ input,
                               scalar_t const* __restrict__ weight,
                               float const rms, float const scale,
                               int32_t const hidden_size,
                               scalar_t* __restrict__ residual = nullptr) {
  int64_t const token_offset = blockIdx.x * static_cast<int64_t>(hidden_size);
  ;

  for (auto i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    float x = static_cast<float>(input[token_offset + i]);
    if constexpr (has_residual) {
      x += static_cast<float>(residual[token_offset + i]);
      residual[token_offset + i] = static_cast<scalar_t>(x);
    }
    // Norm
    x = static_cast<float>(static_cast<scalar_t>(x * rms) * weight[i]);
    // Quant
    output[token_offset + i] =
        ScaledQuant<scalar_out_t, is_scale_inverted>::quant_fn(x, scale);
  }
}

template <typename scalar_t, typename scalar_out_t, bool is_scale_inverted,
          bool has_residual = false>
__device__ void norm_and_quant(scalar_t* __restrict__ output_bf16,
                               scalar_out_t* __restrict__ output,
                               scalar_t const* __restrict__ input,
                               scalar_t const* __restrict__ weight,
                               float const rms, float const scale,
                               int32_t const hidden_size,
                               scalar_t* __restrict__ residual = nullptr) {
  int64_t const token_offset = blockIdx.x * static_cast<int64_t>(hidden_size);
  ;

  for (auto i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    float x = static_cast<float>(input[token_offset + i]);
    if constexpr (has_residual) {
      x += static_cast<float>(residual[token_offset + i]);
      residual[token_offset + i] = static_cast<scalar_t>(x);
    }
    // Norm
    x = static_cast<float>(static_cast<scalar_t>(x * rms) * weight[i]);
    // Quant
    output_bf16[token_offset + i] = static_cast<scalar_t>(x);
    output[token_offset + i] =
        ScaledQuant<scalar_out_t, is_scale_inverted>::quant_fn(x, scale);
  }
}

template<typename T>
__device__ __forceinline__ T WARP_SHFL_XOR(T value, int laneMask, int width = 64) {
    return __shfl_xor_sync(0xffffffffffffffff, value, laneMask, width);
}

template<typename T, int32_t block_dim>
__device__ __forceinline__ void reduce_max_sum(T& val_max, T& val_sum){
  __shared__ T shared_s[64];
  __shared__ T shared_m[64];
  T val_m = val_max;
  T val_s = val_sum;
  int lane = threadIdx.x & 0x3f;
  int wid = threadIdx.x >> 6;

  for (int mask = 32; mask > 0; mask >>= 1) {
    val_s += WARP_SHFL_XOR<T>(val_s, mask);
    val_m = max(val_m, WARP_SHFL_XOR<T>(val_m, mask));
  }

  if(lane == 0) {
    shared_s[wid] = val_s;
    shared_m[wid] = val_m;
  }
  __syncthreads();
  val_s = (lane < ((block_dim + 63) >> 6)) ? shared_s[lane] : static_cast<T>(0.0f);
  val_m = (lane < ((block_dim + 63) >> 6)) ? shared_m[lane] : static_cast<T>(-9999);
  __syncthreads();

  for (int mask = 32; mask > 0; mask >>= 1) {
    val_s += WARP_SHFL_XOR<T>(val_s, mask);
    val_m = max(val_m, WARP_SHFL_XOR<T>(val_m, mask));
  }
  val_max = val_m;
  val_sum = val_s;
}

namespace vectorized {

// Compute 1.0/rms(input)
// hidden_size must be a multiple of 4
template <typename scalar_t, bool has_residual = false>
__device__ void compute_rms(float* rms, scalar_t const* __restrict__ input,
                            int32_t const hidden_size, float const epsilon,
                            scalar_t const* __restrict__ residual = nullptr) {
  int64_t const token_offset = blockIdx.x * static_cast<int64_t>(hidden_size);

  // Vectorized input/output to better utilize memory bandwidth.
  vec4_t<scalar_t> const* vec_input =
      reinterpret_cast<vec4_t<scalar_t> const*>(&input[token_offset]);
  vec4_t<scalar_t> const* vec_residual = nullptr;
  if constexpr (has_residual) {
    vec_residual =
        reinterpret_cast<vec4_t<scalar_t> const*>(&residual[token_offset]);
  }

  // sum of squares
  float ss = 0.0f;

  int32_t const num_vec_elems = hidden_size >> 2;

#pragma unroll 4
  for (auto i = threadIdx.x; i < num_vec_elems; i += blockDim.x) {
    vec4_t<scalar_t> in = vec_input[i];

    vec4_t<float> x;
    x.x = static_cast<float>(in.x);
    x.y = static_cast<float>(in.y);
    x.z = static_cast<float>(in.z);
    x.w = static_cast<float>(in.w);
    if constexpr (has_residual) {
      vec4_t<scalar_t> r = vec_residual[i];
      x.x += static_cast<float>(r.x);
      x.y += static_cast<float>(r.y);
      x.z += static_cast<float>(r.z);
      x.w += static_cast<float>(r.w);
    }

    ss += x.x * x.x;
    ss += x.y * x.y;
    ss += x.z * x.z;
    ss += x.w * x.w;
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  ss = BlockReduce(reduceStore).Reduce(ss, cub::Sum{}, blockDim.x);

  __shared__ float s_rms;
  if (threadIdx.x == 0) {
    s_rms = rsqrtf(ss / hidden_size + epsilon);
  }
  __syncthreads();

  *rms = s_rms;
}

// Vectorized version of vllm::compute_dynamic_per_token_scales
// hidden_size must be a multiple of 4
template <typename scalar_t, typename scalar_out_t, bool has_residual = false>
__device__ void compute_dynamic_per_token_scales(
    float* __restrict__ token_scale, float* __restrict__ all_token_scales,
    scalar_t const* __restrict__ input, scalar_t const* __restrict__ weight,
    float const rms, float const* __restrict__ scale_ub,
    int32_t const hidden_size,
    scalar_t const* __restrict__ residual = nullptr) {
  int64_t const token_offset = blockIdx.x * static_cast<int64_t>(hidden_size);
  ;

  // Vectorized input/weight/residual to better utilize memory bandwidth.
  vec4_t<scalar_t> const* vec_input =
      reinterpret_cast<vec4_t<scalar_t> const*>(&input[token_offset]);
  vec4_t<scalar_t> const* vec_weight =
      reinterpret_cast<vec4_t<scalar_t> const*>(weight);
  vec4_t<scalar_t> const* vec_residual = nullptr;
  if constexpr (has_residual) {
    vec_residual =
        reinterpret_cast<vec4_t<scalar_t> const*>(&residual[token_offset]);
  }

  constexpr scalar_out_t qmax{quant_type_max_v<scalar_out_t>};

  int32_t const num_vec_elems = hidden_size >> 2;
  float block_absmax_val_maybe = 0.0f;

#pragma unroll 4
  for (auto i = threadIdx.x; i < num_vec_elems; i += blockDim.x) {
    vec4_t<scalar_t> in = vec_input[i];
    vec4_t<scalar_t> const w = vec_weight[i];

    vec4_t<float> x;
    x.x = static_cast<float>(in.x);
    x.y = static_cast<float>(in.y);
    x.z = static_cast<float>(in.z);
    x.w = static_cast<float>(in.w);
    if constexpr (has_residual) {
      vec4_t<scalar_t> r = vec_residual[i];
      x.x += static_cast<float>(r.x);
      x.y += static_cast<float>(r.y);
      x.z += static_cast<float>(r.z);
      x.w += static_cast<float>(r.w);
    }

    block_absmax_val_maybe = fmaxf(
        block_absmax_val_maybe, fabs(static_cast<scalar_t>(x.x * rms) * w.x));
    block_absmax_val_maybe = fmaxf(
        block_absmax_val_maybe, fabs(static_cast<scalar_t>(x.y * rms) * w.y));
    block_absmax_val_maybe = fmaxf(
        block_absmax_val_maybe, fabs(static_cast<scalar_t>(x.z * rms) * w.z));
    block_absmax_val_maybe = fmaxf(
        block_absmax_val_maybe, fabs(static_cast<scalar_t>(x.w * rms) * w.w));
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  block_absmax_val_maybe =
      BlockReduce(reduceStore)
          .Reduce(block_absmax_val_maybe, cub::Max{}, blockDim.x);

  __shared__ float s_token_scale;
  if (threadIdx.x == 0) {
    float scale = 0.0f;
    if (scale_ub) {
      scale = min(block_absmax_val_maybe, *scale_ub);
    } else {
      scale = block_absmax_val_maybe;
    }
    // token scale computation
    scale = max(scale / qmax, min_scaling_factor<scalar_out_t>::val());
    s_token_scale = scale;                 // shared memory store
    all_token_scales[blockIdx.x] = scale;  // global output store
  }
  __syncthreads();

  *token_scale = s_token_scale;
}

// hidden_size must be a multiple of 4
template <typename scalar_t, typename scalar_out_t, bool is_scale_inverted,
          bool has_residual = false>
__device__ void norm_and_quant(scalar_out_t* __restrict__ output,
                               scalar_t const* __restrict__ input,
                               scalar_t const* __restrict__ weight,
                               float const rms, float const scale,
                               int32_t const hidden_size,
                               scalar_t* __restrict__ residual = nullptr) {
  int64_t const token_offset = blockIdx.x * static_cast<int64_t>(hidden_size);
  ;

  // Vectorized input/output/weight/residual to better utilize memory bandwidth.
  vec4_t<scalar_t> const* vec_input =
      reinterpret_cast<vec4_t<scalar_t> const*>(&input[token_offset]);
  vec4_t<scalar_t> const* vec_weight =
      reinterpret_cast<vec4_t<scalar_t> const*>(weight);
  q8x4_t<scalar_out_t>* vec_output =
      reinterpret_cast<q8x4_t<scalar_out_t>*>(&output[token_offset]);
  vec4_t<scalar_t>* vec_residual = nullptr;
  if constexpr (has_residual) {
    vec_residual = reinterpret_cast<vec4_t<scalar_t>*>(&residual[token_offset]);
  }

  int32_t const num_vec_elems = hidden_size >> 2;

// TODO(luka/varun) extract into type-agnostic vectorized quant function to
//  replace scaled_fp8_conversion_vec
#pragma unroll 4
  for (auto i = threadIdx.x; i < num_vec_elems; i += blockDim.x) {
    vec4_t<scalar_t> const in = vec_input[i];
    vec4_t<scalar_t> const w = vec_weight[i];

    vec4_t<float> x;
    x.x = static_cast<float>(in.x);
    x.y = static_cast<float>(in.y);
    x.z = static_cast<float>(in.z);
    x.w = static_cast<float>(in.w);
    if constexpr (has_residual) {
      vec4_t<scalar_t> r = vec_residual[i];
      x.x += static_cast<float>(r.x);
      x.y += static_cast<float>(r.y);
      x.z += static_cast<float>(r.z);
      x.w += static_cast<float>(r.w);
      // Update residual
      r.x = static_cast<scalar_t>(x.x);
      r.y = static_cast<scalar_t>(x.y);
      r.z = static_cast<scalar_t>(x.z);
      r.w = static_cast<scalar_t>(x.w);
      vec_residual[i] = r;
    }

    q8x4_t<scalar_out_t> out;
    out.x = ScaledQuant<scalar_out_t, is_scale_inverted>::quant_fn(
        static_cast<scalar_t>(x.x * rms) * w.x, scale);
    out.y = ScaledQuant<scalar_out_t, is_scale_inverted>::quant_fn(
        static_cast<scalar_t>(x.y * rms) * w.y, scale);
    out.z = ScaledQuant<scalar_out_t, is_scale_inverted>::quant_fn(
        static_cast<scalar_t>(x.z * rms) * w.z, scale);
    out.w = ScaledQuant<scalar_out_t, is_scale_inverted>::quant_fn(
        static_cast<scalar_t>(x.w * rms) * w.w, scale);
    vec_output[i] = out;
  }
}



template <typename scalar_t, typename scalar_out_t, bool is_scale_inverted,
          bool has_residual = false>
__device__ void norm_and_quant(scalar_t* __restrict__ output_bf16,
                               scalar_out_t* __restrict__ output,
                               scalar_t const* __restrict__ input,
                               scalar_t const* __restrict__ weight,
                               float const rms, float const scale,
                               int32_t const hidden_size,
                               scalar_t* __restrict__ residual = nullptr) {
  int64_t const token_offset = blockIdx.x * static_cast<int64_t>(hidden_size);
  ;

  // Vectorized input/output/weight/residual to better utilize memory bandwidth.
  vec4_t<scalar_t> const* vec_input =
      reinterpret_cast<vec4_t<scalar_t> const*>(&input[token_offset]);
  vec4_t<scalar_t> const* vec_weight =
      reinterpret_cast<vec4_t<scalar_t> const*>(weight);
  vec4_t<scalar_t>* vec_output_bf16 =
      reinterpret_cast<vec4_t<scalar_t>*>(&output_bf16[token_offset]);
  q8x4_t<scalar_out_t>* vec_output =
      reinterpret_cast<q8x4_t<scalar_out_t>*>(&output[token_offset]);
  vec4_t<scalar_t>* vec_residual = nullptr;
  if constexpr (has_residual) {
    vec_residual = reinterpret_cast<vec4_t<scalar_t>*>(&residual[token_offset]);
  }

  int32_t const num_vec_elems = hidden_size >> 2;

// TODO(luka/varun) extract into type-agnostic vectorized quant function to
//  replace scaled_fp8_conversion_vec
#pragma unroll 4
  for (auto i = threadIdx.x; i < num_vec_elems; i += blockDim.x) {
    vec4_t<scalar_t> const in = vec_input[i];
    vec4_t<scalar_t> const w = vec_weight[i];

    vec4_t<float> x;
    x.x = static_cast<float>(in.x);
    x.y = static_cast<float>(in.y);
    x.z = static_cast<float>(in.z);
    x.w = static_cast<float>(in.w);
    if constexpr (has_residual) {
      vec4_t<scalar_t> r = vec_residual[i];
      x.x += static_cast<float>(r.x);
      x.y += static_cast<float>(r.y);
      x.z += static_cast<float>(r.z);
      x.w += static_cast<float>(r.w);
      // Update residual
      r.x = static_cast<scalar_t>(x.x);
      r.y = static_cast<scalar_t>(x.y);
      r.z = static_cast<scalar_t>(x.z);
      r.w = static_cast<scalar_t>(x.w);
      vec_residual[i] = r;
    }

    q8x4_t<scalar_out_t> out;
    vec4_t<scalar_t> out_bf16;

    out_bf16.x = static_cast<scalar_t>(x.x * rms) * w.x;
    out.x = ScaledQuant<scalar_out_t, is_scale_inverted>::quant_fn(
        out_bf16.x, scale);
    out_bf16.y = static_cast<scalar_t>(x.y * rms) * w.y;
    out.y = ScaledQuant<scalar_out_t, is_scale_inverted>::quant_fn(
        out_bf16.y, scale);
    out_bf16.z = static_cast<scalar_t>(x.z * rms) * w.z;
    out.z = ScaledQuant<scalar_out_t, is_scale_inverted>::quant_fn(
        out_bf16.z, scale);
    out_bf16.w = static_cast<scalar_t>(x.w * rms) * w.w;
    out.w = ScaledQuant<scalar_out_t, is_scale_inverted>::quant_fn(
        out_bf16.w, scale);
    vec_output[i] = out;
    vec_output_bf16[i] = out_bf16;
  }
}

}  // namespace vectorized

}  // namespace vllm
