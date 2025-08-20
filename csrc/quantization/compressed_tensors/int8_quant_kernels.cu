#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>
#include <cmath>

#include "../../dispatch_utils.h"

#ifndef USE_ROCM
  #include <cub/util_type.cuh>
  #include <cub/cub.cuh>
#else
  #include <hipcub/util_type.hpp>
  #include <hipcub/hipcub.hpp>
#endif

static __forceinline__ __device__ int8_t float_to_int8_rn(float x) {
#ifdef USE_ROCM
  static constexpr auto i8_min =
      static_cast<float>(std::numeric_limits<int8_t>::min());
  static constexpr auto i8_max =
      static_cast<float>(std::numeric_limits<int8_t>::max());

  // To match the rounding mode of CUDA, we use nearbyint.
  // It uses the current rounding mode, which is always FE_TONEAREST on HIP.
  // If that changes in the future, we may need to set the rounding mode
  // explicitly, either at runtime or compile time.
  float dst = std::nearbyint(x);

  // saturate

  // See https://github.com/pytorch/pytorch/issues/127666
  // See https://github.com/llvm/llvm-project/issues/95183
  // hip-clang std::clamp __glibcxx_assert_fail host function when building on
  // Arch/gcc14. The following replaces std::clamp usage with similar logic
  // dst = std::clamp(dst, i8_min, i8_max);
  dst = (dst < i8_min) ? i8_min : (dst > i8_max) ? i8_max : dst;
  return static_cast<int8_t>(dst);
#else
  // CUDA path
  //uint32_t dst;
  //asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(dst) : "f"(x));
  //return reinterpret_cast<const int8_t&>(dst);
  int32_t dst;
  dst = __float2int_rn(x);
  dst = min(dst, 127);
  dst = max(dst, -127);
  return reinterpret_cast<const int8_t&>(dst);
#endif
}

static inline __device__ int32_t float_to_int32_rn(float x) {
#ifdef USE_ROCM
  // int32_max is not exactly representable as float.
  // Therefore, we need to be careful and manually return int32_max on overflow.
  // For symmetry, we also do the same for int32_min, even though it is exactly
  // representable as float and the conversion should be exact.
  static constexpr auto i32_min = std::numeric_limits<int32_t>::min();
  static constexpr auto i32_min_f = static_cast<float>(i32_min);
  static constexpr auto i32_max = std::numeric_limits<int32_t>::max();
  static constexpr auto i32_max_f = static_cast<float>(i32_max);

  // To match the rounding mode of CUDA, we use nearbyint.
  // It uses the current rounding mode, which is always FE_TONEAREST on HIP.
  // If that changes in the future, we may need to set the rounding mode
  // explicitly, either at runtime or compile time.
  float dst = std::nearbyint(x);

  // saturate on the higher end.
  if (dst >= i32_max_f) {
    return i32_max;
  }
  // saturate on the lower end.
  if (dst <= i32_min_f) {
    return i32_min;
  }

  return static_cast<int32_t>(dst);
#else
  // CUDA path
  static constexpr auto i32_min = std::numeric_limits<int32_t>::min();
  static constexpr auto i32_min_f = static_cast<float>(i32_min);
  static constexpr auto i32_max = std::numeric_limits<int32_t>::max();
  static constexpr auto i32_max_f = static_cast<float>(i32_max);
  x = min(x, i32_max_f);
  x = max(x, i32_min_f);
  return __float2int_rn(x);
#endif
}

static inline __device__ int8_t int32_to_int8(int32_t x) {
#ifdef USE_ROCM
  static constexpr auto i8_min =
      static_cast<int32_t>(std::numeric_limits<int8_t>::min());
  static constexpr auto i8_max =
      static_cast<int32_t>(std::numeric_limits<int8_t>::max());

  // saturate

  // See https://github.com/pytorch/pytorch/issues/127666
  // See https://github.com/llvm/llvm-project/issues/95183
  // hip-clang std::clamp __glibcxx_assert_fail host function when building on
  // Arch/gcc14. The following replaces std::clamp usage with similar logic
  // int32_t dst = std::clamp(x, i8_min, i8_max);
  int32_t dst = (x < i8_min) ? i8_min : (x > i8_max) ? i8_max : x;
  return static_cast<int8_t>(dst);
#else
  // CUDA path
  static constexpr auto i8_min =
      static_cast<int32_t>(std::numeric_limits<int8_t>::min());
  static constexpr auto i8_max =
      static_cast<int32_t>(std::numeric_limits<int8_t>::max());

  // saturate
  int32_t dst = std::clamp(x, i8_min, i8_max);
  return static_cast<int8_t>(dst);
#endif
}

namespace vllm {

template <typename scalar_t, typename scale_type>
__global__ void static_scaled_int8_quant_kernel(
    scalar_t const* __restrict__ input, int8_t* __restrict__ out,
    scale_type const* scale_ptr, const int hidden_size) {
  int const tid = threadIdx.x;
  int64_t const token_idx = blockIdx.x;
  scale_type const scale = *scale_ptr;

  // Must be performed using 64-bit math to avoid integer overflow.
  out += token_idx * hidden_size;
  input += token_idx * hidden_size;

  for (int i = tid; i < hidden_size; i += blockDim.x) {
    out[i] = float_to_int8_rn(static_cast<float>(input[i]) / scale);
  }
}

template <typename scalar_t, typename scale_type, typename azp_type>
__global__ void static_scaled_int8_azp_quant_kernel(
    scalar_t const* __restrict__ input, int8_t* __restrict__ out,
    scale_type const* scale_ptr, azp_type const* azp_ptr,
    const int hidden_size) {
  int const tid = threadIdx.x;
  int64_t const token_idx = blockIdx.x;
  scale_type const scale = *scale_ptr;
  azp_type const azp = *azp_ptr;

  // Must be performed using 64-bit math to avoid integer overflow.
  out += token_idx * hidden_size;
  input += token_idx * hidden_size;

  for (int i = tid; i < hidden_size; i += blockDim.x) {
    auto const val = static_cast<float>(input[i]);
    auto const quant_val = int32_to_int8(float_to_int32_rn(val / scale) + azp);
    out[i] = quant_val;
  }
}

template <typename scalar_t, typename scale_type, bool WITHMASK>
__global__ void dynamic_scaled_int8_quant_kernel(
    scalar_t const* __restrict__ input, int8_t* __restrict__ out,
    scale_type* scale, const int hidden_size, const int num_tokens, int *mask_buffer = NULL) {
  if constexpr(WITHMASK) {
    __shared__ int sm_max_token;
    if(threadIdx.x == 0) sm_max_token = mask_buffer[blockIdx.y]; 
    __syncthreads();
    if(blockIdx.x >= sm_max_token) return;
  }
  int const tid = threadIdx.x;
  int64_t const token_idx = blockIdx.y * num_tokens + blockIdx.x;
  float absmax_val = 0.0f;
  float const zero = 0.0f;

  for (int i = tid; i < hidden_size; i += blockDim.x) {
    float val = static_cast<float>(input[token_idx * hidden_size + i]);
    val = val > zero ? val : -val;
    absmax_val = val > absmax_val ? val : absmax_val;
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStorage;
  float const block_absmax_val_maybe =
      BlockReduce(reduceStorage).Reduce(absmax_val, cub::Max{}, blockDim.x);
  __shared__ float block_absmax_val;
  if (tid == 0) {
    block_absmax_val = block_absmax_val_maybe;
    scale[token_idx] = block_absmax_val * 0.0078740157;
  }
  __syncthreads();

  float const tmp_scale = 127.0f *__builtin_mxc_rcpf(block_absmax_val);
  for (int i = tid; i < hidden_size; i += blockDim.x) {
    out[token_idx * hidden_size + i] = float_to_int8_rn(
        static_cast<float>(input[token_idx * hidden_size + i]) * tmp_scale);
  }
}

template <typename scalar_t, typename scale_type, typename VT, typename VT1, int NUM_THREADS, bool WITHMASK>
__global__ void dynamic_scaled_int8_quant_kernel_sreg_opt(
    scalar_t const* __restrict__ input, int8_t* __restrict__ out,
    scale_type* scale, const int hidden_size, int num_tokens, int* mask_buffer=NULL) {
  if constexpr(WITHMASK) {
    __shared__ int sm_max_token;
    if(threadIdx.x == 0) sm_max_token = mask_buffer[blockIdx.y]; 
    __syncthreads();
    if(blockIdx.x >= sm_max_token) return;
  }
  int const tid = threadIdx.x;
  int64_t const token_idx = blockIdx.y * num_tokens + blockIdx.x;
  float absmax_val = 0.0f;
  float const zero = 0.0f;
  constexpr int N = sizeof(VT) / sizeof(scalar_t);
  float reg_src0[N];
  scalar_t const* ptr_input = input + token_idx * hidden_size;
  int reg_length = NUM_THREADS * N;
  int length = min(hidden_size, reg_length);
  int index = tid * N;
  if(index < length) {
    VT reg_src;
    reg_src = *(VT*)(ptr_input + index);
    scalar_t* ptr_reg_src = (scalar_t*)&reg_src;
    #pragma unroll N
    for(int i = 0; i < N; i++) {
      reg_src0[i] = (float)ptr_reg_src[i];
    }
    #pragma unroll N
    for(int i = 0; i < N; i++) {
      float val = abs(reg_src0[i]);
      absmax_val = max(absmax_val, val);
    }
  }

  using BlockReduce = cub::BlockReduce<float, NUM_THREADS>;
  __shared__ typename BlockReduce::TempStorage reduceStorage;
  float const block_absmax_val_maybe =
      BlockReduce(reduceStorage).Reduce(absmax_val, cub::Max{}, NUM_THREADS);
  __shared__ float block_absmax_val;
  if (tid == 0) {
    block_absmax_val = block_absmax_val_maybe;
    scale[token_idx] = static_cast<scale_type>(block_absmax_val_maybe * 0.0078740157);
  }
  __syncthreads();
  float const tmp_scale = 127.0f * __builtin_mxc_rcpf(block_absmax_val);
  int8_t* ptr_output = out + token_idx * hidden_size;
  if(index < length) {
    VT1 vdst;
    int8_t* ptr_reg = (int8_t*)&vdst;
    #pragma unroll N
    for(int i = 0; i < N; i++) {
      ptr_reg[i] = float_to_int8_rn(reg_src0[i] * tmp_scale);
    }
    *(VT1*)(ptr_output + index) = vdst;
  }
}

template <typename scalar_t, typename scale_type, typename VT, typename VT1, bool WITHMASK>
__global__ void dynamic_scaled_int8_quant_kernel_reg_opt(
    scalar_t const* __restrict__ input, int8_t* __restrict__ out,
    scale_type* scale, const int hidden_size, int blockDim_x, int num_tokens, int* mask_buffer=NULL) {
  if constexpr(WITHMASK) {
    __shared__ int sm_max_token;
    if(threadIdx.x == 0) sm_max_token = mask_buffer[blockIdx.y]; 
    __syncthreads();
    if(blockIdx.x >= sm_max_token) return;
  }
  int const tid = threadIdx.x;
  int64_t const token_idx = blockIdx.y * num_tokens + blockIdx.x;
  float absmax_val = 0.0f;
  float const zero = 0.0f;
  constexpr int N = sizeof(VT) / sizeof(scalar_t);
  float reg_src0[N];
  float reg_src1[N];
  scalar_t const* ptr_input = input + token_idx * hidden_size;
  int reg_length = 2 * blockDim_x * N;
  int length = min(hidden_size, reg_length);
  int index = 2 * tid * N;
  if(index < length) {
    VT reg_src = *(VT*)(ptr_input + index);
    scalar_t* ptr_reg_src = (scalar_t*)&reg_src;
    #pragma unroll N
    for(int i = 0; i < N; i++) {
      reg_src0[i] = (float)ptr_reg_src[i];
    }
    reg_src = *(VT*)(ptr_input + index + N);
    #pragma unroll N
    for(int i = 0; i < N; i++) {
      reg_src1[i] = (float)ptr_reg_src[i];
    }
    #pragma unroll N
    for(int i = 0; i < N; i++) {
      float val = abs(reg_src0[i]);
      absmax_val =  max(val, absmax_val);
      val = abs(reg_src1[i]);
      absmax_val = max(val, absmax_val);
    }
  }

  using BlockReduce = cub::BlockReduce<float, 512>;
  __shared__ typename BlockReduce::TempStorage reduceStorage;
  float const block_absmax_val_maybe =
      BlockReduce(reduceStorage).Reduce(absmax_val, cub::Max{}, blockDim_x);
  __shared__ float block_absmax_val;
  if (tid == 0) {
    block_absmax_val = (block_absmax_val_maybe);
    scale[token_idx] = static_cast<scale_type>(block_absmax_val * 0.0078740157);
  }
  __syncthreads();
  float const tmp_scale = 127.0f * __builtin_mxc_rcpf(block_absmax_val);
  int8_t* ptr_output = out + token_idx * hidden_size;
  if(index < length) {
    VT1 vdst;
    int8_t* ptr_reg = (int8_t*)&vdst;
    #pragma unroll N
    for(int i = 0; i < N; i++) {
      ptr_reg[i] = float_to_int8_rn(
             reg_src0[i] * tmp_scale);
    }
    ptr_reg = ptr_reg + N;
    #pragma unroll N
    for(int i = 0; i < N; i++) {
      ptr_reg[i] = float_to_int8_rn(
            reg_src1[i] * tmp_scale);
    }
    *(VT1*)(ptr_output + index) = vdst;
  }
}

template <typename scalar_t, typename scale_type, typename VT, typename VT1, int NUM_REG, int NUM_THREADS, bool WITHMASK>
__global__ __launch_bounds__(1024) void dynamic_scaled_int8_quant_kernel_lh_opt(
    scalar_t const* __restrict__ input, int8_t* __restrict__ out,
    scale_type* scale, const int hidden_size, int num_tokens, int* mask_buffer=NULL) {
  if constexpr(WITHMASK) {
    __shared__ int sm_max_token;
    if(threadIdx.x == 0) sm_max_token = mask_buffer[blockIdx.y]; 
    __syncthreads();
    if(blockIdx.x >= sm_max_token) return;
  }
  int const tid = threadIdx.x;
  int64_t const token_idx = blockIdx.y * num_tokens + blockIdx.x;
  float absmax_val = 0.0f;
  float const zero = 0.0f;
  constexpr int N = sizeof(VT) / sizeof(scalar_t);
  int stride = NUM_THREADS * N;
  float reg_src[NUM_REG][N];
  scalar_t const* ptr_input = input + token_idx * hidden_size;
  for(int i = tid * N, k = 0; i < hidden_size; i += stride, k++) {
    VT vsrc = *(VT*)(ptr_input + i);
    scalar_t *ptr_src = (scalar_t*)&vsrc;
    #pragma unroll N
    for(int j = 0; j < N; j++) {
        float val = static_cast<float>(ptr_src[j]);
        reg_src[k][j] = val;
        val = abs(val);
        absmax_val = max(val, absmax_val);
    }
  }
  using BlockReduce = cub::BlockReduce<float, NUM_THREADS>;
  __shared__ typename BlockReduce::TempStorage reduceStorage;
  float const block_absmax_val_maybe =
      BlockReduce(reduceStorage).Reduce(absmax_val, cub::Max{}, NUM_THREADS);
  __shared__ float block_absmax_val;
  if (tid == 0) {
    block_absmax_val = block_absmax_val_maybe;
    scale[token_idx] = block_absmax_val * 0.0078740157;
  }
  
  __syncthreads();

  float const tmp_scale = 127.0f * __builtin_mxc_rcpf(block_absmax_val);
  int8_t* ptr_output = out + token_idx * hidden_size;
  for(int i = tid * N, k = 0; i < hidden_size; i += stride, k++) {
    VT1 vdst;
    int8_t* ptr_reg = (int8_t*)&vdst;
    #pragma unroll N
    for(int j = 0; j < N; j++) {
        ptr_reg[j] = float_to_int8_rn(
            reg_src[k][j] * tmp_scale);
    }
    *(VT1*)(ptr_output + i) = vdst;
  }
}

template <typename scalar_t, typename scale_type, typename VT, typename VT1, bool WITHMASK>
__global__ void dynamic_scaled_int8_quant_kernel_sm_opt(
    scalar_t const* __restrict__ input, int8_t* __restrict__ out,
    scale_type* scale, const int hidden_size, int blockDim_x, int num_tokens, int* mask_buffer=NULL) {
  if constexpr(WITHMASK) {
    __shared__ int sm_max_token;
    if(threadIdx.x == 0) sm_max_token = mask_buffer[blockIdx.y]; 
    __syncthreads();
    if(blockIdx.x >= sm_max_token) return;
  }
  int const tid = threadIdx.x;
  int64_t const token_idx = blockIdx.y * num_tokens + blockIdx.x;
  float absmax_val = 0.0f;
  float const zero = 0.0f;
  constexpr int N = sizeof(VT) / sizeof(scalar_t);
  int stride = blockDim_x * N;
  __shared__ float sm_buffer[8064];
  scalar_t const* ptr_input = input + token_idx * hidden_size;
  for(int i = tid * N; i < hidden_size; i += stride) {
    VT vsrc = *(VT*)(ptr_input + i);
    scalar_t *ptr_src = (scalar_t*)&vsrc;
    float* ptr_sm_buffer = sm_buffer + i;
    #pragma unroll N
    for(int j = 0; j < N; j++) {
        float val = static_cast<float>(ptr_src[j]);
        ptr_sm_buffer[j] = val;
        val = abs(val);
        absmax_val = max(val, absmax_val);
    }
  }
  using BlockReduce = cub::BlockReduce<float, 512>;
  __shared__ typename BlockReduce::TempStorage reduceStorage;
  float const block_absmax_val_maybe =
      BlockReduce(reduceStorage).Reduce(absmax_val, cub::Max{}, blockDim.x);
  __shared__ float block_absmax_val;
  if (tid == 0) {
    block_absmax_val = block_absmax_val_maybe;
    scale[token_idx] = block_absmax_val * 0.0078740157;
  }
  
  __syncthreads();

  float const tmp_scale = 127.0f *__builtin_mxc_rcpf(block_absmax_val);
  int8_t* ptr_output = out + token_idx * hidden_size;
  for(int i = tid * N; i < hidden_size; i += stride) {
    VT1 vdst;
    int8_t* ptr_reg = (int8_t*)&vdst;
    float* ptr_sm_buffer = sm_buffer + i;
    #pragma unroll N
    for(int j = 0; j < N; j++) {
        ptr_reg[j] = float_to_int8_rn(
            ptr_sm_buffer[j] * tmp_scale);
    }
    *(VT1*)(ptr_output + i) = vdst;
  }
}

template <typename scalar_t, typename scale_type, typename VT, typename VT1, bool WITHMASK>
__launch_bounds__(1024) __global__ void dynamic_scaled_int8_quant_kernel_opt(
    scalar_t const* __restrict__ input, int8_t* __restrict__ out,
    scale_type* scale, const int hidden_size, const int blockDim_x, const int num_tokens, int* mask_buffer=NULL) {
  if constexpr(WITHMASK) {
    __shared__ int sm_max_token;
    if(threadIdx.x == 0) sm_max_token = mask_buffer[blockIdx.y]; 
    __syncthreads();
    if(blockIdx.x >= sm_max_token) return;
  }
  constexpr int N = sizeof(VT) / sizeof(scalar_t);
  int const tid = threadIdx.x * N;
  int64_t const token_idx = blockIdx.y * num_tokens + blockIdx.x;
  float absmax_val = 0.0f;
  int stride = blockDim_x * N;
  const scalar_t * ptr_input = input + token_idx * hidden_size;

  for (int i = tid ; i < hidden_size; i += stride) {
    VT vsrc = *(VT*)(ptr_input + i);
    scalar_t *ptr_src = (scalar_t*)&vsrc;
    #pragma unroll N
    for(int j = 0; j < N; j++) {
        float val = static_cast<float>(ptr_src[j]);
        val = abs(val);
        absmax_val = max(val, absmax_val);
    }
  }

    using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStorage;
  float const block_absmax_val_maybe =
      BlockReduce(reduceStorage).Reduce(absmax_val, cub::Max{}, blockDim.x);
  __shared__ float block_absmax_val;
  if (tid == 0) {
    block_absmax_val = block_absmax_val_maybe;
    scale[token_idx] = block_absmax_val * 0.0078740157;
  }
  __syncthreads();

  float const tmp_scale = 127.0f *__builtin_mxc_rcpf(block_absmax_val);
  int8_t* ptr_output = out + token_idx * hidden_size;
  for (int i = tid; i < hidden_size; i += stride) {
    VT vsrc = *(VT*)(ptr_input + i);
    VT1 vdst;
    scalar_t *ptr_src = (scalar_t*)&vsrc;
    int8_t* ptr_dst = (int8_t*)&vdst;
    #pragma unroll N
    for(int j = 0; j < N; ++j) {
        ptr_dst[j] = float_to_int8_rn(
        static_cast<float>(ptr_src[j]) * tmp_scale);
    }
    *(VT1*)(ptr_output + i) = vdst;
  }
}

template <typename scalar_t, typename scale_type, typename azp_type, bool WITHMASK>
__global__ void dynamic_scaled_int8_azp_quant_kernel(
    scalar_t const* __restrict__ input, int8_t* __restrict__ out,
    scale_type* scale, azp_type* azp, const int hidden_size, const int num_tokens, int* mask_buffer=NULL) {
  if constexpr(WITHMASK) {
    __shared__ int sm_max_token;
    if(threadIdx.x == 0) sm_max_token = mask_buffer[blockIdx.y]; 
    __syncthreads();
    if(blockIdx.x >= sm_max_token) return;
  }  
  int64_t const token_idx = blockIdx.y * num_tokens + blockIdx.x;
  // Scan for the min and max value for this token
  float max_val = std::numeric_limits<float>::min();
  float min_val = std::numeric_limits<float>::max();
  for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    auto val = static_cast<float>(input[token_idx * hidden_size + i]);
    max_val = std::max(max_val, val);
    min_val = std::min(min_val, val);
  }

  // Reduce the max and min values across the block
  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStorage;
  max_val = BlockReduce(reduceStorage).Reduce(max_val, cub::Max{}, blockDim.x);
  __syncthreads();  // Make sure min doesn't mess with max shared memory
  min_val = BlockReduce(reduceStorage).Reduce(min_val, cub::Min{}, blockDim.x);

  __shared__ scale_type scale_sh;
  __shared__ azp_type azp_sh;

  // Compute the scale and zero point and store them, only on the first thread
  if (threadIdx.x == 0) {
    float const scale_val = (max_val - min_val) / 255.0f;
    // Use rounding to even (same as torch.round)
    auto const azp_float = std::nearbyint(-128.0f - min_val / scale_val);
    auto const azp_val = static_cast<azp_type>(azp_float);

    // Store the scale and azp into shared and global
    scale[token_idx] = scale_sh = scale_val;
    azp[token_idx] = azp_sh = azp_val;
  }

  // Wait for the scale and azp to be computed
  __syncthreads();

  float const scale_val = scale_sh;
  azp_type const azp_val = azp_sh;

  // Quantize the values
  for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    auto const val = static_cast<float>(input[token_idx * hidden_size + i]);
    auto const quant_val =
        int32_to_int8(float_to_int32_rn(val / scale_val) + azp_val);
    out[token_idx * hidden_size + i] = quant_val;
  }
}

template <typename scalar_t, typename scale_type>
__global__ void dynamic_scaled_int8_quant_mask_kernel(
    scalar_t const* __restrict__ input, int8_t* __restrict__ out,
    scale_type* scale, const int hidden_size, const int num_tokens, const int mask_size, const int grid_size, int * mask) {
  int const tid = threadIdx.x;
  __shared__ int sm_mask[16];
  __shared__ int sm_stride[16];
  if(tid < mask_size) {
    sm_mask[tid] = mask[tid];
  }
  __syncthreads();
  if(tid < mask_size) {
    int tmp = 0;
    for(int i = 0; i < tid; i++) {
      tmp += sm_mask[i];
    }
    sm_stride[tid] = tmp;
  }
  __syncthreads();
  int total_tokens = sm_stride[mask_size - 1] + sm_mask[mask_size - 1];
  for(int idx = blockIdx.x; idx < total_tokens; idx += grid_size) {
    int token_id = mask_size - 1;
    while(idx < sm_stride[token_id]) {
      token_id--;
    }
    int64_t const token_idx = token_id * num_tokens + idx - sm_stride[token_id];
    float absmax_val = 0.0f;
    float const zero = 0.0f;

    for (int i = tid; i < hidden_size; i += blockDim.x) {
      float val = static_cast<float>(input[token_idx * hidden_size + i]);
      val = val > zero ? val : -val;
      absmax_val = val > absmax_val ? val : absmax_val;
    }

    using BlockReduce = cub::BlockReduce<float, 1024>;
    __shared__ typename BlockReduce::TempStorage reduceStorage;
    float const block_absmax_val_maybe =
        BlockReduce(reduceStorage).Reduce(absmax_val, cub::Max{}, blockDim.x);
    __shared__ float block_absmax_val;
    if (tid == 0) {
      block_absmax_val = block_absmax_val_maybe;
      scale[token_idx] = block_absmax_val * 0.0078740157;
    }
    __syncthreads();

    float const tmp_scale = 127.0f *__builtin_mxc_rcpf(block_absmax_val);
    for (int i = tid; i < hidden_size; i += blockDim.x) {
      out[token_idx * hidden_size + i] = float_to_int8_rn(
          static_cast<float>(input[token_idx * hidden_size + i]) * tmp_scale);
    }
  }
}

template <typename scalar_t, typename scale_type, typename VT, typename VT1>
__global__ void dynamic_scaled_int8_quant_mask_kernel_sreg_opt(
    scalar_t const* __restrict__ input, int8_t* __restrict__ out,
    scale_type* scale, const int hidden_size, int blockDim_x, int num_tokens, int mask_size, int grid_size, int* mask=NULL) {
  int const tid = threadIdx.x;
  __shared__ int sm_mask[16];
  __shared__ int sm_stride[16];
  if(tid < mask_size) {
    sm_mask[tid] = mask[tid];
  }
  __syncthreads();
  if(tid < mask_size) {
    int tmp = 0;
    for(int i = 0; i < tid; i++) {
      tmp += sm_mask[i];
    }
    sm_stride[tid] = tmp;
  }
  __syncthreads();
  int total_tokens = sm_stride[mask_size - 1] + sm_mask[mask_size - 1];
  for(int idx = blockIdx.x; idx < total_tokens; idx += grid_size) {
      int token_id = mask_size - 1;
      while(idx < sm_stride[token_id]) {
        token_id--;
      }
      int64_t const token_idx = token_id * num_tokens + idx - sm_stride[token_id];
      scalar_t absmax_val = static_cast<scalar_t>(0.0f);
      float const zero = 0.0f;
      constexpr int N = sizeof(VT) / sizeof(scalar_t);
      scalar_t reg_src0[N];
      scalar_t const* ptr_input = input + token_idx * hidden_size;
      int reg_length = blockDim_x * N;
      int length = min(hidden_size, reg_length);
      int index = tid * N;
      if(index < length) {
        *(VT*)reg_src0 = *(VT*)(ptr_input + index);
        #pragma unroll N
        for(int i = 0; i < N; i++) {
          scalar_t val = abs(reg_src0[i]);
          absmax_val = max(absmax_val, val);
        }
      }

      using BlockReduce = cub::BlockReduce<scalar_t, 512>;
      __shared__ typename BlockReduce::TempStorage reduceStorage;
      float const block_absmax_val_maybe =
          BlockReduce(reduceStorage).Reduce(absmax_val, cub::Max{}, blockDim_x);
      __shared__ scale_type block_absmax_val;
      if (tid == 0) {
        block_absmax_val = static_cast<scale_type>(block_absmax_val_maybe);
        scale[token_idx] = static_cast<scale_type>(block_absmax_val * 0.0078740157);
      }
      __syncthreads();
      float const tmp_scale = 127.0f *__builtin_mxc_rcpf(block_absmax_val);
      int8_t* ptr_output = out + token_idx * hidden_size;
      if(index < length) {
        VT1 vdst;
        int8_t* ptr_reg = (int8_t*)&vdst;
        #pragma unroll N
        for(int i = 0; i < N; i++) {
          ptr_reg[i] = float_to_int8_rn(
                static_cast<float>(reg_src0[i]) * tmp_scale);
        }
        *(VT1*)(ptr_output + index) = vdst;
      }
  }
}

template <typename scalar_t, typename scale_type, typename VT, typename VT1>
__global__ void dynamic_scaled_int8_quant_mask_kernel_reg_opt(
    scalar_t const* __restrict__ input, int8_t* __restrict__ out,
    scale_type* scale, const int hidden_size, int blockDim_x, int num_tokens, int mask_size, int grid_size, int* mask=NULL) {
  int const tid = threadIdx.x;
  __shared__ int sm_mask[16];
  __shared__ int sm_stride[16];
  if(tid < mask_size) {
    sm_mask[tid] = mask[tid];
  }
  __syncthreads();
  if(tid < mask_size) {
    int tmp = 0;
    for(int i = 0; i < tid; i++) {
      tmp += sm_mask[i];
    }
    sm_stride[tid] = tmp;
  }
  __syncthreads();
  int total_tokens = sm_stride[mask_size - 1] + sm_mask[mask_size - 1];
  for(int idx = blockIdx.x; idx < total_tokens; idx += grid_size) 
  {
    int token_id = mask_size - 1;
    while(idx < sm_stride[token_id]) {
      token_id--;
    }
    int64_t const token_idx = token_id * num_tokens + idx - sm_stride[token_id];
    scalar_t absmax_val = static_cast<scalar_t>(0.0f);
    float const zero = 0.0f;
    constexpr int N = sizeof(VT) / sizeof(scalar_t);
    scalar_t reg_src0[N];
    scalar_t reg_src1[N];
    scalar_t const* ptr_input = input + token_idx * hidden_size;
    int reg_length = 2 * blockDim_x * N;
    int length = min(hidden_size, reg_length);
    int index = 2 * tid * N;
    if(index < length) {
      *(VT*)reg_src0 = *(VT*)(ptr_input + index);
      *(VT*)reg_src1 = *(VT*)(ptr_input + index + N);
      #pragma unroll N
      for(int i = 0; i < N; i++) {
        scalar_t val = abs(reg_src0[i]);
        absmax_val =  max(val, absmax_val);
        val = abs(reg_src1[i]);
        absmax_val = max(val, absmax_val);
      }
    }

    using BlockReduce = cub::BlockReduce<scalar_t, 512>;
    __shared__ typename BlockReduce::TempStorage reduceStorage;
    scalar_t const block_absmax_val_maybe =
        BlockReduce(reduceStorage).Reduce(absmax_val, cub::Max{}, blockDim_x);
    __shared__ scale_type block_absmax_val;
    if (tid == 0) {
      block_absmax_val = static_cast<scale_type>(block_absmax_val_maybe);
      scale[token_idx] = block_absmax_val * 0.0078740157;
    }
    __syncthreads();
    float const tmp_scale = 127.0f *__builtin_mxc_rcpf(block_absmax_val);
    int8_t* ptr_output = out + token_idx * hidden_size;
    if(index < length) {
      VT1 vdst;
      int8_t* ptr_reg = (int8_t*)&vdst;
      constexpr int ON = 2 * N;
      #pragma unroll N
      for(int i = 0; i < N; i++) {
        ptr_reg[i] = float_to_int8_rn(
              static_cast<float>(reg_src0[i]) * tmp_scale);
      }
      ptr_reg = ptr_reg + N;
      #pragma unroll N
      for(int i = 0; i < N; i++) {
        ptr_reg[i] = float_to_int8_rn(
              static_cast<float>(reg_src1[i]) * tmp_scale);
      }
      *(VT1*)(ptr_output + index) = vdst;
    }
  }
}

template <typename scalar_t, typename scale_type, typename VT, typename VT1>
__global__ void dynamic_scaled_int8_quant_mask_kernel_sm_opt(
    scalar_t const* __restrict__ input, int8_t* __restrict__ out,
    scale_type* scale, const int hidden_size, int blockDim_x, int num_tokens, int mask_size, int grid_size, int* mask=NULL) {
  int const tid = threadIdx.x;
  __shared__ int sm_mask[16];
  __shared__ int sm_stride[16];
  if(tid < mask_size) {
    sm_mask[tid] = mask[tid];
  }
  __syncthreads();
  if(tid < mask_size) {
    int tmp = 0;
    for(int i = 0; i < tid; i++) {
      tmp += sm_mask[i];
    }
    sm_stride[tid] = tmp;
  }
  __syncthreads();
  int total_tokens = sm_stride[mask_size - 1] + sm_mask[mask_size - 1];
  for(int idx = blockIdx.x; idx < total_tokens; idx += grid_size) {
    int token_id = mask_size - 1;
    while(idx < sm_stride[token_id]) {
      token_id--;
    }
    int64_t const token_idx = token_id * num_tokens + idx - sm_stride[token_id];
    float absmax_val = 0.0f;
    float const zero = 0.0f;
    constexpr int N = sizeof(VT) / sizeof(scalar_t);
    int stride = blockDim_x * N;
    __shared__ float sm_buffer[8064];
    scalar_t const* ptr_input = input + token_idx * hidden_size;
    for(int i = tid * N; i < hidden_size; i += stride) {
      VT vsrc = *(VT*)(ptr_input + i);
      scalar_t *ptr_src = (scalar_t*)&vsrc;
      float* ptr_sm_buffer = sm_buffer + i;
      #pragma unroll N
      for(int j = 0; j < N; j++) {
          float val = static_cast<float>(ptr_src[j]);
          ptr_sm_buffer[j] = val;
          val = val > zero ? val : -val;
          absmax_val = val > absmax_val ? val : absmax_val;
      }
    }
    using BlockReduce = cub::BlockReduce<float, 512>;
    __shared__ typename BlockReduce::TempStorage reduceStorage;
    float const block_absmax_val_maybe =
        BlockReduce(reduceStorage).Reduce(absmax_val, cub::Max{}, blockDim.x);
    __shared__ float block_absmax_val;
    if (tid == 0) {
      block_absmax_val = block_absmax_val_maybe;
      scale[token_idx] = block_absmax_val * 0.0078740157;
    }
    
    __syncthreads();

    float const tmp_scale = 127.0f *__builtin_mxc_rcpf(block_absmax_val);
    int8_t* ptr_output = out + token_idx * hidden_size;
    for(int i = tid * N; i < hidden_size; i += stride) {
      VT1 vdst;
      int8_t* ptr_reg = (int8_t*)&vdst;
      float* ptr_sm_buffer = sm_buffer + i;
      #pragma unroll N
      for(int j = 0; j < N; j++) {
          ptr_reg[j] = float_to_int8_rn(
              ptr_sm_buffer[j] * tmp_scale);
      }
      *(VT1*)(ptr_output + i) = vdst;
    }
  }
}

template <typename scalar_t, typename scale_type, typename VT, typename VT1>
__launch_bounds__(1024) __global__ void dynamic_scaled_int8_quant_mask_kernel_opt(
    scalar_t const* __restrict__ input, int8_t* __restrict__ out,
    scale_type* scale, const int hidden_size, const int blockDim_x, const int num_tokens, int mask_size, int grid_size, int* mask=NULL) {
  constexpr int N = sizeof(VT) / sizeof(scalar_t);
  int const tid = threadIdx.x * N;
  __shared__ int sm_mask[16];
  __shared__ int sm_stride[16];
  if(threadIdx.x < mask_size) {
    sm_mask[threadIdx.x] = mask[threadIdx.x];
  }
  __syncthreads();
  if(threadIdx.x < mask_size) {
    int tmp = 0;
    for(int i = 0; i < threadIdx.x; i++) {
      tmp += sm_mask[i];
    }
    sm_stride[threadIdx.x] = tmp;
  }
  __syncthreads();
  int total_tokens = sm_stride[mask_size - 1] + sm_mask[mask_size - 1];
  
  for(int idx = blockIdx.x; idx < total_tokens; idx += grid_size) {
    int token_id = mask_size - 1;
    while(idx < sm_stride[token_id]) {
      token_id--;
    }
    int64_t const token_idx = token_id * num_tokens + idx - sm_stride[token_id];
    float absmax_val = 0.0f;
    int stride = blockDim_x * N;
    const scalar_t * ptr_input = input + token_idx * hidden_size;

    for (int i = tid ; i < hidden_size; i += stride) {
      VT vsrc = *(VT*)(ptr_input + i);
      scalar_t *ptr_src = (scalar_t*)&vsrc;
      #pragma unroll N
      for(int j = 0; j < N; j++) {
          float val = static_cast<float>(ptr_src[j]);
          val = val > 0 ? val : -val;
          absmax_val = val > absmax_val ? val : absmax_val;
      }
    }

      using BlockReduce = cub::BlockReduce<float, 1024>;
    __shared__ typename BlockReduce::TempStorage reduceStorage;
    float const block_absmax_val_maybe =
        BlockReduce(reduceStorage).Reduce(absmax_val, cub::Max{}, blockDim.x);
    __shared__ float block_absmax_val;
    if (tid == 0) {
      block_absmax_val = block_absmax_val_maybe;
      scale[token_idx] = block_absmax_val * 0.0078740157;
    }
    __syncthreads();

    float const tmp_scale = 127.0f *__builtin_mxc_rcpf(block_absmax_val);
    int8_t* ptr_output = out + token_idx * hidden_size;
    for (int i = tid; i < hidden_size; i += stride) {
      VT vsrc = *(VT*)(ptr_input + i);
      VT1 vdst;
      scalar_t *ptr_src = (scalar_t*)&vsrc;
      int8_t* ptr_dst = (int8_t*)&vdst;
      #pragma unroll N
      for(int j = 0; j < N; ++j) {
          ptr_dst[j] = float_to_int8_rn(
          static_cast<float>(ptr_src[j]) * tmp_scale);
      }
      *(VT1*)(ptr_output + i) = vdst;
    }
  }
}

template<typename T, typename T1, typename VT, typename VT1, int NUM_VT> 
__global__ void silu_and_mul_mask_quant_pack(T* input, T* output,T1* mask, int mask_size, int64_t grid_size, int64_t num_tokens, int64_t hidden_size, int64_t out_stirde, int blockDim_x)
{
    constexpr int N = sizeof(VT) / sizeof(T);
    int const tid = threadIdx.x;
    __shared__ T1 sm_mask[16];
    __shared__ T1 sm_stride[16];
    if(tid < mask_size) {
        sm_mask[tid] = mask[tid];
    }

    int64_t hidden_size2 = hidden_size << 1;
    __syncthreads();
    if(tid < mask_size) {
        T1 tmp = 0;
        for(int i = 0; i < tid; i++) {
        tmp += sm_mask[i];
        }
        sm_stride[tid] = tmp;
    }
    int stride = blockDim_x * N;
    __syncthreads();
    int64_t total_tokens = sm_stride[mask_size - 1] + sm_mask[mask_size - 1];

    for(int64_t idx = blockIdx.x; idx < total_tokens; idx += grid_size) {
        float reg_i[NUM_VT][N];
        int64_t token_id = mask_size - 1;
        while(idx < sm_stride[token_id]) {
            token_id--;
        }
        int64_t const token_idx = token_id * num_tokens + idx - sm_stride[token_id];
        const T* ptr_input0 = input + token_idx * hidden_size2;
        const T* ptr_input1 = ptr_input0 + hidden_size;
        float absmax_val = 0.0f;
        for(int i = tid*N, j = 0; i < hidden_size; i += stride, j++) {
            VT vsrc0, vsrc1;
            vsrc0 = *(VT*)(ptr_input0 + i);
            vsrc1 = *(VT*)(ptr_input1 + i);
            T* ptr_local0 = (T*)&vsrc0;
            T* ptr_local1 = (T*)&vsrc1;
            #pragma unroll N
            for(int k = 0; k < N; k++) {
                float val0 = static_cast<float>(ptr_local0[k]);
                float val1 = static_cast<float>(ptr_local1[k]);
                float sigmoid = val0 * __builtin_mxc_rcpf(1.0f + __builtin_expf(-val0));
                float gate_up = val1 * sigmoid;
                reg_i[j][k] = gate_up;
                absmax_val = max(absmax_val, abs(gate_up));
            }
        }

        using BlockReduce = cub::BlockReduce<float, 512>;
        __shared__ typename BlockReduce::TempStorage reduceStorage;
        float const block_absmax_val_maybe =
            BlockReduce(reduceStorage).Reduce(absmax_val, cub::Max{}, blockDim_x);
        __shared__ float block_absmax_val;
        int8_t* ptr_output = (int8_t*)(output + token_idx * out_stirde);
        float* ptr_scale = (float*)(ptr_output + hidden_size);
        if (tid == 0) {
            block_absmax_val = block_absmax_val_maybe;
            ptr_scale[0] = block_absmax_val * 0.0078740157;
        }
        __syncthreads();
        float const tmp_scale = 127.0f * __builtin_mxc_rcpf(block_absmax_val);
        for (int i = tid*N, k = 0; i < hidden_size; i += stride, k++) {
            VT1 vdst;
            int8_t* ptr_dst = (int8_t*)&vdst;
            #pragma unroll N
            for(int j = 0; j < N; ++j) {
                ptr_dst[j] = float_to_int8_rn(reg_i[k][j] * tmp_scale);
            }
            *(VT1*)(ptr_output + i) = vdst;
        }
    }
}

template<typename T, typename T1, typename VT, typename VT1, int NUM_VT, int NUM_THREADS> 
__global__ void silu_and_mul_mask_quant_pack_1mask(T* input, T* output,T1* mask, int grid_size, int64_t num_tokens, int64_t hidden_size, int64_t out_stirde)
{
    constexpr int N = sizeof(VT) / sizeof(T);
    int const tid = threadIdx.x;
    T1 mask_stride;
    mask_stride = mask[0];
    int64_t hidden_size2 = hidden_size << 1;
    int stride = NUM_THREADS * N;
    int total_tokens = mask_stride;

    for(int idx = blockIdx.x; idx < total_tokens; idx += grid_size) {
        float reg_i[NUM_VT][N];
        int64_t const token_idx = idx;
        const T* ptr_input0 = input + token_idx * hidden_size2;
        const T* ptr_input1 = ptr_input0 + hidden_size;
        float absmax_val = 0.0f;
        for(int i = tid*N, j = 0; i < hidden_size; i += stride, j++) {
            VT vsrc0, vsrc1;
            vsrc0 = *(VT*)(ptr_input0 + i);
            vsrc1 = *(VT*)(ptr_input1 + i);
            T* ptr_local0 = (T*)&vsrc0;
            T* ptr_local1 = (T*)&vsrc1;
            #pragma unroll N
            for(int k = 0; k < N; k++) {
                float val0 = static_cast<float>(ptr_local0[k]);
                float val1 = static_cast<float>(ptr_local1[k]);
                float sigmoid = val0 * __builtin_mxc_rcpf(1.0f + __builtin_expf(-val0));
                float gate_up = val1 * sigmoid;
                reg_i[j][k] = gate_up;
                absmax_val = max(absmax_val, abs(gate_up));
            }
        }

        using BlockReduce = cub::BlockReduce<float, NUM_THREADS>;
        __shared__ typename BlockReduce::TempStorage reduceStorage;
        float const block_absmax_val_maybe =
            BlockReduce(reduceStorage).Reduce(absmax_val, cub::Max{}, NUM_THREADS);
        __shared__ float block_absmax_val;
        int8_t* ptr_output = (int8_t*)(output + token_idx * out_stirde);
        float* ptr_scale = (float*)(ptr_output + hidden_size);
        if (tid == 0) {
            block_absmax_val = block_absmax_val_maybe;
            ptr_scale[0] = block_absmax_val * 0.0078740157;
        }
        __syncthreads();
        float const tmp_scale = 127.0f * __builtin_mxc_rcpf(block_absmax_val);
        for (int i = tid*N, k = 0; i < hidden_size; i += stride, k++) {
            VT1 vdst;
            int8_t* ptr_dst = (int8_t*)&vdst;
            #pragma unroll N
            for(int j = 0; j < N; ++j) {
                ptr_dst[j] = float_to_int8_rn(reg_i[k][j] * tmp_scale);
            }
            *(VT1*)(ptr_output + i) = vdst;
        }
    }
}

template<typename T, typename T1, typename VT, typename VT1, typename VMASK_TYPE, int NUM_VT, int NUM_THREADS> 
__global__ void silu_and_mul_mask_quant_pack_2mask(T* input, T* output,T1* mask, int grid_size, int64_t num_tokens, int64_t hidden_size, int64_t out_stirde)
{
    constexpr int N = sizeof(VT) / sizeof(T);
    int const tid = threadIdx.x;
    VMASK_TYPE vmask_reg = *(VMASK_TYPE*)mask;
    T1 mask_stride[2];
    T1* ptr_mask = (T1*)&vmask_reg;
    mask_stride[0] = 0;
    mask_stride[1] = ptr_mask[0];
    int64_t hidden_size2 = hidden_size << 1;
    int stride = NUM_THREADS * N;
    int total_tokens = ptr_mask[0] + ptr_mask[1];

    for(int idx = blockIdx.x; idx < total_tokens; idx += grid_size) {
        float reg_i[NUM_VT][N];
        int64_t token_id = idx < mask_stride[1] ? 0 : 1;;
        int64_t const token_idx = token_id * num_tokens + idx - mask_stride[token_id];
        const T* ptr_input0 = input + token_idx * hidden_size2;
        const T* ptr_input1 = ptr_input0 + hidden_size;
        float absmax_val = 0.0f;
        for(int i = tid*N, j = 0; i < hidden_size; i += stride, j++) {
            VT vsrc0, vsrc1;
            vsrc0 = *(VT*)(ptr_input0 + i);
            vsrc1 = *(VT*)(ptr_input1 + i);
            T* ptr_local0 = (T*)&vsrc0;
            T* ptr_local1 = (T*)&vsrc1;
            #pragma unroll N
            for(int k = 0; k < N; k++) {
                float val0 = static_cast<float>(ptr_local0[k]);
                float val1 = static_cast<float>(ptr_local1[k]);
                float sigmoid = val0 * __builtin_mxc_rcpf(1.0f + __builtin_expf(-val0));
                float gate_up = val1 * sigmoid;
                reg_i[j][k] = gate_up;
                absmax_val = max(absmax_val, abs(gate_up));
            }
        }

        using BlockReduce = cub::BlockReduce<float, NUM_THREADS>;
        __shared__ typename BlockReduce::TempStorage reduceStorage;
        float const block_absmax_val_maybe =
            BlockReduce(reduceStorage).Reduce(absmax_val, cub::Max{}, NUM_THREADS);
        __shared__ float block_absmax_val;
        int8_t* ptr_output = (int8_t*)(output + token_idx * out_stirde);
        float* ptr_scale = (float*)(ptr_output + hidden_size);
        if (tid == 0) {
            block_absmax_val = block_absmax_val_maybe;
            ptr_scale[0] = block_absmax_val * 0.0078740157;
        }
        __syncthreads();
        float const tmp_scale = 127.0f * __builtin_mxc_rcpf(block_absmax_val);
        for (int i = tid*N, k = 0; i < hidden_size; i += stride, k++) {
            VT1 vdst;
            int8_t* ptr_dst = (int8_t*)&vdst;
            #pragma unroll N
            for(int j = 0; j < N; ++j) {
                ptr_dst[j] = float_to_int8_rn(reg_i[k][j] * tmp_scale);
            }
            *(VT1*)(ptr_output + i) = vdst;
        }
    }
}

template<typename T, typename VT, typename VT1, int NUM_VT> 
__global__ void silu_and_mul_quant(T* input, int8_t* output, float* scale, int64_t hidden_size, int blockDim_x)
{
    constexpr int N = sizeof(VT) / sizeof(T);
    int const tid = threadIdx.x;
    int stride = blockDim_x * N;
    int64_t const token_idx = blockIdx.x;
    int64_t hidden_size2 = hidden_size << 1;
    const T* ptr_input0 = input + token_idx * hidden_size2;
    const T* ptr_input1 = ptr_input0 + hidden_size;
    float absmax_val = 0.0f;
    float reg_i[NUM_VT][N];
    for(int i = tid*N, j = 0; i < hidden_size; i += stride, j++) {
        VT vsrc0, vsrc1;
        vsrc0 = *(VT*)(ptr_input0 + i);
        vsrc1 = *(VT*)(ptr_input1 + i);
        T* ptr_local0 = (T*)&vsrc0;
        T* ptr_local1 = (T*)&vsrc1;
        #pragma unroll N
        for(int k = 0; k < N; k++) {
            float val0 = static_cast<float>(ptr_local0[k]);
            float val1 = static_cast<float>(ptr_local1[k]);
            float sigmoid = val0 * __builtin_mxc_rcpf(1.0f + __builtin_expf(-val0));
            float gate_up = val1 * sigmoid;
            reg_i[j][k] = gate_up;
            absmax_val = max(absmax_val, abs(gate_up));
        }
    }

    using BlockReduce = cub::BlockReduce<float, 512>;
    __shared__ typename BlockReduce::TempStorage reduceStorage;
    float const block_absmax_val_maybe =
        BlockReduce(reduceStorage).Reduce(absmax_val, cub::Max{}, blockDim_x);
    __shared__ float block_absmax_val;
    int8_t* ptr_output = (int8_t*)(output + token_idx * hidden_size);
    if (tid == 0) {
        block_absmax_val = block_absmax_val_maybe;
        scale[token_idx] = block_absmax_val * 0.0078740157;
    }
    __syncthreads();
    float const tmp_scale = 127.0f * __builtin_mxc_rcpf(block_absmax_val);
    for (int i = tid*N, k = 0; i < hidden_size; i += stride, k++) {
        VT1 vdst;
        int8_t* ptr_dst = (int8_t*)&vdst;
        #pragma unroll N
        for(int j = 0; j < N; ++j) {
            ptr_dst[j] = float_to_int8_rn(reg_i[k][j] * tmp_scale);
        }
        *(VT1*)(ptr_output + i) = vdst;
    }
}

template<typename T, typename VT, typename VT1, int NUM_VT> 
__global__ void silu_and_mul_sm_quant(T* input, int8_t* output, float* scale, int64_t hidden_size, int blockDim_x)
{
    constexpr int N = sizeof(VT) / sizeof(T);
    int const tid = threadIdx.x;
    int stride = blockDim_x * N;
    int64_t const token_idx = blockIdx.x;
    int64_t hidden_size2 = hidden_size << 1;
    const T* ptr_input0 = input + token_idx * hidden_size2;
    const T* ptr_input1 = ptr_input0 + hidden_size;
    float absmax_val = 0.0f;
    float reg_i[4][N];
    __shared__ float sm_gate[4096];
    int hidden_size1 = stride * 4;
    for(int i = tid*N, j = 0; i < hidden_size1; i += stride, j++) {
        VT vsrc0, vsrc1;
        vsrc0 = *(VT*)(ptr_input0 + i);
        vsrc1 = *(VT*)(ptr_input1 + i);
        T* ptr_local0 = (T*)&vsrc0;
        T* ptr_local1 = (T*)&vsrc1;
        #pragma unroll N
        for(int k = 0; k < N; k++) {
            float val0 = static_cast<float>(ptr_local0[k]);
            float val1 = static_cast<float>(ptr_local1[k]);
            float sigmoid = val0 * __builtin_mxc_rcpf(1.0f + __builtin_expf(-val0));
            float gate_up = val1 * sigmoid;
            reg_i[j][k] = gate_up;
            absmax_val = max(absmax_val, abs(gate_up));
        }
    }
    const T* ptr_input2 = ptr_input0 + hidden_size1;
    const T* ptr_input3 = ptr_input1 + hidden_size1;
    int remain_hidden_size = hidden_size - hidden_size1;
    for(int i = tid*N; i < remain_hidden_size; i += stride) {
        VT vsrc0, vsrc1;
        vsrc0 = *(VT*)(ptr_input2 + i);
        vsrc1 = *(VT*)(ptr_input3 + i);
        T* ptr_local0 = (T*)&vsrc0;
        T* ptr_local1 = (T*)&vsrc1;
        float* ptr_sm = sm_gate + i;
        #pragma unroll N
        for(int k = 0; k < N; k++) {
            float val0 = static_cast<float>(ptr_local0[k]);
            float val1 = static_cast<float>(ptr_local1[k]);
            float sigmoid = val0 * __builtin_mxc_rcpf(1.0f + __builtin_expf(-val0));
            float gate_up = val1 * sigmoid;
            ptr_sm[k] = gate_up;
            absmax_val = max(absmax_val, abs(gate_up));
        }
    }

    using BlockReduce = cub::BlockReduce<float, 512>;
    __shared__ typename BlockReduce::TempStorage reduceStorage;
    float const block_absmax_val_maybe =
        BlockReduce(reduceStorage).Reduce(absmax_val, cub::Max{}, blockDim_x);
    __shared__ float block_absmax_val;
    int8_t* ptr_output = (int8_t*)(output + token_idx * hidden_size);
    if (tid == 0) {
        block_absmax_val = block_absmax_val_maybe;
        scale[token_idx] = block_absmax_val * 0.0078740157;
    }
    __syncthreads();
    float const tmp_scale = 127.0f * __builtin_mxc_rcpf(block_absmax_val);
    for (int i = tid*N, k = 0; i < hidden_size1; i += stride, k++) {
        VT1 vdst;
        int8_t* ptr_dst = (int8_t*)&vdst;
        #pragma unroll N
        for(int j = 0; j < N; ++j) {
            ptr_dst[j] = float_to_int8_rn(reg_i[k][j] * tmp_scale);
        }
        *(VT1*)(ptr_output + i) = vdst;
    }

    ptr_output = ptr_output + hidden_size1;
    for(int i = tid*N; i < remain_hidden_size; i += stride) {
        VT1 vdst;
        int8_t* ptr_dst = (int8_t*)&vdst;
        float* ptr_sm = sm_gate + i;
        #pragma unroll N
        for(int j = 0; j < N; ++j) {
            ptr_dst[j] = float_to_int8_rn(ptr_sm[j] * tmp_scale);
        }
        *(VT1*)(ptr_output + i) = vdst;
    }
}

template<typename T, typename T1>
void launch_silu_mul_quant_pack(T* input, T* output, T1* mask, int64_t num_tokens, int64_t hidden_size, int64_t out_stride, int64_t mask_size,cudaStream_t stream) {
    int dev = 0;
    cudaGetDevice(&dev);
    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);
    int gridsize = sm_count;
    int64_t inner_hidden_size = hidden_size / 2;
    int blocksize = 512;
    int N = sizeof(float4) / sizeof(T);
    if(mask_size == 1 && N == 8&&(inner_hidden_size & (N - 1)) == 0 && (out_stride & (N -1)) == 0) {
        int base = blocksize * N;
        if(inner_hidden_size <= 64*N) {
          gridsize = gridsize * 8;
          silu_and_mul_mask_quant_pack_1mask<T, T1, float4, float2, 1, 64><<<gridsize, 64,0,stream>>>(input, output, mask, gridsize, num_tokens, inner_hidden_size, out_stride);
        } else if(inner_hidden_size <= 128*N) {
          gridsize = gridsize * 4;
          silu_and_mul_mask_quant_pack_1mask<T, T1, float4, float2, 1, 128><<<gridsize, 128,0,stream>>>(input, output, mask, gridsize, num_tokens, inner_hidden_size, out_stride);
        } else if(inner_hidden_size <= 256*N) {
          gridsize = gridsize * 2;
          silu_and_mul_mask_quant_pack_1mask<T, T1, float4, float2, 1, 256><<<gridsize, 256,0,stream>>>(input, output, mask, gridsize, num_tokens, inner_hidden_size, out_stride);
        } else if(inner_hidden_size <= base) {
            silu_and_mul_mask_quant_pack_1mask<T, T1, float4, float2, 1, 512><<<gridsize, 512,0,stream>>>(input, output, mask, gridsize, num_tokens, inner_hidden_size, out_stride);
        } else if(inner_hidden_size <= base*2) {
            silu_and_mul_mask_quant_pack_1mask<T, T1, float4, float2, 2, 512><<<gridsize, 512,0,stream>>>(input, output, mask, gridsize, num_tokens, inner_hidden_size, out_stride);
        } else if(inner_hidden_size <= base * 3) {
            silu_and_mul_mask_quant_pack_1mask<T, T1, float4, float2, 3, 512><<<gridsize, 512,0,stream>>>(input, output, mask, gridsize, num_tokens, inner_hidden_size, out_stride);
        } else if(inner_hidden_size <= base * 4) {
            silu_and_mul_mask_quant_pack_1mask<T, T1, float4, float2, 4, 512><<<gridsize, 512,0,stream>>>(input, output, mask, gridsize, num_tokens, inner_hidden_size, out_stride);
        } else {
            assert(0);
        }
    } else if(mask_size == 2 && (N == 8&&(inner_hidden_size & (N - 1)) == 0 && (out_stride & (N -1)) == 0)) {
        int base = blocksize * N;
        if(sizeof(T1) == 4) {
          if(inner_hidden_size <= 64 * N) {
            gridsize = gridsize * 8;
            silu_and_mul_mask_quant_pack_2mask<T, T1, float4, float2, float2, 1, 64><<<gridsize, 64,0,stream>>>(input, output, mask, gridsize, num_tokens, inner_hidden_size, out_stride);
          } else if(inner_hidden_size <= 128 * N){
            gridsize = gridsize * 4;
            silu_and_mul_mask_quant_pack_2mask<T, T1, float4, float2, float2, 1, 128><<<gridsize, 128,0,stream>>>(input, output, mask, gridsize, num_tokens, inner_hidden_size, out_stride);
          } else if(inner_hidden_size <= 256 * N) {
            gridsize = gridsize * 2;
            silu_and_mul_mask_quant_pack_2mask<T, T1, float4, float2, float2, 1, 256><<<gridsize, 256,0,stream>>>(input, output, mask, gridsize, num_tokens, inner_hidden_size, out_stride);
          } else if(inner_hidden_size <= base) {
              silu_and_mul_mask_quant_pack_2mask<T, T1, float4, float2, float2, 1, 512><<<gridsize, 512,0,stream>>>(input, output, mask, gridsize, num_tokens, inner_hidden_size, out_stride);
          } else if(inner_hidden_size <= base*2) {
              silu_and_mul_mask_quant_pack_2mask<T, T1, float4, float2, float2, 2, 512><<<gridsize, 512,0,stream>>>(input, output, mask, gridsize, num_tokens, inner_hidden_size, out_stride);
          } else if(inner_hidden_size <= base * 3) {
              silu_and_mul_mask_quant_pack_2mask<T, T1, float4, float2, float2, 3, 512><<<gridsize, 512,0,stream>>>(input, output, mask, gridsize, num_tokens, inner_hidden_size, out_stride);
          } else if(inner_hidden_size <= base * 4) {
              silu_and_mul_mask_quant_pack_2mask<T, T1, float4, float2, float2, 4, 512><<<gridsize, 512,0,stream>>>(input, output, mask, gridsize, num_tokens, inner_hidden_size, out_stride);
          } else {
              assert(0);
          }
        } else if(sizeof(T1) == 8) {
          if(inner_hidden_size <= 64 * N) {
            gridsize = gridsize * 8;
            silu_and_mul_mask_quant_pack_2mask<T, T1, float4, float2, float4, 1, 64><<<gridsize, 64,0,stream>>>(input, output, mask, gridsize, num_tokens, inner_hidden_size, out_stride);
          } else if(inner_hidden_size <= 128 * N) {
            gridsize = gridsize * 4;
            silu_and_mul_mask_quant_pack_2mask<T, T1, float4, float2, float4, 1, 128><<<gridsize, 128,0,stream>>>(input, output, mask, gridsize, num_tokens, inner_hidden_size, out_stride);
          } else if(inner_hidden_size <= 256 * N) {
            gridsize = gridsize * 2;
            silu_and_mul_mask_quant_pack_2mask<T, T1, float4, float2, float4, 1, 256><<<gridsize, 256,0,stream>>>(input, output, mask, gridsize, num_tokens, inner_hidden_size, out_stride);
          } else if(inner_hidden_size <= base) {
              silu_and_mul_mask_quant_pack_2mask<T, T1, float4, float2, float4, 1, 512><<<gridsize, 512,0,stream>>>(input, output, mask, gridsize, num_tokens, inner_hidden_size, out_stride);
          } else if(inner_hidden_size <= base*2) {
              silu_and_mul_mask_quant_pack_2mask<T, T1, float4, float2, float4, 2, 512><<<gridsize, 512,0,stream>>>(input, output, mask, gridsize, num_tokens, inner_hidden_size, out_stride);
          } else if(inner_hidden_size <= base * 3) {
              silu_and_mul_mask_quant_pack_2mask<T, T1, float4, float2, float4, 3,512><<<gridsize, 512,0,stream>>>(input, output, mask, gridsize, num_tokens, inner_hidden_size, out_stride);
          } else if(inner_hidden_size <= base * 4) {
              silu_and_mul_mask_quant_pack_2mask<T, T1, float4, float2, float4, 4, 512><<<gridsize, 512,0,stream>>>(input, output, mask, gridsize, num_tokens, inner_hidden_size, out_stride);
          } else {
              assert(0);
          }
        }
        
    } else if(N == 8&&(inner_hidden_size & (N - 1)) == 0 && (out_stride & (N -1)) == 0) {
        int base = blocksize * N;
        if(inner_hidden_size <= base) {
            silu_and_mul_mask_quant_pack<T, T1, float4, float2, 1><<<gridsize, blocksize,0,stream>>>(input, output, mask, mask_size, gridsize, num_tokens, inner_hidden_size, out_stride, blocksize);
        } else if(inner_hidden_size <= base*2) {
            silu_and_mul_mask_quant_pack<T, T1, float4, float2, 2><<<gridsize, blocksize,0,stream>>>(input, output, mask, mask_size, gridsize, num_tokens, inner_hidden_size, out_stride, blocksize);
        } else if(inner_hidden_size <= base * 3) {
            silu_and_mul_mask_quant_pack<T, T1, float4, float2, 3><<<gridsize, blocksize,0,stream>>>(input, output, mask, mask_size, gridsize, num_tokens, inner_hidden_size, out_stride, blocksize);
        } else if(inner_hidden_size <= base * 4) {
            silu_and_mul_mask_quant_pack<T, T1, float4, float2, 4><<<gridsize, blocksize,0,stream>>>(input, output, mask, mask_size, gridsize, num_tokens, inner_hidden_size, out_stride, blocksize);
        } else {
            assert(0);
        }
    } else {
        assert(0);
    }
}

template<typename T>
void launch_silu_mul_quan_no_mask(T* input, int8_t* output, float* scale, int64_t num_tokens, int64_t hidden_size,cudaStream_t stream) {
    int64_t inner_hidden_size = hidden_size / 2;
    int blocksize = 512;
    int N = sizeof(float4) / sizeof(T);
    if(N == 8&&(inner_hidden_size & (N - 1)) == 0) {
        int base = blocksize * N;
        if(inner_hidden_size <= base) {
            silu_and_mul_quant<T, float4, float2, 1><<<num_tokens, blocksize,0,stream>>>(input, output, scale, inner_hidden_size, blocksize);
        } else if(inner_hidden_size <= base*2) {
            silu_and_mul_quant<T, float4, float2, 2><<<num_tokens, blocksize,0,stream>>>(input, output, scale, inner_hidden_size, blocksize);
        } else if(inner_hidden_size <= base * 3) {
            silu_and_mul_quant<T, float4, float2, 3><<<num_tokens, blocksize,0,stream>>>(input, output, scale, inner_hidden_size, blocksize);
        } else if(inner_hidden_size <= base * 4) {
            silu_and_mul_quant<T, float4, float2, 4><<<num_tokens, blocksize,0,stream>>>(input, output, scale, inner_hidden_size, blocksize);
        } else if(inner_hidden_size <= base*4 + 4096) {
            silu_and_mul_sm_quant<T, float4, float2, 4><<<num_tokens, blocksize,0,stream>>>(input, output, scale, inner_hidden_size, blocksize);
        } else {
            printf("silu_and_mul_quant not support\n");
            assert(0);
        }
    } else if(N == 4 && (inner_hidden_size & (N - 1)) == 0) {
        int base = blocksize * N;
        if(inner_hidden_size <= base) {
            silu_and_mul_quant<T, float4, float, 1><<<num_tokens, blocksize,0,stream>>>(input, output, scale, inner_hidden_size, blocksize);
        } else if(inner_hidden_size <= base*2) {
            silu_and_mul_quant<T, float4, float, 2><<<num_tokens, blocksize,0,stream>>>(input, output, scale, inner_hidden_size, blocksize);
        } else if(inner_hidden_size <= base * 3) {
            silu_and_mul_quant<T, float4, float, 3><<<num_tokens, blocksize,0,stream>>>(input, output, scale, inner_hidden_size, blocksize);
        } else if(inner_hidden_size <= base * 4) {
            silu_and_mul_quant<T, float4, float, 4><<<num_tokens, blocksize,0,stream>>>(input, output, scale, inner_hidden_size, blocksize);
        } else if(inner_hidden_size <= base * 8) {
            silu_and_mul_quant<T, float4, float, 8><<<num_tokens, blocksize,0,stream>>>(input, output, scale, inner_hidden_size, blocksize);
        } else {
            printf("silu_and_mul_quant not support\n");
            assert(0);
        }
    } else {
        assert(0);
    }
}

}  // namespace vllm

void static_scaled_int8_quant(torch::Tensor& out,          // [..., hidden_size]
                              torch::Tensor const& input,  // [..., hidden_size]
                              torch::Tensor const& scale,
                              c10::optional<torch::Tensor> const& azp) {
  TORCH_CHECK(input.is_contiguous());
  TORCH_CHECK(out.is_contiguous());
  TORCH_CHECK(scale.numel() == 1);
  TORCH_CHECK(!azp || azp->numel() == 1);

  int const hidden_size = input.size(-1);
  int const num_tokens = input.numel() / hidden_size;
  dim3 const grid(num_tokens);
  dim3 const block(std::min(hidden_size, 1024));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "static_scaled_int8_quant_kernel", [&] {
        if (!azp) {
          vllm::static_scaled_int8_quant_kernel<scalar_t, float>
              <<<grid, block, 0, stream>>>(
                  input.data_ptr<scalar_t>(), out.data_ptr<int8_t>(),
                  scale.data_ptr<float>(), hidden_size);
        } else {
          vllm::static_scaled_int8_azp_quant_kernel<scalar_t, float, int32_t>
              <<<grid, block, 0, stream>>>(
                  input.data_ptr<scalar_t>(), out.data_ptr<int8_t>(),
                  scale.data_ptr<float>(), azp->data_ptr<int32_t>(),
                  hidden_size);
        }
      });
}

void dynamic_scaled_int8_quant(
    torch::Tensor& out,          // [..., hidden_size]
    torch::Tensor const& input,  // [..., hidden_size]
    torch::Tensor& scales, c10::optional<torch::Tensor> const& azp) {
  TORCH_CHECK(input.is_contiguous());
  TORCH_CHECK(out.is_contiguous());
  TORCH_CHECK(scales.is_contiguous());
  TORCH_CHECK(!azp || azp->is_contiguous());

  int const hidden_size = input.size(-1);
  int const num_tokens = input.numel() / hidden_size;
  dim3 const grid(num_tokens,1,1);
  dim3 const block(std::min(hidden_size, 1024));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "dynamic_scaled_int8_quant_kernel", [&] {
        if (!azp) {
          int n = 16 / sizeof(scalar_t);
          if(hidden_size <= 4096 && ((hidden_size & (n - 1)) == 0) && n == 8) {
            if(hidden_size > 256*n) {
              vllm::dynamic_scaled_int8_quant_kernel_sreg_opt<scalar_t, float, float4, float2, 512, false><<<grid, 512, 0, stream>>>(input.data_ptr<scalar_t>(),out.data_ptr<int8_t>(),scales.data_ptr<float>(),hidden_size, num_tokens);
            } else if(hidden_size > 128*n) {
              vllm::dynamic_scaled_int8_quant_kernel_sreg_opt<scalar_t, float, float4, float2, 256, false><<<grid, 256, 0, stream>>>(input.data_ptr<scalar_t>(),out.data_ptr<int8_t>(),scales.data_ptr<float>(),hidden_size, num_tokens);
            } else if(hidden_size > 64 * n) {
              vllm::dynamic_scaled_int8_quant_kernel_sreg_opt<scalar_t, float, float4, float2, 128, false><<<grid, 128, 0, stream>>>(input.data_ptr<scalar_t>(),out.data_ptr<int8_t>(),scales.data_ptr<float>(),hidden_size, num_tokens);
            } else {
              vllm::dynamic_scaled_int8_quant_kernel_sreg_opt<scalar_t, float, float4, float2, 64, false><<<grid, 64, 0, stream>>>(input.data_ptr<scalar_t>(),out.data_ptr<int8_t>(),scales.data_ptr<float>(),hidden_size, num_tokens);
            }
          } else if(hidden_size > 4096 &&hidden_size <= 8192 && ((hidden_size & (2*n - 1)) == 0) && n == 8) {
            int blocksize = 512;
            vllm::dynamic_scaled_int8_quant_kernel_reg_opt<scalar_t, float, float4, float4, false><<<grid, blocksize, 0, stream>>>(input.data_ptr<scalar_t>(),out.data_ptr<int8_t>(),scales.data_ptr<float>(),hidden_size,blocksize, num_tokens);
          } else if(hidden_size <= 8064 && (hidden_size & (n - 1)) == 0 && n == 8) {
            int blocksize = 512;
            vllm::dynamic_scaled_int8_quant_kernel_sm_opt<scalar_t, float, float4, float2,false><<<grid, blocksize, 0, stream>>>(
              input.data_ptr<scalar_t>(),out.data_ptr<int8_t>(),scales.data_ptr<float>(),hidden_size,blocksize,num_tokens);
          } else if(hidden_size >= 16384 && hidden_size <= 18432 && (hidden_size & (n - 1)) == 0 && n == 8) {
            vllm::dynamic_scaled_int8_quant_kernel_lh_opt<scalar_t, float, float4, float2, 3,1024, false><<<grid, 1024, 0, stream>>>(input.data_ptr<scalar_t>(),out.data_ptr<int8_t>(),scales.data_ptr<float>(),hidden_size, num_tokens);
          } else if (hidden_size > 8064 && ((hidden_size & (n - 1)) == 0 && n == 8)) {
            int blocksize = 1024;
            vllm::dynamic_scaled_int8_quant_kernel_opt<scalar_t, float,float4,float2, false>
                    <<<grid, blocksize, 0, stream>>>(
                        input.data_ptr<scalar_t>(),out.data_ptr<int8_t>(),scales.data_ptr<float>(),hidden_size,blocksize, num_tokens);
          } else {
              vllm::dynamic_scaled_int8_quant_kernel<scalar_t, float, false>
                  <<<grid, block, 0, stream>>>(
                      input.data_ptr<scalar_t>(), out.data_ptr<int8_t>(),
                      scales.data_ptr<float>(), hidden_size, num_tokens);
          }
        } else {
          vllm::dynamic_scaled_int8_azp_quant_kernel<scalar_t, float, int32_t, false>
              <<<grid, block, 0, stream>>>(
                  input.data_ptr<scalar_t>(), out.data_ptr<int8_t>(),
                  scales.data_ptr<float>(), azp->data_ptr<int32_t>(),
                  hidden_size, num_tokens);
        }
      });
}

void dynamic_scaled_int8_mask_quant(
    torch::Tensor& out,          // [..., hidden_size]
    torch::Tensor const& input,  // [..., hidden_size]
    torch::Tensor const &mask,
    torch::Tensor& scales, c10::optional<torch::Tensor> const& azp) {
  TORCH_CHECK(input.is_contiguous());
  TORCH_CHECK(out.is_contiguous());
  TORCH_CHECK(scales.is_contiguous());
  TORCH_CHECK(mask.is_contiguous());
  TORCH_CHECK(!azp || azp->is_contiguous());

  int const hidden_size = input.size(-1);
  int const num_tokens = input.numel() / hidden_size;
  int const mask_size = mask.numel();
  int const num_tokens_batch = num_tokens / mask_size;
  dim3 const grid(num_tokens_batch, mask_size, 1);
  dim3 const block(std::min(hidden_size, 1024));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "dynamic_scaled_int8_quant_kernel_mask", [&] {
        if (!azp) {
          int n = 16 / sizeof(scalar_t);
          if(mask_size < 16) {
            int dev = 0;
            cudaGetDevice(&dev);
            int sm_count = 0;
            cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);
            int gridsize = sm_count;
            if(hidden_size <= 4096 && ((hidden_size & (n - 1)) == 0) && n == 8) {
              int blocksize = 512;
              vllm::dynamic_scaled_int8_quant_mask_kernel_sreg_opt<scalar_t, float, float4, float2><<<gridsize, blocksize, 0, stream>>>(input.data_ptr<scalar_t>(),out.data_ptr<int8_t>(),scales.data_ptr<float>(),hidden_size,blocksize,num_tokens_batch,mask_size,gridsize,mask.data_ptr<int>());
            } else if(hidden_size > 4096 &&hidden_size <= 8192 && ((hidden_size & (2*n - 1)) == 0) && n == 8) {
                int blocksize = 512;
              vllm::dynamic_scaled_int8_quant_mask_kernel_reg_opt<scalar_t, float, float4, float4><<<gridsize, blocksize, 0, stream>>>(input.data_ptr<scalar_t>(),out.data_ptr<int8_t>(),scales.data_ptr<float>(),hidden_size,blocksize,num_tokens_batch,mask_size, gridsize,mask.data_ptr<int>());
            } else if(hidden_size <= 8064 && (hidden_size & (n - 1)) == 0 && n == 8) {
                int blocksize = 512;
                vllm::dynamic_scaled_int8_quant_mask_kernel_sm_opt<scalar_t, float, float4, float2><<<gridsize, blocksize, 0, stream>>>(input.data_ptr<scalar_t>(),out.data_ptr<int8_t>(),scales.data_ptr<float>(),hidden_size,blocksize,num_tokens_batch,mask_size, gridsize, mask.data_ptr<int>());
            } else if(hidden_size > 8064 && ((hidden_size & (n - 1)) == 0 && n == 8)){
                int blocksize = 1024;
                vllm::dynamic_scaled_int8_quant_mask_kernel_opt<scalar_t, float,float4,float2>
                        <<<gridsize, blocksize, 0, stream>>>(
                            input.data_ptr<scalar_t>(),out.data_ptr<int8_t>(),scales.data_ptr<float>(),hidden_size,blocksize,num_tokens_batch,mask_size, gridsize, mask.data_ptr<int>());
            } else {
              vllm::dynamic_scaled_int8_quant_mask_kernel<scalar_t, float>
                    <<<gridsize, block, 0, stream>>>(
                        input.data_ptr<scalar_t>(), out.data_ptr<int8_t>(),
                        scales.data_ptr<float>(), hidden_size, num_tokens_batch, mask_size, gridsize, mask.data_ptr<int>());
            }
          } else {
            if(hidden_size <= 4096 && ((hidden_size & (n - 1)) == 0) && n == 8) {
              if(hidden_size > 256*n) {
                vllm::dynamic_scaled_int8_quant_kernel_sreg_opt<scalar_t, float, float4, float2,512,true><<<grid, 512, 0, stream>>>(input.data_ptr<scalar_t>(),out.data_ptr<int8_t>(),scales.data_ptr<float>(),hidden_size, num_tokens_batch, mask.data_ptr<int>());
              } else if(hidden_size > 128*n) {
                vllm::dynamic_scaled_int8_quant_kernel_sreg_opt<scalar_t, float, float4, float2,256,true><<<grid, 256, 0, stream>>>(input.data_ptr<scalar_t>(),out.data_ptr<int8_t>(),scales.data_ptr<float>(),hidden_size, num_tokens_batch, mask.data_ptr<int>());
              } else if(hidden_size > 64*n) {
                vllm::dynamic_scaled_int8_quant_kernel_sreg_opt<scalar_t, float, float4, float2,128,true><<<grid, 128, 0, stream>>>(input.data_ptr<scalar_t>(),out.data_ptr<int8_t>(),scales.data_ptr<float>(),hidden_size, num_tokens_batch, mask.data_ptr<int>());
              } else {
                vllm::dynamic_scaled_int8_quant_kernel_sreg_opt<scalar_t, float, float4, float2,64,true><<<grid, 64, 0, stream>>>(input.data_ptr<scalar_t>(),out.data_ptr<int8_t>(),scales.data_ptr<float>(),hidden_size, num_tokens_batch, mask.data_ptr<int>());
              }
            } else if(hidden_size > 4096 &&hidden_size <= 8192 && ((hidden_size & (2*n - 1)) == 0) && n == 8) {
              int blocksize = 512;
              vllm::dynamic_scaled_int8_quant_kernel_reg_opt<scalar_t, float, float4, float4, true><<<grid, blocksize, 0, stream>>>(input.data_ptr<scalar_t>(),out.data_ptr<int8_t>(),scales.data_ptr<float>(),hidden_size,blocksize,num_tokens_batch,mask.data_ptr<int>());
            } else if(hidden_size <= 8064 && (hidden_size & (n - 1)) == 0 && n == 8) {
              int blocksize = 512;
              vllm::dynamic_scaled_int8_quant_kernel_sm_opt<scalar_t, float, float4, float2, true><<<grid, blocksize, 0, stream>>>(
                input.data_ptr<scalar_t>(),out.data_ptr<int8_t>(),scales.data_ptr<float>(),hidden_size,blocksize, num_tokens_batch,mask.data_ptr<int>());
            } else if (hidden_size > 8064 && ((hidden_size & (n - 1)) == 0 && n == 8)) {
              int blocksize = 1024;
              vllm::dynamic_scaled_int8_quant_kernel_opt<scalar_t, float,float4,float2, true>
                      <<<grid, blocksize, 0, stream>>>(
                          input.data_ptr<scalar_t>(),out.data_ptr<int8_t>(),scales.data_ptr<float>(),hidden_size,blocksize, num_tokens_batch, mask.data_ptr<int>());
            } else {
                vllm::dynamic_scaled_int8_quant_kernel<scalar_t, float,true>
                    <<<grid, block, 0, stream>>>(
                        input.data_ptr<scalar_t>(), out.data_ptr<int8_t>(),
                        scales.data_ptr<float>(), hidden_size, num_tokens_batch, mask.data_ptr<int>());
            }
          }
        } else {
          vllm::dynamic_scaled_int8_azp_quant_kernel<scalar_t, float, int32_t,true>
              <<<grid, block, 0, stream>>>(
                  input.data_ptr<scalar_t>(), out.data_ptr<int8_t>(),
                  scales.data_ptr<float>(), azp->data_ptr<int32_t>(),
                  hidden_size, num_tokens_batch, mask.data_ptr<int>());
        }
      });
}


void fused_silu_mul_dq_mask_quant_pack(
    torch::Tensor& out,          
    torch::Tensor const& input, 
    torch::Tensor const &mask)
{
  TORCH_CHECK(input.is_contiguous());
  TORCH_CHECK(out.is_contiguous());
  TORCH_CHECK(mask.is_contiguous());
  int64_t const hidden_size = input.size(-1);
  int64_t const num_tokens = input.numel() / hidden_size;
  int64_t const mask_size = mask.numel();
  int64_t const num_tokens_batch = num_tokens / mask_size;
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  int64_t out_stride = ((hidden_size/4 + 2) + 255)/ 256 * 256;
  switch(mask.element_size()) {
    case 8:
      VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "launch_silu_mul_quant_pack", [&] {
        vllm::launch_silu_mul_quant_pack<scalar_t, int64_t>(input.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), mask.data_ptr<int64_t>(), num_tokens_batch, hidden_size, out_stride, mask_size, stream);
      });
    break;
    case 4:
       VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "launch_silu_mul_quant_pack", [&] {
        vllm::launch_silu_mul_quant_pack<scalar_t, int32_t>(input.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), mask.data_ptr<int32_t>(), num_tokens_batch, hidden_size, out_stride, mask_size, stream);
      });
      break;
    default:
    return;
  }
}

void fused_silu_mul_dq_quant_interface(
    torch::Tensor& out,
    torch::Tensor& scale,   
    torch::Tensor const& input)
{
  TORCH_CHECK(input.is_contiguous());
  TORCH_CHECK(scale.is_contiguous());
  TORCH_CHECK(out.is_contiguous());
  int64_t const hidden_size = input.size(-1);
  int64_t const num_tokens = input.numel() / hidden_size;
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "launch_silu_mul_quan_no_mask", [&] {
    vllm::launch_silu_mul_quan_no_mask<scalar_t>(input.data_ptr<scalar_t>(), out.data_ptr<int8_t>(), scale.data_ptr<float>(), num_tokens, hidden_size, stream);
  });
}