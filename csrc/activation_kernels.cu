// 2025 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>

#include <cmath>

#include "cuda_compat.h"
#include "dispatch_utils.h"

namespace vllm {

template <typename scalar_t, scalar_t (*ACT_FN)(const scalar_t&),
          bool act_first>
__device__ __forceinline__ scalar_t compute(const scalar_t& x,
                                            const scalar_t& y) {
  return act_first ? ACT_FN(x) * y : x * ACT_FN(y);
}
// Activation and gating kernel template.

template <typename scalar_t, scalar_t (*ACT_FN)(const scalar_t&),
          bool act_first>
__global__ void act_and_mul_kernel(
    scalar_t* __restrict__ out,          // [..., d]
    const scalar_t* __restrict__ input,  // [..., 2, d]
    const int d) {
  const int64_t token_idx = blockIdx.x;
  for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
    const scalar_t x = VLLM_LDG(&input[token_idx * 2 * d + idx]);
    const scalar_t y = VLLM_LDG(&input[token_idx * 2 * d + d + idx]);
    out[token_idx * d + idx] = compute<scalar_t, ACT_FN, act_first>(x, y);
  }
}

template<typename scalar_t, typename VT, int N, scalar_t (*ACT_FN)(const scalar_t&),
        bool act_first>
__global__ void act_and_mul_kernel_bd_opt(
  scalar_t* __restrict__ out,               // [..., d]
  const scalar_t* __restrict__ input,       // [..., 2, d]
  const int d,
  const int blockDim_x
){
    const int64_t token_idx = blockIdx.y;
    int x_offset = (blockIdx.x * blockDim_x + threadIdx.x) * N;
    if(x_offset >= d) return;
    int64_t offset0 = token_idx * d;
    int64_t offset1 = offset0 << 1;
    const scalar_t* ptr_input = input + offset1;
    const scalar_t* ptr_input0 = ptr_input + x_offset;
    const scalar_t* ptr_input1 = ptr_input0 + d;
    scalar_t* ptr_output = out + offset0 + x_offset;
    VT vsrc0 = *(VT*)(ptr_input0);
    VT vsrc1 = *(VT*)(ptr_input1);
    VT vdst;
    scalar_t* ptr_src0 = (scalar_t*)&vsrc0;
    scalar_t* ptr_src1 = (scalar_t*)&vsrc1;
    scalar_t* ptr_dst = (scalar_t*)&vdst;
    #pragma unroll N
    for(int i = 0; i < N; i++) {
        ptr_dst[i] = compute<scalar_t, ACT_FN, act_first>(ptr_src0[i], ptr_src1[i]);
    }
    *(VT*)(ptr_output) = vdst;
}

template<typename scalar_t, typename VT, int N, scalar_t (*ACT_FN)(const scalar_t&),
        bool act_first>
__global__ void act_and_mul_kernel_sd_opt(
  scalar_t* __restrict__ out,               // [..., d]
  const scalar_t* __restrict__ input,       // [..., 2, d]
  const int d,
  const int blockDim_x,
  const int gridDim_x,
  const int token_per_block,
  const int max_token_num) {
    __shared__ int8_t sm_buffer[16384];
    int token_offset = blockIdx.x * token_per_block;
    int out_offset = token_offset * d;
    int in_offset = out_offset << 1;
    int num_token = min(max_token_num - token_offset, token_per_block);
    if(num_token <= 0) return;
    const scalar_t *ptr_block_input = input + in_offset;
    scalar_t* ptr_block_output = out + out_offset;
    int output_size = num_token * d;
    int input_size = output_size << 1;
    
    scalar_t* ptr_sm_buffer = (scalar_t*)sm_buffer;
    int stride = blockDim_x * N;
    for(int i = threadIdx.x*N; i < input_size; i += stride) {
        *(VT*)(ptr_sm_buffer + i) = *(VT*)(ptr_block_input + i);
    }
    __syncthreads();
    for(int i = threadIdx.x; i < output_size; i += blockDim_x) {
      int token_id = i / d;
      int x_offset = i % d;
      scalar_t *ptr_input0 = ptr_sm_buffer + token_id * d * 2 + x_offset;
      scalar_t *ptr_input1 = ptr_input0 + d;
      *(ptr_block_output + i) = compute<scalar_t, ACT_FN, act_first>(*ptr_input0, ptr_input1[0]);
    }
}

template<typename scalar_t, typename VT, int N, scalar_t (*ACT_FN)(const scalar_t&),
        bool act_first>
__global__ void act_and_mul_kernel_sd_fast_opt(
  scalar_t* __restrict__ out,               // [..., d]
  const scalar_t* __restrict__ input,       // [..., 2, d]
  const int d,
  const int blockDim_x,
  const int gridDim_x,
  const int token_per_block,
  const int max_token_num) {
    __shared__ int8_t sm_buffer[16384];
    int token_offset = blockIdx.x * token_per_block;
    int out_offset = token_offset * d;
    int in_offset = out_offset << 1;
    int num_token = min(max_token_num - token_offset, token_per_block);
    if(num_token <= 0) return;
    const scalar_t *ptr_block_input = input + in_offset;
    scalar_t* ptr_block_output = out + out_offset;
    int output_size = num_token * d;
    int input_size = output_size << 1;
    
    scalar_t* ptr_sm_buffer = (scalar_t*)sm_buffer;
    int stride = blockDim_x * N;
    for(int i = threadIdx.x*N; i < input_size; i += stride) {
        *(VT*)(ptr_sm_buffer + i) = *(VT*)(ptr_block_input + i);
    }
    __syncthreads();
    for(int i = threadIdx.x*N; i < output_size; i += stride) {
      int token_id = i / d;
      int x_offset = i % d;
      scalar_t *ptr_input0 = ptr_sm_buffer + token_id * d * 2 + x_offset;
      scalar_t *ptr_input1 = ptr_input0 + d;
      VT vdst;
      scalar_t *ptr_dst = (scalar_t*)&vdst;
      #pragma unroll N
      for(int j = 0; j < N; j++) {
        ptr_dst[j] = compute<scalar_t, ACT_FN, act_first>(ptr_input0[j], ptr_input1[j]);
      }
      *(VT*)(ptr_block_output + i) = vdst;
    }
}

template <typename T>
__device__ __forceinline__ T silu_kernel(const T& x) {
  // x * sigmoid(x)
  // return (T)(((float)x) / (1.0f + expf((float)-x)));
  float x_f = (float)x;
  return (T) ((x_f) / (1.0f + __builtin_expf(-x_f)));
}

template <typename T>
__device__ __forceinline__ T gelu_kernel(const T& x) {
  // Equivalent to PyTorch GELU with 'none' approximation.
  // Refer to:
  // https://github.com/pytorch/pytorch/blob/8ac9b20d4b090c213799e81acf48a55ea8d437d6/aten/src/ATen/native/cuda/ActivationGeluKernel.cu#L36-L38
  const float f = (float)x;
  constexpr float ALPHA = M_SQRT1_2;
  return (T)(f * 0.5f * (1.0f + ::erf(f * ALPHA)));
}

template <typename T>
__device__ __forceinline__ T gelu_tanh_kernel(const T& x) {
  // Equivalent to PyTorch GELU with 'tanh' approximation.
  // Refer to:
  // https://github.com/pytorch/pytorch/blob/8ac9b20d4b090c213799e81acf48a55ea8d437d6/aten/src/ATen/native/cuda/ActivationGeluKernel.cu#L25-L30
  const float f = (float)x;
  constexpr float BETA = M_SQRT2 * M_2_SQRTPI * 0.5f;
  constexpr float KAPPA = 0.044715;
  float x_cube = f * f * f;
  float inner = BETA * (f + KAPPA * x_cube);
  return (T)(0.5f * f * (1.0f + ::tanhf(inner)));
}

}  // namespace vllm

// Launch activation and gating kernel.
// Use ACT_FIRST (bool) indicating whether to apply the activation function
// first.
#define LAUNCH_ACTIVATION_GATE_KERNEL(KERNEL, ACT_FIRST)                 \
  int d = input.size(-1) / 2;                                            \
  int64_t num_tokens = input.numel() / input.size(-1);                   \
  int n = 16 / input.element_size();                                                          \
  if(((d&(n - 1)) == 0) && d >= 512 * n) {\
    int blocksize = 512;                                                                  \
    dim3 gridsize((d + 512*n - 1) / (512*n), num_tokens,1);                               \
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));                     \
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();                         \
    VLLM_DISPATCH_FLOATING_TYPES(                                                         \
      input.scalar_type(),                                                                \
      "act_and_mul_kernel_bd_opt",                                                               \
      [&] {                                                                               \
        vllm::act_and_mul_kernel_bd_opt<scalar_t, float4, 16 / sizeof(scalar_t), KERNEL<scalar_t>, ACT_FIRST><<<gridsize, blocksize, 0, stream>>>(   \
          out.data_ptr<scalar_t>(),                                                         \
          input.data_ptr<scalar_t>(),                                                       \
          d, blocksize);                                                                    \
      });                                                                                   \
  } else if(d < 512 && (d & (n - 1)) == 0) {                                                \
        int block_token = 16384 / input.element_size() / 2 / d;                             \
        block_token = block_token / n * n;                                                  \
        int blocksize = 512;                                                                \
        int gridsize = (num_tokens + block_token - 1) / block_token;                        \
        const at::cuda::OptionalCUDAGuard device_guard(device_of(input));                   \
        const cudaStream_t stream = at::cuda::getCurrentCUDAStream();                       \
        VLLM_DISPATCH_FLOATING_TYPES(                                                       \
        input.scalar_type(),                                                                \
        "act_and_mul_kernel_sd_fast_opt",                                                   \
        [&] {                                                                               \
        vllm::act_and_mul_kernel_sd_fast_opt<scalar_t, float4, 16 / sizeof(scalar_t), KERNEL<scalar_t>, ACT_FIRST><<<gridsize, blocksize, 0, stream>>>(   \
        out.data_ptr<scalar_t>(),                                                       \
        input.data_ptr<scalar_t>(),                                                     \
        d, blocksize,gridsize,block_token,num_tokens);                                  \
        });                                                                                 \
  } else if(d < 512) {                                                                      \
        int block_token = 16384 / input.element_size() / 2 / d;                             \
        block_token = block_token / n * n;                                                  \
        int blocksize = 512;                                                                \
        int gridsize = (num_tokens + block_token - 1) / block_token;                        \
        const at::cuda::OptionalCUDAGuard device_guard(device_of(input));                   \
        const cudaStream_t stream = at::cuda::getCurrentCUDAStream();                       \
        VLLM_DISPATCH_FLOATING_TYPES(                                                       \
        input.scalar_type(),                                                                \
        "act_and_mul_kernel_sd_opt",                                                        \
        [&] {                                                                               \
          vllm::act_and_mul_kernel_sd_opt<scalar_t, float4, 16 / sizeof(scalar_t), KERNEL<scalar_t>, ACT_FIRST><<<gridsize, blocksize, 0, stream>>>(   \
            out.data_ptr<scalar_t>(),                                                       \
            input.data_ptr<scalar_t>(),                                                     \
            d, blocksize,gridsize,block_token,num_tokens);                                  \
        });                                                                                 \
  } else {                                                                                  \
  dim3 grid(num_tokens);                                                 \
  dim3 block(std::min(d, 1024));                                         \
  if (num_tokens == 0) {                                                 \
    return;                                                              \
  }                                                                      \
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));      \
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();          \
  VLLM_DISPATCH_FLOATING_TYPES(                                          \
      input.scalar_type(),                                                                  \
      "act_and_mul_kernel",                                                                 \
      [&] {                                                                                 \
        vllm::act_and_mul_kernel<scalar_t, KERNEL<scalar_t>, ACT_FIRST><<<grid, block, 0, stream>>>(   \
          out.data_ptr<scalar_t>(),                                                         \
          input.data_ptr<scalar_t>(),                                                       \
          d);                                                                               \
      });                                                                                   \
  }

void silu_and_mul(torch::Tensor& out,    // [..., d]
                  torch::Tensor& input)  // [..., 2 * d]
{
  LAUNCH_ACTIVATION_GATE_KERNEL(vllm::silu_kernel, true);
}

void mul_and_silu(torch::Tensor& out,    // [..., d]
                  torch::Tensor& input)  // [..., 2 * d]
{
  // The difference between mul_and_silu and silu_and_mul is that mul_and_silu
  // applies the silu to the latter half of the input.
  LAUNCH_ACTIVATION_GATE_KERNEL(vllm::silu_kernel, false);
}

void gelu_and_mul(torch::Tensor& out,    // [..., d]
                  torch::Tensor& input)  // [..., 2 * d]
{
  LAUNCH_ACTIVATION_GATE_KERNEL(vllm::gelu_kernel, true);
}

void gelu_tanh_and_mul(torch::Tensor& out,    // [..., d]
                       torch::Tensor& input)  // [..., 2 * d]
{
  LAUNCH_ACTIVATION_GATE_KERNEL(vllm::gelu_tanh_kernel, true);
}

namespace vllm {

template <typename T>
__device__ __forceinline__ T fatrelu_kernel(const T& x, const float threshold) {
  const float f = (float)x;
  return (T)(f > threshold ? f : 0.0f);
}

template <typename scalar_t, scalar_t (*ACT_FN)(const scalar_t&, const float)>
__global__ void act_and_mul_kernel_with_param(
    scalar_t* __restrict__ out, const scalar_t* __restrict__ input, const int d,
    const float param) {
  const int64_t token_idx = blockIdx.x;
  for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
    const scalar_t x = VLLM_LDG(&input[token_idx * 2 * d + idx]);
    const scalar_t y = VLLM_LDG(&input[token_idx * 2 * d + d + idx]);
    out[token_idx * d + idx] = ACT_FN(x, param) * y;
  }
}

}  // namespace vllm

#define LAUNCH_ACTIVATION_GATE_KERNEL_WITH_PARAM(KERNEL, PARAM)         \
  int d = input.size(-1) / 2;                                           \
  int64_t num_tokens = input.numel() / input.size(-1);                  \
  dim3 grid(num_tokens);                                                \
  dim3 block(std::min(d, 1024));                                        \
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));     \
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();         \
  VLLM_DISPATCH_FLOATING_TYPES(                                         \
      input.scalar_type(), "act_and_mul_kernel_with_param", [&] {       \
        vllm::act_and_mul_kernel_with_param<scalar_t, KERNEL<scalar_t>> \
            <<<grid, block, 0, stream>>>(out.data_ptr<scalar_t>(),      \
                                         input.data_ptr<scalar_t>(), d, \
                                         PARAM);                        \
      });

void fatrelu_and_mul(torch::Tensor& out,    // [..., d],
                     torch::Tensor& input,  // [..., 2 * d]
                     double threshold) {
  LAUNCH_ACTIVATION_GATE_KERNEL_WITH_PARAM(vllm::fatrelu_kernel, threshold);
}
namespace vllm {

// Element-wise activation kernel template.
template <typename scalar_t, scalar_t (*ACT_FN)(const scalar_t&)>
__global__ void activation_kernel(
    scalar_t* __restrict__ out,          // [..., d]
    const scalar_t* __restrict__ input,  // [..., d]
    const int d) {
  const int64_t token_idx = blockIdx.x;
  for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
    const scalar_t x = VLLM_LDG(&input[token_idx * d + idx]);
    out[token_idx * d + idx] = ACT_FN(x);
  }
}

}  // namespace vllm

// Launch element-wise activation kernel.
#define LAUNCH_ACTIVATION_KERNEL(KERNEL)                                       \
  int d = input.size(-1);                                                      \
  int64_t num_tokens = input.numel() / d;                                      \
  dim3 grid(num_tokens);                                                       \
  dim3 block(std::min(d, 1024));                                               \
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));            \
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();                \
  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "activation_kernel", [&] { \
    vllm::activation_kernel<scalar_t, KERNEL<scalar_t>>                        \
        <<<grid, block, 0, stream>>>(out.data_ptr<scalar_t>(),                 \
                                     input.data_ptr<scalar_t>(), d);           \
  });

namespace vllm {

template <typename T>
__device__ __forceinline__ T gelu_new_kernel(const T& x) {
  const float x3 = (float)(x * x * x);
  const T t = (T)tanhf((T)(0.79788456f * (float)(x + (T)(0.044715f * x3))));
  return ((T)0.5) * x * (((T)1.0) + t);
}

template <typename T>
__device__ __forceinline__ T gelu_fast_kernel(const T& x) {
  const float f = (float)x;
  const T t =
      (T)tanhf(((T)(f * 0.79788456f)) * (((T)1.0) + (T)(0.044715f * f) * x));
  return ((T)0.5) * x * (((T)1.0) + t);
}

template <typename T>
__device__ __forceinline__ T gelu_quick_kernel(const T& x) {
  // x * sigmoid(1.702 * x)
  return (T)(((float)x) / (1.0f + expf(-1.702f * (float)x)));
}

}  // namespace vllm

void gelu_new(torch::Tensor& out,    // [..., d]
              torch::Tensor& input)  // [..., d]
{
  LAUNCH_ACTIVATION_KERNEL(vllm::gelu_new_kernel);
}

void gelu_fast(torch::Tensor& out,    // [..., d]
               torch::Tensor& input)  // [..., d]
{
  LAUNCH_ACTIVATION_KERNEL(vllm::gelu_fast_kernel);
}

void gelu_quick(torch::Tensor& out,    // [..., d]
                torch::Tensor& input)  // [..., d]
{
  LAUNCH_ACTIVATION_KERNEL(vllm::gelu_quick_kernel);
}
