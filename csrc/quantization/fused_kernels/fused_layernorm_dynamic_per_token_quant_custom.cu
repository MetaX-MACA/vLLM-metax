
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "../../dispatch_utils.h"
#include "layernorm_utils.cuh"
#include "quant_conversions.cuh"

namespace vllm {

template <typename scalar_t, typename scalar_out_t, bool has_residual = false>
__device__ static void rms_norm_dynamic_per_token_quant_vec(
    scalar_out_t* __restrict__ out,       // [..., hidden_size]
    float* __restrict__ scales,           // [num_tokens]
    scalar_t const* __restrict__ input,   // [..., hidden_size]
    scalar_t const* __restrict__ weight,  // [hidden_size]
    float const* scale_ub, float const var_epsilon, int32_t const hidden_size,
    scalar_t* __restrict__ residual = nullptr) {
  float rms = 0.0f;
  float token_scale = 0.0f;

  // Compute rms
  vllm::vectorized::compute_rms<scalar_t, has_residual>(
      &rms, input, hidden_size, var_epsilon, residual);

  // Compute scale
  vllm::vectorized::compute_dynamic_per_token_scales<scalar_t, scalar_out_t,
                                                     has_residual>(
      &token_scale, scales, input, weight, rms, scale_ub, hidden_size,
      residual);

  // RMS Norm + Quant
  if constexpr (std::is_same_v<scalar_out_t, int8_t>) {
    vllm::vectorized::norm_and_quant<scalar_t, scalar_out_t, true,
                                     has_residual>(
        out, input, weight, rms, 1.0f / token_scale, hidden_size, residual);
  } else {
    // FP8 - Do not invert token_scale for exact match with FBGemm
    vllm::vectorized::norm_and_quant<scalar_t, scalar_out_t, false,
                                     has_residual>(
        out, input, weight, rms, token_scale, hidden_size, residual);
  }
}

// RMS norm + quant kernel
template <typename scalar_t, typename scalar_out_t, bool has_residual = false>
__global__ void rms_norm_dynamic_per_token_quant_custom_general_kernel(
    scalar_out_t* __restrict__ out,       // [..., hidden_size]
    scalar_t* __restrict__ out_bf16,
    float* __restrict__ scales,           // [num_tokens]
    scalar_t const* __restrict__ input,   // [..., hidden_size]
    scalar_t const* __restrict__ weight,  // [hidden_size]
    float const* scale_ub, float const var_epsilon, int32_t const hidden_size,
    scalar_t* __restrict__ residual = nullptr) {
  // For vectorization, token_input and token_output pointers need to be
  // aligned at 8-byte and 4-byte addresses respectively.
  bool const can_vectorize = hidden_size % 4 == 0;

  if (can_vectorize) {
    return rms_norm_dynamic_per_token_quant_vec<scalar_t, scalar_out_t,
                                                has_residual>(
        out, scales, input, weight, scale_ub, var_epsilon, hidden_size,
        residual);
  }

  float rms = 0.0f;
  float token_scale = 0.0f;

  // Compute RMS
  vllm::compute_rms<scalar_t, has_residual>(&rms, input, hidden_size,
                                            var_epsilon, residual);
  // Compute Scale
  vllm::compute_dynamic_per_token_scales<scalar_t, scalar_out_t, has_residual>(
      &token_scale, scales, input, weight, rms, scale_ub, hidden_size,
      residual);

  // RMS Norm + Quant
  if constexpr (std::is_same_v<scalar_out_t, int8_t>) {
    vllm::norm_and_quant<scalar_t, scalar_out_t, true, has_residual>(
        out_bf16, out, input, weight, rms, 1.0f / token_scale, hidden_size, residual);
  } else {
    // FP8 - Do not invert s_token_scale for exact match with FBGemm
    vllm::norm_and_quant<scalar_t, scalar_out_t, false, has_residual>(
        out_bf16, out, input, weight, rms, token_scale, hidden_size, residual);
  }
}

namespace rms
{

template <
  typename T,
  int N,
  int Alignment = sizeof(T) * N
>
class alignas(Alignment) AlignedArray {
public:
    T data[N];
};

__forceinline__ __device__ float square(float input)
{
    return input * input;
}

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
  dst = std::clamp(dst, i8_min, i8_max);
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

} // namespace rms

template <typename scalar_t, typename scalar_out_t, int32_t ELTS_PER_LDG, int32_t VPT, bool has_residual = false>
__global__ void rms_norm_dynamic_per_token_quant_wave_kernel(
  scalar_out_t* __restrict__ out,       // [..., hidden_size]
  float* __restrict__ scales,           // [num_tokens]
  scalar_t const* __restrict__ input,   // [..., hidden_size]
  scalar_t const* __restrict__ weight,  // [hidden_size]
  float const* scale_ub, float const var_epsilon,
  int32_t const hidden_size, int32_t num_vec_elems, float min_scale,
  scalar_t* __restrict__ residual = nullptr) {
  static constexpr int32_t WARP_SIZE = 64;
  static constexpr int32_t LDG_PER_THREAD = VPT / ELTS_PER_LDG;
  float ss = 0.0f;
  float block_absmax_val_maybe = -9999.9f;

  using AccessType = rms::AlignedArray<scalar_t, ELTS_PER_LDG>;
  using OutType = rms::AlignedArray<scalar_out_t, ELTS_PER_LDG>;
  const AccessType* vec_input = reinterpret_cast<const AccessType*>(&input[blockIdx.x * hidden_size]);
  const AccessType* vec_weight = reinterpret_cast<const AccessType*>(weight);

  AccessType* vec_residual = nullptr;
  if constexpr (has_residual) {
    vec_residual = reinterpret_cast<AccessType*>(&residual[blockIdx.x * hidden_size]);
  }
  float row_chunk[VPT];
  constexpr scalar_out_t qmax{std::numeric_limits<scalar_out_t>::max()};

  #pragma unroll
  for (int32_t ii = threadIdx.x, idx = 0; ii < num_vec_elems; ii += WARP_SIZE, ++idx) {
    AccessType vec_x1 = vec_input[ii];
    AccessType vec_w = vec_weight[ii];
    #pragma unroll
    for (int32_t jj = 0; jj < ELTS_PER_LDG; ++jj) {
      row_chunk[idx * ELTS_PER_LDG + jj] = static_cast<float>(vec_x1.data[jj]);
    }
    if (has_residual) {
      AccessType vec_res = vec_residual[ii];
      #pragma unroll
      for (int jj = 0; jj < ELTS_PER_LDG; ++jj) {
        row_chunk[idx * ELTS_PER_LDG + jj] += static_cast<float>(vec_res.data[jj]);
        vec_x1.data[jj] = static_cast<scalar_t>(row_chunk[idx * ELTS_PER_LDG + jj]);
      }
      vec_residual[ii] = vec_x1;
    }
    #pragma unroll
    for (int32_t jj = 0; jj < ELTS_PER_LDG; ++jj) {
      ss += rms::square(row_chunk[idx * ELTS_PER_LDG + jj]);
      row_chunk[idx * ELTS_PER_LDG + jj] *= static_cast<float>(vec_w.data[jj]);
      block_absmax_val_maybe = fmaxf(block_absmax_val_maybe, fabs(row_chunk[idx * ELTS_PER_LDG + jj]));
    }
  }

  for (int32_t step = 1; step < WARP_SIZE; step <<= 1) {
    block_absmax_val_maybe = max(block_absmax_val_maybe, __shfl_xor_sync(0xffffffffffffffff, block_absmax_val_maybe, step, WARP_SIZE));
    ss += __shfl_xor_sync(0xffffffffffffffff, ss, step, WARP_SIZE);
  }

  float s_rms;
  float s_token_scale;
  if (threadIdx.x == 0) {
    s_rms = rsqrtf(ss / hidden_size + var_epsilon);
    float scale = block_absmax_val_maybe * s_rms;
    scale = max(scale / qmax, min_scale);
    s_token_scale = 1.0f / scale;
    scales[blockIdx.x] = scale;
  }
  s_rms = __shfl_sync(0xffffffffffffffff, s_rms, 0, WARP_SIZE);
  s_token_scale = __shfl_sync(0xffffffffffffffff, s_token_scale, 0, WARP_SIZE);

  OutType* vec_output = reinterpret_cast<OutType*>(&out[blockIdx.x * hidden_size]);
  #pragma unroll
  for (int32_t ii = threadIdx.x, idx = 0; ii < num_vec_elems; ii += WARP_SIZE, ++idx){
    OutType out;
    #pragma unroll
    for (int32_t jj = 0; jj < ELTS_PER_LDG; ++jj) {
      out.data[jj] = ScaledQuant<scalar_out_t, true>::quant_fn(static_cast<scalar_t>(row_chunk[idx * ELTS_PER_LDG + jj] * s_rms), s_token_scale);
    }
    vec_output[ii] = out;
  }
}

template <typename scalar_t, typename scalar_out_t, int32_t ELTS_PER_LDG, int32_t VPT, int32_t BLOCK_DIM, bool has_residual = false>
__global__ void rms_norm_dynamic_per_token_quant_custom_kernel(
    scalar_out_t* __restrict__ out,       // [..., hidden_size]
    scalar_t* __restrict__ out_bf16,
    float* __restrict__ scales,           // [num_tokens]
    scalar_t const* __restrict__ input,   // [..., hidden_size]
    scalar_t const* __restrict__ weight,  // [hidden_size]
    float const* scale_ub, float const var_epsilon,
    int32_t const hidden_size, int32_t num_vec_elems, float min_scale,
    scalar_t* __restrict__ residual = nullptr) {
  static constexpr int32_t LDG_PER_THREAD = VPT / ELTS_PER_LDG;
  float ss = 0.0f;
  float block_absmax_val_maybe = -9999.9f;

  using AccessType = rms::AlignedArray<scalar_t, ELTS_PER_LDG>;
  using OutType = rms::AlignedArray<scalar_out_t, ELTS_PER_LDG>;
  const AccessType* vec_input = reinterpret_cast<const AccessType*>(&input[blockIdx.x * hidden_size]);
  const AccessType* vec_weight = reinterpret_cast<const AccessType*>(weight);
  AccessType* vec_residual = nullptr;
  if constexpr (has_residual) {
    vec_residual = reinterpret_cast<AccessType*>(&residual[blockIdx.x * hidden_size]);
  }
  float row_chunk[VPT];
  constexpr scalar_out_t qmax{std::numeric_limits<scalar_out_t>::max()};

  #pragma unroll
  for (int32_t ii = threadIdx.x, idx = 0; ii < num_vec_elems ; ii += BLOCK_DIM, ++idx) {
    AccessType vec_x1 = vec_input[ii];
    AccessType vec_w = vec_weight[ii];
    #pragma unroll
    for (int32_t jj = 0; jj < ELTS_PER_LDG; ++jj) {
      row_chunk[idx * ELTS_PER_LDG + jj] = static_cast<float>(vec_x1.data[jj]);
    }
    if (has_residual) {
      AccessType vec_res = vec_residual[ii];
      #pragma unroll
      for (int jj = 0; jj < ELTS_PER_LDG; ++jj) {
        row_chunk[idx * ELTS_PER_LDG + jj] += static_cast<float>(vec_res.data[jj]);
        vec_x1.data[jj] = static_cast<scalar_t>(row_chunk[idx * ELTS_PER_LDG + jj]);
      }
      vec_residual[ii] = vec_x1;
    }
    #pragma unroll
    for (int32_t jj = 0; jj < ELTS_PER_LDG; ++jj) {
      ss += rms::square(row_chunk[idx * ELTS_PER_LDG + jj]);
      row_chunk[idx * ELTS_PER_LDG + jj] *= static_cast<float>(vec_w.data[jj]);
      block_absmax_val_maybe = fmaxf(block_absmax_val_maybe, fabs(row_chunk[idx * ELTS_PER_LDG + jj]));
    }
  }
  reduce_max_sum<float, BLOCK_DIM>(block_absmax_val_maybe, ss);

  __shared__ float s_rms;
  __shared__ float s_token_scale;
  if (threadIdx.x == 0) {
    s_rms = rsqrtf(ss / hidden_size + var_epsilon);
    float scale = block_absmax_val_maybe * s_rms;
    scale = max(scale / qmax, min_scale);
    s_token_scale = 1.0f / scale;
    scales[blockIdx.x] = scale;
  }
  __syncthreads();

  OutType* vec_output = reinterpret_cast<OutType*>(&out[blockIdx.x * hidden_size]);
  AccessType* vec_output_bf16 = reinterpret_cast<AccessType*>(&out_bf16[blockIdx.x * hidden_size]);
  #pragma unroll
  for (int32_t ii = threadIdx.x, idx = 0; ii < num_vec_elems; ii += BLOCK_DIM, ++idx){
    OutType out_;
    AccessType out_bf16_;
    #pragma unroll
    for (int32_t jj = 0; jj < ELTS_PER_LDG; ++jj) {
      out_bf16_.data[jj] = static_cast<scalar_t>(row_chunk[idx * ELTS_PER_LDG + jj] * s_rms);
      out_.data[jj] = static_cast<scalar_out_t>(rms::float_to_int8_rn(static_cast<float>(out_bf16_.data[jj]) * s_token_scale));
    }
    vec_output[ii] = out_;
    vec_output_bf16[ii] = out_bf16_;
  }
}
}  // namespace vllm

// Residual add + RMS norm + dynamic per token
template <typename scalar_in_t>
void rms_norm_dynamic_per_token_quant_custom_dispatch(
    torch::Tensor& out,           // [..., hidden_size]
    torch::Tensor& out_bf16,
    torch::Tensor const& input,   // [..., hidden_size]
    torch::Tensor const& weight,  // [hidden_size]
    torch::Tensor& scales,        // [num_tokens]
    double const var_epsilon,     // Variance epsilon used in norm calculation
    std::optional<at::Tensor> const& scale_ub,
    std::optional<at::Tensor>& residual) {
  int32_t hidden_size = input.size(-1);
  int32_t num_tokens = input.numel() / hidden_size;
  static constexpr int ELTS_PER_LDG = 4;

  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, 1024));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  bool can_opt = hidden_size % ELTS_PER_LDG == 0;
  if (can_opt) {
    const int32_t num_vec_elems = hidden_size / ELTS_PER_LDG;
    const float min_scaling_factor =
    out.dtype() == torch::kInt8
        ? std::numeric_limits<float>::epsilon()
        : 1.0f / (std::numeric_limits<c10::Float8_e4m3fn>::max() * 512.f);
    static constexpr int32_t ELTS_PER_LDG = 4;

    if (residual.has_value()) {
      if (hidden_size <= 2048) {
        constexpr int32_t BLOCK_DIM = 256;
        constexpr int32_t VPT = 8;
        VLLM_DISPATCH_QUANT_TYPES(
        out.scalar_type(), "rms_norm_dynamic_per_token_quant_custom_kernel", [&] {
          vllm::rms_norm_dynamic_per_token_quant_custom_kernel<scalar_in_t, scalar_t, ELTS_PER_LDG, VPT, BLOCK_DIM,
                                                        true>
              <<<grid, BLOCK_DIM, 0, stream>>>(
                  out.data_ptr<scalar_t>(), out_bf16.data_ptr<scalar_in_t>(), scales.data_ptr<float>(),
                  input.data_ptr<scalar_in_t>(), weight.data_ptr<scalar_in_t>(),
                  scale_ub.has_value() ? scale_ub->data_ptr<float>() : nullptr,
                  var_epsilon, hidden_size, num_vec_elems, min_scaling_factor, residual->data_ptr<scalar_in_t>());
        });
      } else if (hidden_size <= 4096) {
        constexpr int32_t BLOCK_DIM = 512;
        constexpr int32_t VPT = 8;
        VLLM_DISPATCH_QUANT_TYPES(
        out.scalar_type(), "rms_norm_dynamic_per_token_quant_custom_kernel", [&] {
          vllm::rms_norm_dynamic_per_token_quant_custom_kernel<scalar_in_t, scalar_t, ELTS_PER_LDG, VPT, BLOCK_DIM,
                                                        true>
              <<<grid, BLOCK_DIM, 0, stream>>>(
                  out.data_ptr<scalar_t>(), out_bf16.data_ptr<scalar_in_t>(), scales.data_ptr<float>(),
                  input.data_ptr<scalar_in_t>(), weight.data_ptr<scalar_in_t>(),
                  scale_ub.has_value() ? scale_ub->data_ptr<float>() : nullptr,
                  var_epsilon, hidden_size, num_vec_elems, min_scaling_factor, residual->data_ptr<scalar_in_t>());
        });
      } else if (hidden_size <= 8192) {
        constexpr int32_t BLOCK_DIM = 512;
        constexpr int32_t VPT = 16;
        VLLM_DISPATCH_QUANT_TYPES(
        out.scalar_type(), "rms_norm_dynamic_per_token_quant_custom_kernel", [&] {
          vllm::rms_norm_dynamic_per_token_quant_custom_kernel<scalar_in_t, scalar_t, ELTS_PER_LDG, VPT, BLOCK_DIM,
                                                        true>
              <<<grid, BLOCK_DIM, 0, stream>>>(
                  out.data_ptr<scalar_t>(), out_bf16.data_ptr<scalar_in_t>(), scales.data_ptr<float>(),
                  input.data_ptr<scalar_in_t>(), weight.data_ptr<scalar_in_t>(),
                  scale_ub.has_value() ? scale_ub->data_ptr<float>() : nullptr,
                  var_epsilon, hidden_size, num_vec_elems, min_scaling_factor, residual->data_ptr<scalar_in_t>());
        });
      } else {
        VLLM_DISPATCH_QUANT_TYPES(
          out.scalar_type(), "rms_norm_dynamic_per_token_quant_custom_general_kernel", [&] {
            vllm::rms_norm_dynamic_per_token_quant_custom_general_kernel<scalar_in_t, scalar_t,
                                                          true>
                <<<grid, block, 0, stream>>>(
                    out.data_ptr<scalar_t>(), out_bf16.data_ptr<scalar_in_t>(), scales.data_ptr<float>(),
                    input.data_ptr<scalar_in_t>(), weight.data_ptr<scalar_in_t>(),
                    scale_ub.has_value() ? scale_ub->data_ptr<float>() : nullptr,
                    var_epsilon, hidden_size, residual->data_ptr<scalar_in_t>());
          });
      }
    } else {
      if (hidden_size <= 2048) {
        constexpr int32_t BLOCK_DIM = 256;
        constexpr int32_t VPT = 8;
        VLLM_DISPATCH_QUANT_TYPES(
        out.scalar_type(), "rms_norm_dynamic_per_token_quant_custom_kernel", [&] {
          vllm::rms_norm_dynamic_per_token_quant_custom_kernel<scalar_in_t, scalar_t, ELTS_PER_LDG, VPT, BLOCK_DIM,
                                                        false>
              <<<grid, BLOCK_DIM, 0, stream>>>(
                  out.data_ptr<scalar_t>(), out_bf16.data_ptr<scalar_in_t>(), scales.data_ptr<float>(),
                  input.data_ptr<scalar_in_t>(), weight.data_ptr<scalar_in_t>(),
                  scale_ub.has_value() ? scale_ub->data_ptr<float>() : nullptr,
                  var_epsilon, hidden_size, num_vec_elems, min_scaling_factor, nullptr);
        });
      } else if (hidden_size <= 4096) {
        constexpr int32_t BLOCK_DIM = 512;
        constexpr int32_t VPT = 8;
        VLLM_DISPATCH_QUANT_TYPES(
        out.scalar_type(), "rms_norm_dynamic_per_token_quant_custom_kernel", [&] {
          vllm::rms_norm_dynamic_per_token_quant_custom_kernel<scalar_in_t, scalar_t, ELTS_PER_LDG, VPT, BLOCK_DIM,
                                                        false>
              <<<grid, BLOCK_DIM, 0, stream>>>(
                  out.data_ptr<scalar_t>(), out_bf16.data_ptr<scalar_in_t>(), scales.data_ptr<float>(),
                  input.data_ptr<scalar_in_t>(), weight.data_ptr<scalar_in_t>(),
                  scale_ub.has_value() ? scale_ub->data_ptr<float>() : nullptr,
                  var_epsilon, hidden_size, num_vec_elems, min_scaling_factor, nullptr);
        });
      } else if (hidden_size <= 8192) {
        constexpr int32_t BLOCK_DIM = 512;
        constexpr int32_t VPT = 16;
        VLLM_DISPATCH_QUANT_TYPES(
        out.scalar_type(), "rms_norm_dynamic_per_token_quant_custom_kernel", [&] {
          vllm::rms_norm_dynamic_per_token_quant_custom_kernel<scalar_in_t, scalar_t, ELTS_PER_LDG, VPT, BLOCK_DIM,
                                                        false>
              <<<grid, BLOCK_DIM, 0, stream>>>(
                  out.data_ptr<scalar_t>(), out_bf16.data_ptr<scalar_in_t>(), scales.data_ptr<float>(),
                  input.data_ptr<scalar_in_t>(), weight.data_ptr<scalar_in_t>(),
                  scale_ub.has_value() ? scale_ub->data_ptr<float>() : nullptr,
                  var_epsilon, hidden_size, num_vec_elems, min_scaling_factor, nullptr);
        });
      } else {
        VLLM_DISPATCH_QUANT_TYPES(
          out.scalar_type(), "rms_norm_dynamic_per_token_quant_custom_general_kernel", [&] {
            vllm::rms_norm_dynamic_per_token_quant_custom_general_kernel<scalar_in_t, scalar_t,
                                                          false>
                <<<grid, block, 0, stream>>>(
                    out.data_ptr<scalar_t>(), out_bf16.data_ptr<scalar_in_t>(), scales.data_ptr<float>(),
                    input.data_ptr<scalar_in_t>(), weight.data_ptr<scalar_in_t>(),
                    scale_ub.has_value() ? scale_ub->data_ptr<float>() : nullptr,
                    var_epsilon, hidden_size, nullptr);
          });
      }

    }
  } else {
    if (residual.has_value()) {
      VLLM_DISPATCH_QUANT_TYPES(
          out.scalar_type(), "rms_norm_dynamic_per_token_quant_custom_general_kernel", [&] {
            vllm::rms_norm_dynamic_per_token_quant_custom_general_kernel<scalar_in_t, scalar_t,
                                                          true>
                <<<grid, block, 0, stream>>>(
                    out.data_ptr<scalar_t>(), out_bf16.data_ptr<scalar_in_t>(), scales.data_ptr<float>(),
                    input.data_ptr<scalar_in_t>(), weight.data_ptr<scalar_in_t>(),
                    scale_ub.has_value() ? scale_ub->data_ptr<float>() : nullptr,
                    var_epsilon, hidden_size, residual->data_ptr<scalar_in_t>());
          });

    } else {
      VLLM_DISPATCH_QUANT_TYPES(
          out.scalar_type(), "rms_norm_dynamic_per_token_quant_custom_general_kernel", [&] {
            vllm::rms_norm_dynamic_per_token_quant_custom_general_kernel<scalar_in_t, scalar_t,
                                                          false>
                <<<grid, block, 0, stream>>>(
                    out.data_ptr<scalar_t>(), out_bf16.data_ptr<scalar_in_t>(), scales.data_ptr<float>(),
                    input.data_ptr<scalar_in_t>(), weight.data_ptr<scalar_in_t>(),
                    scale_ub.has_value() ? scale_ub->data_ptr<float>() : nullptr,
                    var_epsilon, hidden_size, nullptr);
          });
    }
  }
}

void rms_norm_dynamic_per_token_quant_custom(
    torch::Tensor& out,           // [..., hidden_size]
    torch::Tensor& out_bf16,
    torch::Tensor const& input,   // [..., hidden_size]
    torch::Tensor const& weight,  // [hidden_size]
    torch::Tensor& scales,        // [num_tokens]
    double const var_epsilon,     // Variance epsilon used in norm calculation
    std::optional<at::Tensor> scale_ub, std::optional<at::Tensor> residual) {
  static c10::ScalarType kFp8Type = is_fp8_ocp()
                                        ? c10::ScalarType::Float8_e4m3fn
                                        : c10::ScalarType::Float8_e4m3fnuz;
  TORCH_CHECK(out.dtype() == kFp8Type || out.dtype() == torch::kInt8);
  TORCH_CHECK(out.is_contiguous() && input.is_contiguous());

  if (scale_ub.has_value()) {
    TORCH_CHECK(out.dtype() == kFp8Type);
  }
  TORCH_CHECK(scales.dtype() == torch::kFloat32);

  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "rms_norm_dynamic_per_token_quant_dispatch", [&] {
        rms_norm_dynamic_per_token_quant_custom_dispatch<scalar_t>(
            out, out_bf16, input, weight, scales, var_epsilon, scale_ub, residual);
      });
}
