// 2025 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved. 
#pragma once

#include "../../attention/attention_dtypes.h"
#include <assert.h>
#include <float.h>
#include <stdint.h>
#include <type_traits>

namespace vllm {

namespace fp8 {

template <typename Tout, typename Tin, Fp8KVCacheDataType kv_dt>
__inline__ __device__ Tout convert(const Tin& x) {
  assert(false);
  __builtin_unreachable();  // Suppress missing return statement warning
}

template <typename Tout, typename Tin, Fp8KVCacheDataType kv_dt>
__inline__ __device__ Tout scaled_convert(const Tin& x, const float scale) {
  assert(false);
  __builtin_unreachable();  // Suppress missing return statement warning
}

// The following macro is used to dispatch the conversion function based on
// the data type of the key and value cache. The FN is a macro that calls a
// function with template<typename scalar_t, typename cache_t,
// Fp8KVCacheDataType kv_dt>.
#define DISPATCH_BY_KV_CACHE_DTYPE(SRC_DTYPE, KV_DTYPE, FN)                  \
  if (KV_DTYPE == "auto") {                                                  \
    if (SRC_DTYPE == at::ScalarType::Float) {                                \
      FN(float, float, vllm::Fp8KVCacheDataType::kAuto);                     \
    } else if (SRC_DTYPE == at::ScalarType::Half) {                          \
      FN(uint16_t, uint16_t, vllm::Fp8KVCacheDataType::kAuto);               \
    } else if (SRC_DTYPE == at::ScalarType::BFloat16) {                      \
      FN(__nv_bfloat16, __nv_bfloat16, vllm::Fp8KVCacheDataType::kAuto);     \
    } else {                                                                 \
      TORCH_CHECK(false, "Unsupported input type of kv cache: ", SRC_DTYPE); \
    }                                                                        \
  } else {                                                                   \
    TORCH_CHECK(false, "Unsupported data type of kv cache: ", KV_DTYPE);     \
  }

}  // namespace fp8
}  // namespace vllm
