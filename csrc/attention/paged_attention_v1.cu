// 2025 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
/*
 * Adapted from
 * https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/kernels/decoder_masked_multihead_attention/decoder_masked_multihead_attention_template.hpp
 * Copyright (c) 2023, The vLLM team.
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "attention_kernels.cuh"

#ifndef USE_ROCM
  #define WARP_SIZE 32
#else
  #define WARP_SIZE warpSize
#endif

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define DIVIDE_ROUND_UP(a, b) (((a) + (b) - 1) / (b))

#define LAUNCH_PAGED_ATTENTION_V1(HEAD_SIZE)                                \
  VLLM_DevFuncAttribute_SET_MaxDynamicSharedMemorySize(                     \
      ((void*)vllm::paged_attention_v1_kernel<T, CACHE_T, HEAD_SIZE,        \
                                              BLOCK_SIZE, NUM_THREADS,      \
                                              KV_DTYPE, IS_BLOCK_SPARSE>),  \
      shared_mem_size);                                                     \
  vllm::paged_attention_v1_kernel<T, CACHE_T, HEAD_SIZE, BLOCK_SIZE,        \
                                  NUM_THREADS, KV_DTYPE, IS_BLOCK_SPARSE>   \
      <<<grid, block, shared_mem_size, stream>>>(                           \
          out_ptr, query_ptr, key_cache_ptr, value_cache_ptr, num_kv_heads, \
          scale, block_tables_ptr, seq_lens_ptr, max_num_blocks_per_seq,    \
          alibi_slopes_ptr, q_stride, kv_block_stride, kv_head_stride,      \
          k_scale_ptr, v_scale_ptr, tp_rank, blocksparse_local_blocks,      \
          blocksparse_vert_stride, blocksparse_block_size,                  \
          blocksparse_head_sliding_step);


#define LAUNCH_PAGED_ATTENTION_V1_32N(HEAD_SIZE)                                \
  VLLM_DevFuncAttribute_SET_MaxDynamicSharedMemorySize(                     \
      ((void*)vllm::paged_attention_v1_32N_kernel<T, CACHE_T, HEAD_SIZE,        \
                                              BLOCK_SIZE, NUM_THREADS,      \
                                              KV_DTYPE, IS_BLOCK_SPARSE>),  \
      shared_mem_size);                                                     \
  vllm::paged_attention_v1_32N_kernel<T, CACHE_T, HEAD_SIZE, BLOCK_SIZE,        \
                                  NUM_THREADS, KV_DTYPE, IS_BLOCK_SPARSE>   \
      <<<grid, block, shared_mem_size, stream>>>(                           \
          out_ptr, query_ptr, key_cache_ptr, value_cache_ptr, num_kv_heads, \
          scale, block_tables_ptr, seq_lens_ptr, max_num_blocks_per_seq,    \
          alibi_slopes_ptr, q_stride, kv_block_stride, kv_head_stride,      \
          k_scale_ptr, v_scale_ptr, tp_rank, blocksparse_local_blocks,                      \
          blocksparse_vert_stride, blocksparse_block_size,                  \
          blocksparse_head_sliding_step, 1, num_heads);

template< typename scalar_t>
__global__ void reshape_k_layout_new(scalar_t * __restrict__ k_buffer, scalar_t* k_output,int num_blocks,int num_kv_heads, int head_size,int block_size, int x,int dst_x) {
  int k_head_stride = head_size * block_size;
  scalar_t *ptr_k_buffer = k_buffer + blockIdx.x * k_head_stride;
  scalar_t *ptr_output = k_output + blockIdx.x * k_head_stride;
  for(int t = threadIdx.x; t < k_head_stride; t += blockDim.x) {
    int heightId = t / (block_size * dst_x);
    int remain = t % (block_size * dst_x);
    int blockId = remain / dst_x;
    int wId = remain % dst_x;
    int inId = heightId * dst_x + wId;
    int in_y = inId / x;
    int in_x = inId % x;
    int inIndex = in_y  * block_size * x + blockId * x + in_x;
    ptr_output[t] = ptr_k_buffer[inIndex];
  }
}
// [num_blocks, num_kv_heads, head_size, block_size] -->   [num_blocks,  num_kv_heads, block_size,head_size]
template<typename scalar_t>
__global__ void reshape_v_layout(scalar_t * __restrict__ v_buffer, scalar_t* v_output,int num_blocks,int num_kv_heads, int head_size,int block_size) {
      int v_block_stride = head_size * block_size * num_kv_heads;
      int v_head_stride = head_size * block_size;
      scalar_t *ptr_in = v_buffer + blockIdx.x * v_block_stride;
      scalar_t *ptr_output = v_output + blockIdx.x * v_block_stride;
      for(int t = threadIdx.x; t < v_block_stride; t += blockDim.x) {
        int num_kv_headIdx = t / v_head_stride;
        int remain = t % v_head_stride;
        int headId_H = remain / block_size;
        remain = remain % block_size;
        int out_idx = num_kv_headIdx * head_size * block_size + remain * head_size + headId_H;
        ptr_output[out_idx] = ptr_in[t];
      }
}

template<
  typename CACHE_T,
  int BLOCK_SIZE>
void reshape_kv_cache(
  torch::Tensor& key_cache,
  torch::Tensor& value_cache,
  torch::Tensor& key_cache_new_layer,
  torch::Tensor& value_cache_new_layer,
  int num_seqs,
  int num_heads,
  int head_size,
  int num_kv_heads) {
  int kv_block_stride = key_cache.stride(0); // NU ,BLC ,HEAD, HEAD_DIM
  int kv_head_stride = key_cache.stride(1);

  CACHE_T* key_cache_ptr = reinterpret_cast<CACHE_T*>(key_cache.data_ptr());
  CACHE_T* value_cache_ptr = reinterpret_cast<CACHE_T*>(value_cache.data_ptr());
  CACHE_T* key_cache_tmp = reinterpret_cast<CACHE_T*>(key_cache_new_layer.data_ptr());
  CACHE_T* value_cache_tmp = reinterpret_cast<CACHE_T*>(value_cache_new_layer.data_ptr());

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  reshape_k_layout_new<CACHE_T><<<dim3(key_cache.size(0)*num_kv_heads,1,1),dim3(256,1,1),0,stream>>>(key_cache_ptr,key_cache_tmp,key_cache.size(0),num_kv_heads,head_size,BLOCK_SIZE,8,16);
  reshape_v_layout<CACHE_T><<<dim3(key_cache.size(0),1,1),dim3(256,1,1),0,stream>>>(value_cache_ptr,value_cache_tmp,key_cache.size(0),num_kv_heads,head_size,BLOCK_SIZE);
}
#define CALL_RESHAPE_LAUNCHER(CACHE_T, BLOCK_SIZE)       \
  reshape_kv_cache<CACHE_T, BLOCK_SIZE>( \
    key_cache,                                                               \
    value_cache,                                                             \
    key_cache_new_layer,                                                     \
    value_cache_new_layer,                                                   \
    num_seqs,\
    num_heads,\
    head_size,\
    num_kv_heads);

#define CALL_RESHAPE_BLOCK_SIZE(CACHE_T) \
  switch (block_size) {                                               \
    case 8:                                                           \
      CALL_RESHAPE_LAUNCHER(CACHE_T, 8);          \
      break;                                                          \
    case 16:                                                          \
      CALL_RESHAPE_LAUNCHER(CACHE_T, 16);         \
      break;                                                          \
    case 32:                                                          \
      CALL_RESHAPE_LAUNCHER(CACHE_T, 32);         \
      break;                                                          \
    default:                                                          \
      TORCH_CHECK(false, "Unsupported block size: ", block_size);     \
      break;                                                          \
  }
void page_reshape_kv_cache(
  torch::Tensor& key_cache,       // [num_blocks, num_heads, head_size/x, block_size, x]
  torch::Tensor& value_cache,     // [num_blocks, num_heads, head_size, block_size]
  torch::Tensor& key_cache_new_layer, //[num_blocks, num_heads, head_size/16, block_size, 16]
  torch::Tensor& value_cache_new_layer,//[num_blocks, num_heads, block_size, head_size]
  int64_t num_seqs,
  int64_t num_heads,
  int64_t head_size,
  int64_t num_kv_heads,               // [num_heads]
  int64_t block_size,
  const std::string& kv_cache_dtype) {
  if (kv_cache_dtype == "auto") {
    if (sizeof(key_cache.dtype())==4) {
      CALL_RESHAPE_BLOCK_SIZE(float);
    } else if (sizeof(key_cache.dtype()) == 2) {
      CALL_RESHAPE_BLOCK_SIZE(uint16_t);
    } else {
      TORCH_CHECK(false, "Unsupported data type: ", key_cache.dtype());
    }
  }  else {
    TORCH_CHECK(false, "Unsupported data type of kv cache: ", kv_cache_dtype);
  }
}

// TODO(woosuk): Tune NUM_THREADS.
template <typename T, typename CACHE_T, int BLOCK_SIZE,
          vllm::Fp8KVCacheDataType KV_DTYPE, bool IS_BLOCK_SPARSE,
          int NUM_THREADS = 256>
void paged_attention_v1_launcher(
    torch::Tensor& out, torch::Tensor& query, torch::Tensor& key_cache,
    torch::Tensor& value_cache, int num_kv_heads, float scale,
    torch::Tensor& block_tables, torch::Tensor& seq_lens, int max_seq_len,
    const std::optional<torch::Tensor>& alibi_slopes, torch::Tensor& k_scale,
    torch::Tensor& v_scale, const int tp_rank,
    const int blocksparse_local_blocks, const int blocksparse_vert_stride,
    const int blocksparse_block_size, const int blocksparse_head_sliding_step) {
  int num_seqs = query.size(0);
  int num_heads = query.size(1);
  int head_size = query.size(2);
  int max_num_blocks_per_seq = block_tables.size(1);
  int q_stride = query.stride(0);  //num head head_dim 
  int kv_block_stride = key_cache.stride(0);   // NU ,BLC ,HEAD, HEAD_DIM
  int kv_head_stride = key_cache.stride(1);

  [[maybe_unused]] int thread_group_size = MAX(WARP_SIZE / BLOCK_SIZE, 1);
  assert(head_size % thread_group_size == 0);
  assert((head_size & 7) == 0);

  // NOTE: alibi_slopes is optional.
  const float* alibi_slopes_ptr =
      alibi_slopes
          ? reinterpret_cast<const float*>(alibi_slopes.value().data_ptr())
          : nullptr;

  T* out_ptr = reinterpret_cast<T*>(out.data_ptr());
  T* query_ptr = reinterpret_cast<T*>(query.data_ptr());
  CACHE_T* key_cache_ptr = reinterpret_cast<CACHE_T*>(key_cache.data_ptr());
  CACHE_T* value_cache_ptr = reinterpret_cast<CACHE_T*>(value_cache.data_ptr());
  int* block_tables_ptr = block_tables.data_ptr<int>();
  int* seq_lens_ptr = seq_lens.data_ptr<int>();
  const float* k_scale_ptr = reinterpret_cast<const float*>(k_scale.data_ptr());
  const float* v_scale_ptr = reinterpret_cast<const float*>(v_scale.data_ptr());

  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  int padded_max_seq_len =
      DIVIDE_ROUND_UP(max_seq_len, BLOCK_SIZE) * BLOCK_SIZE;
  int logits_size = padded_max_seq_len * sizeof(float);

  int V_VEC_SIZE = 16 / sizeof(CACHE_T); 
  int NUM_V_VECS_PER_THREAD = head_size / V_VEC_SIZE; 
  int NUM_COLS_PER_ITER = MAX(WARP_SIZE / NUM_V_VECS_PER_THREAD, 1); 

  int outputs_size = (NUM_WARPS / 2) * head_size * sizeof(float) * NUM_COLS_PER_ITER;
  // Python-side check in vllm.worker.worker._check_if_can_support_max_seq_len
  // Keep that in sync with the logic here!
  int shared_mem_size = std::max(logits_size, outputs_size);

  dim3 grid(num_heads, num_seqs, 1);
  dim3 block(NUM_THREADS);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(query));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  switch (head_size) {
    // NOTE(woosuk): To reduce the compilation time, we only compile for the
    // head sizes that we use in the model. However, we can easily extend this
    // to support any head size which is a multiple of 16.
    case 32:
      LAUNCH_PAGED_ATTENTION_V1(32);
      break;
    case 64:
      LAUNCH_PAGED_ATTENTION_V1_32N(64);
      break;
    case 80:
      LAUNCH_PAGED_ATTENTION_V1(80);
      break;
    case 96:
      LAUNCH_PAGED_ATTENTION_V1(96);
      break;
    case 112:
      LAUNCH_PAGED_ATTENTION_V1(112);
      break;
    case 120:
      LAUNCH_PAGED_ATTENTION_V1(120);
      break;
    case 128:
      LAUNCH_PAGED_ATTENTION_V1_32N(128);
      break;
    case 160:
      LAUNCH_PAGED_ATTENTION_V1(160);
      break;
    case 192:
      LAUNCH_PAGED_ATTENTION_V1(192);
      break;
    case 256:
      LAUNCH_PAGED_ATTENTION_V1_32N(256);
      break;
    default:
      TORCH_CHECK(false, "Unsupported head size: ", head_size);
      break;
  }
}

#define CALL_V1_LAUNCHER(T, CACHE_T, BLOCK_SIZE, KV_DTYPE, IS_BLOCK_SPARSE)  \
  paged_attention_v1_launcher<T, CACHE_T, BLOCK_SIZE, KV_DTYPE,              \
                              IS_BLOCK_SPARSE>(                              \
      out, query, key_cache, value_cache, num_kv_heads, scale, block_tables, \
      seq_lens, max_seq_len, alibi_slopes, k_scale, v_scale, tp_rank,        \
      blocksparse_local_blocks, blocksparse_vert_stride,                     \
      blocksparse_block_size, blocksparse_head_sliding_step);

#define CALL_V1_LAUNCHER_SPARSITY(T, CACHE_T, BLOCK_SIZE, IS_FP8_KV_CACHE) \
  if (is_block_sparse) {                                                   \
    CALL_V1_LAUNCHER(T, CACHE_T, BLOCK_SIZE, IS_FP8_KV_CACHE, true);       \
  } else {                                                                 \
    CALL_V1_LAUNCHER(T, CACHE_T, BLOCK_SIZE, IS_FP8_KV_CACHE, false);      \
  }

// NOTE(woosuk): To reduce the compilation time, we omitted block sizes
// 1, 2, 4, 64, 128, 256.
#define CALL_V1_LAUNCHER_BLOCK_SIZE(T, CACHE_T, KV_DTYPE)         \
  switch (block_size) {                                           \
    case 8:                                                       \
      CALL_V1_LAUNCHER_SPARSITY(T, CACHE_T, 8, KV_DTYPE);         \
      break;                                                      \
    case 16:                                                      \
      CALL_V1_LAUNCHER_SPARSITY(T, CACHE_T, 16, KV_DTYPE);        \
      break;                                                      \
    case 32:                                                      \
      CALL_V1_LAUNCHER_SPARSITY(T, CACHE_T, 32, KV_DTYPE);        \
      break;                                                      \
    default:                                                      \
      TORCH_CHECK(false, "Unsupported block size: ", block_size); \
      break;                                                      \
  }

void paged_attention_v1(
    torch::Tensor& out,    // [num_seqs, num_heads, head_size]
    torch::Tensor& query,  // [num_seqs, num_heads, head_size]
    torch::Tensor&
        key_cache,  // [num_blocks, num_heads, head_size/x, block_size, x]
    torch::Tensor&
        value_cache,       // [num_blocks, num_heads, head_size, block_size]
    int64_t num_kv_heads,  // [num_heads]
    double scale,
    torch::Tensor& block_tables,  // [num_seqs, max_num_blocks_per_seq]
    torch::Tensor& seq_lens,      // [num_seqs]
    int64_t block_size, int64_t max_seq_len,
    const std::optional<torch::Tensor>& alibi_slopes,
    const std::string& kv_cache_dtype, torch::Tensor& k_scale,
    torch::Tensor& v_scale, const int64_t tp_rank,
    const int64_t blocksparse_local_blocks,
    const int64_t blocksparse_vert_stride, const int64_t blocksparse_block_size,
    const int64_t blocksparse_head_sliding_step) {
  const bool is_block_sparse = (blocksparse_vert_stride > 1);

  DISPATCH_BY_KV_CACHE_DTYPE(query.dtype(), kv_cache_dtype,
                             CALL_V1_LAUNCHER_BLOCK_SIZE)
}

#undef WARP_SIZE
#undef MAX
#undef MIN
#undef DIVIDE_ROUND_UP