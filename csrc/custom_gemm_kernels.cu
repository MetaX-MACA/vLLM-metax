// SPDX-License-Identifier: Apache-2.0
// 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Custom BF16 GEMM kernel optimized for MACA C500 architecture.
// Computes out = a @ b  (A: M×K row-major, B: K×N row-major, out: M×N float).
// Uses __builtin_mxc_mma_16x16x16bf16 WMMA intrinsics with software-managed
// shared-memory double buffering and warp-level cooperative fetch.
//
// Block config: BM=32, BN=32, BK=16, 4 warps (2×2 warp tile grid).
// Supported dtypes: BF16 in, FP32 out.

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "cuda_compat.h"
#include "dispatch_utils.h"

// MACA architecture warp size (64 threads per warp on C500).
#ifndef WARP_SIZE
#define WARP_SIZE 64
#endif

#define div_ceil(a, b) (((a) + (b) - 1) / (b))

// Async copy pipeline macros — MACA cu-bridge does not provide built-in
// equivalents for cp.async.commit_group / cp.async.wait_group, so we
// make them no-ops.  Correctness is preserved by the __syncthreads()
// barrier that follows each group; only the pipelining overlap is lost.
#define CP_ASYNC_COMMIT_GROUP()
#define CP_ASYNC_WAIT_GROUP(n)

// ---- MACA vector types for BF16 MMA ----
typedef __NATIVE_VECTOR__(4, _Float16) v4f16;
typedef __NATIVE_VECTOR__(4, float) v4f32;
// BF16 MMA fragment: 4×bf16 = 64 bits → packed as 2×uint32_t.
typedef __NATIVE_VECTOR__(2, uint32_t) v4bf16_as_u32;

namespace vllm {

// ---- helper: broadcast a scalar into all lanes of a float4 fragment ----
__device__ inline void fill_fragment(v4f32 &frag, float val) {
  frag[0] = val;
  frag[1] = val;
  frag[2] = val;
  frag[3] = val;
}

// =============================================================
// BF16 warp-level matrix load / store (MACA tensor-core layout)
// =============================================================

// Load a 16×16 tile of A (row-major, with stride "Stride" in elements).
template <int Stride>
__device__ inline void load_matrix_sync_bf16(v4bf16_as_u32 &frag,
                                              nv_bfloat16 *ptr) {
  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  int lane = tid % WARP_SIZE;

  // MACA matrix_a row-major mapping
  int row = lane & 0xf;
  int col = ((lane >> 4) << 2);

  frag = *reinterpret_cast<v4bf16_as_u32*>(&ptr[row * Stride + col]);
}

// Load a 16×16 tile of B (column-major, with stride "Stride" in elements).
template <int Stride>
__device__ inline void load_matrix_sync_col_bf16(v4bf16_as_u32 &frag,
                                                  nv_bfloat16 *ptr) {
  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  int lane = tid % WARP_SIZE;

  int row = ((lane >> 4) << 2);
  int col = lane & 0xf;

  nv_bfloat16* base_ptr = &ptr[row * Stride + col];

  nv_bfloat16 vals[4];
  vals[0] = base_ptr[0];
  vals[1] = base_ptr[Stride];
  vals[2] = base_ptr[2 * Stride];
  vals[3] = base_ptr[3 * Stride];

  frag = *reinterpret_cast<v4bf16_as_u32*>(vals);
}

// Store a float32 16×16 fragment back to global memory.
__device__ inline void store_matrix_sync_fp32(
    float *ptr, v4f32 frag, int stride,
    int start_m, int start_n, int M, int N) {
  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  int lane = tid % WARP_SIZE;

  int row = ((lane >> 4) << 2);
  int col = lane & 0xf;

  float* base_ptr = &ptr[row * stride + col];

  if (start_m + row + 0 < M && start_n + col < N) base_ptr[0] = frag[0];
  if (start_m + row + 1 < M && start_n + col < N) base_ptr[stride] = frag[1];
  if (start_m + row + 2 < M && start_n + col < N) base_ptr[2 * stride] = frag[2];
  if (start_m + row + 3 < M && start_n + col < N) base_ptr[3 * stride] = frag[3];
}

// =============================================================
// MMA intrinsic wrapper
// =============================================================

__device__ inline void mma_sync_bf16(v4f32 &d, v4bf16_as_u32 a,
                                      v4bf16_as_u32 b, v4f32 c) {
  d = __builtin_mxc_mma_16x16x16bf16(a, b, c);
}

// =============================================================
// Global → shared-memory load (safe, with out-of-bounds padding)
// =============================================================

// Load 4 BF16 elements from A into shared memory (per thread).
__device__ inline void load_global_a_safe_bf16(
    nv_bfloat16* smem_ptr, const nv_bfloat16* A,
    int m, int k, int M, int K) {
  const nv_bfloat16* src_ptr = &A[m * K + k];

  if (m < M && k + 4 <= K) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      smem_ptr[i] = src_ptr[i];
    }
  } else {
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      smem_ptr[i] = (m < M && k + i < K) ? src_ptr[i] : __float2bfloat16(0.0f);
    }
  }
}

// Load 8 BF16 elements from B into shared memory (per thread).
// 128-bit aligned path via int4 when the source address is 16-byte aligned.
__device__ inline void load_global_b_safe_bf16(
    nv_bfloat16* smem_ptr, const nv_bfloat16* B,
    int k, int n, int K, int N) {
  const nv_bfloat16* src_ptr = &B[k * N + n];

  if (k < K && n + 8 <= N) {
    if (((uint64_t)src_ptr % 16) == 0) {
      // 128-bit aligned load (8×bf16 = 128 bits = int4)
      *reinterpret_cast<int4*>(smem_ptr) = *reinterpret_cast<const int4*>(src_ptr);
    } else {
      #pragma unroll
      for (int i = 0; i < 8; ++i) {
        smem_ptr[i] = src_ptr[i];
      }
    }
  } else {
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      smem_ptr[i] = (k < K && n + i < N) ? src_ptr[i] : __float2bfloat16(0.0f);
    }
  }
}

// =============================================================
// BF16 GEMM Kernel (output FP32)
//
// Block tile:  BM = BLOCK_ROW_WARPS × (WMMA_M × WARP_TILE_M) = 32
//               BN = BLOCK_COL_WARPS × (WMMA_N × WARP_TILE_N) = 32
//               BK = WMMA_K  = 16
// Shared mem:  s_a[2][BM][BK + OFFSET] + s_b[2][BK][BN + OFFSET]
// =============================================================

template <const int WMMA_M = 16, const int WMMA_N = 16, const int WMMA_K = 16,
          const int WMMA_TILE_M = 4, const int WMMA_TILE_N = 2,
          const int WARP_TILE_M = 1, const int WARP_TILE_N = 1,
          const int OFFSET = 8>
__global__ void bf16_gemm_kernel(
    const nv_bfloat16* __restrict__ A,
    const nv_bfloat16* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K) {

  constexpr int THREADS_PER_BLOCK = 256;
  constexpr int NUM_WARPS = THREADS_PER_BLOCK / WARP_SIZE;          // 4

  constexpr int BLOCK_COL_WARPS = 2;
  constexpr int BLOCK_ROW_WARPS = NUM_WARPS / BLOCK_COL_WARPS;     // 2

  constexpr int BM = BLOCK_ROW_WARPS * (WMMA_M * WARP_TILE_M);     // 32
  constexpr int BN = BLOCK_COL_WARPS * (WMMA_N * WARP_TILE_N);     // 32
  constexpr int BK = WMMA_K;                                        // 16

  static_assert(NUM_WARPS % BLOCK_COL_WARPS == 0,
                "Warp count must be divisible by column warp layout");

  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int NUM_K_TILES = div_ceil(K, WMMA_K);

  __shared__ nv_bfloat16 s_a[2][BM][BK + OFFSET];   // [2][32][24]
  __shared__ nv_bfloat16 s_b[2][BK][BN + OFFSET];   // [2][16][40]

  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int warp_id = tid / WARP_SIZE;

  const int warp_m = warp_id / BLOCK_COL_WARPS;
  const int warp_n = warp_id % BLOCK_COL_WARPS;

  // ---- A tile addressing ----
  constexpr int THREADS_PER_ROW_A = THREADS_PER_BLOCK / BM;  // 8
  constexpr int LOAD_VEC_A = BK / THREADS_PER_ROW_A;         // 2

  int load_smem_a_m = tid / THREADS_PER_ROW_A;
  int load_smem_a_k = (tid % THREADS_PER_ROW_A) * LOAD_VEC_A;

  // ---- B tile addressing ----
  constexpr int THREADS_PER_ROW_B = THREADS_PER_BLOCK / BK;  // 16
  constexpr int LOAD_VEC_B = BN / THREADS_PER_ROW_B;         // 2

  int load_smem_b_k = tid / THREADS_PER_ROW_B;
  int load_smem_b_n = (tid % THREADS_PER_ROW_B) * LOAD_VEC_B;

  int load_gmem_a_m = by * BM + load_smem_a_m;
  int load_gmem_b_n = bx * BN + load_smem_b_n;

  // Warp-level accumulators (16×16 fragments).
  v4f32 C_frag[WARP_TILE_M][WARP_TILE_N];  // [1][1]
  v4bf16_as_u32 A_frag[WARP_TILE_M];        // [1]
  v4bf16_as_u32 B_frag[WARP_TILE_N];        // [1]

  // Initialize accumulators to zero.
  #pragma unroll
  for (int i = 0; i < WARP_TILE_M; ++i) {
    #pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      fill_fragment(C_frag[i][j], 0.0f);
    }
  }

  // ---- Prologue: load first K tile into smem[0] ----
  {
    int load_gmem_a_k = load_smem_a_k;
    int load_gmem_b_k = load_smem_b_k;

    load_global_a_safe_bf16(
        &s_a[0][load_smem_a_m][load_smem_a_k],
        A, load_gmem_a_m, load_gmem_a_k, M, K);
    load_global_b_safe_bf16(
        &s_b[0][load_smem_b_k][load_smem_b_n],
        B, load_gmem_b_k, load_gmem_b_n, K, N);

    CP_ASYNC_COMMIT_GROUP();
    CP_ASYNC_WAIT_GROUP(0);
  }
  __syncthreads();

  // ---- Main loop: double-buffer K tiles 1 .. NUM_K_TILES-1 ----
  for (int k = 1; k < NUM_K_TILES; ++k) {
    int smem_sel = (k - 1) & 1;
    int smem_sel_next = k & 1;

    int load_gmem_a_k = k * WMMA_K + load_smem_a_k;
    int load_gmem_b_k = k * WMMA_K + load_smem_b_k;

    // Issue global→shared loads for the next K tile.
    load_global_a_safe_bf16(
        &s_a[smem_sel_next][load_smem_a_m][load_smem_a_k],
        A, load_gmem_a_m, load_gmem_a_k, M, K);
    load_global_b_safe_bf16(
        &s_b[smem_sel_next][load_smem_b_k][load_smem_b_n],
        B, load_gmem_b_k, load_gmem_b_n, K, N);

    CP_ASYNC_COMMIT_GROUP();

    // Load A fragments from current smem buffer.
    #pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      const int warp_smem_a_m = warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
      load_matrix_sync_bf16<BK + OFFSET>(
          A_frag[i], &s_a[smem_sel][warp_smem_a_m][0]);
    }

    // Load B fragments from current smem buffer.
    #pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      const int warp_smem_b_n = warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
      load_matrix_sync_col_bf16<BN + OFFSET>(
          B_frag[j], &s_b[smem_sel][0][warp_smem_b_n]);
    }

    // MMA on the current tile while the next tile is in flight.
    #pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      #pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        mma_sync_bf16(C_frag[i][j], A_frag[i], B_frag[j], C_frag[i][j]);
      }
    }

    CP_ASYNC_WAIT_GROUP(0);
    __syncthreads();
  }

  // ---- Epilogue: MMA on the last K tile (already in smem) ----
  {
    int smem_sel = (NUM_K_TILES - 1) & 1;

    #pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      const int warp_smem_a_m = warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
      load_matrix_sync_bf16<BK + OFFSET>(
          A_frag[i], &s_a[smem_sel][warp_smem_a_m][0]);
    }

    #pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      const int warp_smem_b_n = warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
      load_matrix_sync_col_bf16<BN + OFFSET>(
          B_frag[j], &s_b[smem_sel][0][warp_smem_b_n]);
    }

    #pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      #pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        mma_sync_bf16(C_frag[i][j], A_frag[i], B_frag[j], C_frag[i][j]);
      }
    }
  }

  // ---- Store accumulator fragments back to global memory ----
  #pragma unroll
  for (int i = 0; i < WARP_TILE_M; ++i) {
    #pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      const int store_m = by * BM + warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
      const int store_n = bx * BN + warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;

      if (store_m < M && store_n < N) {
        store_matrix_sync_fp32(
            C + store_m * N + store_n, C_frag[i][j],
            N, store_m, store_n, M, N);
      }
    }
  }
}

}  // namespace vllm

// =============================================================
// Host launch function (global scope — matches ops.h declaration)
// =============================================================

void custom_gemm(
    torch::Tensor& out,
    const torch::Tensor& a,
    const torch::Tensor& b,
    int64_t M, int64_t N, int64_t K) {

  // Device checks
  TORCH_CHECK(a.is_cuda(), "custom_gemm: input a must be on CUDA device");
  TORCH_CHECK(b.is_cuda(), "custom_gemm: input b must be on CUDA device");
  TORCH_CHECK(out.is_cuda(), "custom_gemm: out must be on CUDA device");
  TORCH_CHECK(a.get_device() == b.get_device(),
              "custom_gemm: a and b must be on the same device");
  TORCH_CHECK(out.get_device() == a.get_device(),
              "custom_gemm: out must be on the same device as inputs");

  // Dtype checks — this kernel only supports BF16×BF16 → FP32
  TORCH_CHECK(a.scalar_type() == at::kBFloat16,
              "custom_gemm: input a must be bfloat16, got ",
              a.scalar_type());
  TORCH_CHECK(b.scalar_type() == at::kBFloat16,
              "custom_gemm: input b must be bfloat16, got ",
              b.scalar_type());
  TORCH_CHECK(out.scalar_type() == at::kFloat,
              "custom_gemm: out must be float32, got ",
              out.scalar_type());

  // Shape checks
  TORCH_CHECK(a.dim() == 2 && a.size(0) == M && a.size(1) == K,
              "custom_gemm: a shape mismatch, expected [", M, ", ", K,
              "] got ", a.sizes());
  TORCH_CHECK(b.dim() == 2 && b.size(0) == K && b.size(1) == N,
              "custom_gemm: b shape mismatch, expected [", K, ", ", N,
              "] got ", b.sizes());
  TORCH_CHECK(out.dim() == 2 && out.size(0) == M && out.size(1) == N,
              "custom_gemm: out shape mismatch, expected [", M, ", ", N,
              "] got ", out.sizes());

  if (M == 0 || N == 0 || K == 0) return;

  at::cuda::CUDAGuard device_guard{a.get_device()};
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // Fixed tile config (matching the kernel template parameters).
  constexpr int WMMA_M = 16;
  constexpr int WMMA_N = 16;
  constexpr int WMMA_K = 16;
  constexpr int WMMA_TILE_M = 4;
  constexpr int WMMA_TILE_N = 2;
  constexpr int WARP_TILE_M = 1;
  constexpr int WARP_TILE_N = 1;

  constexpr int THREADS_PER_BLOCK = 256;
  constexpr int NUM_WARPS = THREADS_PER_BLOCK / WARP_SIZE;           // 4
  constexpr int BLOCK_COL_WARPS = 2;
  constexpr int BLOCK_ROW_WARPS = NUM_WARPS / BLOCK_COL_WARPS;      // 2

  constexpr int BM = BLOCK_ROW_WARPS * (WMMA_M * WARP_TILE_M);      // 32
  constexpr int BN = BLOCK_COL_WARPS * (WMMA_N * WARP_TILE_N);      // 32

  const dim3 block(THREADS_PER_BLOCK);
  const dim3 grid(div_ceil(N, BN), div_ceil(M, BM));

  printf("[CUSTOM_GEMM] launching kernel: M=%ld N=%ld K=%ld "
         "grid=(%d,%d) block=%d a_dtype=%.*s device=%d\n",
         M, N, K,
         static_cast<int>(grid.x), static_cast<int>(grid.y),
         THREADS_PER_BLOCK,
         static_cast<int>(a.dtype().name().size()), a.dtype().name().data(),
         static_cast<int>(a.get_device()));

  vllm::bf16_gemm_kernel<WMMA_M, WMMA_N, WMMA_K,
                         WMMA_TILE_M, WMMA_TILE_N,
                         WARP_TILE_M, WARP_TILE_N, 8>
      <<<grid, block, 0, stream>>>(
          reinterpret_cast<const nv_bfloat16*>(a.data_ptr()),
          reinterpret_cast<const nv_bfloat16*>(b.data_ptr()),
          reinterpret_cast<float*>(out.data_ptr()),
          static_cast<int>(M), static_cast<int>(N), static_cast<int>(K));
}
