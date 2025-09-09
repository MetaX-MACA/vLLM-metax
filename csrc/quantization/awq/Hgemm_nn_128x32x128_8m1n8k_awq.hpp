// 2024 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights
// Reserved.
#pragma once
#include "../gptq/Hgemm_common.cuh"
#include "awq_4bits.cuh"
#include "maca_fp16.h"

#define input_type __half
#define output_type __half
#define scalar_type float
#define acc_type float

#define SEL0 0x01000504
#define SEL1 0x03020706

#define SWIZZLE_INDEX(index, phase) (index ^ phase)
#define ONE_DIM_INDEX(x, y, lda) (y * lda + x)
#define CLAMP(value, bound, align) (value < bound ? value : bound - align)

#define LDG_B                                                   \
  {                                                             \
    ldg_b128_bsm_async(b_sts, dB[0], pred_n0, true);            \
    ldg_b128_bsm_async(b_sts + 128 * 16, dB[1], pred_n1, true); \
  }

#define LDG_B_HEAD                                                        \
  {                                                                       \
    bool pred_k = rowB_swizzle < ktail;                                   \
    ldg_b128_bsm_async(b_sts, dB[0], pred_n0 && pred_k, true);            \
    ldg_b128_bsm_async(b_sts + 128 * 16, dB[1], pred_n1 && pred_k, true); \
  }

#define MMA_ABC1                                                     \
  {                                                                  \
    for (int index_n = 0; index_n < 2; ++index_n) {                  \
      for (int index_m = 0; index_m < 8; ++index_m) {                \
        mma_16x16x16f16(cast_b64(rgA)[ONE_DIM_INDEX(index_m, 0, 8)], \
                        cast_b64(rgB)[ONE_DIM_INDEX(0, index_n, 2)], \
                        rgC[ONE_DIM_INDEX(index_m, index_n, 8)]);    \
      }                                                              \
    }                                                                \
  }

#define MMA_ABC2                                                     \
  {                                                                  \
    for (int index_n = 0; index_n < 2; ++index_n) {                  \
      for (int index_m = 0; index_m < 8; ++index_m) {                \
        mma_16x16x16f16(cast_b64(rgA)[ONE_DIM_INDEX(index_m, 1, 8)], \
                        cast_b64(rgB)[ONE_DIM_INDEX(1, index_n, 2)], \
                        rgC[ONE_DIM_INDEX(index_m, index_n, 8)]);    \
      }                                                              \
    }                                                                \
  }

#define LDS_B                                      \
  {                                                \
    cast_b128(rgB)[0] = cast_b128(b_lds)[0];       \
    cast_b128(rgB)[1] = cast_b128(b_lds)[16 * 16]; \
  }

#define PERM_C(C2perm)                     \
  {                                        \
    Float4VecType C_tmp[8];                \
    float* ptri = (float*)C2perm;          \
    float* ptro = (float*)C_tmp;           \
    for (int j = 0; j < 4; ++j) {          \
      for (int i = 0; i < 8; ++i) {        \
        ptro[j * 8 + i] = ptri[j + i * 4]; \
      }                                    \
    }                                      \
    for (int i = 0; i < 8; ++i) {          \
      C2perm[i] = C_tmp[i];                \
    }                                      \
  }

#define STS_C(phase)                                        \
  {                                                         \
    cast_b128(c_sts[0])[0] = cast_b128(rgC)[0 + phase * 8]; \
    cast_b128(c_sts[1])[0] = cast_b128(rgC)[1 + phase * 8]; \
    cast_b128(c_sts[2])[0] = cast_b128(rgC)[2 + phase * 8]; \
    cast_b128(c_sts[3])[0] = cast_b128(rgC)[3 + phase * 8]; \
    cast_b128(c_sts[4])[0] = cast_b128(rgC)[4 + phase * 8]; \
    cast_b128(c_sts[5])[0] = cast_b128(rgC)[5 + phase * 8]; \
    cast_b128(c_sts[6])[0] = cast_b128(rgC)[6 + phase * 8]; \
    cast_b128(c_sts[7])[0] = cast_b128(rgC)[7 + phase * 8]; \
  }

#define REDUCE_C(phase)                                               \
  {                                                                   \
    cast_b128(rgC)[0 + phase * 8] = cast_b128(c_lds[0])[0];           \
    cast_b128(rgC)[1 + phase * 8] = cast_b128(c_lds[1])[0];           \
    cast_b128(rgC)[2 + phase * 8] = cast_b128(c_lds[0])[1 * 16 * 32]; \
    cast_b128(rgC)[3 + phase * 8] = cast_b128(c_lds[1])[1 * 16 * 32]; \
    cast_b128(rgC)[4 + phase * 8] = cast_b128(c_lds[0])[2 * 16 * 32]; \
    cast_b128(rgC)[5 + phase * 8] = cast_b128(c_lds[1])[2 * 16 * 32]; \
    cast_b128(rgC)[6 + phase * 8] = cast_b128(c_lds[0])[3 * 16 * 32]; \
    cast_b128(rgC)[7 + phase * 8] = cast_b128(c_lds[1])[3 * 16 * 32]; \
    float* reduc_c = reinterpret_cast<float*>(rgC);                   \
    for (int loop = 0; loop < 8; ++loop) {                            \
      float acc = 0;                                                  \
      acc += reduc_c[loop + phase * 32 + 0 * 8];                      \
      acc += reduc_c[loop + phase * 32 + 1 * 8];                      \
      acc += reduc_c[loop + phase * 32 + 2 * 8];                      \
      acc += reduc_c[loop + phase * 32 + 3 * 8];                      \
      reduc_c[loop + phase * 8] = acc;                                \
    }                                                                 \
  }

template <Operation_t TransA, Operation_t TransB, bool IsBetaZero, int tileM,
          int tileN, int tileK, bool splitk = false, bool Swap = false>
__global__ __launch_bounds__(256) void Hgemm_nn_128x32x128_8m1n8k_awq_4bit(
    int m, int n, int k, const scalar_type alpha, const scalar_type beta,
    const quant_packed_type* dA_input, int lda, const input_type* dB_input,
    int ldb, output_type* dC_input, output_type* dC_output, int ldc,
    quant_packed_type* d_zeros, input_type* d_scales, int splitk_iters = 1,
    acc_type* d_acc_tmp = nullptr) {
  __shared__ uint8_t smem_base[0x8000];
  int bidx = Swap ? blockIdx.y : blockIdx.x;
  int bidy = Swap ? blockIdx.x : blockIdx.y;
  int bidz = blockIdx.z;
  int tid = threadIdx.x;

  uint64_t arowstride = TransA == OP_N ? 1 : lda;
  uint64_t acolstride = TransA == OP_N ? lda : 1;
  uint64_t browstride = TransB == OP_N ? 1 : ldb;
  uint64_t bcolstride = TransB == OP_N ? ldb : 1;

  const int malign = 8;
  const int nalign = 1;
  const int kalign = 8;
  int align_m = (m + malign - 1) / malign * malign;
  int align_n = (n + nalign - 1) / nalign * nalign;
  int align_k = (k + kalign - 1) / kalign * kalign;

  int k_num = (align_k + splitk_iters - 1) / splitk_iters;
  k_num =
      (k_num + tileK - 1) / tileK * tileK;  // k_num should be aligned to tileK
  int k_begin = bidz * k_num;
  int k_end = (bidz + 1) * k_num;
  k_end = k_end > align_k ? align_k : k_end;
  k_num = k_end - k_begin;
  int ktail = k_num % tileK;
  int kloop = k_begin;
  if (k_begin > align_k) {
    return;
  }

  int slot = tid / 64;
  int lane = tid & 63;
  int m64d16 = lane / 16;
  int m64m16 = lane % 16;
  output_type* c_base_i = dC_input;
  output_type* c_base_o = dC_output;

  quant_packed_type rga[8];
  b128VecType rgb[2];

  b64VecType rgA[16], rgB[4], rgCi[4];
  Float4VecType rgC[16];
  uint32_t rgZeros[1];
  b128VecType rgScales[1];

  quant_packed_type* dA[1];
  input_type* dB[2];

  // ldg A/B head

  int rowA = m64m16 * 8;
  int colA = m64d16 * 8 + slot * 32;
  uint current_m = bidx * tileM + rowA;
  bool pred_m = current_m < align_m;
  dA[0] = (quant_packed_type*)dA_input +
          (uint64_t)(current_m / PACK_RATIO) * (uint64_t)(arowstride) +
          (uint64_t)(colA + k_begin) * (uint64_t)(acolstride / PACK_RATIO);

  int colB = m64d16 + slot * 4;
  int rowB = m64m16 * 8;
  int rowB_swizzle = SWIZZLE_INDEX(rowB, (colB % 4 * 8));
  int current_n = bidy * tileN + colB;
  bool pred_n0 = current_n < align_n;
  dB[0] = (input_type*)dB_input +
          (uint64_t)(rowB_swizzle + k_begin) * (uint64_t)(browstride) +
          (uint64_t)(current_n) * (uint64_t)bcolstride;

  current_n += 16;
  bool pred_n1 = current_n < align_n;
  dB[1] = (input_type*)dB_input +
          (uint64_t)(rowB_swizzle + k_begin) * (uint64_t)(browstride) +
          (uint64_t)(current_n) * (uint64_t)bcolstride;

  // ldgB to BSM need swizzle
  input_type* b_sts = (input_type*)smem_base;
  b_sts += ONE_DIM_INDEX(rowB, colB, 128);

  // lds need swizzle
  input_type* b_lds = (input_type*)smem_base;
  int colB_lds = m64m16;
  int rowB_lds = m64d16 * 8 + slot * 32;
  b_lds +=
      ONE_DIM_INDEX(SWIZZLE_INDEX(rowB_lds, (colB_lds % 4 * 8)), colB_lds, 128);

  // ldg zeros and scales
  quant_packed_type* ldg_zeros_offset = d_zeros +
                                        (lda / PACK_RATIO) * (kloop / tileK) +
                                        bidx * (tileM / PACK_RATIO) + tid;
  input_type* ldg_scales_offset =
      d_scales + lda * (kloop / tileK) + bidx * tileM + tid * 2;
  quant_packed_type* sts_zeros_offset =
      (quant_packed_type*)(smem_base + tileK * tileN * sizeof(input_type)) +
      tid;
  input_type* sts_scales_offset =
      (input_type*)(smem_base + tileK * tileN * sizeof(input_type) +
                    tileM / PACK_RATIO * sizeof(quant_packed_type)) +
      tid * 2;
  quant_packed_type* lds_zeros_offset =
      (quant_packed_type*)(smem_base + tileK * tileN * sizeof(input_type)) +
      m64m16;
  input_type* lds_scales_offset =
      (input_type*)(smem_base + tileK * tileN * sizeof(input_type) +
                    tileM / PACK_RATIO * sizeof(quant_packed_type)) +
      m64m16 * 8;

  // clear C registers
  rgC[0] = 0;
  rgC[1] = 0;
  rgC[2] = 0;
  rgC[3] = 0;
  rgC[4] = 0;
  rgC[5] = 0;
  rgC[6] = 0;
  rgC[7] = 0;
  rgC[8] = 0;
  rgC[9] = 0;
  rgC[10] = 0;
  rgC[11] = 0;
  rgC[12] = 0;
  rgC[13] = 0;
  rgC[14] = 0;
  rgC[15] = 0;

  int aincr = tileK * acolstride / PACK_RATIO;
  int bincr = tileK * browstride;

  if (k_num > ktail) {
    LDG_B;
    LDG_ZEROS;
    LDG_SCALES;
    LDG_A;

    dA[0] += aincr;
    dB[0] += bincr;
    dB[1] += bincr;

    // lds_A, need swizzle
    arrive_gvmcnt(8);
    barrier();
    LDS_B;
    LDS_ZEROS;
    LDS_SCALES;
    arrive_bsmcnt(0);
    barrier();

    ldg_zeros_offset += lda / PACK_RATIO;
    ldg_scales_offset += lda;

// KLOOP
#pragma unroll 1
    for (kloop = tileK; kloop + tileK <= k_num; kloop += tileK) {
      // load next B
      LDG_B;
      LDG_ZEROS;
      LDG_SCALES;

      // MMA_ABC1;
      arrive_gvmcnt(4);
      PERM_A;
      LDG_A;
      MMA;

      // sts && lds
      arrive_gvmcnt(8);
      barrier();
      LDS_B;
      LDS_ZEROS;
      LDS_SCALES;
      arrive_bsmcnt(0);
      barrier();

      dA[0] += aincr;
      dB[0] += bincr;
      dB[1] += bincr;
      ldg_zeros_offset += lda / PACK_RATIO;
      ldg_scales_offset += lda;
    }

    arrive_gvmcnt(0);
    PERM_A;
    MMA;
  }

  // final tail kloop
  if (ktail > 0) {
    LDG_B_HEAD;
    LDG_ZEROS;
    LDG_SCALES;
    LDG_A_HEAD;
    arrive_gvmcnt(8);
    barrier();
    LDS_B;
    LDS_ZEROS;
    LDS_SCALES;
    arrive_bsmcnt(0);
    barrier();
    arrive_gvmcnt(0);
    PERM_A;
    MMA;
  }

  // store C registers into BSM & do reduction
  int colC = m64m16 + slot * 16;
  int rowC = m64d16 * 32;
  scalar_type* c_sts[8];
  c_sts[0] = (scalar_type*)smem_base +
             ONE_DIM_INDEX(SWIZZLE_INDEX(rowC, (colC % 4 * 4)), colC, 128);
  c_sts[1] = (scalar_type*)smem_base +
             ONE_DIM_INDEX(SWIZZLE_INDEX(rowC + 4, (colC % 4 * 4)), colC, 128);
  c_sts[2] = (scalar_type*)smem_base +
             ONE_DIM_INDEX(SWIZZLE_INDEX(rowC + 8, (colC % 4 * 4)), colC, 128);
  c_sts[3] = (scalar_type*)smem_base +
             ONE_DIM_INDEX(SWIZZLE_INDEX(rowC + 12, (colC % 4 * 4)), colC, 128);
  c_sts[4] = (scalar_type*)smem_base +
             ONE_DIM_INDEX(SWIZZLE_INDEX(rowC + 16, (colC % 4 * 4)), colC, 128);
  c_sts[5] = (scalar_type*)smem_base +
             ONE_DIM_INDEX(SWIZZLE_INDEX(rowC + 20, (colC % 4 * 4)), colC, 128);
  c_sts[6] = (scalar_type*)smem_base +
             ONE_DIM_INDEX(SWIZZLE_INDEX(rowC + 24, (colC % 4 * 4)), colC, 128);
  c_sts[7] = (scalar_type*)smem_base +
             ONE_DIM_INDEX(SWIZZLE_INDEX(rowC + 28, (colC % 4 * 4)), colC, 128);

  colC = m64d16 + slot * 4;
  rowC = m64m16 * 4;
  scalar_type* c_lds[2];
  c_lds[0] = (scalar_type*)smem_base +
             ONE_DIM_INDEX(SWIZZLE_INDEX(rowC, (colC % 4 * 4)), colC, 128);
  c_lds[1] = (scalar_type*)smem_base +
             ONE_DIM_INDEX(SWIZZLE_INDEX(rowC + 64, (colC % 4 * 4)), colC, 128);

  PERM_C(rgC);
  STS_C(0);
  arrive_bsmcnt(0);
  barrier();
  REDUCE_C(0);

  arrive_bsmcnt(0);
  barrier();

  PERM_C((rgC + 8))
  STS_C(1);
  arrive_bsmcnt(0);
  barrier();
  REDUCE_C(1);

  arrive_bsmcnt(0);

  // read C_input if beta!=0
  colC = m64d16 + slot * 4 + bidy * tileN;
  rowC = m64m16 * 4 + bidx * tileM;
  int colC1 = colC + 16;
  int rowC1 = rowC + 64;

  if constexpr (!IsBetaZero && !splitk) {
    ldg_b64_reg_async(
        cast_b64(rgCi)[0],
        c_base_i + (uint64_t)rowC + (uint64_t)(colC) * (uint64_t)(ldc),
        rowC < align_m && colC < align_n, true);
    ldg_b64_reg_async(cast_b64(rgCi)[1],
                      c_base_i + rowC1 + (uint64_t)(colC) * (uint64_t)(ldc),
                      rowC1 < align_m && colC < align_n, true);
    ldg_b64_reg_async(cast_b64(rgCi)[2],
                      c_base_i + rowC + (uint64_t)(colC1) * (uint64_t)(ldc),
                      rowC < align_m && colC1 < align_n, true);
    ldg_b64_reg_async(cast_b64(rgCi)[3],
                      c_base_i + rowC1 + (uint64_t)(colC1) * (uint64_t)(ldc),
                      rowC1 < align_m && colC1 < align_n, true);
    arrive_gvmcnt(0);
  }

  if constexpr (IsBetaZero && !splitk) {
    for (int i = 0; i < 4; ++i) rgCi[i] = static_cast<b64VecType>(0);
  }

  if constexpr (!splitk) {
    output_type* ptrCi = reinterpret_cast<output_type*>(rgCi);
    b64VecType* ptrO = reinterpret_cast<b64VecType*>(
        c_base_o + (uint64_t)(rowC) + (uint64_t)(colC) * (uint64_t)(ldc));
    if (rowC < align_m && colC < align_n) {
      output_type result[4];
      result[0] = static_cast<output_type>(
          alpha * rgC[0][0] + beta * static_cast<scalar_type>(ptrCi[0]));
      result[1] = static_cast<output_type>(
          alpha * rgC[0][1] + beta * static_cast<scalar_type>(ptrCi[1]));
      result[2] = static_cast<output_type>(
          alpha * rgC[0][2] + beta * static_cast<scalar_type>(ptrCi[2]));
      result[3] = static_cast<output_type>(
          alpha * rgC[0][3] + beta * static_cast<scalar_type>(ptrCi[3]));
      ptrO[0] = cast_b64(result)[0];
    }
    ptrO = reinterpret_cast<b64VecType*>(c_base_o + (uint64_t)(rowC1) +
                                         (uint64_t)(colC) * (uint64_t)(ldc));
    if (rowC1 < align_m && colC < align_n) {
      output_type result[4];
      result[0] = static_cast<output_type>(
          alpha * rgC[1][0] + beta * static_cast<scalar_type>(ptrCi[0]));
      result[1] = static_cast<output_type>(
          alpha * rgC[1][1] + beta * static_cast<scalar_type>(ptrCi[1]));
      result[2] = static_cast<output_type>(
          alpha * rgC[1][2] + beta * static_cast<scalar_type>(ptrCi[2]));
      result[3] = static_cast<output_type>(
          alpha * rgC[1][3] + beta * static_cast<scalar_type>(ptrCi[3]));
      ptrO[0] = cast_b64(result)[0];
    }
    ptrO = reinterpret_cast<b64VecType*>(c_base_o + (uint64_t)(rowC) +
                                         (uint64_t)(colC1) * (uint64_t)(ldc));
    if (rowC < align_m && colC1 < align_n) {
      output_type result[4];
      result[0] = static_cast<output_type>(
          alpha * rgC[2][0] + beta * static_cast<scalar_type>(ptrCi[0]));
      result[1] = static_cast<output_type>(
          alpha * rgC[2][1] + beta * static_cast<scalar_type>(ptrCi[1]));
      result[2] = static_cast<output_type>(
          alpha * rgC[2][2] + beta * static_cast<scalar_type>(ptrCi[2]));
      result[3] = static_cast<output_type>(
          alpha * rgC[2][3] + beta * static_cast<scalar_type>(ptrCi[3]));
      ptrO[0] = cast_b64(result)[0];
    }
    ptrO = reinterpret_cast<b64VecType*>(c_base_o + (uint64_t)(rowC1) +
                                         (uint64_t)(colC1) * (uint64_t)(ldc));
    if (rowC1 < align_m && colC1 < align_n) {
      output_type result[4];
      result[0] = static_cast<output_type>(
          alpha * rgC[3][0] + beta * static_cast<scalar_type>(ptrCi[0]));
      result[1] = static_cast<output_type>(
          alpha * rgC[3][1] + beta * static_cast<scalar_type>(ptrCi[1]));
      result[2] = static_cast<output_type>(
          alpha * rgC[3][2] + beta * static_cast<scalar_type>(ptrCi[2]));
      result[3] = static_cast<output_type>(
          alpha * rgC[3][3] + beta * static_cast<scalar_type>(ptrCi[3]));
      ptrO[0] = cast_b64(result)[0];
    }
  } else {
    acc_type* ptrAcc =
        d_acc_tmp + (uint64_t)(rowC) + (uint64_t)(colC) * (uint64_t)(ldc);
    if (rowC < align_m && colC < align_n) {
      atomicAdd(&ptrAcc[0], static_cast<acc_type>(alpha) *
                                static_cast<acc_type>(rgC[0][0]));
      atomicAdd(&ptrAcc[1], static_cast<acc_type>(alpha) *
                                static_cast<acc_type>(rgC[0][1]));
      atomicAdd(&ptrAcc[2], static_cast<acc_type>(alpha) *
                                static_cast<acc_type>(rgC[0][2]));
      atomicAdd(&ptrAcc[3], static_cast<acc_type>(alpha) *
                                static_cast<acc_type>(rgC[0][3]));
    }
    ptrAcc = d_acc_tmp + (uint64_t)(rowC1) + (uint64_t)(colC) * (uint64_t)(ldc);
    if (rowC1 < align_m && colC < align_n) {
      atomicAdd(&ptrAcc[0], static_cast<acc_type>(alpha) *
                                static_cast<acc_type>(rgC[1][0]));
      atomicAdd(&ptrAcc[1], static_cast<acc_type>(alpha) *
                                static_cast<acc_type>(rgC[1][1]));
      atomicAdd(&ptrAcc[2], static_cast<acc_type>(alpha) *
                                static_cast<acc_type>(rgC[1][2]));
      atomicAdd(&ptrAcc[3], static_cast<acc_type>(alpha) *
                                static_cast<acc_type>(rgC[1][3]));
    }
    ptrAcc = d_acc_tmp + (uint64_t)(rowC) + (uint64_t)(colC1) * (uint64_t)(ldc);
    if (rowC < align_m && colC1 < align_n) {
      atomicAdd(&ptrAcc[0], static_cast<acc_type>(alpha) *
                                static_cast<acc_type>(rgC[2][0]));
      atomicAdd(&ptrAcc[1], static_cast<acc_type>(alpha) *
                                static_cast<acc_type>(rgC[2][1]));
      atomicAdd(&ptrAcc[2], static_cast<acc_type>(alpha) *
                                static_cast<acc_type>(rgC[2][2]));
      atomicAdd(&ptrAcc[3], static_cast<acc_type>(alpha) *
                                static_cast<acc_type>(rgC[2][3]));
    }
    ptrAcc =
        d_acc_tmp + (uint64_t)(rowC1) + (uint64_t)(colC1) * (uint64_t)(ldc);
    if (rowC1 < align_m && colC1 < align_n) {
      atomicAdd(&ptrAcc[0], static_cast<acc_type>(alpha) *
                                static_cast<acc_type>(rgC[3][0]));
      atomicAdd(&ptrAcc[1], static_cast<acc_type>(alpha) *
                                static_cast<acc_type>(rgC[3][1]));
      atomicAdd(&ptrAcc[2], static_cast<acc_type>(alpha) *
                                static_cast<acc_type>(rgC[3][2]));
      atomicAdd(&ptrAcc[3], static_cast<acc_type>(alpha) *
                                static_cast<acc_type>(rgC[3][3]));
    }
  }
}