// 2025 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#pragma once

#include "Hgemm_common.cuh"
#include "dequant.cuh"
#define quant_packed_type uint32_t

#define Q4BITS 4
#define PACK_RATIO_4BITS (32 / Q4BITS)

#define Q8BITS 8
#define PACK_RATIO_8BITS (32 / Q8BITS)

#define input_type __half
#define output_type __half
#define scalar_type float
//#define acc_type float
#define acc_type __half

#define SEL0 0x01000504
#define SEL1 0x03020706

#define SWIZZLE_INDEX(index, phase) (index ^ phase)
#define ONE_DIM_INDEX(x, y, lda) (y * lda + x)
#define CLAMP(value, bound, align) (value < bound ? value : bound - align)

#define LDG_A1                                                        \
    {                                                                 \
        ldg_b128_reg_async(cast_b128(rga)[0], dA[0], predm[0], true); \
    }

#define LDG_A2                                                        \
    {                                                                 \
        ldg_b128_reg_async(cast_b128(rga)[1], dA[1], predm[1], true); \
    }

#define LDG_A3                                                        \
    {                                                                 \
        ldg_b128_reg_async(cast_b128(rga)[2], dA[2], predm[0], true); \
    }

#define LDG_A4                                                        \
    {                                                                 \
        ldg_b128_reg_async(cast_b128(rga)[3], dA[3], predm[1], true); \
    }

#define LDG_A1_4BITS LDG_A1
#define LDG_A2_4BITS LDG_A2
#define LDG_A1_8BITS LDG_A1
#define LDG_A2_8BITS LDG_A2
#define LDG_A3_8BITS LDG_A3
#define LDG_A4_8BITS LDG_A4

#define LDG_A1_HEAD_4BITS                                                      \
    {                                                                          \
        bool predk = colA < ktail / PACK_RATIO_4BITS;                          \
        ldg_b128_reg_async(cast_b128(rga)[0], dA[0], predm[0] && predk, true); \
    }

#define LDG_A2_HEAD_4BITS                                                      \
    {                                                                          \
        bool predk = colA < ktail / PACK_RATIO_4BITS;                          \
        ldg_b128_reg_async(cast_b128(rga)[1], dA[1], predm[1] && predk, true); \
    }

#define LDG_A1_HEAD_8BITS                                                      \
    {                                                                          \
        bool predk = colA < ktail / PACK_RATIO_8BITS;                          \
        ldg_b128_reg_async(cast_b128(rga)[0], dA[0], predm[0] && predk, true); \
    }

#define LDG_A2_HEAD_8BITS                                                      \
    {                                                                          \
        bool predk = colA < ktail / PACK_RATIO_8BITS;                          \
        ldg_b128_reg_async(cast_b128(rga)[1], dA[1], predm[1] && predk, true); \
    }

#define LDG_A3_HEAD_8BITS                                                      \
    {                                                                          \
        bool predk = colA < ktail / PACK_RATIO_8BITS;                          \
        ldg_b128_reg_async(cast_b128(rga)[2], dA[2], predm[0] && predk, true); \
    }

#define LDG_A4_HEAD_8BITS                                                      \
    {                                                                          \
        bool predk = colA < ktail / PACK_RATIO_8BITS;                          \
        ldg_b128_reg_async(cast_b128(rga)[3], dA[3], predm[1] && predk, true); \
    }

#define LDG_ZEROS_4BITS                                                                                                                        \
    {                                                                                                                                          \
        ldg_b32_bsm_async(sts_zeros_offset, ldg_zeros_offset, tid < tileM / PACK_RATIO_4BITS && (bidx * tileM + tid * PACK_RATIO_4BITS < align_m), false); \
    }

#define LDG_ZEROS_8BITS                                                                                                                        \
    {                                                                                                                                          \
        ldg_b32_bsm_async(sts_zeros_offset, ldg_zeros_offset, tid < tileM / PACK_RATIO_8BITS && (bidx * tileM + tid * PACK_RATIO_8BITS < align_m), false); \
    }

#define LDG_SCALES                                                                                                      \
    {                                                                                                                   \
        ldg_b32_bsm_async(sts_scales_offset, ldg_scales_offset, tid < 64 && (bidx * tileM + tid * 2 < align_m), false); \
    }

#define LDG_SCALES_4BITS LDG_SCALES
#define LDG_SCALES_8BITS LDG_SCALES

#define LDS_ZEROS_4BITS                              \
    {                                                \
        cast_b16(rgZeros)[0] = lds_zeros_offset[0];  \
        cast_b16(rgZeros)[1] = lds_zeros_offset[16]; \
    }

#define LDS_ZEROS_8BITS                              \
    {                                                \
        cast_b32(rgZeros)[0] = lds_zeros_offset[0];  \
        cast_b32(rgZeros)[1] = lds_zeros_offset[16]; \
    }

#define LDS_SCALES                                               \
    {                                                            \
        cast_b64(rgScales)[0] = cast_b64(lds_scales_offset)[0];  \
        cast_b64(rgScales)[1] = cast_b64(lds_scales_offset)[16]; \
    }

#define LDS_SCALES_4BITS LDS_SCALES
#define LDS_SCALES_8BITS LDS_SCALES

#define PERM_ELEM_4BITS(index)                                                                                                          \
    {                                                                                                                                   \
        __half_raw elem;                                                                                                                \
        if constexpr (index & 0x1)                                                                                                      \
        {                                                                                                                               \
            elem.x = rgScales[index / 4][index % 4 / 2] >> 16;                                                                          \
        }                                                                                                                               \
        else                                                                                                                            \
        {                                                                                                                               \
            elem.x = rgScales[index / 4][index % 4 / 2] & 0xffff;                                                                       \
        }                                                                                                                               \
        __half scale = __half(elem);                                                                                                    \
        uint32_t zero = __builtin_mxc_ubfe(rgZeros[0], Q4BITS * index, Q4BITS) + 1;                                                     \
        if constexpr (doShuffle)                                                                                                        \
        {                                                                                                                               \
            cast_half(rgA)[index * 4 + 0] = __hmul((__half)__builtin_mxc_i16_to_f16(__builtin_mxc_ubfe(rga[index], Q4BITS * 0, Q4BITS) -       \
                                                                            zero) ,                                                     \
                                            scale);                                                                                      \
            cast_half(rgA)[index * 4 + 1] = __hmul((__half)__builtin_mxc_i16_to_f16(__builtin_mxc_ubfe(rga[index], Q4BITS * 4, Q4BITS) -       \
                                                                            zero) ,                                                     \
                                            scale);                                                                                      \
            cast_half(rgA)[index * 4 + 2] = __hmul((__half)__builtin_mxc_i16_to_f16(__builtin_mxc_ubfe(rga[index], Q4BITS * 1, Q4BITS) -       \
                                                                            zero) ,                                                     \
                                            scale);                                                                                      \
            cast_half(rgA)[index * 4 + 3] = __hmul((__half)__builtin_mxc_i16_to_f16(__builtin_mxc_ubfe(rga[index], Q4BITS * 5, Q4BITS) -       \
                                                                            zero) ,                                                     \
                                            scale);                                                                                      \
            cast_half(rgA)[index * 4 + 32] = __hmul((__half)__builtin_mxc_i16_to_f16(__builtin_mxc_ubfe(rga[index], Q4BITS * 2, Q4BITS) -      \
                                                                            zero) ,                                                     \
                                            scale);                                                                                      \
            cast_half(rgA)[index * 4 + 33] = __hmul((__half)__builtin_mxc_i16_to_f16(__builtin_mxc_ubfe(rga[index], Q4BITS * 6, Q4BITS) -      \
                                                                            zero) ,                                                     \
                                            scale);                                                                                      \
            cast_half(rgA)[index * 4 + 34] = __hmul((__half)__builtin_mxc_i16_to_f16(__builtin_mxc_ubfe(rga[index], Q4BITS * 3, Q4BITS) -      \
                                                                            zero) ,                                                     \
                                            scale);                                                                                      \
            cast_half(rgA)[index * 4 + 35] = __hmul((__half)__builtin_mxc_i16_to_f16(__builtin_mxc_ubfe(rga[index], Q4BITS * 7, Q4BITS) -      \
                                                                            zero) ,                                                     \
                                            scale);                                                                                      \
        }                                                                                                                               \
        else                                                                                                                            \
        {                                                                                                                               \
            cast_half(rgA)[index * 4 + 0] = __hmul((__half)__builtin_mxc_i16_to_f16(__builtin_mxc_ubfe(rga[index], Q4BITS * 0, Q4BITS) -       \
                                                                            zero) ,                                                     \
                                            scale);                                                                                      \
            cast_half(rgA)[index * 4 + 1] = __hmul((__half)__builtin_mxc_i16_to_f16(__builtin_mxc_ubfe(rga[index], Q4BITS * 1, Q4BITS) -       \
                                                                            zero) ,                                                     \
                                            scale);                                                                                      \
            cast_half(rgA)[index * 4 + 2] = __hmul((__half)__builtin_mxc_i16_to_f16(__builtin_mxc_ubfe(rga[index], Q4BITS * 2, Q4BITS) -       \
                                                                            zero) ,                                                     \
                                            scale);                                                                                      \
            cast_half(rgA)[index * 4 + 3] = __hmul((__half)__builtin_mxc_i16_to_f16(__builtin_mxc_ubfe(rga[index], Q4BITS * 3, Q4BITS) -       \
                                                                            zero) ,                                                     \
                                            scale);                                                                                      \
            cast_half(rgA)[index * 4 + 32] = __hmul((__half)__builtin_mxc_i16_to_f16(__builtin_mxc_ubfe(rga[index], Q4BITS * 4, Q4BITS) -      \
                                                                            zero) ,                                                     \
                                            scale);                                                                                      \
            cast_half(rgA)[index * 4 + 33] = __hmul((__half)__builtin_mxc_i16_to_f16(__builtin_mxc_ubfe(rga[index], Q4BITS * 5, Q4BITS) -      \
                                                                            zero) ,                                                     \
                                            scale);                                                                                      \
            cast_half(rgA)[index * 4 + 34] = __hmul((__half)__builtin_mxc_i16_to_f16(__builtin_mxc_ubfe(rga[index], Q4BITS * 6, Q4BITS) -      \
                                                                            zero) ,                                                     \
                                            scale);                                                                                      \
            cast_half(rgA)[index * 4 + 35] = __hmul((__half)__builtin_mxc_i16_to_f16(__builtin_mxc_ubfe(rga[index], Q4BITS * 7, Q4BITS) -      \
                                                                            zero) ,                                                     \
                                            scale);                                                                                      \
        }                                                                                                                               \
    }

#define PERM_A1_4BITS       \
    {                       \
        PERM_ELEM_4BITS(0)  \
        PERM_ELEM_4BITS(1)  \
        PERM_ELEM_4BITS(2)  \
        PERM_ELEM_4BITS(3)  \
    }

#define PERM_A2_4BITS       \
    {                       \
        PERM_ELEM_4BITS(4)  \
        PERM_ELEM_4BITS(5)  \
        PERM_ELEM_4BITS(6)  \
        PERM_ELEM_4BITS(7)  \
    }

#define PERM_ELEM_8BITS(index)                                                                                                          \
    {                                                                                                                                   \
        __half_raw elem;                                                                                                                \
        if constexpr (index & 0x1)                                                                                                      \
        {                                                                                                                               \
            elem.x = rgScales[index / 4][index % 4 / 2] >> 16;                                                                          \
        }                                                                                                                               \
        else                                                                                                                            \
        {                                                                                                                               \
            elem.x = rgScales[index / 4][index % 4 / 2] & 0xffff;                                                                       \
        }                                                                                                                               \
        __half scale = __half(elem);                                                                                                    \
        uint32_t zero = __builtin_mxc_ubfe(rgZeros[index/4], Q8BITS * index, Q8BITS) + 1;                                               \
                                                                                                                                        \
        cast_half(rgA)[index * 4 + 0] = __hmul((__half)__builtin_mxc_i16_to_f16(__builtin_mxc_ubfe(rga[index], Q8BITS * 0, Q8BITS) -           \
                                                                        zero) ,                                                         \
                                        scale);                                                                                          \
        cast_half(rgA)[index * 4 + 1] = __hmul((__half)__builtin_mxc_i16_to_f16(__builtin_mxc_ubfe(rga[index], Q8BITS * 1, Q8BITS) -           \
                                                                        zero) ,                                                         \
                                        scale);                                                                                          \
        cast_half(rgA)[index * 4 + 2] = __hmul((__half)__builtin_mxc_i16_to_f16(__builtin_mxc_ubfe(rga[index], Q8BITS * 2, Q8BITS) -           \
                                                                        zero) ,                                                         \
                                        scale);                                                                                          \
        cast_half(rgA)[index * 4 + 3] = __hmul((__half)__builtin_mxc_i16_to_f16(__builtin_mxc_ubfe(rga[index], Q8BITS * 3, Q8BITS) -           \
                                                                        zero) ,                                                         \
                                        scale);                                                                                          \
        cast_half(rgA)[index * 4 + 32] = __hmul((__half)__builtin_mxc_i16_to_f16(__builtin_mxc_ubfe(rga[index+8], Q8BITS * 0, Q8BITS) -        \
                                                                        zero) ,                                                         \
                                        scale);                                                                                          \
        cast_half(rgA)[index * 4 + 33] = __hmul((__half)__builtin_mxc_i16_to_f16(__builtin_mxc_ubfe(rga[index+8], Q8BITS * 1, Q8BITS) -        \
                                                                        zero) ,                                                         \
                                        scale);                                                                                          \
        cast_half(rgA)[index * 4 + 34] = __hmul((__half)__builtin_mxc_i16_to_f16(__builtin_mxc_ubfe(rga[index+8], Q8BITS * 2, Q8BITS) -        \
                                                                        zero) ,                                                         \
                                        scale);                                                                                          \
        cast_half(rgA)[index * 4 + 35] = __hmul((__half)__builtin_mxc_i16_to_f16(__builtin_mxc_ubfe(rga[index+8], Q8BITS * 3, Q8BITS) -        \
                                                                        zero) ,                                                         \
                                        scale);                                                                                          \
    }

#define PERM_A1A3_8BITS     \
    {                       \
        PERM_ELEM_8BITS(0)  \
        PERM_ELEM_8BITS(1)  \
        PERM_ELEM_8BITS(2)  \
        PERM_ELEM_8BITS(3)  \
    }

#define PERM_A2A4_8BITS     \
    {                       \
        PERM_ELEM_8BITS(4)  \
        PERM_ELEM_8BITS(5)  \
        PERM_ELEM_8BITS(6)  \
        PERM_ELEM_8BITS(7)  \
    }

#define MMA_ELEM(index_m)                                        \
    mma_16x16x16f16(cast_b64(rgA)[ONE_DIM_INDEX(index_m, 0, 8)], \
                    cast_b64(rgB)[ONE_DIM_INDEX(0, 0, 2)],       \
                    rgC[ONE_DIM_INDEX(index_m, 0, 8)]);          \
    mma_16x16x16f16(cast_b64(rgA)[ONE_DIM_INDEX(index_m, 0, 8)], \
                    cast_b64(rgB)[ONE_DIM_INDEX(0, 1, 2)],       \
                    rgC[ONE_DIM_INDEX(index_m, 1, 8)]);          \
    mma_16x16x16f16(cast_b64(rgA)[ONE_DIM_INDEX(index_m, 1, 8)], \
                    cast_b64(rgB)[ONE_DIM_INDEX(1, 0, 2)],       \
                    rgC[ONE_DIM_INDEX(index_m, 0, 8)]);          \
    mma_16x16x16f16(cast_b64(rgA)[ONE_DIM_INDEX(index_m, 1, 8)], \
                    cast_b64(rgB)[ONE_DIM_INDEX(1, 1, 2)],       \
                    rgC[ONE_DIM_INDEX(index_m, 1, 8)]);

#define MMA1    \
    MMA_ELEM(0) \
    MMA_ELEM(1) \
    MMA_ELEM(2) \
    MMA_ELEM(3)

#define MMA2    \
    MMA_ELEM(4) \
    MMA_ELEM(5) \
    MMA_ELEM(6) \
    MMA_ELEM(7)

#define LDG_B                                                        \
    {                                                                \
        ldg_b128_bsm_async(b_sts, dB[0], predn[0], true);            \
        ldg_b128_bsm_async(b_sts + 128 * 16, dB[1], predn[1], true); \
    }

#define LDG_B_HEAD                                                            \
    {                                                                         \
        bool predk = rowB_swizzle < ktail;                                    \
        ldg_b128_bsm_async(b_sts, dB[0], predn[0] && predk, true);            \
        ldg_b128_bsm_async(b_sts + 128 * 16, dB[1], predn[1] && predk, true); \
    }

#define LDS_B                                          \
    {                                                  \
        cast_b128(rgB)[0] = cast_b128(b_lds)[0];       \
        cast_b128(rgB)[1] = cast_b128(b_lds)[16 * 16]; \
    }

#define PERM_C(C2perm)                                       \
    {                                                        \
        Float4VecType C_tmp[8];                              \
        float *ptri = (float *)C2perm;                       \
        float *ptro = (float *)C_tmp;                        \
        for (int j = 0; j < 4; ++j)                          \
        {                                                    \
            for (int i = 0; i < 4; ++i)                      \
            {                                                \
                ptro[j * 4 + i] = ptri[j + i * 4];           \
                ptro[j * 4 + i + 16] = ptri[j + i * 4 + 16]; \
            }                                                \
        }                                                    \
        for (int i = 0; i < 8; ++i)                          \
        {                                                    \
            C2perm[i] = C_tmp[i];                            \
        }                                                    \
    }

#define STS_C(phase)                                            \
    {                                                           \
        cast_b128(c_sts[0])[0] = cast_b128(rgC)[0 + phase * 8]; \
        cast_b128(c_sts[1])[0] = cast_b128(rgC)[1 + phase * 8]; \
        cast_b128(c_sts[2])[0] = cast_b128(rgC)[2 + phase * 8]; \
        cast_b128(c_sts[3])[0] = cast_b128(rgC)[3 + phase * 8]; \
        cast_b128(c_sts[4])[0] = cast_b128(rgC)[4 + phase * 8]; \
        cast_b128(c_sts[5])[0] = cast_b128(rgC)[5 + phase * 8]; \
        cast_b128(c_sts[6])[0] = cast_b128(rgC)[6 + phase * 8]; \
        cast_b128(c_sts[7])[0] = cast_b128(rgC)[7 + phase * 8]; \
    }

#define REDUCE_C(phase)                                                   \
    {                                                                     \
        cast_b128(rgC)[0 + phase * 8] = cast_b128(c_lds[0])[0];           \
        cast_b128(rgC)[1 + phase * 8] = cast_b128(c_lds[1])[0];           \
        cast_b128(rgC)[2 + phase * 8] = cast_b128(c_lds[0])[1 * 16 * 32]; \
        cast_b128(rgC)[3 + phase * 8] = cast_b128(c_lds[1])[1 * 16 * 32]; \
        cast_b128(rgC)[4 + phase * 8] = cast_b128(c_lds[0])[2 * 16 * 32]; \
        cast_b128(rgC)[5 + phase * 8] = cast_b128(c_lds[1])[2 * 16 * 32]; \
        cast_b128(rgC)[6 + phase * 8] = cast_b128(c_lds[0])[3 * 16 * 32]; \
        cast_b128(rgC)[7 + phase * 8] = cast_b128(c_lds[1])[3 * 16 * 32]; \
        float *reduc_c = reinterpret_cast<float *>(rgC);                  \
        for (int loop = 0; loop < 8; ++loop)                              \
        {                                                                 \
            float acc = 0;                                                \
            acc += reduc_c[loop + phase * 32 + 0 * 8];                    \
            acc += reduc_c[loop + phase * 32 + 1 * 8];                    \
            acc += reduc_c[loop + phase * 32 + 2 * 8];                    \
            acc += reduc_c[loop + phase * 32 + 3 * 8];                    \
            reduc_c[loop + phase * 8] = acc;                              \
        }                                                                 \
    }
