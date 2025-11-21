// 2025 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#pragma once

#include "../gptq/Hgemm_common.cuh"
#include "dequant.cuh"
#define quant_packed_type uint32_t

#define QBITS 4
#define PACK_RATIO (32 / QBITS)

#define LDG_A                                                                         \
    {                                                                                 \
        ldg_b32_reg_async(rga[0], dA[0] + acolstride * 0 / PACK_RATIO, pred_m, true); \
        ldg_b32_reg_async(rga[1], dA[0] + acolstride * 1 / PACK_RATIO, pred_m, true); \
        ldg_b32_reg_async(rga[2], dA[0] + acolstride * 2 / PACK_RATIO, pred_m, true); \
        ldg_b32_reg_async(rga[3], dA[0] + acolstride * 3 / PACK_RATIO, pred_m, true); \
        ldg_b32_reg_async(rga[4], dA[0] + acolstride * 4 / PACK_RATIO, pred_m, true); \
        ldg_b32_reg_async(rga[5], dA[0] + acolstride * 5 / PACK_RATIO, pred_m, true); \
        ldg_b32_reg_async(rga[6], dA[0] + acolstride * 6 / PACK_RATIO, pred_m, true); \
        ldg_b32_reg_async(rga[7], dA[0] + acolstride * 7 / PACK_RATIO, pred_m, true); \
    }

#define LDG_A_HEAD                                                                    \
    {                                                                                 \
        bool predk = colA < ktail;                                                    \
        ldg_b32_reg_async(rga[0], dA[0] + acolstride * 0 / PACK_RATIO, pred_m && predk, true); \
        ldg_b32_reg_async(rga[1], dA[0] + acolstride * 1 / PACK_RATIO, pred_m && predk, true); \
        ldg_b32_reg_async(rga[2], dA[0] + acolstride * 2 / PACK_RATIO, pred_m && predk, true); \
        ldg_b32_reg_async(rga[3], dA[0] + acolstride * 3 / PACK_RATIO, pred_m && predk, true); \
        ldg_b32_reg_async(rga[4], dA[0] + acolstride * 4 / PACK_RATIO, pred_m && predk, true); \
        ldg_b32_reg_async(rga[5], dA[0] + acolstride * 5 / PACK_RATIO, pred_m && predk, true); \
        ldg_b32_reg_async(rga[6], dA[0] + acolstride * 6 / PACK_RATIO, pred_m && predk, true); \
        ldg_b32_reg_async(rga[7], dA[0] + acolstride * 7 / PACK_RATIO, pred_m && predk, true); \
    }

// 每个block中用tileM/PACK_RATIO个线程去加载tileM个zeros，zeros为int4类型，每个32bits寄存器可以保存8个zeros
#define LDG_ZEROS                                                                                                                              \
    {                                                                                                                                          \
        ldg_b32_bsm_async(sts_zeros_offset, ldg_zeros_offset, tid < tileM / PACK_RATIO && (bidx * tileM + tid * PACK_RATIO < align_m), false); \
    }

// 每个block中用tileM/(sizeof(uint32_t)/sizeof(__half))个线程去加载tileM个scales，scale为fp16类型，每个32bits寄存器可以保存2个scales
#define LDG_SCALES                                                                                                      \
    {                                                                                                                   \
        ldg_b32_bsm_async(sts_scales_offset, ldg_scales_offset, tid < 64 && (bidx * tileM + tid * 2 < align_m), false); \
    }

#define LDS_ZEROS                         \
    {                                     \
        rgZeros[0] = lds_zeros_offset[0]; \
    }

#define LDS_SCALES                                     \
    {                                                  \
        rgScales[0] = cast_b128(lds_scales_offset)[0]; \
    }

// zeros中从低字节到高字节分别对应第 0 2 4 6 1 3 5 7 列， 所以index_shfl = index % 2 * 4 + index / 2
//    提取出int4的zeros后转成int32，再把weight（矩阵A）还原成fp16，Weight_Q=Scale*(Weight_4Bit-ZeroPoint)
#define PERM_ELEM(index)                                                                                                          \
    {                                                                                                                             \
        __half_raw elem;                                                                                                          \
        if constexpr (index & 0x1)                                                                                                \
        {                                                                                                                         \
            elem.x = cast_b32(rgScales)[index / 2] >> 16;                                                                         \
        }                                                                                                                         \
        else                                                                                                                      \
        {                                                                                                                         \
            elem.x = cast_b32(rgScales)[index / 2] & 0xffff;                                                                      \
        }                                                                                                                         \
        __half scale = __half(elem);                                                                                              \
        constexpr int index_shfl = index % 2 * 4 + index / 2;                                                                     \
        uint32_t zero = __builtin_mxc_ubfe(rgZeros[0], QBITS * index_shfl, QBITS);                                                \
        cast_half(rgA)[index * 4 + 0] = __hmul((__half)__builtin_mxc_i16_to_f16(__builtin_mxc_ubfe(rga[0], QBITS * index_shfl, QBITS) -  \
                                                                         zero) , scale);                                                                                    \
        cast_half(rgA)[index * 4 + 1] = __hmul((__half)__builtin_mxc_i16_to_f16(__builtin_mxc_ubfe(rga[1], QBITS * index_shfl, QBITS) -  \
                                                                         zero) , scale);                                                                                    \
        cast_half(rgA)[index * 4 + 2] = __hmul((__half)__builtin_mxc_i16_to_f16(__builtin_mxc_ubfe(rga[2], QBITS * index_shfl, QBITS) -  \
                                                                         zero) , scale);                                                                                    \
        cast_half(rgA)[index * 4 + 3] = __hmul((__half)__builtin_mxc_i16_to_f16(__builtin_mxc_ubfe(rga[3], QBITS * index_shfl, QBITS) -  \
                                                                         zero) , scale);                                                                                    \
        cast_half(rgA)[index * 4 + 32] = __hmul((__half)__builtin_mxc_i16_to_f16(__builtin_mxc_ubfe(rga[4], QBITS * index_shfl, QBITS) - \
                                                                          zero) , scale);                                                                                   \
        cast_half(rgA)[index * 4 + 33] = __hmul((__half)__builtin_mxc_i16_to_f16(__builtin_mxc_ubfe(rga[5], QBITS * index_shfl, QBITS) - \
                                                                          zero) , scale);                                                                                   \
        cast_half(rgA)[index * 4 + 34] = __hmul((__half)__builtin_mxc_i16_to_f16(__builtin_mxc_ubfe(rga[6], QBITS * index_shfl, QBITS) - \
                                                                          zero) , scale);                                                                                   \
        cast_half(rgA)[index * 4 + 35] = __hmul((__half)__builtin_mxc_i16_to_f16(__builtin_mxc_ubfe(rga[7], QBITS * index_shfl, QBITS) - \
                                                                          zero) , scale);                                                                                   \
    }

#define PERM_A       \
    {                \
        PERM_ELEM(0) \
        PERM_ELEM(1) \
        PERM_ELEM(2) \
        PERM_ELEM(3) \
        PERM_ELEM(4) \
        PERM_ELEM(5) \
        PERM_ELEM(6) \
        PERM_ELEM(7) \
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

#define MMA     \
    MMA_ELEM(0) \
    MMA_ELEM(1) \
    MMA_ELEM(2) \
    MMA_ELEM(3) \
    MMA_ELEM(4) \
    MMA_ELEM(5) \
    MMA_ELEM(6) \
    MMA_ELEM(7)
