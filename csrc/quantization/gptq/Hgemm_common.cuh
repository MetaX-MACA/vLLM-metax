#pragma once
#include "maca_fp16.h"

using b32VecType = uint32_t;
using b64VecType = __NATIVE_VECTOR__(2, uint32_t);
using b128VecType = __NATIVE_VECTOR__(4, uint32_t);
using b128VecType_i = __NATIVE_VECTOR__(4, int32_t);
using Float4VecType = __NATIVE_VECTOR__(4, float);
typedef enum { OP_N = 0, OP_T = 1 } Operation_t;

#define cast_half(ptr) reinterpret_cast<__half*>(ptr)
#define cast_b16(ptr) reinterpret_cast<uint16_t*>(ptr)
#define cast_b32(ptr) reinterpret_cast<uint32_t*>(ptr)
#define cast_b64(ptr) reinterpret_cast<b64VecType*>(ptr)
#define cast_u64(ptr) reinterpret_cast<uint64_t*>(ptr)
#define cast_b128(ptr) reinterpret_cast<b128VecType*>(ptr)
#define cast_b128_i(ptr) reinterpret_cast<b128VecType_i*>(ptr)

#define ldg_b32_reg_noasync(dst, base, pred, ret0_en)                      \
  dst = __builtin_mxc_ldg_b32_predicator(cast_b32(base), 0, ret0_en, true, \
                                         false, false, pred, 1,            \
                                         MACA_ICMP_EQ)[0];
#define ldg_b64_reg_noasync(dst, base, pred, ret0_en)                      \
  dst = __builtin_mxc_ldg_b64_predicator(cast_b64(base), 0, ret0_en, true, \
                                         false, false, pred, 1, MACA_ICMP_EQ);
#define ldg_b128_reg_noasync(dst, base, pred, ret0_en) \
  dst = __builtin_mxc_ldg_b128_predicator(             \
      cast_b128(base), 0, ret0_en, true, false, false, pred, 1, MACA_ICMP_EQ);

#define ldg_b32_reg_async(dst, base, pred, ret0_en)                      \
  dst =                                                                  \
      __builtin_mxc_ldg_b32_predicator(cast_b32(base), 0, ret0_en, true, \
                                       false, true, pred, 1, MACA_ICMP_EQ)[0];
#define ldg_b64_reg_async(dst, base, pred, ret0_en)                        \
  dst = __builtin_mxc_ldg_b64_predicator(cast_b64(base), 0, ret0_en, true, \
                                         false, true, pred, 1, MACA_ICMP_EQ);
#define ldg_b128_reg_async(dst, base, pred, ret0_en)                         \
  dst = __builtin_mxc_ldg_b128_predicator(cast_b128(base), 0, ret0_en, true, \
                                          false, true, pred, 1, MACA_ICMP_EQ);
#define ldg_b64_v4h_reg_async(dst, base, pred, ret0_en)                    \
  dst = __builtin_mxc_ldg_b64_predicator(cast_b64(base), 0, ret0_en, true, \
                                         false, true, pred, 1, MACA_ICMP_EQ);

#define ldg_b32_bsm_noasync(saddr, base, pred, ret0_en)                      \
  __builtin_mxc_ldg_b32_bsm_predicator(cast_b32(saddr), cast_b32(base), 0,   \
                                       ret0_en, true, false, false, pred, 1, \
                                       MACA_ICMP_EQ);
#define ldg_b64_bsm_noasync(saddr, base, pred, ret0_en)                      \
  __builtin_mxc_ldg_b64_bsm_predicator(cast_b64(saddr), cast_b64(base), 0,   \
                                       ret0_en, true, false, false, pred, 1, \
                                       MACA_ICMP_EQ);
#define ldg_b128_bsm_noasync(saddr, base, pred, ret0_en)                      \
  __builtin_mxc_ldg_b128_bsm_predicator(cast_b128(saddr), cast_b128(base), 0, \
                                        ret0_en, true, false, false, pred, 1, \
                                        MACA_ICMP_EQ);

#define ldg_b32_bsm_async(saddr, base, pred, ret0_en)                       \
  __builtin_mxc_ldg_b32_bsm_predicator(cast_b32(saddr), cast_b32(base), 0,  \
                                       ret0_en, true, false, true, pred, 1, \
                                       MACA_ICMP_EQ);
#define ldg_b64_bsm_async(saddr, base, pred, ret0_en)                       \
  __builtin_mxc_ldg_b64_bsm_predicator(cast_b64(saddr), cast_b64(base), 0,  \
                                       ret0_en, true, false, true, pred, 1, \
                                       MACA_ICMP_EQ);
#define ldg_b128_bsm_async(saddr, base, pred, ret0_en)                        \
  __builtin_mxc_ldg_b128_bsm_predicator(cast_b128(saddr), cast_b128(base), 0, \
                                        ret0_en, true, false, true, pred, 1,  \
                                        MACA_ICMP_EQ);

#define stg_b32_async(data, base, pred)                                        \
  __builtin_mxc_stg_b32_predicator(cast_b32(base), 0, data, true, false, true, \
                                   pred, 1, MACA_ICMP_EQ);
#define stg_b64_async(data, base, pred)                                        \
  __builtin_mxc_stg_b64_predicator(cast_b64(base), 0, data, true, false, true, \
                                   pred, 1, MACA_ICMP_EQ);
#define stg_b128_async(data, base, pred)                                   \
  __builtin_mxc_stg_b128_predicator(cast_b128(base), 0, data, true, false, \
                                    true, pred, 1, MACA_ICMP_EQ);

#define perm_b32(dst, reg1, reg2, selector) \
  dst = __builtin_mxc_byte_perm(reg1, reg2, selector)
#define mma_16x16x16f16(a_reg, b_reg, c_reg) \
  c_reg = __builtin_mxc_mma_16x16x16f16(a_reg, b_reg, c_reg)

#define mma_16x16x16bf16(a_reg, b_reg, c_reg) \
  c_reg = __builtin_mxc_mma_16x16x16bf16(a_reg, b_reg, c_reg)

#define FENCE__ asm volatile(";")
#define arrive_gvmcnt(num) __builtin_mxc_arrive(64 + num)
#define arrive_bsmcnt(num) __builtin_mxc_arrive(4096 + 128 * num)
#define arrive_gvm_bsmcnt(gvm, bsm) \
  __builtin_mxc_arrive(4096 | (128 * bsm) | 64 | gvm)
#define barrier __builtin_mxc_barrier_inst
#define barrier_all __builtin_mxc_barrier_ex(0)
#define barrier_bsm __builtin_mxc_barrier_ex(1)
#define barrier_inst __builtin_mxc_barrier_ex(2)
