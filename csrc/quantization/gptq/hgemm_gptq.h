// 2025 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
/*
hgemm gptq 4bits

hgemm inputs:
half/tf16 a[m*k];
uint32_t b[k*n/pack_ratio]; //pack_ratio = sizeof(uint32_t) / 4;
uint32_t zp[k*n] with kU4 or zp[0] with kU4B8;
half scales[m*k/quant_group];

MMA accepts a m16xk16 b k16xn16
A tile should be m16n64k128 or m32n64k128
When m>=32, we should load a when compute, implement this stragedge later?

We will make a block size 256, which means 4 waves, we need all these 4 waves execute on one PEU
So the maximum shared memory used by a block use not larger than 16K, and MTRegisters should be
less than 128.

A tile will be divied into serveral sub_tiles, a minimum sub_tile is m16n16k32, so we can have
m_blocks*k_blocks when loading a and k_blocks*n_blocks when loading b.

We will have n*k/TILE_N/TILE_K tiles in b, and these TILES are diveded into ITERS where ITERS*PEUS>=TILES
for a proper shape.
Of course some small shapes will nerver have chance to use all the PEUS

Parallism is not suitable for C500 as parallism will dequant b more than once, that is not acceptable
*/
#include <iostream>
#include <algorithm>

#include <mc_runtime.h>
#include <maca_fp16.h>
#include <maca_bfloat16.h>

#include "Hgemm_common.cuh"
#include "scalar_type.hpp"

using cudaStream_t = mcStream_t;

#define WAVES_PER_BLOCK (THREADS/WAVE)
#define TILE_M (BLOCKS_M*SLICE_M)
#define TILE_N (BLOCKS_N*SLICE_N)
#define TILE_K (BLOCKS_K*SLICE_K)
#define N_ITERS (TILE_N / (WAVES_PER_BLOCK*SLOT))
#define LOADING_A_LOOP SLICE_K * TILE_M / (sizeof(PackType) / sizeof(scalar_t)) / THREADS
#define AS_PTR_B128(x) ((PackTypeInt4*)x)
#define AS_PTR_B64(x) ((PackTypeInt2*)x)
#define AS_PTR_B32(x) ((float*)x)
#define AS_PTR_B16(x) ((half*)x)
#define AS_PTR_B8(x) ((uint8_t*)x)

#define BF16_HIGH_PRECISION

#define div_ceil(x, y) (x + y - 1) / (y)

//Although quant_group can be any positive value, but it is not a good idea
//to set quant_group to values that cannot fit the SLICE_K as we will get a
//very low performance, and we are not ready to support these values
//Here we annouce that we support quant_group = 32, 64, 128, but actually
//quant_group = 2^n where n >= 5 is also supported, for very large k.
static int get_power2(uint32_t v) {
    uint32_t power = 0;
    uint32_t mask = 0x00000001;
    while (power < 32) {
        if ((v & mask) > 0) break;
        power++;
        mask <<= 1;
    }
    if ((1 << power) != v) return -1;
    return static_cast<int>(power);
}



namespace hgemm_marlin_gptq {

constexpr static int clean_kernel_thread_num = 512;
constexpr static int clean_kernel_pack_num = 4;
constexpr static int reduce_kernel_thread_num = 512;
constexpr static int reduce_kernel_pack_num = 4;

//#define DEBUG
using PackTypeInt4 = b128VecType;
using PackTypeInt2 = b64VecType;
using PackType = uint32_t;

template<class scalar_t>
__device__ __forceinline__ void mma_16x16x16(PackTypeInt2& a, PackTypeInt2& b, PackTypeInt4& c) {
}

template<>
__device__ __forceinline__ void mma_16x16x16<half>(PackTypeInt2& a, PackTypeInt2& b, PackTypeInt4& c) {
    mma_16x16x16f16(a, b, c);
}

template<>
__device__ __forceinline__ void mma_16x16x16<__maca_bfloat16>(PackTypeInt2& a, PackTypeInt2& b, PackTypeInt4& c) {
    mma_16x16x16bf16(a, b, c);
}

#if 0
#ifdef BF16_HIGH_PRECISION
__global__ void vectorized_elementwise_fp32tobf16(float* input, __maca_bfloat16* output, int N) {
    uint64_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        // printf("tid = %d, input = %f, output = %f\n", tid, input[tid], (float)(__maca_bfloat16)input[tid]);
        *(__maca_bfloat16*)(output+tid) = (__maca_bfloat16)input[tid];
    }
}
#else
__global__ void vectorized_elementwise_fp16tobf16(__maca_bfloat16* input, int N) {
    uint64_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        input[tid] = (float)(*(half*)(input+tid));
    }
}
#endif
#endif

template<
    const int THREADS, // number of threads in a threadblock
    const int PACK_NUM
    >
__global__ void clean_zero(float* data, size_t num_elem) {
    int bidx = blockIdx.x;
    int tidx = threadIdx.x;
    int idx = (bidx * THREADS + tidx) * PACK_NUM;
    float zeros[4] = {0.0, 0.0, 0.0, 0.0};
    if (idx < num_elem) {
        *((b128VecType*)(&data[idx])) = *((b128VecType*)zeros);
    }
}

template<
    const int THREADS, // number of threads in a threadblock
    const int PACK_NUM,
    const bool USE_C = false
    >
__global__ void all_reduce(float* in_data, maca_bfloat16* out_data, size_t num_elem) {
    int bidx = blockIdx.x;
    int tidx = threadIdx.x;
    int idx = (bidx * THREADS + tidx) * PACK_NUM;

    if constexpr(PACK_NUM == 4) {
        if constexpr(USE_C == true) {
            float temp_in_fp32[PACK_NUM];
            float temp_out_fp32[PACK_NUM];
            maca_bfloat16 temp_out_bf16[PACK_NUM];

            bool pred = idx < num_elem;
            ldg_b64_reg_noasync(*((b64VecType*)temp_out_bf16), ((b64VecType*)(out_data + idx)), pred, true);
            ldg_b128_reg_noasync(*((b128VecType*)temp_in_fp32), ((b128VecType*)(in_data + idx)), pred, true);

            #pragma unroll
            for(int i = 0; i < PACK_NUM; i++) {
                temp_out_fp32[i] = __bfloat162float(temp_out_bf16[i]);
                temp_out_fp32[i] += temp_in_fp32[i];
                temp_out_bf16[i] = __float2bfloat16(temp_out_fp32[i]);
            }

            if (pred) {
                *((b64VecType*)(out_data + idx)) = *((b64VecType*)temp_out_bf16);
            }
        } else {
            float temp_in_fp32[PACK_NUM];
            maca_bfloat16 temp_out_bf16[PACK_NUM];

            bool pred = idx < num_elem;
            ldg_b128_reg_noasync(*((b128VecType*)temp_in_fp32), ((b128VecType*)(in_data + idx)), pred, true);

            #pragma unroll
            for(int i = 0; i < PACK_NUM; i++) {
                temp_out_bf16[i] = __float2bfloat16(temp_in_fp32[i]);
            }

            if (pred) {
                *((b64VecType*)(out_data + idx)) = *((b64VecType*)temp_out_bf16);
            }
        }
    }
}

typedef __NATIVE_VECTOR__(2, float) v2f;
using PackTypeFloat2 = v2f;
constexpr static int Q4BITS = 4;
constexpr static int Q8BITS = 8;
constexpr static int PACK_RATIO_4BITS = sizeof(PackType) * 8 / Q4BITS;
constexpr static int PACK_RATIO_8BITS = sizeof(PackType) * 8 / Q8BITS;
constexpr static int SLICE_M = 16;
constexpr static int SLICE_N = 16;
constexpr static int SLICE_K = 32;
constexpr static int PAD_SLICE_K = 40;
constexpr static int SLOT    = 16;
constexpr static int WAVE    = 64;
constexpr static int WAVE_SLOTS = 4;
constexpr static int PEUS = 13*8*4; //For C500, There are 8 DPC and each DPC have 13 APs, each AP have 4 PEUs
constexpr static int MAX_BLOCKS_M = 4;
constexpr static uint32_t seil = 0x03020706u;

__device__ __forceinline__ void f32x2_cvt_bf16x2(uint32_t& dst, float src[2]) {
    uint32_t tmp[2];
    tmp[0] = __builtin_mxc_ubfe(*(reinterpret_cast<uint32_t*>(src)), 16, 1);
    tmp[0] = tmp[0] + *reinterpret_cast<uint32_t*>(src);
    tmp[0] = (uint32_t)0x7fff + tmp[0];
    tmp[1] = __builtin_mxc_ubfe(*(reinterpret_cast<uint32_t*>(src + 1)), 16, 1);
    tmp[1] = tmp[1] + *(reinterpret_cast<uint32_t*>(src + 1));
    tmp[1] = (uint32_t)0x7fff + tmp[1];
    dst = __builtin_mxc_byte_perm(tmp[0], tmp[1], seil);
}


// #define CVT_B0TOF32(q, out) asm volatile("cvt_b0tof32 %0,%1;\n":"=r"(out):"r"(q));
// #define CVT_B1TOF32(q, out) asm volatile("cvt_b1tof32 %0,%1;\n":"=r"(out):"r"(q));
// #define CVT_B2TOF32(q, out) asm volatile("cvt_b2tof32 %0,%1;\n":"=r"(out):"r"(q));
// #define CVT_B3TOF32(q, out) asm volatile("cvt_b3tof32 %0,%1;\n":"=r"(out):"r"(q));

#define CVT_B0TOF32(q, out) out = __builtin_mxc_b0_cast_to_f32(q);
#define CVT_B1TOF32(q, out) out = __builtin_mxc_b1_cast_to_f32(q);
#define CVT_B2TOF32(q, out) out = __builtin_mxc_b2_cast_to_f32(q);
#define CVT_B3TOF32(q, out) out = __builtin_mxc_b3_cast_to_f32(q);

//FIXME: We'd rather a quant group will not divided into serveral blocks
template<int BLOCKS_M, int BLOCKS_N, int BLOCKS_K>
struct TileManager {
    int tile_start_row;
    int tile_start_col;
    int tiles_k;
    int my_iters = 0;
    bool global_pred = true;

    __device__ __forceinline__ void init(int m, int n, int k, int bidx, int iters) {
        //Calculate tile start row and cols so we can calculate the offset address of a b and c
        int tile_idx = iters * bidx;
        int tiles_n = div_ceil(n, TILE_N);
        tiles_k = div_ceil(k, TILE_K);
        //if (tile_idx >= tiles_n*tiles_k) return false;
        global_pred = tile_idx < tiles_n * tiles_k;
        int tile_col = tile_idx / tiles_k;
        int tile_row = tile_idx - tile_col * tiles_k;
        tile_start_col = tile_col;
        tile_start_row = tile_row;
        my_iters = tile_idx + iters >= tiles_n*tiles_k ? tiles_n*tiles_k - tile_idx : iters;
        my_iters = global_pred ? my_iters : 0;
    }

    __device__ __forceinline__ void next_tile() {
        tile_start_col = tile_start_row + 1 == tiles_k ? tile_start_col + 1 : tile_start_col;
        tile_start_row = tile_start_row + 1 == tiles_k ? 0 : tile_start_row + 1;
        --my_iters;
        global_pred = my_iters > 0;
    }

    __device__ __host__ __forceinline__ bool need_save_data() {
        if (global_pred && my_iters == 1) return true;
        if (global_pred && tile_start_row + 1 == tiles_k) return true;
        return false;
    }

    //support for preloading next tile in current tile calculation
    //The point is when all quanted values are dequanted and all a are stored to bsm already
    //Then the registers for scales, zeros, and a_caches are free to use, bsm for scales are already
    //free to use.
    //The main problem we will face is that no more registers can be used to cache tile information
    //when we want to preload next tile data
    bool flag_save_data = false;
    int tile_start_col_cache; //Kept for saving result n calculation

    __device__ __forceinline__ void next_tile_pre() {
        flag_save_data = need_save_data();
        tile_start_col_cache = tile_start_col;
        next_tile();
    }

    __device__ __forceinline__ bool need_save_data_pre() {
        return flag_save_data;
    }
};

struct ThreadView {
    int tid;
    int wave_idx;
    int wave_tid;
    int slot_idx;
    int slot_tid;

    __device__ __forceinline__ void init() {
        tid = threadIdx.x;
        wave_idx = tid / WAVE;
        wave_tid = tid % WAVE;
        slot_idx = wave_tid / SLOT;
        slot_tid = wave_tid % SLOT;
    }
};

/*
Currently, we develop a version that TILE size is m64n128k64
So every TILE we load 64x64x2=8192bytes A, 128x64/8x4=4096bytes B, 128x4=512 scales, 128x4=512 zeros if needed

If more batches is needed, I think it is better to run dequant again. We do not count on situation that m>128 now.

Memory View:
There are 1/2/4 waves according to BLOCKS_N, wave_count = BLOCKS_N, mostly, BLOCKS_N=4, and THREADS=256
Each wave will process a m16n16k32 sub tile with two mma instructions, we should re-order data in bsm so
we can load data from shared memory only with tid or wave_tid, where we can reduce the instructions of 
calculate address offsets of each data:

BSM A order(assume that BLOCKS_M=4):
|---------------------------------8x4x16 half value ------------------------------------------------------------------------|
|-use by wave  slotidx 0 slottid 0------|-use by wave  slotidx 0 slottid 1------|...|-use by wave  slotidx 0 slottid 15-----|
|-8 half--|-8 half--|-8 half--|-8 half--|-8 half--|-8 half--|-8 half--|-8 half--|...|-8 half--|-8 half--|-8 half--|-8 half--|
|m00k00~07|m16k00~07|m32k00~07|m48k00~07|m01k00~07|m17k00~07|m33k00~07|m49k00~07|...|m15k00~07|m31k00~07|m47k00~07|m63k00~07|

|-use by wave  slotidx 1 slottid 0------|-use by wave  slotidx 1 slottid 1------|...|-use by wave  slotidx 1 slottid 15-----|
|-8 half--|-8 half--|-8 half--|-8 half--|-8 half--|-8 half--|-8 half--|-8 half--|...|-8 half--|-8 half--|-8 half--|-8 half--|
|m00k08~15|m16k08~15|m32k08~15|m48k08~15|m01k08~15|m17k08~15|m33k08~15|m49k08~15|...|m15k08~15|m31k08~15|m47k08~15|m63k08~15|

...

|-use by wave  slotidx 3 slottid 0------|-use by wave  slotidx 3 slottid 1------|...|-use by wave  slotidx 3 slottid 15-----|
|-8 half--|-8 half--|-8 half--|-8 half--|-8 half--|-8 half--|-8 half--|-8 half--|...|-8 half--|-8 half--|-8 half--|-8 half--|
|m00k24~31|m16k24~31|m32k24~31|m48k24~31|m01k24~31|m17k24~31|m33k24~31|m49k24~31|...|m15k24~31|m31k24~31|m47k24~31|m63k24~31|

---- next k ----
|---------------------------------8x4x16 half value ------------------------------------------------------------------------|
|-use by wave  slotidx 0 slottid 0------|-use by wave  slotidx 0 slottid 1------|...|-use by wave  slotidx 0 slottid 15-----|
|-8 half--|-8 half--|-8 half--|-8 half--|-8 half--|-8 half--|-8 half--|-8 half--|...|-8 half--|-8 half--|-8 half--|-8 half--|
|m00k32~39|m16k32~39|m32k32~39|m48k32~39|m01k32~39|m17k32~39|m33k32~39|m49k32~39|...|m15k32~39|m31k32~39|m47k32~39|m63k32~39|

...

|---------------------------------8x4x16 half value ------------------------------------------------------------------------|
|-use by wave  slotidx 3 slottid 0------|-use by wave  slotidx 3 slottid 1------|...|-use by wave  slotidx 3 slottid 15-----|
|-8 half--|-8 half--|-8 half--|-8 half--|-8 half--|-8 half--|-8 half--|-8 half--|...|-8 half--|-8 half--|-8 half--|-8 half--|
|m00k56~63|m16k56~63|m32k56~63|m48k56~63|m01k56~63|m17k56~63|m33k56~63|m49k56~63|...|m15k56~63|m31k56~63|m47k56~63|m63k56~63|

When loading a from bsm, each thread only read 4x8 half value with offset wave_tid:
b128 out[4];
b128 *bsm_a_ptr;
out[0] = bsm_a_ptr[wave_tid*BLOCKS_M];
out[1] = bsm_a_ptr[wave_tid*BLOCKS_M+1];
out[2] = bsm_a_ptr[wave_tid*BLOCKS_M+2];
out[3] = bsm_a_ptr[wave_tid*BLOCKS_M+3];


BSM B order: here k means the quanted k, which means each k represents 8 dequanted k data
//Loop = BLOCKS_N*SLICE_N/16/WAVES
Loop 0:
|---------16 uint32_t loaded by wave0-|
k00n00 k00n01 k00n02 k00n03 ... k00n15
k01n00 k01n01 k01n02 k01n03 ... k01n15
k02n00 k02n01 k02n02 k02n03 ... k02n15
k03n00 k03n01 k03n02 k03n03 ... k03n15
|---------16 uint32_t loaded by wave1-|
k00n16 k00n17 k00n18 k00n19 ... k00n31
...
k03n16 k03n17 k03n18 k03n19 ... k03n31
|---------16 uint32_t loaded by wave2-|
k00n32 k00n33 k00n34 k00n35 ... k00n47
...
k03n32 k03n33 k03n34 k03n35 ... k03n47
|---------16 uint32_t loaded by wave3-|
k00n48 k00n49 k00n50 k00n51 ... k00n63
...
k03n48 k03n49 k03n50 k03n51 ... k03n63
*/
template<class scalar_t>
__device__ __forceinline__ void awq_dequant_4bits(const uint32_t& p, scalar_t (&out)[8], scalar_t (&scale)[8], const uint32_t& scale_zero) {
    if constexpr(std::is_same_v<scalar_t, __maca_bfloat16>) {
        v2f v2z,v2scale, v2neg;
        v2neg[0] = -1.0f; v2neg[1] = -1.0f;
        v2z[0] = 0.0f; v2z[1] = 0.0f;
        v2f v2zero;
        float tmp[2];
        int p0 = p & 0x0f0f0f0f;
        int z0 = scale_zero & 0x0f0f0f0f;
        CVT_B0TOF32(p0, tmp[0]);
        CVT_B2TOF32(p0, tmp[1]);
        CVT_B0TOF32(z0, v2zero[0]);
        CVT_B2TOF32(z0, v2zero[1]);
        *((v2f*)tmp) = __builtin_mxc_pk_fma_f32(v2zero, v2neg, *((v2f*)tmp));
        v2scale[0] = (float)scale[0]; v2scale[1] = (float)scale[1];
        *((v2f*)tmp) = __builtin_mxc_pk_fma_f32(*((v2f*)tmp), v2scale, v2z);
        f32x2_cvt_bf16x2(*((uint32_t*)out), tmp);

        CVT_B1TOF32(p0, tmp[0]);
        CVT_B3TOF32(p0, tmp[1]);
        CVT_B1TOF32(z0, v2zero[0]);
        CVT_B3TOF32(z0, v2zero[1]);
        *((v2f*)tmp) = __builtin_mxc_pk_fma_f32(v2zero, v2neg, *((v2f*)tmp));
        v2scale[0] = (float)scale[4]; v2scale[1] = (float)scale[5];
        *((v2f*)tmp) = __builtin_mxc_pk_fma_f32(*((v2f*)tmp), v2scale, v2z);
        f32x2_cvt_bf16x2(*((uint32_t*)(out + 4)), tmp);

        p0 = (p >> 4) & 0x0f0f0f0f;
        z0 = (scale_zero >> 4) & 0x0f0f0f0f;
        CVT_B0TOF32(p0, tmp[0]);
        CVT_B2TOF32(p0, tmp[1]);
        CVT_B0TOF32(z0, v2zero[0]);
        CVT_B2TOF32(z0, v2zero[1]);
        *((v2f*)tmp) = __builtin_mxc_pk_fma_f32(v2zero, v2neg, *((v2f*)tmp));
        v2scale[0] = (float)scale[2]; v2scale[1] = (float)scale[3];
        *((v2f*)tmp) = __builtin_mxc_pk_fma_f32(*((v2f*)tmp), v2scale, v2z);
        f32x2_cvt_bf16x2(*((uint32_t*)(out + 2)), tmp);

        CVT_B1TOF32(p0, tmp[0]);
        CVT_B3TOF32(p0, tmp[1]);
        CVT_B1TOF32(z0, v2zero[0]);
        CVT_B3TOF32(z0, v2zero[1]);

        *((v2f*)tmp) = __builtin_mxc_pk_fma_f32(v2zero, v2neg, *((v2f*)tmp));
        v2scale[0] = (float)scale[6]; v2scale[1] = (float)scale[7];
        *((v2f*)tmp) = __builtin_mxc_pk_fma_f32(*((v2f*)tmp), v2scale, v2z);
        f32x2_cvt_bf16x2(*((uint32_t*)(out + 6)), tmp);
    } else {
        v2f a0, v2z, v2scale, v2zero, v2neg;
        v2z[0] = 0; v2z[1] = 0;
        v2neg[0] = -1.0f; v2neg[1] = -1.0f;
        int p0 = p & 0x0f0f0f0f;
        int z0 = scale_zero & 0x0f0f0f0f;
        CVT_B0TOF32(p0, a0.x);
        CVT_B2TOF32(p0, a0.y);
        CVT_B0TOF32(z0, v2zero[0]);
        CVT_B2TOF32(z0, v2zero[1]);
        a0 = __builtin_mxc_pk_fma_f32(v2zero, v2neg, a0);
        v2scale[0] = (float)scale[0]; v2scale[1] = (float)scale[1];
        a0 = __builtin_mxc_pk_fma_f32(a0, v2scale, v2z);
        out[0] = (scalar_t)a0.x;
        out[1] = (scalar_t)a0.y;

        CVT_B1TOF32(p0, a0.x);
        CVT_B3TOF32(p0, a0.y);
        CVT_B1TOF32(z0, v2zero[0]);
        CVT_B3TOF32(z0, v2zero[1]);
        a0 = __builtin_mxc_pk_fma_f32(v2zero, v2neg, a0);
        v2scale[0] = (float)scale[4]; v2scale[1] = (float)scale[5];
        a0 = __builtin_mxc_pk_fma_f32(a0, v2scale, v2z);
        out[4] = (scalar_t)a0.x;
        out[5] = (scalar_t)a0.y;

        p0 = (p >> 4) & 0x0f0f0f0f;
        z0 = (scale_zero >> 4) & 0x0f0f0f0f;
        CVT_B0TOF32(p0, a0.x);
        CVT_B2TOF32(p0, a0.y);
        CVT_B0TOF32(z0, v2zero[0]);
        CVT_B2TOF32(z0, v2zero[1]);
        a0 = __builtin_mxc_pk_fma_f32(v2zero, v2neg, a0);
        v2scale[0] = (float)scale[2]; v2scale[1] = (float)scale[3];
        a0 = __builtin_mxc_pk_fma_f32(a0, v2scale, v2z);
        out[2] = (scalar_t)a0.x;
        out[3] = (scalar_t)a0.y;

        CVT_B1TOF32(p0, a0.x);
        CVT_B3TOF32(p0, a0.y);
        CVT_B1TOF32(z0, v2zero[0]);
        CVT_B3TOF32(z0, v2zero[1]);
        a0 = __builtin_mxc_pk_fma_f32(v2zero, v2neg, a0);
        v2scale[0] = (float)scale[6]; v2scale[1] = (float)scale[7];
        a0 = __builtin_mxc_pk_fma_f32(a0, v2scale, v2z);
        out[6] = (scalar_t)a0.x;
        out[7] = (scalar_t)a0.y;
    }
}

//deqaunt a uint32_t
template<class scalar_t>
__device__ __forceinline__ void dequant_gptq_4bits(const PackType& p, scalar_t (&out)[8], const v2f& scale, const v2f& scale_zero) {
    if constexpr(std::is_same_v<scalar_t, __maca_bfloat16>) {
        v2f a0;
        float tmp[2];
        int p0 = p & 0x0f0f0f0f;

        // CVT_B0TOF32(p0, a0.x);
        // CVT_B2TOF32(p0, a0.y);
        // a0 = __builtin_mxc_pk_fma_f32(a0, scale, scale_zero);
        // out[0] = __float2bfloat16(a0.x);
        // out[1] = __float2bfloat16(a0.y);

        CVT_B0TOF32(p0, tmp[0]);
        CVT_B2TOF32(p0, tmp[1]);
        *((v2f*)tmp) = __builtin_mxc_pk_fma_f32(*((v2f*)tmp), scale, scale_zero);
        f32x2_cvt_bf16x2(*((uint32_t*)out), tmp);

        // CVT_B1TOF32(p0, a0.x);
        // CVT_B3TOF32(p0, a0.y);
        // a0 = __builtin_mxc_pk_fma_f32(a0, scale, scale_zero);
        // out[4] = __float2bfloat16(a0.x);
        // out[5] = __float2bfloat16(a0.y);

        CVT_B1TOF32(p0, tmp[0]);
        CVT_B3TOF32(p0, tmp[1]);
        *((v2f*)tmp) = __builtin_mxc_pk_fma_f32(*((v2f*)tmp), scale, scale_zero);
        f32x2_cvt_bf16x2(*((uint32_t*)(out + 4)), tmp);

        p0 = (p >> 4) & 0x0f0f0f0f;

        // CVT_B0TOF32(p0, a0.x);
        // CVT_B2TOF32(p0, a0.y);
        // a0 = __builtin_mxc_pk_fma_f32(a0, scale, scale_zero);
        // out[2] = __float2bfloat16(a0.x);
        // out[3] = __float2bfloat16(a0.y);

        CVT_B0TOF32(p0, tmp[0]);
        CVT_B2TOF32(p0, tmp[1]);
        *((v2f*)tmp) = __builtin_mxc_pk_fma_f32(*((v2f*)tmp), scale, scale_zero);
        f32x2_cvt_bf16x2(*((uint32_t*)(out + 2)), tmp);

        // CVT_B1TOF32(p0, a0.x);
        // CVT_B3TOF32(p0, a0.y);
        // a0 = __builtin_mxc_pk_fma_f32(a0, scale, scale_zero);
        // out[6] = __float2bfloat16(a0.x);
        // out[7] = __float2bfloat16(a0.y);

        CVT_B1TOF32(p0, tmp[0]);
        CVT_B3TOF32(p0, tmp[1]);
        *((v2f*)tmp) = __builtin_mxc_pk_fma_f32(*((v2f*)tmp), scale, scale_zero);
        f32x2_cvt_bf16x2(*((uint32_t*)(out + 6)), tmp);
    } else {
        v2f a0;
        int p0 = p & 0x0f0f0f0f;
        CVT_B0TOF32(p0, a0.x);
        CVT_B2TOF32(p0, a0.y);
        a0 = __builtin_mxc_pk_fma_f32(a0, scale, scale_zero);
        out[0] = (scalar_t)a0.x;
        out[1] = (scalar_t)a0.y;

        CVT_B1TOF32(p0, a0.x);
        CVT_B3TOF32(p0, a0.y);
        a0 = __builtin_mxc_pk_fma_f32(a0, scale, scale_zero);
        out[4] = (scalar_t)a0.x;
        out[5] = (scalar_t)a0.y;

        p0 = (p >> 4) & 0x0f0f0f0f;
        CVT_B0TOF32(p0, a0.x);
        CVT_B2TOF32(p0, a0.y);
        a0 = __builtin_mxc_pk_fma_f32(a0, scale, scale_zero);
        out[2] = (scalar_t)a0.x;
        out[3] = (scalar_t)a0.y;

        CVT_B1TOF32(p0, a0.x);
        CVT_B3TOF32(p0, a0.y);
        a0 = __builtin_mxc_pk_fma_f32(a0, scale, scale_zero);
        out[6] = (scalar_t)a0.x;
        out[7] = (scalar_t)a0.y;
    }
}

template<class scalar_t>
__device__ __forceinline__ void dequant_gptq_8bits(const PackType& p, scalar_t (&out)[4], const v2f& scale, const v2f& scale_zero) {
    v2f a0;
    CVT_B0TOF32(p, a0.x);
    CVT_B1TOF32(p, a0.y);
    a0 = __builtin_mxc_pk_fma_f32(a0, scale, scale_zero);
    out[0] = (scalar_t)a0.x;
    out[1] = (scalar_t)a0.y;

    CVT_B2TOF32(p, a0.x);
    CVT_B3TOF32(p, a0.y);
    a0 = __builtin_mxc_pk_fma_f32(a0, scale, scale_zero);
    out[2] = (scalar_t)a0.x;
    out[3] = (scalar_t)a0.y;
}

// decompress zero
__device__ __forceinline__ void decompress_zero_4bits(const PackType& zp, float (&out)[8]) {
    v2f a0;
    int p0 = zp & 0x0f0f0f0f;
    CVT_B0TOF32(p0, a0.x);
    CVT_B2TOF32(p0, a0.y);
    out[0] = -a0.x;
    out[1] = -a0.y;

    CVT_B1TOF32(p0, a0.x);
    CVT_B3TOF32(p0, a0.y);
    out[4] = -a0.x;
    out[5] = -a0.y;

    p0 = (zp >> 4) & 0x0f0f0f0f;
    CVT_B0TOF32(p0, a0.x);
    CVT_B2TOF32(p0, a0.y);
    out[2] = -a0.x;
    out[3] = -a0.y;

    CVT_B1TOF32(p0, a0.x);
    CVT_B3TOF32(p0, a0.y);
    out[6] = -a0.x;
    out[7] = -a0.y;
}

namespace __hgemm_singular_blocks_k {
template<typename scalar_t, const vllm::ScalarTypeId w_type_id, int THREADS, int BLOCKS_M, int BLOCKS_N, int BLOCKS_K, bool HAS_ACT, bool HAS_ZP, bool HAS_M_PRED, bool HAS_NK_PRED>
struct LoadingManager {
    constexpr static int FragACount = 2;
    using FragA = PackType;
    constexpr static int FragBCount = 1;
    using FragB = PackType;
    constexpr static int FragCCount = 4;
    #ifdef BF16_HIGH_PRECISION
    using FragC = scalar_t;
    #else
    //Directly use half as the final atomic type:
    //1. Half precision and data range satisfies need of deepseek gemm
    //2. C500 has no atomic instructions for bfloat16, we cannot atomic a bfloat16 memory
    //3. The perfect precision type of atomic should be fp32, but the cost is too high to allocate a temp memory for float atomic
    using atomic_type = half;
    using FragC = atomic_type;
    #endif
    const FragA* A;
    const FragA* A_loading;
    const FragB* B;
    const FragB* B_loading;
    FragC* C;
    float* C_temp;
    using FragScaleLoading = half2;
    using FragZeroLoading = uint32_t;
    const FragScaleLoading* scales;
    const FragScaleLoading* scales_loading;
    const FragZeroLoading* zeros;
    const FragZeroLoading* zeros_loading;
    
    int m;
    int n;
    int k;
    int quant_group_power2;
    uint8_t* smem_base;
    int bidx;

    PackTypeInt4* bsm_a_ptr;
    scalar_t* bsm_scales_ptr;
    float* bsm_zeros_ptr;
    float* remaining_bsm_ptr;

    PackTypeInt2 local_a[BLOCKS_M][2];
    PackType local_b[N_ITERS];
    PackType local_b_cache[N_ITERS];
    scalar_t local_dequanted_b_8bits[N_ITERS][2][PACK_RATIO_8BITS];
    scalar_t local_dequanted_b[N_ITERS][PACK_RATIO_4BITS];
    v2f local_scales[N_ITERS];
    v2f local_zeros[N_ITERS];
    FragScaleLoading temp_scales;
    PackType temp_zeros;
    float output[BLOCKS_M][N_ITERS][4];
    FragA temp_a[LOADING_A_LOOP];

    TileManager<BLOCKS_M, BLOCKS_N, BLOCKS_K> tile_manager;
    ThreadView tv;

    __device__ __forceinline__ void set_address(const PackTypeInt4* a,
        const PackTypeInt4* b,
        PackTypeInt4* c,
        PackTypeInt4* c_temp,
        const PackTypeInt4* scale_ptr,
        const PackTypeInt4* zp_ptr = nullptr) {
            A = (const FragA*)a;
            B = (const FragB*)b;
            C = (FragC*)c;
            C_temp = (float*)c_temp;
            scales = (const FragScaleLoading*)scale_ptr;
            if constexpr(w_type_id == vllm::kU4.id()) {
                zeros = (const FragZeroLoading*)zp_ptr;
            }
    }

    __device__ __forceinline__ bool debug() {
        #ifdef DEBUG
        bool do_print = tv.wave_idx == 1 && tv.slot_idx == 0 && tv.slot_tid == 0;
        return do_print;
        #else
        return false;
        #endif
    }

    __device__ __forceinline__ void next_k() {
        //Update only bsm_a_ptr
        bsm_a_ptr = (PackTypeInt4*)smem_base;
        bsm_a_ptr += tv.slot_tid * (PAD_SLICE_K / (sizeof(PackTypeInt4) / sizeof(scalar_t))) + tv.slot_idx;
    }

    __device__ __forceinline__ void next_k_pre() {
        A_loading += SLICE_K / FragACount;
        //B_loading += SLICE_K / PACK_RATIO_4BITS * n;
        if constexpr(w_type_id == vllm::kU4.id() || w_type_id == vllm::kU4B8.id()) {
            B_loading += SLICE_K / PACK_RATIO_4BITS * n;
        } else if constexpr(w_type_id == vllm::kU8.id() || w_type_id == vllm::kU8B128.id()) {
            B_loading += SLICE_K / PACK_RATIO_8BITS * n;
        }
    }

    __device__ __forceinline__ void ldg_a(int k_idx) {
        //32x64/2/256 = 16 / 4 = 4
        int t = tv.tid;
        int k_broad = tile_manager.tile_start_row * TILE_K + k_idx * SLICE_K;
        #pragma unroll LOADING_A_LOOP
        for (int i = 0; i < LOADING_A_LOOP; i++)  {
            int reading_m = t / (SLICE_K / FragACount);
            int reading_k = t % (SLICE_K / FragACount);
            int gvm_offset = reading_m * k / FragACount + reading_k;
            FragA* gvm_addr = (FragA*)A_loading + gvm_offset;
            //FIXME: we cannot do slice k pad as ldg_b32_bsm_async seems does not support padding
            if constexpr(HAS_M_PRED && HAS_NK_PRED) {
                bool pred = reading_m < m;
                bool pred_k = k_broad + reading_k * FragACount < k;
                pred = pred && pred_k && tile_manager.global_pred;
                ldg_b32_reg_noasync(temp_a[i], gvm_addr, pred, true);
            } else if constexpr(HAS_M_PRED) {
                bool pred = reading_m < m && tile_manager.global_pred;
                ldg_b32_reg_noasync(temp_a[i], gvm_addr, pred, true);
            } else if constexpr(HAS_NK_PRED) {
                bool pred_k = k_broad + reading_k * FragACount < k && tile_manager.global_pred;
                ldg_b32_reg_noasync(temp_a[i], gvm_addr, pred_k, true);
            } else {
                ldg_b32_reg_noasync(temp_a[i], gvm_addr, tile_manager.global_pred, true);
            }
            t += THREADS;
        }
    }

    __device__ __forceinline__ void sts_a() {
        FragA* to_bsm_a_ptr = (FragA*)smem_base;
        int t = tv.tid;
        #pragma unroll LOADING_A_LOOP
        for (int i = 0; i < LOADING_A_LOOP; i++)  {
            int reading_m = t / (SLICE_K / FragACount);
            int reading_k = t % (SLICE_K / FragACount);
            int bsm_offset = reading_m * (PAD_SLICE_K / FragACount) + reading_k;
            *(to_bsm_a_ptr + bsm_offset) = temp_a[i];
            t += THREADS;
        }
    }

    __device__ __forceinline__ void lds_a(int midx) {
        *((PackTypeInt4*)local_a[midx]) = *bsm_a_ptr;
        bsm_a_ptr += SLOT * (PAD_SLICE_K / (sizeof(PackTypeInt4) / sizeof(scalar_t)));
    }

    //TODO: implement when N_ITERS==1 or N_ITERS==3
    __device__ __forceinline__ void ldg_b(int k_idx, int korder = 0) {
        if constexpr(HAS_NK_PRED) {
            bool pred_k = tile_manager.tile_start_row * TILE_K + k_idx * SLICE_K + tv.slot_idx * PACK_RATIO_4BITS + korder < k;
            bool pred_n = tile_manager.tile_start_col * TILE_N + (tv.wave_idx * SLOT + tv.slot_tid)*N_ITERS < n;
            bool pred = pred_n && pred_k && tile_manager.global_pred;
            FragB* addr =  (FragB*)B_loading + korder * n;
            if constexpr(N_ITERS == 2) {
                ldg_b64_reg_noasync(*((PackTypeInt2*)local_b_cache), ((PackTypeInt2*)addr), pred, true);
            } else if constexpr(N_ITERS == 4) {
                ldg_b128_reg_noasync(*((PackTypeInt4*)local_b_cache), ((PackTypeInt4*)addr), pred, true);
            }
        } else {
            FragB* addr =  (FragB*)B_loading + korder * n;
            if constexpr(N_ITERS == 2) {
                ldg_b64_reg_noasync(*((PackTypeInt2*)local_b_cache), ((PackTypeInt2*)addr), tile_manager.global_pred, true);
            } else if constexpr(N_ITERS == 4) {
                ldg_b128_reg_noasync(*((PackTypeInt4*)local_b_cache), ((PackTypeInt4*)addr), tile_manager.global_pred, true);
            }
        }
    }

    __device__ __forceinline__ void swap_b_cache(int i) {
        local_b[i] = local_b_cache[i];
    }

    __device__ __forceinline__ void ldg_scales() {
        bool pred = tv.tid < TILE_N / (sizeof(FragScaleLoading) / sizeof(scalar_t)) && tile_manager.global_pred;
        if constexpr(HAS_NK_PRED) {
            pred = pred && tv.tid < (n - tile_manager.tile_start_col * TILE_N) / (sizeof(FragScaleLoading) / sizeof(scalar_t));
        }
        //FragScaleLoading *scale_bsm = (FragScaleLoading*)(smem_base + 0x2000) + tv.tid;
        FragScaleLoading *gvm_addr = (FragScaleLoading*)scales_loading + tv.tid;
        //ldg_b32_bsm_async(scale_bsm, gvm_addr, pred, false);
        ldg_b32_reg_noasync(*((PackType*)&temp_scales), gvm_addr, pred, true);
    }

    __device__ __forceinline__ void ldg_zp() {
        if constexpr(w_type_id == vllm::kU4.id()) {
            bool pred = (tv.tid < TILE_N / PACK_RATIO_4BITS) && tile_manager.global_pred;
            if constexpr(HAS_NK_PRED) {
                pred = pred && tv.tid < ((n - tile_manager.tile_start_col * TILE_N) / PACK_RATIO_4BITS);
            }
            FragZeroLoading *gvm_addr = (FragZeroLoading*)zeros_loading + tv.tid;
            ldg_b32_reg_noasync(*((PackType*)&temp_zeros), gvm_addr, pred, true);
        }
    }

    __device__ __forceinline__ void sts_scales() {
        FragScaleLoading *scale_bsm = (FragScaleLoading*)(smem_base + 0x2000) + tv.tid;
        *scale_bsm = temp_scales;
    }

    __device__ __forceinline__ void sts_zeros() {
        if constexpr(w_type_id == vllm::kU4.id()) {
            bool pred = (tv.tid < TILE_N / PACK_RATIO_4BITS) && tile_manager.global_pred;
            if (pred) {
                float temp[PACK_RATIO_4BITS];
                decompress_zero_4bits(temp_zeros, temp);
                float *scale_bsm = (float*)(smem_base + 0x3000) + tv.tid * PACK_RATIO_4BITS;
                for (int i = 0; i < PACK_RATIO_4BITS; i++) {
                    *(scale_bsm + i) = temp[i];
                }
            }
        }
    }

    //TODO: implement when N_ITERS==1 or N_ITERS==3
    __device__ __forceinline__ void lds_scales() {
        if constexpr(N_ITERS==2) {
            *((half2*)local_dequanted_b[0]) = *((half2*)bsm_scales_ptr);
        } else if constexpr(N_ITERS==4) {
            *((PackTypeInt2*)local_dequanted_b[0]) = *((PackTypeInt2*)bsm_scales_ptr);
        }
    }

    __device__ __forceinline__ void pack_scales() {
        if constexpr(w_type_id == vllm::kU4B8.id()) {
            #pragma unroll
            for (int i = 0; i < N_ITERS; i++) {
                float s = local_dequanted_b[0][i];
                float z = -8 * s;
                local_scales[i] = {s, s};
                local_zeros[i] = {z, z};
            }
        } else if constexpr(w_type_id == vllm::kU4.id()) {
            #pragma unroll
            for (int i = 0; i < N_ITERS; i++) {
                float s;
                if constexpr(std::is_same_v<scalar_t, __maca_bfloat16>) {
                    s = __bfloat162float(local_dequanted_b[0][i]);
                } else {
                    s = local_dequanted_b[0][i];
                }
                float z = *(bsm_zeros_ptr + i);
                z = z * s;
                local_scales[i] = {s, s};
                local_zeros[i] = {z, z};
            }
        } else if constexpr(w_type_id == vllm::kU8B128.id()) {
            #pragma unroll
            for (int i = 0; i < N_ITERS; i++) {
                float s = local_dequanted_b[0][i];
                float z = -128 * s;
                local_scales[i] = {s, s};
                local_zeros[i] = {z, z};
            }
        } else if constexpr(w_type_id == vllm::kU8.id()) {
            // should apply zeros
        }
    }

    __device__ __forceinline__ void dequant(int kdx, int korder = 0) {
        if constexpr(w_type_id == vllm::kU4.id() || w_type_id == vllm::kU4B8.id()) {
            dequant_gptq_4bits<scalar_t>(local_b[kdx], local_dequanted_b[kdx], local_scales[kdx], local_zeros[kdx]);
        } else if constexpr(w_type_id == vllm::kU8.id() || w_type_id == vllm::kU8B128.id()) {
            dequant_gptq_8bits<scalar_t>(local_b[kdx], local_dequanted_b_8bits[kdx][korder], local_scales[kdx], local_zeros[kdx]);
        }
    }

    __device__ __forceinline__ void matmul(int mdx) {
        if constexpr(w_type_id == vllm::kU4.id() || w_type_id == vllm::kU4B8.id()) {
            #pragma unroll
            for (int i = 0; i < N_ITERS; i++) {
                mma_16x16x16<scalar_t>(local_a[mdx][0], *((PackTypeInt2*)local_dequanted_b[i]), *((PackTypeInt4*)output[mdx][i]));
                mma_16x16x16<scalar_t>(local_a[mdx][1], *((PackTypeInt2*)local_dequanted_b[i] + 1), *((PackTypeInt4*)output[mdx][i]));
	    }
        } else if constexpr(w_type_id == vllm::kU8.id() || w_type_id == vllm::kU8B128.id()) {
            #pragma unroll
            for (int i = 0; i < N_ITERS; i++) {
                mma_16x16x16<scalar_t>(local_a[mdx][0], *((PackTypeInt2*)local_dequanted_b_8bits[i][0]), *((PackTypeInt4*)output[mdx][i]));
                mma_16x16x16<scalar_t>(local_a[mdx][1], *((PackTypeInt2*)local_dequanted_b_8bits[i][1]), *((PackTypeInt4*)output[mdx][i]));
            }
        }
    }

    __device__ __forceinline__ void clear_c() {
        #pragma unroll
        for (int miter = 0; miter < BLOCKS_M; miter++) {
            #pragma unroll
            for (int niter = 0; niter < N_ITERS; niter++) {
                #pragma unroll
                for (int miter2 = 0; miter2 < 4; miter2++) {
                    output[miter][niter][miter2] = 0;
                }
            }
        }
    }

    //functions for preloading next tile data
    __device__ __forceinline__ void init_address_pre(int _m, int _n, int _k, int _quant_group_power2, int _bidx, int _iters, uint8_t *_smem_base) {
        tv.init();
        m = _m;
        n = _n;
        k = _k;
        quant_group_power2 = _quant_group_power2;
        bidx = _bidx;
        smem_base = _smem_base;
        tile_manager.init(m, n, k, bidx, _iters);
        init_tile_pre(tile_manager.tile_start_col, tile_manager.tile_start_row);
    }

    __device__ __forceinline__ void init_tile_pre(int col, int row) {
        //Initialize start slice address and set them to A_loading and B_loading
        int offset_n = col * TILE_N;
        int offset_k = row * TILE_K;
        //A_loading address will always be valid
        A_loading = A + offset_k / (FragACount);
        //B_loading = B + (offset_k / PACK_RATIO_4BITS * n + offset_n) / (FragBCount) + tv.slot_idx * n + (tv.wave_idx * SLOT + tv.slot_tid)*N_ITERS;
        if constexpr(w_type_id == vllm::kU4.id() || w_type_id == vllm::kU4B8.id()) {
            B_loading = B + (offset_k / PACK_RATIO_4BITS * n + offset_n) / (FragBCount) + tv.slot_idx * n + (tv.wave_idx * SLOT + tv.slot_tid)*N_ITERS;
        } else if constexpr(w_type_id == vllm::kU8.id() || w_type_id == vllm::kU8B128.id()) {
            B_loading = B + (offset_k / PACK_RATIO_8BITS * n + offset_n) / (FragBCount) + tv.slot_idx * n * 2 + (tv.wave_idx * SLOT + tv.slot_tid)*N_ITERS;
        }
        scales_loading = scales + ((offset_k >> quant_group_power2) * n + offset_n) / (sizeof(FragScaleLoading)/sizeof(scalar_t));
        if constexpr(w_type_id == vllm::kU4.id()) {
            zeros_loading = zeros + ((offset_k >> quant_group_power2) * n + offset_n) / PACK_RATIO_4BITS;
        }
    }

    __device__ __forceinline__ void next_tile_pre() {
        tile_manager.next_tile_pre();
        init_tile_pre(tile_manager.tile_start_col, tile_manager.tile_start_row);
    }

    __device__ __forceinline__ void init_bsm_addr() {
        bsm_a_ptr = (PackTypeInt4*)smem_base;           //use 8k bytes, will load at most 32x128*sizeof(half), either m32k128 or m128k32
        remaining_bsm_ptr = (float*)(smem_base + 0x2000 + 0x1000); //3k bytes
        bsm_a_ptr += tv.slot_tid * (PAD_SLICE_K / (sizeof(PackTypeInt4) / sizeof(scalar_t))) + tv.slot_idx;
        bsm_scales_ptr = (scalar_t*)(smem_base + 0x2000);      //use 128xsizeof(float)*2 = 1k
        bsm_scales_ptr += (tv.wave_idx * SLOT + tv.slot_tid) * N_ITERS;
        if constexpr(w_type_id == vllm::kU4.id()) {
            bsm_zeros_ptr = (float*)(smem_base + 0x3000);
            bsm_zeros_ptr += (tv.wave_idx * SLOT + tv.slot_tid) * N_ITERS;
        }
    }

    __device__ __forceinline__ void write_c(int offset, const float& v) {
        if constexpr(std::is_same_v<scalar_t, __maca_bfloat16>) {
            #ifdef BF16_HIGH_PRECISION
            atomicAdd(C_temp + offset, v);
            #else
            atomicAdd(C+offset, (atomic_type)v);
            #endif
        } else {
            atomicAdd(C + offset, (scalar_t)v);
        }
    }

    //atomic write to c
    __device__ __forceinline__ void write_c_pre() {
        int k_broad = tv.slot_idx * 4;
        int n_broad = (tv.wave_idx * SLOT + tv.slot_tid) * N_ITERS + tile_manager.tile_start_col_cache * TILE_N;
        #pragma unroll
        for (int miter = 0; miter < BLOCKS_M; miter++) {
            #pragma unroll
            for (int niter = 0; niter < N_ITERS; niter++) {
                int store_n = n_broad + niter;
                #pragma unroll
                for (int miter2 = 0; miter2 < 4; miter2++) {
                    int store_m = k_broad + miter * SLICE_M + miter2;
                    if constexpr(HAS_M_PRED && HAS_NK_PRED) {
                        if (store_m < m && store_n < n) {
                            write_c(store_m * n + store_n, output[miter][niter][miter2]);
			            }
                    } else if constexpr(HAS_M_PRED) {
                        if (store_m < m) {
                            write_c(store_m * n + store_n, output[miter][niter][miter2]);
			            }
                    } else if constexpr(HAS_NK_PRED) {
                        if (store_n < n) {
                            write_c(store_m * n + store_n, output[miter][niter][miter2]);
			            }
                    } else {
                        write_c(store_m * n + store_n, output[miter][niter][miter2]);
		            }
                }
            }
        }
    }

    template<bool KTAIL>
    __device__ __forceinline__ void on_dequant_niters1(int k_idx) {
        if constexpr(w_type_id == vllm::kU4.id() || w_type_id == vllm::kU4B8.id()) {
            swap_b_cache(0);
            sts_a(); // reorder data_a from reg(gvm resouce) to bsm
            if constexpr(!KTAIL) {
                next_k_pre(); // preload gvm a/b
                ldg_b(k_idx + 1); //preload b for next k
                ldg_a(k_idx + 1); //preload a for next k
            }
            barrier_bsm; //sync threads for sts_a
            lds_a(0); //load first 16 batches into registers
            dequant(0); //dequant b0
        } else if constexpr(w_type_id == vllm::kU8.id() || w_type_id == vllm::kU8B128.id()) {
            swap_b_cache(0);
            ldg_b(k_idx, 1);
            sts_a(); // reorder data_a from reg(gvm resouce) to bsm
            dequant(0);
            swap_b_cache(0);
            if constexpr(!KTAIL) {
                next_k_pre(); // preload gvm a/b
                ldg_b(k_idx + 1); //preload b for next k
                ldg_a(k_idx + 1); //preload a for next k
            }
            barrier_bsm; //sync threads for sts_a
            lds_a(0); //load first 16 batches into registers
            dequant(0); //dequant b0
        }
    }

    template<bool KTAIL>
    __device__ __forceinline__ void on_dequant_niters2(int k_idx) {
        if constexpr(w_type_id == vllm::kU4.id() || w_type_id == vllm::kU4B8.id()) {
            swap_b_cache(0);
            dequant(0); //dequant b0
            sts_a(); // reorder data_a from reg(gvm resouce) to bsm
            swap_b_cache(1);
            if constexpr(!KTAIL) {
                next_k_pre(); // preload gvm a/b
                ldg_b(k_idx + 1); //preload b for next k
                ldg_a(k_idx + 1); //preload a for next k
            }
            barrier_bsm; //sync threads for sts_a
            lds_a(0); //load first 16 batches into registers
            dequant(1);   //dequant b64
        } else if constexpr(w_type_id == vllm::kU8.id() || w_type_id == vllm::kU8B128.id()) {
            swap_b_cache(0);
            swap_b_cache(1);
            sts_a(); // reorder data_a from reg(gvm resouce) to bsm
            ldg_b(k_idx, 1);
            dequant(0); //dequant b0
            dequant(1);
            swap_b_cache(0);
            swap_b_cache(1);
            if constexpr(!KTAIL) {
                next_k_pre(); // preload gvm a/b
                ldg_b(k_idx + 1); //preload b for next k
                ldg_a(k_idx + 1); //preload a for next k
            }
            barrier_bsm; //sync threads for sts_a
            lds_a(0); //load first 16 batches into registers
            dequant(0, 1);
            dequant(1, 1);   //dequant b64
        }
    }

    template<bool KTAIL>
    __device__ __forceinline__ void on_dequant_niters3(int k_idx) {
        if constexpr(w_type_id == vllm::kU4.id() || w_type_id == vllm::kU4B8.id()) {
            swap_b_cache(0);
            dequant(0); //dequant b0
            sts_a(); // reorder data_a from reg(gvm resouce) to bsm
            swap_b_cache(1);
            dequant(1);
            swap_b_cache(2);
            if constexpr(!KTAIL) {
                next_k_pre(); // preload gvm a/b
                ldg_b(k_idx + 1); //preload b for next k
                ldg_a(k_idx + 1); //preload a for next k
            }
            barrier_bsm; //sync threads for sts_a
            lds_a(0); //load first 16 batches into registers
            dequant(2); //dequant b0
        } else if constexpr(w_type_id == vllm::kU8.id() || w_type_id == vllm::kU8B128.id()) {
            swap_b_cache(0);
            swap_b_cache(1);
            swap_b_cache(2);
            sts_a(); // reorder data_a from reg(gvm resouce) to bsm
            ldg_b(k_idx, 1);
            dequant(0); //dequant b0
            dequant(1); //dequant b0
            dequant(2); //dequant b0
            swap_b_cache(0);
            swap_b_cache(1);
            swap_b_cache(2);
            if constexpr(!KTAIL) {
                next_k_pre(); // preload gvm a/b
                ldg_b(k_idx + 1); //preload b for next k
                ldg_a(k_idx + 1); //preload a for next k
            }
            barrier_bsm; //sync threads for sts_a
            lds_a(0); //load first 16 batches into registers
            dequant(0, 1); //dequant b0
            dequant(1, 1); //dequant b0
            dequant(2, 1); //dequant b0
        }
    }

    template<bool KTAIL>
    __device__ __forceinline__ void on_dequant_niters4(int k_idx) {
        if constexpr(w_type_id == vllm::kU4.id() || w_type_id == vllm::kU4B8.id()) {
            swap_b_cache(0);
            dequant(0); //dequant b0
            sts_a(); // reorder data_a from reg(gvm resouce) to bsm
            swap_b_cache(1);
            dequant(1); //dequant b1
            swap_b_cache(2);
            swap_b_cache(3);
            dequant(2); //dequant b2
            if constexpr(!KTAIL) {
                next_k_pre(); // preload gvm a/b
                ldg_b(k_idx + 1); //preload b for next k
                ldg_a(k_idx + 1); //preload a for next k
            }
            barrier_bsm; //sync threads for sts_a
            lds_a(0); //load first 16 batches into registers
            dequant(3); //dequant b3
        } else {
            swap_b_cache(0);
            swap_b_cache(1);
            swap_b_cache(2);
            swap_b_cache(3);
            dequant(0); //dequant b0
            sts_a(); // reorder data_a from reg(gvm resouce) to bsm
            ldg_b(k_idx, 1);
            dequant(1); //dequant b0
            dequant(2); //dequant b0
            dequant(3); //dequant b0
            swap_b_cache(0);
            swap_b_cache(1);
            swap_b_cache(2);
            swap_b_cache(3);
            if constexpr(!KTAIL) {
                next_k_pre(); // preload gvm a/b
                ldg_b(k_idx + 1); //preload b for next k
                ldg_a(k_idx + 1); //preload a for next k
            }
            barrier_bsm; //sync threads for sts_a
            lds_a(0); //load first 16 batches into registers
            dequant(0, 1); //dequant b3
            dequant(1, 1); //dequant b3
            dequant(2, 1); //dequant b3
            dequant(3, 1); //dequant b3
        }
    }

    template<bool KTAIL>
    __device__ __forceinline__ void on_dequant(int kdx) {
        if constexpr(N_ITERS == 1) on_dequant_niters1<KTAIL>(kdx);
        else if constexpr(N_ITERS == 2) on_dequant_niters2<KTAIL>(kdx);
        else if constexpr(N_ITERS == 3) on_dequant_niters3<KTAIL>(kdx);
        else if constexpr(N_ITERS == 4) on_dequant_niters4<KTAIL>(kdx);
    }
};

template<typename scalar_t,
    const vllm::ScalarTypeId w_type_id,
    const int THREADS,          // number of threads in a threadblock
    const int BLOCKS_M,         // number of 16x16 blocks in the m
                                // dimension (batchsize) of the
                                // threadblock
    const int BLOCKS_N,         // same for n dimension (output)
    const int BLOCKS_K,         // same for k dimension (reduction)
    const bool HAS_ACT_ORDER,   // whether act_order is enabled
    const bool HAS_ZP,          // whether zero-points are enabled
    const bool HAS_M_PRED = true,  //If we should use predictors to load m from gvm
    const bool HAS_NK_PRED = true  //If we should use predictors to load nk from gvm
    >
__global__ void hgemm_gptq(
    const PackTypeInt4* __restrict__ A,  // fp16 input matrix of shape mxk
    const PackTypeInt4* __restrict__ B,  // 4bit quantized weight matrix of shape kxn
    PackTypeInt4* __restrict__ C,        // fp16 output buffer of shape mxn
    PackTypeInt4* __restrict__ C_tmp,    // fp32 tmp output buffer (for reduce)
    const PackTypeInt4* __restrict__ scales_ptr,  // fp16 quantization scales of shape
                                          // (k/groupsize)xn
    const PackTypeInt4* __restrict__ zp_ptr,      // 4bit packed zero-points of shape
                                          // (k/groupsize)x(n/pack_factor)
    const int* __restrict__ g_idx,        // int32 group indices of shape k
    int prob_m,           // batch dimension m
    int prob_n,           // output dimension n
    int prob_k,           // reduction dimension k
    int quant_group_power2, // quant group means how many quanted values share the same scale and zero, this value restricts to 2^x where x >= 5
    int max_iters,        // max tile iterations for one block
    int* locks,           // extra global storage for barrier synchronization
    bool use_fp32_reduce  // whether to use fp32 global reduce
) {
    int bidx = blockIdx.x;
    __shared__ uint8_t smem_base[0x4000]; //4x16x256 = 16Kbytes
    using LoadingManagerType = LoadingManager<scalar_t, w_type_id, THREADS, BLOCKS_M, BLOCKS_N, BLOCKS_K, HAS_ACT_ORDER, HAS_ZP, HAS_M_PRED, HAS_NK_PRED>;
    LoadingManagerType loading_manager;
    A += blockIdx.y * (MAX_BLOCKS_M * SLICE_M) * prob_k / (sizeof(PackTypeInt4) / sizeof(scalar_t));
    C += blockIdx.y * (MAX_BLOCKS_M * SLICE_M) * prob_n / (sizeof(PackTypeInt4) / sizeof(scalar_t));
    #ifdef BF16_HIGH_PRECISION
    if constexpr(std::is_same_v<scalar_t, __maca_bfloat16>) {
        C_tmp += blockIdx.y * (MAX_BLOCKS_M * SLICE_M) * prob_n / (sizeof(PackTypeInt4) / sizeof(float));
    }
    #endif
    loading_manager.set_address(A, B, C, C_tmp, scales_ptr, zp_ptr);
    //loading_manager.init_address(prob_m, prob_n, prob_k, bidx, max_iters, smem_base);
    loading_manager.init_address_pre(std::min<int>(MAX_BLOCKS_M*SLICE_M, prob_m - blockIdx.y * (MAX_BLOCKS_M * SLICE_M)), prob_n, prob_k, quant_group_power2, bidx, max_iters, smem_base);
    loading_manager.clear_c();

    while (max_iters > 0) {
        loading_manager.init_bsm_addr(); //reset all bsm address for current tile
        loading_manager.ldg_scales(); //Load all scales to bsm
        loading_manager.ldg_zp();
        loading_manager.ldg_b(0);    //load b0 and b64, two gvm
        loading_manager.ldg_a(0);    //Load first k0~31 and all m, one ldg_b128, heavy load
        loading_manager.sts_scales();
        loading_manager.sts_zeros();
        barrier_bsm;
        loading_manager.lds_scales(); //load scale0 and scale64
        loading_manager.pack_scales(); //pack scales into two v2f structure

        int k_idx = 0;
        if constexpr(BLOCKS_K > 1) {
            #pragma unroll BLOCKS_K - 1
            for (; k_idx < BLOCKS_K - 1; k_idx++) {
                int m_idx = 0;
                loading_manager.template on_dequant<false>(k_idx);
                //Loop for 3 times so that we can add some loading instructions before matmul
                if constexpr(BLOCKS_M > 1) {
                    #pragma unroll BLOCKS_M - 1
                    for (; m_idx < BLOCKS_M - 1; m_idx++) {
                        loading_manager.lds_a(m_idx+1); //pre load next 16 batches
                        loading_manager.matmul(m_idx); //do matmul
                    }
                }
                barrier_bsm; //wait for the last 16 batches ready, becasue of k_idx will be change
                loading_manager.next_k(); //modify gvm/bsm address of a and b
                loading_manager.matmul(m_idx); //do matmul
            }
        }
        int m_idx = 0;
        loading_manager.template on_dequant<true>(k_idx);
        if constexpr(BLOCKS_M > 1) {
            #pragma unroll BLOCKS_M - 1
            for (; m_idx < BLOCKS_M - 1; m_idx++) {
                loading_manager.lds_a(m_idx+1); //pre load next 16 batches
                loading_manager.matmul(m_idx); //do matmul
            }
        }
        loading_manager.next_tile_pre();

        loading_manager.matmul(m_idx); //do matmul
        max_iters--;

        if (loading_manager.tile_manager.need_save_data_pre()) {
            loading_manager.write_c_pre(); // reduce and write back
            loading_manager.clear_c();
        }

        barrier_bsm;
    }
}

} //end of namespace __hgemm_singular_blocks_k

namespace __hgemm_even_blocks_k {
template<typename scalar_t, const vllm::ScalarTypeId w_type_id, int THREADS, int BLOCKS_M, int BLOCKS_N, int BLOCKS_K, bool HAS_ACT, bool HAS_ZP, bool HAS_M_PRED, bool HAS_NK_PRED>
struct LoadingManager {
    constexpr static int FragACount = 4;
    using FragA = PackTypeInt2;
    constexpr static int FragBCount = 1;
    using FragB = PackType;
    constexpr static int FragCCount = 4;
    #ifdef BF16_HIGH_PRECISION
    using FragC = scalar_t;
    #else
    //Directly use half as the final atomic type:
    //1. Half precision and data range satisfies need of deepseek gemm
    //2. C500 has no atomic instructions for bfloat16, we cannot atomic a bfloat16 memory
    //3. The perfect precision type of atomic should be fp32, but the cost is too high to allocate a temp memory for float atomic
    using atomic_type = half;
    using FragC = atomic_type;
    #endif
    const FragA* A;
    const FragA* A_loading;
    const FragB* B;
    const FragB* B_loading;
    FragC* C;
    float* C_temp;
    using FragScaleLoading = half2;
    using FragZeroLoading = uint32_t;
    const FragScaleLoading* scales;
    const FragScaleLoading* scales_loading;
    const FragZeroLoading* zeros;
    const FragZeroLoading* zeros_loading;

    constexpr static int DOUBLE_SLICE_K = SLICE_K * 2;
    constexpr static int DOUBLE_PAD_SLICE_K = SLICE_K * 2 + sizeof(PackTypeInt4) / sizeof(scalar_t);

    int m;
    int n;
    int k;
    int quant_group_power2;
    uint8_t* smem_base;
    int bidx;

    PackTypeInt4* bsm_a_ptr;
    scalar_t* bsm_scales_ptr;
    float* bsm_zeros_ptr;
    //float* remaining_bsm_ptr;

    PackTypeInt2 local_a[BLOCKS_M][2];
    PackType local_b[N_ITERS];
    PackType local_b_cache[N_ITERS];
    scalar_t local_dequanted_b[N_ITERS][PACK_RATIO_4BITS];
    scalar_t local_dequanted_b_8bits[N_ITERS][2][PACK_RATIO_8BITS];
    v2f local_scales[N_ITERS];
    v2f local_zeros[N_ITERS];
    FragScaleLoading temp_scales;
    PackType temp_zeros;
    float output[BLOCKS_M][N_ITERS][4];
    FragA temp_a[LOADING_A_LOOP];

    TileManager<BLOCKS_M, BLOCKS_N, BLOCKS_K> tile_manager;
    ThreadView tv;

    __device__ __forceinline__ void set_address(const PackTypeInt4* a,
        const PackTypeInt4* b,
        PackTypeInt4* c,
        PackTypeInt4* c_temp,
        const PackTypeInt4* scale_ptr,
        const PackTypeInt4* zp_ptr = nullptr) {
            A = (const FragA*)a;
            B = (const FragB*)b;
            C = (FragC*)c;
            C_temp = (float*)c_temp;
            scales = (const FragScaleLoading*)scale_ptr;
            if constexpr(w_type_id == vllm::kU4.id()) {
                zeros = (const FragZeroLoading*)zp_ptr;
            }
    }

    __device__ __forceinline__ bool debug() {
        #ifdef DEBUG
        bool do_print = tv.wave_idx == 1 && tv.slot_idx == 0 && tv.slot_tid == 0;
        return do_print;
        #else
        return false;
        #endif
    }

    __device__ __forceinline__ void next_k0() {
        //reset bsm a to base
        bsm_a_ptr = (PackTypeInt4*)smem_base;
        bsm_a_ptr += tv.slot_tid * (DOUBLE_PAD_SLICE_K / (sizeof(PackTypeInt4) / sizeof(scalar_t))) + tv.slot_idx;
    }

    __device__ __forceinline__ void next_k1() {
        //Update only bsm_a_ptr
        bsm_a_ptr = (PackTypeInt4*)smem_base;
        bsm_a_ptr += tv.slot_tid * (DOUBLE_PAD_SLICE_K / (sizeof(PackTypeInt4) / sizeof(scalar_t))) + tv.slot_idx + WAVE_SLOTS;
        //load k32~k63
        //bsm_a_ptr += 4;
    }

    __device__ __forceinline__ void next_k0_pre() {
        //A_loading += SLICE_K / FragACount;
        //B_loading += SLICE_K / PACK_RATIO_4BITS * n;
        if constexpr(w_type_id == vllm::kU4.id() || w_type_id == vllm::kU4B8.id()) {
            B_loading += SLICE_K / PACK_RATIO_4BITS * n;
        } else if constexpr(w_type_id == vllm::kU8.id() || w_type_id == vllm::kU8B128.id()) {
            B_loading += SLICE_K / PACK_RATIO_8BITS * n;
        }
    }

    __device__ __forceinline__ void next_k1_pre() {
        A_loading += DOUBLE_SLICE_K / FragACount;
        if constexpr(w_type_id == vllm::kU4.id() || w_type_id == vllm::kU4B8.id()) {
            B_loading += SLICE_K / PACK_RATIO_4BITS * n;
        } else if constexpr(w_type_id == vllm::kU8.id() || w_type_id == vllm::kU8B128.id()) {
            B_loading += SLICE_K / PACK_RATIO_8BITS * n;
        }
    }

    __device__ __forceinline__ void ldg_a(int k_idx) {
        //32x64/2/256 = 16 / 4 = 4
        int t = tv.tid;
        int k_broad = tile_manager.tile_start_row * TILE_K + k_idx * SLICE_K;
        #pragma unroll LOADING_A_LOOP
        for (int i = 0; i < LOADING_A_LOOP; i++)  {
            int reading_m = t / (DOUBLE_SLICE_K / FragACount);
            int reading_k = t % (DOUBLE_SLICE_K / FragACount);
            int gvm_offset = reading_m * k / FragACount + reading_k;
            FragA* gvm_addr = (FragA*)A_loading + gvm_offset;
            //FIXME: we cannot do slice k pad as ldg_b32_bsm_async seems does not support padding
            if constexpr(HAS_M_PRED && HAS_NK_PRED) {
                bool pred = reading_m < m;
                bool pred_k = k_broad + reading_k * FragACount < k;
                pred = pred && pred_k && tile_manager.global_pred;
                ldg_b64_reg_noasync(temp_a[i], gvm_addr, pred, true);
            } else if constexpr(HAS_M_PRED) {
                bool pred = reading_m < m && tile_manager.global_pred;
                ldg_b64_reg_noasync(temp_a[i], gvm_addr, pred, true);
            } else if constexpr(HAS_NK_PRED) {
                bool pred_k = k_broad + reading_k * FragACount < k && tile_manager.global_pred;
                ldg_b64_reg_noasync(temp_a[i], gvm_addr, pred_k, true);
            } else {
                ldg_b64_reg_noasync(temp_a[i], gvm_addr, tile_manager.global_pred, true);
            }
            t += THREADS;
        }
    }

    __device__ __forceinline__ void sts_a() {
        FragA* to_bsm_a_ptr = (FragA*)smem_base;
        int t = tv.tid;
        #pragma unroll LOADING_A_LOOP
        for (int i = 0; i < LOADING_A_LOOP; i++)  {
            int reading_m = t / (DOUBLE_SLICE_K / FragACount);
            int reading_k = t % (DOUBLE_SLICE_K / FragACount);
            int bsm_offset = reading_m * (DOUBLE_PAD_SLICE_K / FragACount) + reading_k;
            *(to_bsm_a_ptr + bsm_offset) = temp_a[i];
            t += THREADS;
        }
    }

    __device__ __forceinline__ void lds_a(int midx) {
        *((PackTypeInt4*)local_a[midx]) = *bsm_a_ptr;
        bsm_a_ptr += SLOT * (DOUBLE_PAD_SLICE_K / (sizeof(PackTypeInt4) / sizeof(scalar_t)));
    }

    //TODO: implement when N_ITERS==1 or N_ITERS==3
    //korder used gptq_8bits, ldg_b will load two times in one SLICE_K
    //For example, t0 loads packed_k0, packed_k1, and packed_k0 represents a packed 4 ks in first line of B,
    //and packed_k1 represents a packed 4 ks in second line of B
    __device__ __forceinline__ void ldg_b(int k_idx, int korder = 0) {
        if constexpr(HAS_NK_PRED) {
            bool pred_k = tile_manager.tile_start_row * TILE_K + k_idx * SLICE_K + tv.slot_idx * PACK_RATIO_4BITS + korder < k;
            bool pred_n = tile_manager.tile_start_col * TILE_N + (tv.wave_idx * SLOT + tv.slot_tid)*N_ITERS < n;
            bool pred = pred_n && pred_k && tile_manager.global_pred;
            FragB* addr =  (FragB*)B_loading + korder * n;
            if constexpr(N_ITERS == 2) {
                ldg_b64_reg_noasync(*((PackTypeInt2*)local_b_cache), ((PackTypeInt2*)addr), pred, true);
            } else if constexpr(N_ITERS == 4) {
                ldg_b128_reg_noasync(*((PackTypeInt4*)local_b_cache), ((PackTypeInt4*)addr), pred, true);
            }
        } else {
            FragB* addr =  (FragB*)B_loading + korder * n;
            if constexpr(N_ITERS == 2) {
                ldg_b64_reg_noasync(*((PackTypeInt2*)local_b_cache), ((PackTypeInt2*)addr), tile_manager.global_pred, true);
            } else if constexpr(N_ITERS == 4) {
                ldg_b128_reg_noasync(*((PackTypeInt4*)local_b_cache), ((PackTypeInt4*)addr), tile_manager.global_pred, true);
            }
        }
    }

    __device__ __forceinline__ void swap_b_cache(int i) {
        local_b[i] = local_b_cache[i];
    }

    __device__ __forceinline__ void ldg_scales() {
        bool pred = tv.tid < TILE_N / (sizeof(FragScaleLoading) / sizeof(scalar_t)) && tile_manager.global_pred;
        if constexpr(HAS_NK_PRED) {
            pred = pred && tv.tid < (n - tile_manager.tile_start_col * TILE_N) / (sizeof(FragScaleLoading) / sizeof(scalar_t));
        }
        //FragScaleLoading *scale_bsm = (FragScaleLoading*)(smem_base + 0x2000) + tv.tid;
        FragScaleLoading *gvm_addr = (FragScaleLoading*)scales_loading + tv.tid;
        //ldg_b32_bsm_async(scale_bsm, gvm_addr, pred, false);
        ldg_b32_reg_noasync(*((PackType*)&temp_scales), gvm_addr, pred, true);
    }

    __device__ __forceinline__ void ldg_zp() {
        if constexpr(w_type_id == vllm::kU4.id()) {
            bool pred = (tv.tid < TILE_N / PACK_RATIO_4BITS) && tile_manager.global_pred;
            if constexpr(HAS_NK_PRED) {
                pred = pred && tv.tid < ((n - tile_manager.tile_start_col * TILE_N) / PACK_RATIO_4BITS);
            }
            FragZeroLoading *gvm_addr = (FragZeroLoading*)zeros_loading + tv.tid;
            ldg_b32_reg_noasync(*((PackType*)&temp_zeros), gvm_addr, pred, true);
        }
    }

    __device__ __forceinline__ void sts_scales() {
        FragScaleLoading *scale_bsm = (FragScaleLoading*)(smem_base + 0x3000) + tv.tid;
        *scale_bsm = temp_scales;
    }

    __device__ __forceinline__ void sts_zeros() {
        if constexpr(w_type_id == vllm::kU4.id()) {
            bool pred = (tv.tid < TILE_N / PACK_RATIO_4BITS) && tile_manager.global_pred;
            if (pred) {
                float temp[PACK_RATIO_4BITS];
                decompress_zero_4bits(temp_zeros, temp);
                float *zeros_bsm = (float*)(smem_base + 0x3400) + tv.tid * PACK_RATIO_4BITS;
                for (int i = 0; i < PACK_RATIO_4BITS; i++) {
                    *(zeros_bsm + i) = temp[i];
                }
            }
        }
    }

    //TODO: implement when N_ITERS==1 or N_ITERS==3
    __device__ __forceinline__ void lds_scales() {
        if constexpr(N_ITERS==2) {
            *((half2*)local_dequanted_b[0]) = *((half2*)bsm_scales_ptr);
        } else if constexpr(N_ITERS==4) {
            *((PackTypeInt2*)local_dequanted_b[0]) = *((PackTypeInt2*)bsm_scales_ptr);
        }
    }

    __device__ __forceinline__ void pack_scales() {
        if constexpr(w_type_id == vllm::kU4B8.id()) {
            #pragma unroll
            for (int i = 0; i < N_ITERS; i++) {
                float s = local_dequanted_b[0][i];
                float z = -8 * s;
                local_scales[i] = {s, s};
                local_zeros[i] = {z, z};
            }
        } else if constexpr(w_type_id == vllm::kU4.id()) {
            #pragma unroll
            for (int i = 0; i < N_ITERS; i++) {
                float s;
                if constexpr(std::is_same_v<scalar_t, __maca_bfloat16>) {
                    s = __bfloat162float(local_dequanted_b[0][i]);
                } else {
                    s = local_dequanted_b[0][i];
                }
                float z = *(bsm_zeros_ptr + i);
                z = z * s;
                local_scales[i] = {s, s};
                local_zeros[i] = {z, z};
            }
        } else if constexpr(w_type_id == vllm::kU8B128.id()) {
            #pragma unroll
            for (int i = 0; i < N_ITERS; i++) {
                float s = local_dequanted_b[0][i];
                float z = -128 * s;
                local_scales[i] = {s, s};
                local_zeros[i] = {z, z};
            }
        } else if constexpr(w_type_id == vllm::kU8.id()) {
            // should apply zeros
        }
    }

    __device__ __forceinline__ void dequant(int kdx, int korder = 0) {
        if constexpr(w_type_id == vllm::kU4.id() || w_type_id == vllm::kU4B8.id()) {
            dequant_gptq_4bits<scalar_t>(local_b[kdx], local_dequanted_b[kdx], local_scales[kdx], local_zeros[kdx]);
        } else if constexpr(w_type_id == vllm::kU8.id() || w_type_id == vllm::kU8B128.id()) {
            dequant_gptq_8bits<scalar_t>(local_b[kdx], local_dequanted_b_8bits[kdx][korder], local_scales[kdx], local_zeros[kdx]);
        }
    }

    __device__ __forceinline__ void matmul(int mdx) {
        if constexpr(w_type_id == vllm::kU4.id() || w_type_id == vllm::kU4B8.id()) {
            #pragma unroll
            for (int i = 0; i < N_ITERS; i++) {
                mma_16x16x16<scalar_t>(local_a[mdx][0], *((PackTypeInt2*)local_dequanted_b[i]), *((PackTypeInt4*)output[mdx][i]));
                mma_16x16x16<scalar_t>(local_a[mdx][1], *((PackTypeInt2*)local_dequanted_b[i] + 1), *((PackTypeInt4*)output[mdx][i]));
            }
        } else if constexpr(w_type_id == vllm::kU8.id() || w_type_id == vllm::kU8B128.id()) {
            #pragma unroll
            for (int i = 0; i < N_ITERS; i++) {
                mma_16x16x16<scalar_t>(local_a[mdx][0], *((PackTypeInt2*)local_dequanted_b_8bits[i][0]), *((PackTypeInt4*)output[mdx][i]));
                mma_16x16x16<scalar_t>(local_a[mdx][1], *((PackTypeInt2*)local_dequanted_b_8bits[i][1]), *((PackTypeInt4*)output[mdx][i]));
            }
        }
    }

    __device__ __forceinline__ void clear_c() {
        #pragma unroll
        for (int miter = 0; miter < BLOCKS_M; miter++) {
            #pragma unroll
            for (int niter = 0; niter < N_ITERS; niter++) {
                #pragma unroll
                for (int miter2 = 0; miter2 < 4; miter2++) {
                    output[miter][niter][miter2] = 0;
                }
            }
        }
    }

    //functions for preloading next tile data
    __device__ __forceinline__ void init_address_pre(int _m, int _n, int _k, int _quant_group_power2, int _bidx, int _iters, uint8_t *_smem_base) {
        tv.init();
        m = _m;
        n = _n;
        k = _k;
        quant_group_power2 = _quant_group_power2;
        bidx = _bidx;
        smem_base = _smem_base;
        tile_manager.init(m, n, k, bidx, _iters);
        init_tile_pre(tile_manager.tile_start_col, tile_manager.tile_start_row);
    }

     __device__ __forceinline__ void init_tile_pre(int col, int row) {
        //Initialize start slice address and set them to A_loading and B_loading
        int offset_n = col * TILE_N;
        int offset_k = row * TILE_K;
        //A_loading address will always be valid
        A_loading = A + offset_k / (FragACount);
        if constexpr(w_type_id == vllm::kU4.id() || w_type_id == vllm::kU4B8.id()) {
            B_loading = B + (offset_k / PACK_RATIO_4BITS * n + offset_n) / (FragBCount) + tv.slot_idx * n + (tv.wave_idx * SLOT + tv.slot_tid)*N_ITERS;
        } else if constexpr(w_type_id == vllm::kU8.id() || w_type_id == vllm::kU8B128.id()) {
            B_loading = B + (offset_k / PACK_RATIO_8BITS * n + offset_n) / (FragBCount) + tv.slot_idx * n * 2 + (tv.wave_idx * SLOT + tv.slot_tid)*N_ITERS;
        }
        scales_loading = scales + ((offset_k >> quant_group_power2) * n + offset_n) / (sizeof(FragScaleLoading)/sizeof(scalar_t));
        if constexpr(w_type_id == vllm::kU4.id()) {
            zeros_loading = zeros + ((offset_k >> quant_group_power2) * n + offset_n) / PACK_RATIO_4BITS;
        }
    }

    __device__ __forceinline__ void next_tile_pre() {
        tile_manager.next_tile_pre();
        init_tile_pre(tile_manager.tile_start_col, tile_manager.tile_start_row);
    }

    __device__ __forceinline__ void init_bsm_addr() {
        bsm_a_ptr = (PackTypeInt4*)smem_base;           //use 8k bytes, will load at most 32x128*sizeof(half), either m32k128 or m128k32
        //remaining_bsm_ptr = (float*)(smem_base + 0x2000 + 0x1000); //3k bytes
        bsm_a_ptr += tv.slot_tid * (DOUBLE_PAD_SLICE_K / (sizeof(PackTypeInt4) / sizeof(scalar_t))) + tv.slot_idx;
        bsm_scales_ptr = (scalar_t*)(smem_base + 0x3000);      //use 128xsizeof(float)*2 = 1k
        bsm_scales_ptr += (tv.wave_idx * SLOT + tv.slot_tid) * N_ITERS;
        if constexpr(w_type_id == vllm::kU4.id()) {
            bsm_zeros_ptr = (float*)(smem_base + 0x3400);      //use 128xsizeof(float)*2 = 1k
            bsm_zeros_ptr += (tv.wave_idx * SLOT + tv.slot_tid) * N_ITERS;
        }
    }

    __device__ __forceinline__ void write_c(int offset, const float& v) {
        if constexpr(std::is_same_v<scalar_t, __maca_bfloat16>) {
            #ifdef BF16_HIGH_PRECISION
	        atomicAdd(C_temp + offset, v);
            #else
            atomicAdd(C + offset, (atomic_type)v);
            #endif
        } else {
            atomicAdd(C + offset, (scalar_t)v);
        }
    }

    //atomic write to c
    __device__ __forceinline__ void write_c_pre() {
        int k_broad = tv.slot_idx * 4;
        int n_broad = (tv.wave_idx * SLOT + tv.slot_tid) * N_ITERS + tile_manager.tile_start_col_cache * TILE_N;
        #pragma unroll
        for (int miter = 0; miter < BLOCKS_M; miter++) {
            #pragma unroll
            for (int niter = 0; niter < N_ITERS; niter++) {
                int store_n = n_broad + niter;
                #pragma unroll
                for (int miter2 = 0; miter2 < 4; miter2++) {
                    int store_m = k_broad + miter * SLICE_M + miter2;
                    if constexpr(HAS_M_PRED && HAS_NK_PRED) {
                        if (store_m < m && store_n < n) {
			                write_c(store_m * n + store_n, output[miter][niter][miter2]);
                        }
                    } else if constexpr(HAS_M_PRED) {
                        if (store_m < m) {
			                write_c(store_m * n + store_n, output[miter][niter][miter2]);
                        }
                    } else if constexpr(HAS_NK_PRED) {
                        if (store_n < n) {
			                write_c(store_m * n + store_n, output[miter][niter][miter2]);
                        }
                    } else {
			            write_c(store_m * n + store_n, output[miter][niter][miter2]);
                    }
                }
            }
        }
    }

    template<int K, bool KTAIL>
    __device__ __forceinline__ void on_sts_a() {
        if constexpr(K == 0) {
            sts_a(); // reorder data_a from reg(gvm resouce) to bsm
        }
    }

    template<int K, bool KTAIL>
    __device__ __forceinline__ void on_preload(int k_idx) {
        if constexpr(!KTAIL) {
            if constexpr(K == 0) {
                next_k0_pre(); // preload gvm a/b
            } else {
                next_k1_pre(); // preload gvm a/b
            }
            ldg_b(k_idx + K + 1); //preload b for next k
            if constexpr(K == 1) {
                ldg_a(k_idx + K + 1); //preload a for next k
            }
        } else {
            next_tile_pre();
            ldg_scales();
            ldg_zp();
            ldg_b(0);
            ldg_a(0);
        }
    }

    template<int K, bool KTAIL>
    __device__ __forceinline__ void on_dequant_niters1(int k_idx) {
        if constexpr(w_type_id == vllm::kU4.id() || w_type_id == vllm::kU4B8.id()) {
            swap_b_cache(0);
            on_sts_a<K,KTAIL>(); // reorder data_a from reg(gvm resouce) to bsm
            on_preload<K,KTAIL>(k_idx);
            barrier_bsm; //sync threads for sts_a
            lds_a(0); //load first 16 batches into registers
            dequant(0); //dequant b0
        } else if constexpr(w_type_id == vllm::kU8.id() || w_type_id == vllm::kU8B128.id()) {
            swap_b_cache(0);
            ldg_b(0, 1);
            dequant(0, 0);
            on_sts_a<K,KTAIL>(); // reorder data_a from reg(gvm resouce) to bsm
            swap_b_cache(0);
            on_preload<K,KTAIL>(k_idx);
            barrier_bsm; //sync threads for sts_a
            lds_a(0); //load first 16 batches into registers
            dequant(0, 1); //dequant b0
        }
    }

    template<int K, bool KTAIL>
    __device__ __forceinline__ void on_dequant_niters2(int k_idx) {
        if constexpr(w_type_id == vllm::kU4.id() || w_type_id == vllm::kU4B8.id()) {
            swap_b_cache(0);
            dequant(0); //dequant b0
            on_sts_a<K,KTAIL>();
            swap_b_cache(1);
            on_preload<K,KTAIL>(k_idx);
            barrier_bsm; //sync threads for sts_a
            lds_a(0); //load first 16 batches into registers
            dequant(1);   //dequant b64
        } else if constexpr(w_type_id == vllm::kU8.id() || w_type_id == vllm::kU8B128.id()) {
            swap_b_cache(0);
            swap_b_cache(1);
            ldg_b(0, 1);
            dequant(0); //dequant b0
            on_sts_a<K,KTAIL>();
            dequant(1);
            swap_b_cache(0);
            swap_b_cache(1);
            on_preload<K,KTAIL>(k_idx);
            barrier_bsm; //sync threads for sts_a
            lds_a(0); //load first 16 batches into registers
            dequant(0, 1);   //dequant b64
            dequant(1, 1);   //dequant b64
        }
    }

    template<int K, bool KTAIL>
    __device__ __forceinline__ void on_dequant_niters3(int k_idx) {
        if constexpr(w_type_id == vllm::kU4.id() || w_type_id == vllm::kU4B8.id()) {
            swap_b_cache(0);
            dequant(0); //dequant b0
            on_sts_a<K,KTAIL>();
            swap_b_cache(1);
            dequant(1);
            swap_b_cache(2);
            on_preload<K,KTAIL>(k_idx);
            barrier_bsm; //sync threads for sts_a
            lds_a(0); //load first 16 batches into registers
            dequant(2); //dequant b0
        } else if constexpr(w_type_id == vllm::kU8.id() || w_type_id == vllm::kU8B128.id()) {
            swap_b_cache(0);
            swap_b_cache(1);
            swap_b_cache(2);
            ldg_b(0, 1);
            dequant(0); //dequant b0
            on_sts_a<K,KTAIL>();
            swap_b_cache(1);
            dequant(1);
            dequant(2);
            swap_b_cache(0);
            swap_b_cache(1);
            swap_b_cache(2);
            on_preload<K,KTAIL>(k_idx);
            barrier_bsm; //sync threads for sts_a
            lds_a(0); //load first 16 batches into registers
            dequant(0, 1); //dequant b0
            dequant(1, 1); //dequant b0
            dequant(2, 1); //dequant b0
        }
    }

    template<int K, bool KTAIL>
    __device__ __forceinline__ void on_dequant_niters4(int k_idx) {
        if constexpr(w_type_id == vllm::kU4.id() || w_type_id == vllm::kU4B8.id()) {
            swap_b_cache(0);
            dequant(0); //dequant b0
            on_sts_a<K,KTAIL>();
            swap_b_cache(1);
            dequant(1); //dequant b1
            swap_b_cache(2);
            swap_b_cache(3);
            dequant(2); //dequant b2
            on_preload<K,KTAIL>(k_idx);
            barrier_bsm; //sync threads for sts_a
            lds_a(0); //load first 16 batches into registers
            dequant(3); //dequant b3
        } else if constexpr(w_type_id == vllm::kU8.id() || w_type_id == vllm::kU8B128.id()) {
            swap_b_cache(0);
            swap_b_cache(1);
            swap_b_cache(2);
            swap_b_cache(3);
            ldg_b(0, 1);
            dequant(0); //dequant b0
            on_sts_a<K,KTAIL>();
            dequant(1); //dequant b1
            dequant(2); //dequant b2
            dequant(3); //dequant b2
            swap_b_cache(0);
            swap_b_cache(1);
            swap_b_cache(2);
            swap_b_cache(3);
            on_preload<K,KTAIL>(k_idx);
            barrier_bsm; //sync threads for sts_a
            lds_a(0); //load first 16 batches into registers
            dequant(0, 1); //dequant b0
            dequant(1, 1); //dequant b1
            dequant(2, 1); //dequant b2
            dequant(3, 1); //dequant b3
        }
    }

    template<int K, bool KTAIL>
    __device__ __forceinline__ void on_dequant(int kdx) {
        if constexpr(N_ITERS == 1) on_dequant_niters1<K,KTAIL>(kdx);
        else if constexpr(N_ITERS == 2) on_dequant_niters2<K,KTAIL>(kdx);
        else if constexpr(N_ITERS == 3) on_dequant_niters3<K,KTAIL>(kdx);
        else if constexpr(N_ITERS == 4) on_dequant_niters4<K,KTAIL>(kdx);
    }
};

template<typename scalar_t,
    const vllm::ScalarTypeId w_type_id,
    const int THREADS,          // number of threads in a threadblock
    const int BLOCKS_M,         // number of 16x16 blocks in the m
                                // dimension (batchsize) of the
                                // threadblock
    const int BLOCKS_N,         // same for n dimension (output)
    const int BLOCKS_K,         // same for k dimension (reduction)
    const bool HAS_ACT_ORDER,   // whether act_order is enabled
    const bool HAS_ZP,          // whether zero-points are enabled
    const bool HAS_M_PRED = true,  //If we should use predictors to load m from gvm
    const bool HAS_NK_PRED = true  //If we should use predictors to load nk from gvm
    >
__global__ void hgemm_gptq(
    const PackTypeInt4* __restrict__ A,  // fp16 input matrix of shape mxk
    const PackTypeInt4* __restrict__ B,  // 4bit quantized weight matrix of shape kxn
    PackTypeInt4* __restrict__ C,        // fp16 output buffer of shape mxn
    PackTypeInt4* __restrict__ C_tmp,    // fp32 tmp output buffer (for reduce)
    const PackTypeInt4* __restrict__ scales_ptr,  // fp16 quantization scales of shape
                                          // (k/groupsize)xn
    const PackTypeInt4* __restrict__ zp_ptr,      // 4bit packed zero-points of shape
                                          // (k/groupsize)x(n/pack_factor)
    const int* __restrict__ g_idx,        // int32 group indices of shape k
    int prob_m,           // batch dimension m
    int prob_n,           // output dimension n
    int prob_k,           // reduction dimension k
    int quant_group_power2,
    int max_iters,        // max tile iterations for one block
    int* locks,           // extra global storage for barrier synchronization
    bool use_fp32_reduce  // whether to use fp32 global reduce
) {
    int bidx = blockIdx.x;
    __shared__ uint8_t smem_base[0x4000]; //4x16x256 = 16Kbytes
    using LoadingManagerType = LoadingManager<scalar_t, w_type_id, THREADS, BLOCKS_M, BLOCKS_N, BLOCKS_K, HAS_ACT_ORDER, HAS_ZP, HAS_M_PRED, HAS_NK_PRED>;
    LoadingManagerType loading_manager;
    A += blockIdx.y * (MAX_BLOCKS_M * SLICE_M) * prob_k / (sizeof(PackTypeInt4) / sizeof(scalar_t));
    C += blockIdx.y * (MAX_BLOCKS_M * SLICE_M) * prob_n / (sizeof(PackTypeInt4) / sizeof(scalar_t));
    #ifdef BF16_HIGH_PRECISION
    if constexpr(std::is_same_v<scalar_t, __maca_bfloat16>) {
        C_tmp += blockIdx.y * (MAX_BLOCKS_M * SLICE_M) * prob_n / (sizeof(PackTypeInt4) / sizeof(float));
    }
    #endif
    loading_manager.set_address(A, B, C, C_tmp, scales_ptr, zp_ptr);
    //loading_manager.init_address(prob_m, prob_n, prob_k, bidx, max_iters, smem_base);
    loading_manager.init_address_pre(std::min<int>(MAX_BLOCKS_M*SLICE_M, prob_m - blockIdx.y * (MAX_BLOCKS_M * SLICE_M)), prob_n, prob_k, quant_group_power2, bidx, max_iters, smem_base);

    loading_manager.ldg_scales(); //Load all scales to bsm
    loading_manager.ldg_zp();
    loading_manager.ldg_b(0);    //load b in k0~31
    loading_manager.ldg_a(0);    //Load first k0~63 and all m
    loading_manager.clear_c();

    while (max_iters > 0) {
        loading_manager.init_bsm_addr(); //reset all bsm address for current tile
        loading_manager.sts_scales();
        loading_manager.sts_zeros();
        barrier_bsm;
        loading_manager.lds_scales(); //load scale0 and scale64
        loading_manager.pack_scales(); //pack scales into two v2f structure

        int k_idx = 0;
        if constexpr(BLOCKS_K / 2 - 1 > 0) {
            #pragma unroll BLOCKS_K / 2 - 1
            for (int kloop = 0; kloop < BLOCKS_K / 2 - 1; kloop++) {
                int m_idx = 0;
                loading_manager.template on_dequant<0, false>(k_idx);
                if constexpr(BLOCKS_M > 1) {
                    #pragma unroll BLOCKS_M - 1
                    for (; m_idx < BLOCKS_M - 1; m_idx++) {
                        loading_manager.lds_a(m_idx+1); //pre load next 16 batches
                        loading_manager.matmul(m_idx); //do matmul
                    }
                }
                barrier_bsm; //wait for the last 16 batches ready, becasue of k_idx will be change
                loading_manager.next_k1(); //modify gvm/bsm address of a and b
                loading_manager.matmul(m_idx); //do matmul
                m_idx = 0;
                loading_manager.template on_dequant<1, false>(k_idx);
                if constexpr(BLOCKS_M > 1) {
                    #pragma unroll BLOCKS_M - 1
                    for (; m_idx < BLOCKS_M - 1; m_idx++) {
                        loading_manager.lds_a(m_idx+1); //pre load next 16 batches
                        loading_manager.matmul(m_idx); //do matmul
                    }
                }
                barrier_bsm; //wait for the last 16 batches ready, becasue of k_idx will be change
                loading_manager.next_k0(); //modify gvm/bsm address of a and b
                loading_manager.matmul(m_idx); //do matmul
                k_idx += 2;
            }
        }

        int m_idx = 0;
        loading_manager.template on_dequant<0, false>(k_idx);
        if constexpr(BLOCKS_M > 1) {
            #pragma unroll BLOCKS_M - 1
            for (; m_idx < BLOCKS_M - 1; m_idx++) {
                loading_manager.lds_a(m_idx+1); //pre load next 16 batches
                loading_manager.matmul(m_idx); //do matmul
            }
        }
        barrier_bsm; //wait for the last 16 batches ready, becasue of k_idx will be change
        loading_manager.next_k1(); //modify gvm/bsm address of a and b
        loading_manager.matmul(m_idx); //do matmul
        m_idx = 0;
        loading_manager.template on_dequant<1, true>(k_idx);
        if constexpr(BLOCKS_M > 1) {
            #pragma unroll BLOCKS_M - 1
            for (; m_idx < BLOCKS_M - 1; m_idx++) {
                loading_manager.lds_a(m_idx+1); //pre load next 16 batches
                loading_manager.matmul(m_idx); //do matmul
            }
        }
        barrier_bsm; //wait for the last 16 batches ready, becasue of k_idx will be change
        //loading_manager.next_tile_pre(); should move into on_dequant ?
        loading_manager.matmul(m_idx); //do matmul

        max_iters--;

        if (loading_manager.tile_manager.need_save_data_pre()) {
            loading_manager.write_c_pre(); // reduce and write back
            loading_manager.clear_c();
        }

        barrier_bsm;
    }
}
} //end of namespace __hgemm_even_blocks_k

template<typename scalar_t,
    const vllm::ScalarTypeId w_type_id,
    int THREADS,
    int BLOCKS_M,
    int BLOCKS_N,
    int BLOCKS_K,
    bool HAS_ACT_ORDER,
    bool HAS_ZP,
    bool HAS_M_PRED,
    bool HAS_NK_PRED>
bool launch_gemm_gptq_kernel(const PackTypeInt4* A,
    const PackTypeInt4* B,
    PackTypeInt4* C,
    PackTypeInt4* C_temp,
    const PackTypeInt4* scales,
    const PackTypeInt4* zeros,
    int* g_idx, int m, int n, int k, int quant_group, int chunks, cudaStream_t stream = nullptr) {
    int tiles_m = div_ceil(m, TILE_M);
    int tiles_n = div_ceil(n, TILE_N);
    int tiles_k = div_ceil(k, TILE_K);
    if (TILE_K > quant_group && TILE_K % quant_group != 0) {
        printf("Invalid TILE_K %d that can not be dived by QUANT_GROUP %d\n", TILE_K, quant_group);
        return false;
    }

    int total_tiles = tiles_n * tiles_k;
    int blocks = PEUS;
    int iters = div_ceil(total_tiles, PEUS);
    if (total_tiles < PEUS) {
        if (TILE_K < quant_group) {
            iters = quant_group / TILE_K;
            blocks = div_ceil(total_tiles, iters);
        } else {
            iters = 1;
            blocks = total_tiles;
        }
    } else {
        if (TILE_K < quant_group) {
            iters = div_ceil(iters, quant_group / TILE_K) * quant_group / TILE_K;
            blocks = div_ceil(total_tiles, iters);
        }
    }
    while (iters * blocks - total_tiles >= iters) {
        blocks -= 1;
    }

    if (total_tiles < blocks) {
        printf("total slice %d < blocks %d, Invalid configure\n", total_tiles, blocks);
        return false;
    }
    // printf("Launching hgemm_gptq_4bits THREADS=%d, BLOCKS_M=%d, BLOCKS_N=%d, BLOCKS_K=%d, HAS_M_PRED=%d, HAS_NK_PRED=%d, tiles_n=%d, tiles_k = %d, total_tiles = %d, iters = %d, blocks = %d, chunks = %d, TILE=m%dn%dk%d\n",
    // THREADS, BLOCKS_M, BLOCKS_N, BLOCKS_K, HAS_M_PRED, HAS_NK_PRED, tiles_n, tiles_k, total_tiles, iters, blocks, chunks, BLOCKS_M*SLICE_M, BLOCKS_N*SLICE_N, BLOCKS_K*SLICE_K
    // );

    if constexpr(std::is_same_v<scalar_t, __maca_bfloat16>) {
        size_t num_elem = size_t(m) * size_t(n);
        size_t clean_blocks = std::max(size_t(1), (num_elem + clean_kernel_thread_num * clean_kernel_pack_num - 1)/ (clean_kernel_thread_num * clean_kernel_pack_num));
        clean_zero<clean_kernel_thread_num, clean_kernel_pack_num><<<clean_blocks, clean_kernel_thread_num, 0, stream>>>((float*)C_temp, num_elem);
    }

    //It is better to do perm before launch kernel
    if constexpr(BLOCKS_K % 2 == 1) {
        __hgemm_singular_blocks_k::hgemm_gptq<scalar_t,
            w_type_id,
            THREADS,
            BLOCKS_M, BLOCKS_N, BLOCKS_K,
            HAS_ACT_ORDER, HAS_ZP, HAS_M_PRED, HAS_NK_PRED
            ><<<dim3(std::max(blocks,1), std::max(chunks,1), 1), THREADS, 0, stream>>>(A, B, C, C_temp, scales, zeros, g_idx, m, n, k, get_power2(quant_group), iters, nullptr, false);
    } else {
        __hgemm_even_blocks_k::hgemm_gptq<scalar_t,
            w_type_id,
            THREADS,
            BLOCKS_M, BLOCKS_N, BLOCKS_K,
            HAS_ACT_ORDER, HAS_ZP, HAS_M_PRED, HAS_NK_PRED
            ><<<dim3(std::max(blocks,1), std::max(chunks,1), 1), THREADS, 0, stream>>>(A, B, C, C_temp, scales, zeros, g_idx, m, n, k, get_power2(quant_group), iters, nullptr, false);
    }

    if constexpr(std::is_same_v<scalar_t, __maca_bfloat16>) {
        size_t num_elem = size_t(m) * size_t(n);
        size_t reduce_blocks = std::max(size_t(1), (num_elem  + reduce_kernel_thread_num * reduce_kernel_pack_num - 1) / (reduce_kernel_thread_num * reduce_kernel_pack_num));
        all_reduce<reduce_kernel_thread_num, reduce_kernel_pack_num, false><<<reduce_blocks, reduce_kernel_thread_num, 0, stream>>>((float*)C_temp, (maca_bfloat16*)C, num_elem);
    }

    return true;
}

}

