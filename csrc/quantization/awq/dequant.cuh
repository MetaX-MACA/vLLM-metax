// 2025 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#pragma once

#define BFE(source, bit_start, num_bits) __builtin_mxc_ubfe(source, bit_start, num_bits);

template <typename outputT, typename inputT, int qbits>
__device__ __forceinline__ outputT extract(inputT *input, size_t loc)
{
    constexpr int widthPerElem = sizeof(inputT) * 8 / qbits;
    return (outputT)BFE(input[loc / widthPerElem], loc % widthPerElem, qbits);
}