/*
Adapted from https://github.com/mit-han-lab/llm-awq
Modified from NVIDIA FasterTransformer:
https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/cutlass_extensions/include/cutlass_extensions/interleaved_numeric_conversion.h
@article{lin2023awq,
  title={AWQ: Activation-aware Weight Quantization for LLM Compression and
Acceleration}, author={Lin, Ji and Tang, Jiaming and Tang, Haotian and Yang,
Shang and Dang, Xingyu and Han, Song}, journal={arXiv}, year={2023}
}
*/

#pragma once

namespace vllm {
namespace awq {

template<typename VT>
__device__ uint4 dequantize_s4_to_fp16x2(uint32_t const& source) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 750
  assert(false);
#else
  uint4 result;

  uint32_t* h = reinterpret_cast<uint32_t*>(&result);
  uint32_t const i4s = reinterpret_cast<uint32_t const&>(source);

  // First, we extract the i4s and construct an intermediate fp16 number.
  static constexpr uint32_t immLut = (0xf0 & 0xcc) | 0xaa;
  static constexpr uint32_t BOTTOM_MASK = 0x000f000f;
  static constexpr uint32_t TOP_MASK = 0x00f000f0;
  static constexpr uint32_t I4s_TO_F16s_MAGIC_NUM = 0x64006400;

  // Note that the entire sequence only requires 1 shift instruction. This is
  // thanks to the register packing format and the fact that we force our
  // integers to be unsigned, and account for this in the fp16 subtractions. In
  // addition, I exploit the fact that sub and fma have the same throughput in
  // order to convert elt_23 and elt_67 to fp16 without having to shift them to
  // the bottom bits before hand.

  // Shift right by 8 to now consider elt_45 and elt_67. Issue first to hide RAW
  // dependency if we issue immediately before required.
  const uint32_t top_i4s = i4s >> 8;
#ifndef USE_MACA
  // Extract elt_01 - (i4s & 0x000f000f) | 0x64006400
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(h[0])
               : "r"(i4s), "n"(BOTTOM_MASK), "n"(I4s_TO_F16s_MAGIC_NUM),
                 "n"(immLut));
  // Extract elt_23 (i4s & 0x00f000f0) | 0x64006400
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(h[1])
               : "r"(i4s), "n"(TOP_MASK), "n"(I4s_TO_F16s_MAGIC_NUM),
                 "n"(immLut));
  // Extract elt_45 (top_i4s & 0x000f000f) | 0x64006400
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(h[2])
               : "r"(top_i4s), "n"(BOTTOM_MASK), "n"(I4s_TO_F16s_MAGIC_NUM),
                 "n"(immLut));
  // Extract elt_67 (top_i4s & 0x00f000f0) | 0x64006400
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(h[3])
               : "r"(top_i4s), "n"(TOP_MASK), "n"(I4s_TO_F16s_MAGIC_NUM),
                 "n"(immLut));
#else
      // >>>> PTX2CPP Success <<<<
{
(h[0])=0;
if((immLut)&0x01)(h[0])|=~(i4s)&~(BOTTOM_MASK)&~(I4s_TO_F16s_MAGIC_NUM);
if((immLut)&0x02)(h[0])|=~(i4s)&~(BOTTOM_MASK)& (I4s_TO_F16s_MAGIC_NUM);
if((immLut)&0x04)(h[0])|=~(i4s)& (BOTTOM_MASK)&~(I4s_TO_F16s_MAGIC_NUM);
if((immLut)&0x08)(h[0])|=~(i4s)& (BOTTOM_MASK)& (I4s_TO_F16s_MAGIC_NUM);
if((immLut)&0x10)(h[0])|= (i4s)&~(BOTTOM_MASK)&~(I4s_TO_F16s_MAGIC_NUM);
if((immLut)&0x20)(h[0])|= (i4s)&~(BOTTOM_MASK)& (I4s_TO_F16s_MAGIC_NUM);
if((immLut)&0x40)(h[0])|= (i4s)& (BOTTOM_MASK)&~(I4s_TO_F16s_MAGIC_NUM);
if((immLut)&0x80)(h[0])|= (i4s)& (BOTTOM_MASK)& (I4s_TO_F16s_MAGIC_NUM);
}

    // Extract elt_23 (i4s & 0x00f000f0) | 0x64006400

// >>>> PTX2CPP Success <<<<
{
(h[1])=0;
if((immLut)&0x01)(h[1])|=~(i4s)&~(TOP_MASK)&~(I4s_TO_F16s_MAGIC_NUM);
if((immLut)&0x02)(h[1])|=~(i4s)&~(TOP_MASK)& (I4s_TO_F16s_MAGIC_NUM);
if((immLut)&0x04)(h[1])|=~(i4s)& (TOP_MASK)&~(I4s_TO_F16s_MAGIC_NUM);
if((immLut)&0x08)(h[1])|=~(i4s)& (TOP_MASK)& (I4s_TO_F16s_MAGIC_NUM);
if((immLut)&0x10)(h[1])|= (i4s)&~(TOP_MASK)&~(I4s_TO_F16s_MAGIC_NUM);
if((immLut)&0x20)(h[1])|= (i4s)&~(TOP_MASK)& (I4s_TO_F16s_MAGIC_NUM);
if((immLut)&0x40)(h[1])|= (i4s)& (TOP_MASK)&~(I4s_TO_F16s_MAGIC_NUM);
if((immLut)&0x80)(h[1])|= (i4s)& (TOP_MASK)& (I4s_TO_F16s_MAGIC_NUM);
}
// >>>> PTX2CPP Success <<<<
{
(h[2])=0;
if((immLut)&0x01)(h[2])|=~(top_i4s)&~(BOTTOM_MASK)&~(I4s_TO_F16s_MAGIC_NUM);
if((immLut)&0x02)(h[2])|=~(top_i4s)&~(BOTTOM_MASK)& (I4s_TO_F16s_MAGIC_NUM);
if((immLut)&0x04)(h[2])|=~(top_i4s)& (BOTTOM_MASK)&~(I4s_TO_F16s_MAGIC_NUM);
if((immLut)&0x08)(h[2])|=~(top_i4s)& (BOTTOM_MASK)& (I4s_TO_F16s_MAGIC_NUM);
if((immLut)&0x10)(h[2])|= (top_i4s)&~(BOTTOM_MASK)&~(I4s_TO_F16s_MAGIC_NUM);
if((immLut)&0x20)(h[2])|= (top_i4s)&~(BOTTOM_MASK)& (I4s_TO_F16s_MAGIC_NUM);
if((immLut)&0x40)(h[2])|= (top_i4s)& (BOTTOM_MASK)&~(I4s_TO_F16s_MAGIC_NUM);
if((immLut)&0x80)(h[2])|= (top_i4s)& (BOTTOM_MASK)& (I4s_TO_F16s_MAGIC_NUM);
}
    // Extract elt_67 (top_i4s & 0x00f000f0) | 0x64006400

// >>>> PTX2CPP Success <<<<
{
(h[3])=0;
if((immLut)&0x01)(h[3])|=~(top_i4s)&~(TOP_MASK)&~(I4s_TO_F16s_MAGIC_NUM);
if((immLut)&0x02)(h[3])|=~(top_i4s)&~(TOP_MASK)& (I4s_TO_F16s_MAGIC_NUM);
if((immLut)&0x04)(h[3])|=~(top_i4s)& (TOP_MASK)&~(I4s_TO_F16s_MAGIC_NUM);
if((immLut)&0x08)(h[3])|=~(top_i4s)& (TOP_MASK)& (I4s_TO_F16s_MAGIC_NUM);
if((immLut)&0x10)(h[3])|= (top_i4s)&~(TOP_MASK)&~(I4s_TO_F16s_MAGIC_NUM);
if((immLut)&0x20)(h[3])|= (top_i4s)&~(TOP_MASK)& (I4s_TO_F16s_MAGIC_NUM);
if((immLut)&0x40)(h[3])|= (top_i4s)& (TOP_MASK)&~(I4s_TO_F16s_MAGIC_NUM);
if((immLut)&0x80)(h[3])|= (top_i4s)& (TOP_MASK)& (I4s_TO_F16s_MAGIC_NUM);
}


#endif //USE_MACA

  // I use inline PTX below because I am not sure if the compiler will emit
  // float2half instructions if I use the half2 ctor. In this case, I chose
  // performance reliability over code readability.

  // This is the half2 {1032, 1032} represented as an integer.
  // static constexpr uint32_t FP16_TOP_MAGIC_NUM = 0x64086408;
  // Haotian: subtract {1024, 1024} instead, we do not need to map to [-8, 7]
  static constexpr uint32_t FP16_TOP_MAGIC_NUM = 0x64006400;
  // This is the half2 {1 / 16, 1 / 16} represented as an integer.
  static constexpr uint32_t ONE_SIXTEENTH = 0x2c002c00;
  // This is the half2 {-72, -72} represented as an integer.
  // static constexpr uint32_t NEG_72 = 0xd480d480;
  // Haotian: Let's use {-64, -64}.
  static constexpr uint32_t NEG_64 = 0xd400d400;

  // Finally, we construct the output numbers.
  // Convert elt_01
#ifndef USE_MACA
  asm volatile("sub.f16x2 %0, %1, %2;\n"
               : "=r"(h[0])
               : "r"(h[0]), "r"(FP16_TOP_MAGIC_NUM));
  // Convert elt_23
  asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n"
               : "=r"(h[1])
               : "r"(h[1]), "r"(ONE_SIXTEENTH), "r"(NEG_64));
  // Convert elt_45
  asm volatile("sub.f16x2 %0, %1, %2;\n"
               : "=r"(h[2])
               : "r"(h[2]), "r"(FP16_TOP_MAGIC_NUM));
  // Convert elt_67
  asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n"
               : "=r"(h[3])
               : "r"(h[3]), "r"(ONE_SIXTEENTH), "r"(NEG_64));
#else
    // >>>> PTX2CPP Success <<<<
{
{
unsigned int __a=(h[0]);
unsigned int __b=(FP16_TOP_MAGIC_NUM);
VT __d=__hsub2(*(VT*)&__a,*(VT*)&__b);
(h[0])=*(unsigned int*)&__d;
}
}

    // Convert elt_23

// >>>> PTX2CPP Success <<<<
{
{
unsigned int __a=(h[1]);
unsigned int __b=(ONE_SIXTEENTH);
unsigned int __c=(NEG_64);
VT __d=__hfma2(*(VT*)&__a,*(VT*)&__b,*(VT*)&__c);
(h[1])=*(unsigned int*)&__d;
}
}
    // Convert elt_45
// >>>> PTX2CPP Success <<<<
{
{
unsigned int __a=(h[2]);
unsigned int __b=(FP16_TOP_MAGIC_NUM);
VT __d=__hsub2(*(VT*)&__a,*(VT*)&__b);
(h[2])=*(unsigned int*)&__d;
}
}

    // Convert elt_67

// >>>> PTX2CPP Success <<<<
{
{
unsigned int __a=(h[3]);
unsigned int __b=(ONE_SIXTEENTH);
unsigned int __c=(NEG_64);
VT __d=__hfma2(*(VT*)&__a,*(VT*)&__b,*(VT*)&__c);
(h[3])=*(unsigned int*)&__d;
}
}

#endif // USE_MACA
  return result;
#endif
  __builtin_unreachable();  // Suppress missing return statement warning
}

}  // namespace awq
}  // namespace vllm
