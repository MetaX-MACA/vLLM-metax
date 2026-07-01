#include <stddef.h>

#include <cstdint>
#include <optional>

#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include "libtorch_stable/torch_utils.h"
#include "mctlass/epilogue/thread/scale_type.h"
#include "mctlass/mctlass.h"
#include "scaled_mm_c2x.cuh"

void cutlass_scaled_mm_sm75(torch::stable::Tensor& out,
                            torch::stable::Tensor const& a,
                            torch::stable::Tensor const& b,
                            torch::stable::Tensor const& a_scales,
                            torch::stable::Tensor const& b_scales,
                            std::optional<torch::stable::Tensor> const& bias) {
  int32_t m = a.size(0);
  int32_t n = b.size(1);
  int32_t k = a.size(1);
  int32_t batch_count = 1;
  if (a.dim() == 3 && b.dim() == 3) {
    // a.size = [batch_size, M, K], b.size = [batch_size, K, N]
    m = a.size(1);
    n = b.size(2);
    k = a.size(2);
    batch_count = a.size(0);
  }

  using ArchTag = mctlass::arch::Sm80;
  using ElementA = int8_t;
  using ElementB = int8_t;
  using ElementC = mctlass::half_t;
  using ElementCompute = float;
  using LayoutA = mctlass::layout::RowMajor;
  using LayoutB = mctlass::layout::ColumnMajor;
  using LayoutC = mctlass::layout::RowMajor;

  const auto stream = get_current_cuda_stream(a.get_device_index());
  const auto a_ptr = static_cast<ElementA const*>(a.const_data_ptr());
  const auto b_ptr = static_cast<ElementB const*>(b.const_data_ptr());
  const auto scale_a = a_scales.const_data_ptr<float>();
  const auto scale_b = b_scales.const_data_ptr<float>();

  if (out.scalar_type() == torch::headeronly::ScalarType::BFloat16) {
    auto c_ptr = static_cast<maca_bfloat16*>(out.mutable_data_ptr());
    if (bias) {
      constexpr auto scale_type =
          mctlass::epilogue::thread::ScaleType::ScaleAvBvBias;
      using MctlassGemmScaleOp =
          mctlassGemmScale<ElementA, LayoutA, ElementB, LayoutB, maca_bfloat16,
                           LayoutC, ElementCompute, ArchTag, scale_type>;
      auto bias_t = static_cast<maca_bfloat16*>(bias->mutable_data_ptr());
      MctlassGemmScaleOp mctlass_op;
      mctlass::gemm::GemmCoord problem_size(m, n, k);
      typename MctlassGemmScaleOp::Arguments arguments{
          mctlass::gemm::GemmUniversalMode::kGemm,
          problem_size,
          batch_count,
          {scale_a, scale_b, bias_t},
          a_ptr,
          b_ptr,
          c_ptr,
          c_ptr,
          m * k,
          n * k,
          m * n,
          m * n,
          k,
          n,
          n,
          n};
      mctlass_op(arguments, nullptr, stream);
    } else {
      constexpr auto scale_type =
          mctlass::epilogue::thread::ScaleType::ScaleAvBv;
      using MctlassGemmScaleOp =
          mctlassGemmScale<ElementA, LayoutA, ElementB, LayoutB, maca_bfloat16,
                           LayoutC, ElementCompute, ArchTag, scale_type>;
      MctlassGemmScaleOp mctlass_op;
      mctlass::gemm::GemmCoord problem_size(m, n, k);
      typename MctlassGemmScaleOp::Arguments arguments{
          mctlass::gemm::GemmUniversalMode::kGemm,
          problem_size,
          batch_count,
          {scale_a, scale_b, nullptr},
          a_ptr,
          b_ptr,
          c_ptr,
          c_ptr,
          m * k,
          n * k,
          m * n,
          m * n,
          k,
          n,
          n,
          n};
      mctlass_op(arguments, nullptr, stream);
    }
  } else {
    auto c_ptr = static_cast<ElementC*>(out.mutable_data_ptr());
    if (bias) {
      constexpr auto scale_type =
          mctlass::epilogue::thread::ScaleType::ScaleAvBvBias;
      using MctlassGemmScaleOp =
          mctlassGemmScale<ElementA, LayoutA, ElementB, LayoutB, ElementC,
                           LayoutC, ElementCompute, ArchTag, scale_type>;
      auto bias_t = static_cast<ElementC*>(bias->mutable_data_ptr());
      MctlassGemmScaleOp mctlass_op;
      mctlass::gemm::GemmCoord problem_size(m, n, k);
      typename MctlassGemmScaleOp::Arguments arguments{
          mctlass::gemm::GemmUniversalMode::kGemm,
          problem_size,
          batch_count,
          {scale_a, scale_b, bias_t},
          a_ptr,
          b_ptr,
          c_ptr,
          c_ptr,
          m * k,
          n * k,
          m * n,
          m * n,
          k,
          n,
          n,
          n};
      mctlass_op(arguments, nullptr, stream);
    } else {
      constexpr auto scale_type =
          mctlass::epilogue::thread::ScaleType::ScaleAvBv;
      using MctlassGemmScaleOp =
          mctlassGemmScale<ElementA, LayoutA, ElementB, LayoutB, ElementC,
                           LayoutC, ElementCompute, ArchTag, scale_type>;
      MctlassGemmScaleOp mctlass_op;
      mctlass::gemm::GemmCoord problem_size(m, n, k);
      typename MctlassGemmScaleOp::Arguments arguments{
          mctlass::gemm::GemmUniversalMode::kGemm,
          problem_size,
          batch_count,
          {scale_a, scale_b, nullptr},
          a_ptr,
          b_ptr,
          c_ptr,
          c_ptr,
          m * k,
          n * k,
          m * n,
          m * n,
          k,
          n,
          n,
          n};
      mctlass_op(arguments, nullptr, stream);
    }
  }
}

void cutlass_scaled_mm_azp_sm75(
    torch::stable::Tensor& out, torch::stable::Tensor const& a,
    torch::stable::Tensor const& b, torch::stable::Tensor const& a_scales,
    torch::stable::Tensor const& b_scales, torch::stable::Tensor const& azp_adj,
    std::optional<torch::stable::Tensor> const& azp,
    std::optional<torch::stable::Tensor> const& bias) {
  int32_t m = a.size(0);
  int32_t n = b.size(1);
  int32_t k = a.size(1);
  int32_t batchsize = 1;
  if (a.dim() == 3 && b.dim() == 3) {
    // a.size = [batch_size, M, K], b.size = [batch_size, K, N]
    m = a.size(1);
    n = b.size(2);
    k = a.size(2);
    batchsize = a.size(0);
  }

  using ArchTag = mctlass::arch::Sm80;
  using ElementA = int8_t;
  using ElementB = int8_t;
  using ElementCompute = float;
  using ElementAccumulator = int32_t;
  using LayoutA = mctlass::layout::RowMajor;
  using LayoutB = mctlass::layout::ColumnMajor;
  using LayoutC = mctlass::layout::RowMajor;

  const auto stream = get_current_cuda_stream(a.get_device_index());
  const auto a_ptr = static_cast<ElementA const*>(a.const_data_ptr());
  const auto b_ptr = static_cast<ElementB const*>(b.const_data_ptr());
  const auto scale_a = a_scales.const_data_ptr<float>();
  const auto scale_b = b_scales.const_data_ptr<float>();
  auto* azp_ptr =
      azp ? static_cast<ElementAccumulator*>(azp->mutable_data_ptr()) : nullptr;
  auto azp_adj_ptr =
      static_cast<ElementAccumulator*>(azp_adj.mutable_data_ptr());

  STD_TORCH_CHECK(bias.has_value(),
                  "cutlass_scaled_mm_azp_sm75 requires a bias tensor. Pass a "
                  "zero bias tensor when the public op receives None.");

  if (out.scalar_type() == torch::headeronly::ScalarType::BFloat16) {
    using ElementC = maca_bfloat16;
    using ElementOutput = ElementC;

    auto c_ptr = static_cast<ElementC*>(out.mutable_data_ptr());
    auto bias_t = static_cast<ElementOutput*>(bias->mutable_data_ptr());

    if (azp) {
      constexpr auto scale_type =
          mctlass::epilogue::thread::ScaleType::ScaleAvBvBiasAzpPerTorken;
      using MctlassGemmScaleOp =
          mctlassGemmScale<ElementA, LayoutA, ElementB, LayoutB, ElementC,
                           LayoutC, ElementCompute, ArchTag, scale_type>;
      MctlassGemmScaleOp mctlass_op;
      mctlass::gemm::GemmCoord problem_size(m, n, k);
      typename MctlassGemmScaleOp::Arguments arguments{
          mctlass::gemm::GemmUniversalMode::kGemm,
          problem_size,
          batchsize,
          {scale_a, scale_b, bias_t, azp_adj_ptr, azp_ptr},
          a_ptr,
          b_ptr,
          c_ptr,
          c_ptr,
          m * k,
          n * k,
          m * n,
          m * n,
          k,
          n,
          n,
          n};
      mctlass_op(arguments, nullptr, stream);
    } else {
      constexpr auto scale_type =
          mctlass::epilogue::thread::ScaleType::ScaleAsBvBiasAzp;
      using MctlassGemmScaleOp =
          mctlassGemmScale<ElementA, LayoutA, ElementB, LayoutB, ElementC,
                           LayoutC, ElementCompute, ArchTag, scale_type>;
      MctlassGemmScaleOp mctlass_op;
      mctlass::gemm::GemmCoord problem_size(m, n, k);
      typename MctlassGemmScaleOp::Arguments arguments{
          mctlass::gemm::GemmUniversalMode::kGemm,
          problem_size,
          batchsize,
          {scale_a, scale_b, bias_t, azp_adj_ptr, azp_ptr},
          a_ptr,
          b_ptr,
          c_ptr,
          c_ptr,
          m * k,
          n * k,
          m * n,
          m * n,
          k,
          n,
          n,
          n};
      mctlass_op(arguments, nullptr, stream);
    }
  } else {
    using ElementC = mctlass::half_t;
    using ElementOutput = ElementC;

    auto c_ptr = static_cast<ElementC*>(out.mutable_data_ptr());
    auto bias_t = static_cast<ElementOutput*>(bias->mutable_data_ptr());

    if (azp) {
      constexpr auto scale_type =
          mctlass::epilogue::thread::ScaleType::ScaleAvBvBiasAzpPerTorken;
      using MctlassGemmScaleOp =
          mctlassGemmScale<ElementA, LayoutA, ElementB, LayoutB, ElementC,
                           LayoutC, ElementCompute, ArchTag, scale_type>;
      MctlassGemmScaleOp mctlass_op;
      mctlass::gemm::GemmCoord problem_size(m, n, k);
      typename MctlassGemmScaleOp::Arguments arguments{
          mctlass::gemm::GemmUniversalMode::kGemm,
          problem_size,
          batchsize,
          {scale_a, scale_b, bias_t, azp_adj_ptr, azp_ptr},
          a_ptr,
          b_ptr,
          c_ptr,
          c_ptr,
          m * k,
          n * k,
          m * n,
          m * n,
          k,
          n,
          n,
          n};
      mctlass_op(arguments, nullptr, stream);
    } else {
      constexpr auto scale_type =
          mctlass::epilogue::thread::ScaleType::ScaleAsBvBiasAzp;
      using MctlassGemmScaleOp =
          mctlassGemmScale<ElementA, LayoutA, ElementB, LayoutB, ElementC,
                           LayoutC, ElementCompute, ArchTag, scale_type>;
      MctlassGemmScaleOp mctlass_op;
      mctlass::gemm::GemmCoord problem_size(m, n, k);
      typename MctlassGemmScaleOp::Arguments arguments{
          mctlass::gemm::GemmUniversalMode::kGemm,
          problem_size,
          batchsize,
          {scale_a, scale_b, bias_t, azp_adj_ptr, azp_ptr},
          a_ptr,
          b_ptr,
          c_ptr,
          c_ptr,
          m * k,
          n * k,
          m * n,
          m * n,
          k,
          n,
          n,
          n};
      mctlass_op(arguments, nullptr, stream);
    }
  }
}
