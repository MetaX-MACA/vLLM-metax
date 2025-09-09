#include <stddef.h>
#include <torch/all.h>
#include "mctlass/mctlass.h"
#include "mctlass/epilogue/thread/scale_type.h"
#include "scaled_mm_c2x.cuh"

void cutlass_scaled_mm_sm75(torch::Tensor& out, torch::Tensor const& a,
                            torch::Tensor const& b,
                            torch::Tensor const& a_scales,
                            torch::Tensor const& b_scales,
                            std::optional<torch::Tensor> const& bias) {
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
  // using LayoutB = mctlass::layout::RowMajor;
  using LayoutB = mctlass::layout::ColumnMajor;
  using LayoutC = mctlass::layout::RowMajor;

  if (out.dtype() == torch::kBFloat16) {
    auto a_ptr = static_cast<ElementA const*>(a.data_ptr());
    auto b_ptr = static_cast<ElementB const*>(b.data_ptr());
    auto c_ptr = static_cast<maca_bfloat16*>(out.data_ptr());
    auto scale_a = a_scales.data_ptr<float>();
    auto scale_b = b_scales.data_ptr<float>();
    auto stream = at::cuda::getCurrentCUDAStream(a.get_device());
    if (bias) {
      mctlass::epilogue::thread::ScaleType::ScaleBiasKind const scale_type =
          mctlass::epilogue::thread::ScaleType::ScaleAvBvBias;
      using mctlassGemmScaleOp =
          mctlassGemmScale<ElementA, LayoutA, ElementB, LayoutB, maca_bfloat16,
                           LayoutC, ElementCompute, ArchTag, scale_type>;
      maca_bfloat16* bias_t;
      bias_t = static_cast<maca_bfloat16*>(bias.value().data_ptr());
      mctlassGemmScaleOp mctlass_op;
      mctlass::gemm::GemmCoord problem_size(m, n, k);
      typename mctlassGemmScaleOp::Arguments arguments{
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
      mctlass_op(arguments, NULL, stream);
    } else {
      mctlass::epilogue::thread::ScaleType::ScaleBiasKind const scale_type =
          mctlass::epilogue::thread::ScaleType::ScaleAvBv;
      using mctlassGemmScaleOp =
          mctlassGemmScale<ElementA, LayoutA, ElementB, LayoutB, maca_bfloat16,
                           LayoutC, ElementCompute, ArchTag, scale_type>;
      mctlassGemmScaleOp mctlass_op;
      mctlass::gemm::GemmCoord problem_size(m, n, k);
      typename mctlassGemmScaleOp::Arguments arguments{
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
      mctlass_op(arguments, NULL, stream);
    }
  } else {
    auto a_ptr = static_cast<ElementA const*>(a.data_ptr());
    auto b_ptr = static_cast<ElementB const*>(b.data_ptr());
    auto c_ptr = static_cast<ElementC*>(out.data_ptr());
    auto scale_a = a_scales.data_ptr<float>();
    auto scale_b = b_scales.data_ptr<float>();
    auto stream = at::cuda::getCurrentCUDAStream(a.get_device());
    if (bias) {
      mctlass::epilogue::thread::ScaleType::ScaleBiasKind const scale_type =
          mctlass::epilogue::thread::ScaleType::ScaleAvBvBias;
      using mctlassGemmScaleOp =
          mctlassGemmScale<ElementA, LayoutA, ElementB, LayoutB, ElementC,
                           LayoutC, ElementCompute, ArchTag, scale_type>;
      ElementC* bias_t;
      bias_t = static_cast<ElementC*>(bias.value().data_ptr());
      mctlassGemmScaleOp mctlass_op;
      mctlass::gemm::GemmCoord problem_size(m, n, k);
      typename mctlassGemmScaleOp::Arguments arguments{
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
      mctlass_op(arguments, NULL, stream);
    } else {
      mctlass::epilogue::thread::ScaleType::ScaleBiasKind const scale_type =
          mctlass::epilogue::thread::ScaleType::ScaleAvBv;
      using mctlassGemmScaleOp =
          mctlassGemmScale<ElementA, LayoutA, ElementB, LayoutB, ElementC,
                           LayoutC, ElementCompute, ArchTag, scale_type>;
      mctlassGemmScaleOp mctlass_op;
      mctlass::gemm::GemmCoord problem_size(m, n, k);
      typename mctlassGemmScaleOp::Arguments arguments{
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
      mctlass_op(arguments, NULL, stream);
    }
  }
}

void cutlass_scaled_mm_azp_sm75(torch::Tensor& out, torch::Tensor const& a,
                                torch::Tensor const& b,
                                torch::Tensor const& a_scales,
                                torch::Tensor const& b_scales,
                                torch::Tensor const& azp_adj,
                                std::optional<torch::Tensor> const& azp,
                                std::optional<torch::Tensor> const& bias) {
#if (MACA_VERSION_MAJOR * 100 + MACA_VERSION_MINOR) >= \
    231  // MACA version >= 2.31.0.x
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

  auto stream = at::cuda::getCurrentCUDAStream(a.get_device());

  if (out.dtype() == torch::kBFloat16) {
    using ElementC = maca_bfloat16;
    using ElementOutput = ElementC;

    auto a_ptr = static_cast<ElementA const*>(a.data_ptr());
    auto b_ptr = static_cast<ElementB const*>(b.data_ptr());
    auto c_ptr = static_cast<ElementC*>(out.data_ptr());
    auto scale_a = a_scales.data_ptr<float>();
    auto scale_b = b_scales.data_ptr<float>();

    ElementAccumulator* azp_ptr = NULL;
    auto azp_adj_ptr = azp_adj.data_ptr<ElementAccumulator>();
    ElementOutput* bias_t =
        static_cast<ElementOutput*>(bias.value().data_ptr());

    if (azp) {
      azp_ptr = static_cast<ElementAccumulator*>(azp.value().data_ptr());
      mctlass::epilogue::thread::ScaleType::ScaleBiasKind const scale_type =
          mctlass::epilogue::thread::ScaleType::ScaleAvBvBiasAzpPerTorken;
      using mctlassGemmScaleOp =
          mctlassGemmScale<ElementA, LayoutA, ElementB, LayoutB, ElementC,
                           LayoutC, ElementCompute, ArchTag, scale_type>;
      mctlassGemmScaleOp mctlass_op;
      mctlass::gemm::GemmCoord problem_size(m, n, k);
      typename mctlassGemmScaleOp::Arguments arguments{
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
      mctlass_op(arguments, NULL, stream);
    } else {
      mctlass::epilogue::thread::ScaleType::ScaleBiasKind const scale_type =
          mctlass::epilogue::thread::ScaleType::ScaleAsBvBiasAzp;
      using mctlassGemmScaleOp =
          mctlassGemmScale<ElementA, LayoutA, ElementB, LayoutB, ElementC,
                           LayoutC, ElementCompute, ArchTag, scale_type>;
      mctlassGemmScaleOp mctlass_op;
      mctlass::gemm::GemmCoord problem_size(m, n, k);
      typename mctlassGemmScaleOp::Arguments arguments{
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
      mctlass_op(arguments, NULL, stream);
    }
  } else {
    using ElementC = mctlass::half_t;
    using ElementOutput = ElementC;

    auto a_ptr = static_cast<ElementA const*>(a.data_ptr());
    auto b_ptr = static_cast<ElementB const*>(b.data_ptr());
    auto c_ptr = static_cast<ElementC*>(out.data_ptr());
    auto scale_a = a_scales.data_ptr<float>();
    auto scale_b = b_scales.data_ptr<float>();

    ElementAccumulator* azp_ptr = nullptr;
    auto azp_adj_ptr = azp_adj.data_ptr<ElementAccumulator>();
    ElementOutput* bias_t =
        static_cast<ElementOutput*>(bias.value().data_ptr());

    if (azp) {
      azp_ptr = static_cast<ElementAccumulator*>(azp.value().data_ptr());

      mctlass::epilogue::thread::ScaleType::ScaleBiasKind const scale_type =
          mctlass::epilogue::thread::ScaleType::ScaleAvBvBiasAzpPerTorken;
      using mctlassGemmScaleOp =
          mctlassGemmScale<ElementA, LayoutA, ElementB, LayoutB, ElementC,
                           LayoutC, ElementCompute, ArchTag, scale_type>;
      mctlassGemmScaleOp mctlass_op;
      mctlass::gemm::GemmCoord problem_size(m, n, k);
      typename mctlassGemmScaleOp::Arguments arguments{
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
      mctlass_op(arguments, NULL, stream);
    } else {
      mctlass::epilogue::thread::ScaleType::ScaleBiasKind const scale_type =
          mctlass::epilogue::thread::ScaleType::ScaleAsBvBiasAzp;
      using mctlassGemmScaleOp =
          mctlassGemmScale<ElementA, LayoutA, ElementB, LayoutB, ElementC,
                           LayoutC, ElementCompute, ArchTag, scale_type>;
      mctlassGemmScaleOp mctlass_op;
      mctlass::gemm::GemmCoord problem_size(m, n, k);
      typename mctlassGemmScaleOp::Arguments arguments{
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
      mctlass_op(arguments, NULL, stream);
    }
  }
#endif  // USE_MACA
}
