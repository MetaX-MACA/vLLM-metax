// 2025 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved. 
#include <stddef.h>
#include <torch/all.h>
#include "mctlass/half.h"
#include "mctlass/mctlass.h"
#include "mctlass/epilogue/thread/scale_type.h"
#include "scaled_mm_c2x.cuh"

void cutlass_moe_mm_sm75(torch::Tensor& out, torch::Tensor const& a, torch::Tensor const& b,
                        torch::Tensor const& moe_weight,
                        torch::Tensor const& token_ids, torch::Tensor const& expert_ids, 
                        torch::Tensor const& num_tokens_post_padded, int64_t num_valid_tokens, 
                        int64_t topk, bool mul_routed_weight)  {

  TORCH_CHECK(a.size(0) > 0, "No input A tensors provided.");
  TORCH_CHECK(b.size(0) > 0, "No input B tensors provided.");
  TORCH_CHECK(out.size(0) > 0, "No output tensors provided.");

  TORCH_CHECK(a.dtype() == torch::kBFloat16,
              "A tensors must be of type kBFloat16.");
  TORCH_CHECK(b.dtype() == torch::kBFloat16,
              "B tensors must be of type kBFloat16.");

  TORCH_CHECK(a.dtype() == torch::kBFloat16);
  TORCH_CHECK(b.dtype() == torch::kBFloat16);


  using ElementA = maca_bfloat16;
  using ElementB = maca_bfloat16;
  using ElementC = maca_bfloat16;
  using ElementCompute = float;
  using LayoutA = mctlass::layout::RowMajor;
  using LayoutB = mctlass::layout::ColumnMajor;
  using LayoutC = mctlass::layout::RowMajor;

  using ArchTag = mctlass::arch::Sm80;

  using mctlassMoeGemmBf16Op = mctlassMoeGemm<
      ElementA, LayoutA,
      ElementB, LayoutB,
      ElementC, LayoutC,
      ElementCompute,
      ArchTag
  >;

  auto a_ptr = static_cast<ElementA const*>(a.data_ptr());
  auto b_ptr = static_cast<ElementB const*>(b.data_ptr());
  auto c_ptr = static_cast<ElementC*>(out.data_ptr());
  auto moe_weight_ptr = moe_weight.data_ptr<float>();
  auto token_ids_ptr = token_ids.data_ptr<int32_t>();
  auto expert_ids_ptr = expert_ids.data_ptr<int32_t>();
  auto num_tokens_post_padded_ptr = num_tokens_post_padded.data_ptr<int32_t>();

  int64_t const m = a.size(0);
  int64_t const n = b.size(1);
  int64_t const k = a.size(1);
  int64_t const num_experts = b.size(0);
  int32_t const num_tokens_post_padded_size = token_ids.size(0);

  mctlassMoeGemmBf16Op mctlass_op;
  mctlass::gemm::BatchedGemmCoord problem_size(num_valid_tokens, n, k, num_experts);

  mctlass::gemm::GemmCoord kernel_size;
  mctlass::Status status = mctlass_op.gemm_kernel_mnk(problem_size, kernel_size, 0, 0, mul_routed_weight);
  if (status != mctlass::Status::kSuccess) {
      printf("Error: Not find supported kernel!\n");
      return;
  }

  typename mctlassMoeGemmBf16Op::Arguments arguments{
      mctlass::gemm::GemmUniversalMode::kGemm,
      problem_size,
      {moe_weight_ptr},
      a_ptr,
      b_ptr,
      c_ptr,
      {token_ids_ptr,
      expert_ids_ptr,
      num_tokens_post_padded_ptr,
      num_tokens_post_padded_size,
      static_cast<int32_t>(topk),
      mul_routed_weight}
  };

  auto stream = at::cuda::getCurrentCUDAStream(a.get_device());
  mctlass::Status is_success = mctlass_op(arguments, NULL, stream);
  assert(mctlass::Status::kSuccess == is_success);
}

int64_t cutlass_moe_mm_gemm_kernel_m_w8a8_sm75(int64_t num_valid_tokens, int64_t N, int64_t K, int64_t group) {
    using ElementA = int8_t;
    using ElementB = int8_t;
    using ElementC = maca_bfloat16;
    using ElementCompute = float;
    using LayoutA = mctlass::layout::RowMajor;
    using LayoutB = mctlass::layout::ColumnMajor;
    using LayoutC = mctlass::layout::RowMajor;
    using ArchTag = mctlass::arch::Sm80;

    using mctlassMoeGemmOp = mctlassMoeGemm<
    ElementA,
    LayoutA,
    ElementB,
    LayoutB,
    ElementC,
    LayoutC,
    ElementCompute,
    ArchTag
    >;

    mctlassMoeGemmOp mctlass_op;
    mctlass::gemm::BatchedGemmCoord problem_size(num_valid_tokens, N, K, group);
    mctlass::gemm::GemmCoord kernel_size;
    mctlass::Status status = mctlass_op.gemm_kernel_mnk(problem_size, kernel_size);
    if (status != mctlass::Status::kSuccess) {
        return -1;
    }

    return static_cast<int64_t>(kernel_size.m());
}

void cutlass_moe_mm_w8a8_sm75(torch::Tensor const& a, torch::Tensor const& b, torch::Tensor& c,
                              torch::Tensor const& a_scales, torch::Tensor const& b_scales, torch::Tensor const& moe_weight,
                              torch::Tensor const& token_ids, torch::Tensor const& expert_ids, 
                              torch::Tensor const& num_tokens_post_padded, 
                              int64_t N, int64_t K, int64_t EM, int64_t num_valid_tokens, int64_t topk, bool mul_routed_weight) {
    using ElementA = int8_t;
    using ElementB = int8_t;
    using ElementC = maca_bfloat16;
    using ElementCompute = float;
    using LayoutA = mctlass::layout::RowMajor;
    using LayoutB = mctlass::layout::ColumnMajor;
    using LayoutC = mctlass::layout::RowMajor;
    using ArchTag = mctlass::arch::Sm80;

    using mctlassMoeGemmOp = mctlassMoeGemm<
    ElementA,
    LayoutA,
    ElementB,
    LayoutB,
    ElementC,
    LayoutC,
    ElementCompute,
    ArchTag
    >;

    int64_t num_experts = b.size(0);

    auto a_ptr = static_cast<ElementA const*>(a.data_ptr());
    auto b_ptr = static_cast<ElementB const*>(b.data_ptr());
    auto c_ptr = static_cast<ElementC*>(c.data_ptr());
    auto scale_a_ptr = a_scales.data_ptr<float>();
    auto scale_b_ptr = b_scales.data_ptr<float>();
    auto moe_weight_ptr = moe_weight.data_ptr<float>();
    auto token_ids_ptr = token_ids.data_ptr<int32_t>();
    auto expert_ids_ptr = expert_ids.data_ptr<int32_t>();
    auto num_tokens_post_padded_ptr = num_tokens_post_padded.data_ptr<int32_t>();

    mctlassMoeGemmOp mctlass_op;
    mctlass::gemm::BatchedGemmCoord problem_size(num_valid_tokens, N, K, num_experts);

    auto stream = at::cuda::getCurrentCUDAStream(a.get_device());
    // mctlass
    typename mctlassMoeGemmOp::Arguments arguments{
        mctlass::gemm::GemmUniversalMode::kGemm,
        problem_size,
        {scale_a_ptr, scale_b_ptr, moe_weight_ptr},
        a_ptr,
        b_ptr,
        c_ptr,
        {token_ids_ptr,
        expert_ids_ptr,
        num_tokens_post_padded_ptr,
        static_cast<int32_t>(EM),
        static_cast<int32_t>(topk),
        mul_routed_weight}};
    
    mctlass::Status is_success = mctlass_op(arguments, NULL, stream);
    assert(mctlass::Status::kSuccess == is_success);
}

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
}
