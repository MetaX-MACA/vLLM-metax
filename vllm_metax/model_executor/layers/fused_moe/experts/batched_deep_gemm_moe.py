# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceDelegate,
)
from vllm.model_executor.layers.fused_moe.utils import (
    _resize_cache,
    moe_kernel_quantize_input,
    swiglu_limit_func,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    get_fp8_min_max,
    kFp8Dynamic128Sym,
    kFp8Static128BlockSym,
    kInt8DynamicTokenSym,
    kInt8StaticChannelSym,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm_metax.utils.deep_gemm import (
    DeepGemmQuantScaleFMT,
    fp8_m_grouped_gemm_nt_masked,
    get_mk_alignment_for_contiguous_layout,
    int8_m_grouped_gemm_nt_masked,
    is_deep_gemm_e8m0_used,
    is_deep_gemm_supported,
)
from vllm.utils.math_utils import cdiv, round_up

logger = init_logger(__name__)


def _normalize_int8_scale_2d(
    scale: torch.Tensor | None,
    num_experts: int,
    rows: int,
    name: str,
) -> torch.Tensor:
    """Normalize per-token/per-channel INT8 scales for DeepGEMM."""
    assert scale is not None, f"{name} must be provided"
    assert scale.dtype == torch.float32, (
        f"{name} must have dtype torch.float32, got {scale.dtype}"
    )
    assert scale.numel() == num_experts * rows, (
        f"Invalid {name} shape {scale.shape}; expected {num_experts * rows} values"
    )
    valid_shapes = {
        (num_experts, rows),
        (num_experts, rows, 1),
        (num_experts * rows, 1),
    }
    assert tuple(scale.shape) in valid_shapes, (
        f"Invalid {name} layout {scale.shape}; expected one of {valid_shapes}"
    )
    return scale.contiguous().reshape(num_experts, rows)


def scales_shape_stride_dtype(
    E: int, T: int, G: int, quant_scale_fmt: DeepGemmQuantScaleFMT
) -> tuple[tuple[int, ...], tuple[int, ...], torch.dtype]:
    shape = (E, T, G)
    strides = (T * G, 1, T)
    if quant_scale_fmt in [
        DeepGemmQuantScaleFMT.FLOAT32,
        DeepGemmQuantScaleFMT.FLOAT32_CEIL_UE8M0,
    ]:
        return shape, strides, torch.float32

    assert quant_scale_fmt == DeepGemmQuantScaleFMT.UE8M0
    shape = (E, T, cdiv(G, 4))
    strides = (T * cdiv(G, 4), 1, T)
    return shape, strides, torch.int32


@triton.jit
def _silu_mul_fp8_quant_deep_gemm(
    # Pointers ------------------------------------------------------------
    input_ptr,  # 16-bit activations (E, T, 2*H)
    y_q_ptr,  # fp8 quantized activations (E, T, H)
    y_s_ptr,  # 16-bit scales (E, T, G)
    counts_ptr,  # int32 num tokens per expert (E)
    # Sizes ---------------------------------------------------------------
    H: tl.constexpr,  # hidden dimension (per output)
    GROUP_SIZE: tl.constexpr,  # elements per group (usually 128)
    # Strides for input (elements) ---------------------------------------
    stride_i_e,
    stride_i_t,
    stride_i_h,
    # Strides for y_q (elements) -----------------------------------------
    stride_yq_e,
    stride_yq_t,
    stride_yq_h,
    # Strides for y_s (elements) -----------------------------------------
    stride_ys_e,
    stride_ys_t,
    stride_ys_g,
    # Stride for counts (elements)
    stride_counts_e,
    # Numeric params ------------------------------------------------------
    eps: tl.constexpr,
    fp8_min: tl.constexpr,
    fp8_max: tl.constexpr,
    ceil_ue8m0: tl.constexpr,
    # Meta ---------------------------------------------------------------
    BLOCK: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    G = H // GROUP_SIZE

    # map program id -> (e, g)
    pid = tl.program_id(0)
    e = pid // G
    g = pid % G

    e = e.to(tl.int64)
    g = g.to(tl.int64)

    # number of valid tokens for this expert
    n_tokens = tl.load(counts_ptr + e * stride_counts_e).to(tl.int64)

    cols = tl.arange(0, BLOCK).to(tl.int64)
    mask = cols < BLOCK

    base_input_offset = e * stride_i_e + g * GROUP_SIZE * stride_i_h
    base_gate_offset = base_input_offset + cols * stride_i_h
    base_up_offset = base_input_offset + H * stride_i_h + cols * stride_i_h
    base_yq_offset = e * stride_yq_e + g * GROUP_SIZE * stride_yq_h + cols * stride_yq_h
    base_ys_offset = e * stride_ys_e + g * stride_ys_g

    for t in tl.range(0, n_tokens, num_stages=NUM_STAGES):
        gate = tl.load(
            input_ptr + base_gate_offset + t * stride_i_t, mask=mask, other=0.0
        ).to(tl.float32)
        up = tl.load(input_ptr + base_up_offset + t * stride_i_t, mask=mask, other=0.0)

        gate = gate * (1.0 / (1.0 + tl.exp(-gate)))
        y = gate * up

        # Use multiply-by-reciprocal to match PyTorch's tensor/scalar
        # division precision (Triton GPU fast-division for constexpr
        # divisors can introduce 1-ULP error).
        y_s = tl.maximum(tl.max(tl.abs(y)), eps) * (1.0 / fp8_max)
        if ceil_ue8m0:
            y_s = tl.exp2(tl.ceil(tl.log2(y_s)))

        y_q = tl.clamp(y / y_s, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

        tl.store(y_q_ptr + base_yq_offset + t * stride_yq_t, y_q, mask=mask)
        tl.store(y_s_ptr + base_ys_offset + t * stride_ys_t, y_s)


def persistent_masked_m_silu_mul_quant(
    y: torch.Tensor,  # (E, T, 2*H)
    tokens_per_expert: torch.Tensor,  # (E,) number of valid tokens per expert
    num_parallel_tokens=16,
    group_size: int = 128,
    quant_scale_fmt: DeepGemmQuantScaleFMT = DeepGemmQuantScaleFMT.FLOAT32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize silu(y[..., :H]) * y[..., H:] to FP8 with group per-token scales
    y has shape (E, T, 2*H). The first half of the last dimension is
    silu-activated, multiplied by the second half, then quantized into FP8.
    We launch a fixed grid of threads to accommodate CUDA graphs. Let `P2`
    be a parallelization factor for persistent_masked_m_silu_mul_quant over the
    hidden dimension.

    Let `expert_offsets = [0] + [num_tokens.cumsum()]` and
    `total_tokens = expert_offsets[-1]`.
    persistent_masked_m_silu_mul_quant launches `total_tokens x P2` number of
    thread blocks. Each thread block contains `NUM_WARPS` warps.

    Every thread block needs to find it's corresponding expert by warp-parallel scanning
    over the `expert_offsets` array.

    The i-th warp in the first thread block processes
    `[i * warp_chunk_size, (i + 1) * warp_chunk_size]` groups
    sequentially, where `warp_chunk_size = ((H / GROUP_SIZE) / P2) / NUM_WARPS`,
    pipelining loads and computes.

    The shared memory layout for 4 warps with a 2-stage pipeline for SiLU V2
    can is visualized like so:

                         stage0                              stage1
    ┌─────┬───┬─────┬───┬─────┬───┬─────┬───┬─────┬───┬─────┬───┬─────┬───┬─────┬───┐
    │gate0│up0│gate1│up1│gate2│up2│gate3│up3│gate0│up0│gate1│up1│gate2│up2│gate3│up3│
    └─────┴───┴─────┴───┴─────┴───┴─────┴───┴─────┴───┴─────┴───┴─────┴───┴─────┴───┘

    with the main difference between V1 and V2 being the global load
    stride between warps, and between half-warps. Regarding the latter stride,
    we assign the first half warp of every warp for `gate` loads and the second
    half-warp to `up` loads.

    Returns `(y_q, y_s)` where
    * `y_q`: FP8 tensor, shape (E, T, H), same layout as y[..., :H]
    * `y_s` depends on quant_scale_fmt,
      - quant_scale_fmt == FLOAT32,
         `y_s`: FP32 tensor, shape (E, T, H // group_size), strides (T*G, 1, T)
      - quant_scale_fmt == E8M0,
         `y_s`: Int32 tensor, shape (E, T, H // group_size // 4), strides (T*G, 1, T)
      - quant_scale_fmt == E8M0_FLOAT32_SPARSE
         `y_s`: FP32 tensor, shape (E, T, H // group_size), strides (T*G, 1, T)
    Let NUM_WARPS be the number of warps in a single thread block and
    `GROUP_SIZE = 128` be the size of the quantization group.
    """
    assert y.ndim == 3, "y must be (E, T, 2*H)"
    E, T, H2 = y.shape
    assert H2 % 2 == 0, "last dim of y must be even (2*H)"
    H = H2 // 2
    G = (H + group_size - 1) // group_size
    assert H % 8 == 0, "H must be divisible by 8"
    assert group_size == 128, "H must be divisible by 8"
    assert tokens_per_expert.ndim == 1 and tokens_per_expert.shape[0] == E

    tokens_per_expert = tokens_per_expert.to(device=y.device, dtype=torch.int32)

    fp8_dtype = current_platform.fp8_dtype()
    y_q = torch.empty((E, T, H), dtype=fp8_dtype, device=y.device)

    ys_shape, ys_strides, ys_dtype = scales_shape_stride_dtype(E, T, G, quant_scale_fmt)
    y_s = torch.empty_strided(
        ys_shape,
        ys_strides,
        dtype=ys_dtype,
        device=y.device,
    )

    ceil_ue8m0 = quant_scale_fmt in [
        DeepGemmQuantScaleFMT.FLOAT32_CEIL_UE8M0,
        DeepGemmQuantScaleFMT.UE8M0,
    ]

    device_capability = current_platform.get_device_capability(device_id=y.device.index)
    assert device_capability is not None
    cuda_arch = device_capability.to_int()

    if current_platform.is_cuda() and cuda_arch >= 80:
        torch.ops._C.persistent_masked_m_silu_mul_quant(
            y, tokens_per_expert, y_q, y_s, ceil_ue8m0
        )
    else:
        # Triton fallback for ROCm -- the C++ kernel is guarded by
        # #ifndef USE_ROCM in activation_kernels.cu.
        # https://github.com/ROCm/aiter/issues/2420
        stride_cnt_e = tokens_per_expert.stride()[0]

        # Static grid over experts and H-groups.
        # A loop inside the kernel handles the token dim
        grid = (E * G,)
        # strides (elements)
        stride_i_e, stride_i_t, stride_i_h = y.stride()
        stride_yq_e, stride_yq_t, stride_yq_h = y_q.stride()

        fp8_min, fp8_max = get_fp8_min_max()
        eps: float = 1e-10
        assert y_s.dtype == torch.float32, (
            "_silu_mul_fp8_quant_deep_gemm Triton fallback does not "
            f"support {y_s.dtype} scales. Only torch.float32 supported."
        )
        _silu_mul_fp8_quant_deep_gemm[grid](
            y,
            y_q,
            y_s,
            tokens_per_expert,
            H,
            group_size,
            stride_i_e,
            stride_i_t,
            stride_i_h,
            stride_yq_e,
            stride_yq_t,
            stride_yq_h,
            ys_strides[0],
            ys_strides[1],
            ys_strides[2],
            stride_cnt_e,
            eps,
            fp8_min,
            fp8_max,
            ceil_ue8m0,
            BLOCK=group_size,
            NUM_STAGES=4,
            num_warps=1,
        )

    return y_q, y_s


@triton.jit
def _swiglustep_mul_fp8_quant_deep_gemm(
    # Pointers ------------------------------------------------------------
    input_ptr,  # 16-bit activations (E, T, 2*H)
    y_q_ptr,  # fp8 quantized activations (E, T, H)
    y_s_ptr,  # 16-bit scales (E, T, G)
    counts_ptr,  # int32 num tokens per expert (E)
    # Sizes ---------------------------------------------------------------
    H: tl.constexpr,  # hidden dimension (per output)
    GROUP_SIZE: tl.constexpr,  # elements per group (usually 128)
    # Strides for input (elements) ---------------------------------------
    stride_i_e,
    stride_i_t,
    stride_i_h,
    # Strides for y_q (elements) -----------------------------------------
    stride_yq_e,
    stride_yq_t,
    stride_yq_h,
    # Strides for y_s (elements) -----------------------------------------
    stride_ys_e,
    stride_ys_t,
    stride_ys_g,
    # Stride for counts (elements)
    stride_counts_e,
    limit,
    # Numeric params ------------------------------------------------------
    eps: tl.constexpr,
    fp8_min: tl.constexpr,
    fp8_max: tl.constexpr,
    ceil_ue8m0: tl.constexpr,
    # Meta ---------------------------------------------------------------
    BLOCK: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    G = H // GROUP_SIZE

    # map program id -> (e, g)
    pid = tl.program_id(0)
    e = pid // G
    g = pid % G

    e = e.to(tl.int64)
    g = g.to(tl.int64)

    # number of valid tokens for this expert
    n_tokens = tl.load(counts_ptr + e * stride_counts_e).to(tl.int64)

    cols = tl.arange(0, BLOCK).to(tl.int64)
    mask = cols < BLOCK

    base_input_offset = e * stride_i_e + g * GROUP_SIZE * stride_i_h
    base_gate_offset = base_input_offset + cols * stride_i_h
    base_up_offset = base_input_offset + H * stride_i_h + cols * stride_i_h
    base_yq_offset = e * stride_yq_e + g * GROUP_SIZE * stride_yq_h + cols * stride_yq_h
    base_ys_offset = e * stride_ys_e + g * stride_ys_g

    for t in tl.range(0, n_tokens, num_stages=NUM_STAGES):
        gate = tl.load(
            input_ptr + base_gate_offset + t * stride_i_t, mask=mask, other=0.0
        ).to(tl.float32)
        up = tl.load(
            input_ptr + base_up_offset + t * stride_i_t, mask=mask, other=0.0
        ).to(tl.float32)

        gate = gate * (1.0 / (1.0 + tl.exp(-gate)))

        gate = tl.minimum(gate, limit)  # clamp gate
        up = tl.clamp(up, -limit, limit)  # clamp up
        y = gate * up

        # Use multiply-by-reciprocal to match PyTorch's tensor/scalar
        # division precision (Triton GPU fast-division for constexpr
        # divisors can introduce 1-ULP error).
        y_s = tl.maximum(tl.max(tl.abs(y)), eps) * (1.0 / fp8_max)
        if ceil_ue8m0:
            y_s = tl.exp2(tl.ceil(tl.log2(y_s)))

        y_q = tl.clamp(y / y_s, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

        tl.store(y_q_ptr + base_yq_offset + t * stride_yq_t, y_q, mask=mask)
        tl.store(y_s_ptr + base_ys_offset + t * stride_ys_t, y_s)


def persistent_masked_m_swiglustep_mul_quant(
    y: torch.Tensor,  # (E, T, 2*H)
    tokens_per_expert: torch.Tensor,  # (E,) number of valid tokens per expert
    num_parallel_tokens=16,
    limit: float = 7.0,
    group_size: int = 128,
    quant_scale_fmt: DeepGemmQuantScaleFMT = DeepGemmQuantScaleFMT.FLOAT32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize silu(y[..., :H]) * y[..., H:] to FP8 with group per-token scales
    y has shape (E, T, 2*H). The first half of the last dimension is
    silu-activated, multiplied by the second half, then quantized into FP8.
    We launch a fixed grid of threads to accommodate CUDA graphs. Let `P2`
    be a parallelization factor for persistent_masked_m_swiglustep_mul_quant over the
    hidden dimension.

    Let `expert_offsets = [0] + [num_tokens.cumsum()]` and
    `total_tokens = expert_offsets[-1]`.
    persistent_masked_m_swiglustep_mul_quant launches `total_tokens x P2` number of
    thread blocks. Each thread block contains `NUM_WARPS` warps.

    Every thread block needs to find it's corresponding expert by warp-parallel scanning
    over the `expert_offsets` array.

    The i-th warp in the first thread block processes
    `[i * warp_chunk_size, (i + 1) * warp_chunk_size]` groups
    sequentially, where `warp_chunk_size = ((H / GROUP_SIZE) / P2) / NUM_WARPS`,
    pipelining loads and computes.

    The shared memory layout for 4 warps with a 2-stage pipeline for SiLU V2
    can is visualized like so:

                         stage0                              stage1
    ┌─────┬───┬─────┬───┬─────┬───┬─────┬───┬─────┬───┬─────┬───┬─────┬───┬─────┬───┐
    │gate0│up0│gate1│up1│gate2│up2│gate3│up3│gate0│up0│gate1│up1│gate2│up2│gate3│up3│
    └─────┴───┴─────┴───┴─────┴───┴─────┴───┴─────┴───┴─────┴───┴─────┴───┴─────┴───┘

    with the main difference between V1 and V2 being the global load
    stride between warps, and between half-warps. Regarding the latter stride,
    we assign the first half warp of every warp for `gate` loads and the second
    half-warp to `up` loads.

    Returns `(y_q, y_s)` where
    * `y_q`: FP8 tensor, shape (E, T, H), same layout as y[..., :H]
    * `y_s` depends on quant_scale_fmt,
      - quant_scale_fmt == FLOAT32,
         `y_s`: FP32 tensor, shape (E, T, H // group_size), strides (T*G, 1, T)
      - quant_scale_fmt == E8M0,
         `y_s`: Int32 tensor, shape (E, T, H // group_size // 4), strides (T*G, 1, T)
      - quant_scale_fmt == E8M0_FLOAT32_SPARSE
         `y_s`: FP32 tensor, shape (E, T, H // group_size), strides (T*G, 1, T)
    Let NUM_WARPS be the number of warps in a single thread block and
    `GROUP_SIZE = 128` be the size of the quantization group.
    """
    assert y.ndim == 3, "y must be (E, T, 2*H)"
    E, T, H2 = y.shape
    assert H2 % 2 == 0, "last dim of y must be even (2*H)"
    H = H2 // 2
    G = (H + group_size - 1) // group_size
    assert H % 8 == 0, "H must be divisible by 8"
    assert group_size == 128, "H must be divisible by 8"
    assert tokens_per_expert.ndim == 1 and tokens_per_expert.shape[0] == E

    tokens_per_expert = tokens_per_expert.to(device=y.device, dtype=torch.int32)

    fp8_dtype = current_platform.fp8_dtype()
    y_q = torch.empty((E, T, H), dtype=fp8_dtype, device=y.device)

    ys_shape, ys_strides, ys_dtype = scales_shape_stride_dtype(E, T, G, quant_scale_fmt)
    y_s = torch.empty_strided(
        ys_shape,
        ys_strides,
        dtype=ys_dtype,
        device=y.device,
    )

    ceil_ue8m0 = quant_scale_fmt in [
        DeepGemmQuantScaleFMT.FLOAT32_CEIL_UE8M0,
        DeepGemmQuantScaleFMT.UE8M0,
    ]

    device_capability = current_platform.get_device_capability(device_id=y.device.index)
    assert device_capability is not None
    cuda_arch = device_capability.to_int()

    # Maca does not have the kernel
    if current_platform.is_cuda() and cuda_arch >= 80:
        torch.ops._C.persistent_masked_m_swiglustep_mul_quant(
            y, tokens_per_expert, y_q, y_s, ceil_ue8m0
        )
    else:
        # Triton fallback for ROCm -- the C++ kernel is guarded by
        # #ifndef USE_ROCM in activation_kernels.cu.
        # https://github.com/ROCm/aiter/issues/2420
        stride_cnt_e = tokens_per_expert.stride()[0]

        # Static grid over experts and H-groups.
        # A loop inside the kernel handles the token dim
        grid = (E * G,)
        # strides (elements)
        stride_i_e, stride_i_t, stride_i_h = y.stride()
        stride_yq_e, stride_yq_t, stride_yq_h = y_q.stride()

        fp8_min, fp8_max = get_fp8_min_max()
        eps: float = 1e-10
        assert y_s.dtype == torch.float32, (
            "_swiglustep_mul_fp8_quant_deep_gemm Triton fallback does not "
            f"support {y_s.dtype} scales. Only torch.float32 supported."
        )
        _swiglustep_mul_fp8_quant_deep_gemm[grid](
            y,
            y_q,
            y_s,
            tokens_per_expert,
            H,
            group_size,
            stride_i_e,
            stride_i_t,
            stride_i_h,
            stride_yq_e,
            stride_yq_t,
            stride_yq_h,
            ys_strides[0],
            ys_strides[1],
            ys_strides[2],
            stride_cnt_e,
            limit,
            eps,
            fp8_min,
            fp8_max,
            ceil_ue8m0,
            BLOCK=group_size,
            NUM_STAGES=4,
            num_warps=1,
        )

    return y_q, y_s


class BatchedDeepGemmExperts(mk.FusedMoEExpertsModular):
    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int,
        num_dispatchers: int,
    ):
        """
        max_num_tokens: Maximum number of tokens from a DP Rank
        num_dispatchers: The number of DP dispatchers.
        quant_config: Quantization configuration
        """
        super().__init__(
            moe_config=moe_config,
            quant_config=quant_config,
            max_num_tokens=max_num_tokens,
            num_dispatchers=num_dispatchers,
        )
        self.gemm1_clamp_limit = (
            quant_config.gemm1_clamp_limit
            if quant_config.gemm1_clamp_limit is not None
            else moe_config.swiglu_limit
        )
        self.gemm1_alpha = (
            quant_config.gemm1_alpha
            if quant_config.gemm1_alpha is not None
            else (
                moe_config.swiglu_alpha if moe_config.swiglu_alpha is not None else 1.0
            )
        )
        self.gemm1_beta = (
            quant_config.gemm1_beta
            if quant_config.gemm1_beta is not None
            else (moe_config.swiglu_beta if moe_config.swiglu_beta is not None else 0.0)
        )

        self.is_fp8 = quant_config.use_fp8_w8a8
        self.is_int8 = quant_config.use_int8_w8a8

        if self.is_fp8 == self.is_int8:
            raise ValueError(
                "BatchedDeepGemmExperts requires exactly one of FP8 W8A8 or INT8 W8A8"
            )

        if self.is_fp8:
            assert self.block_shape == get_mk_alignment_for_contiguous_layout()
        else:
            assert self.block_shape is None
            assert self.per_act_token_quant
            assert self.w1_scale is not None
            assert self.w2_scale is not None
            assert self.w1_zp is None and self.w2_zp is None
            assert self.w1_bias is None and self.w2_bias is None
            if moe_config.in_dtype != torch.bfloat16:
                raise ValueError(
                    "BatchedDeepGemmExperts INT8 kernels require bfloat16 "
                    f"input/output, got {moe_config.in_dtype}"
                )

        if (
            moe_config.activation == MoEActivation.SWIGLUSTEP
            and self.gemm1_clamp_limit is None
        ):
            raise ValueError("SWIGLUSTEP requires swiglu_limit in moe_config")

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.BatchedExperts

    @staticmethod
    def is_supported_config(
        cls: type[mk.FusedMoEExperts],
        moe_config: FusedMoEConfig,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
        activation_format: mk.FusedMoEActivationFormat,
    ) -> tuple[bool, str | None]:
        int8_w8a8 = (weight_key, activation_key) == (
            kInt8StaticChannelSym,
            kInt8DynamicTokenSym,
        )
        if (
            moe_config.activation == MoEActivation.SWIGLUSTEP
            and moe_config.swiglu_limit is None
        ):
            return False, "kernel requires swiglu_limit for SWIGLUSTEP"
        if int8_w8a8 and moe_config.in_dtype != torch.bfloat16:
            return (
                False,
                f"kernel does not support {moe_config.in_dtype} input/output dtype",
            )
        if int8_w8a8 and moe_config.has_bias:
            return False, "kernel does not support bias"

        return mk.FusedMoEExperts.is_supported_config(
            cls,
            moe_config,
            weight_key,
            activation_key,
            activation_format,
        )

    @staticmethod
    def _supports_current_device() -> bool:
        return is_deep_gemm_supported()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        supported_w_a = [
            (kFp8Static128BlockSym, kFp8Dynamic128Sym),
            (kInt8StaticChannelSym, kInt8DynamicTokenSym),
        ]
        return (weight_key, activation_key) in supported_w_a

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in [MoEActivation.SILU, MoEActivation.SWIGLUSTEP]

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        return True

    def supports_packed_ue8m0_act_scales(self) -> bool:
        """
        DeepGemm supports packed ue8m0 activation scales on Blackwell-family
        GPUs (SM100 datacenter and SM120 consumer).
        """
        return self.is_fp8 and is_deep_gemm_e8m0_used()

    def workspace_dtype(self, act_dtype: torch.dtype) -> torch.dtype:
        if self.is_int8:
            # MetaX INT8 masked grouped GEMM only supports BF16 output.
            return torch.bfloat16
        return super().workspace_dtype(act_dtype)

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        # Let PrepareAndFinalize::finalize() decide the impl.
        return TopKWeightAndReduceDelegate()

    def activation(
        self,
        activation: MoEActivation,
        output: torch.Tensor,
        input: torch.Tensor,
        clamp_limit: float | None = None,
        alpha: float = 1.0,
        beta: float = 0.0,
    ) -> None:
        if activation == MoEActivation.SILU and clamp_limit is not None:
            swiglu_limit_func(output, input, float(clamp_limit))
            return

        if activation == MoEActivation.SWIGLUSTEP:
            from vllm.model_executor.layers.activation import (
                swiglustep_and_mul_triton,
            )

            assert clamp_limit is not None, (
                "SWIGLUSTEP requires swiglu_limit in moe_config"
            )
            # Note: super().activation() call swiglustep_and_mul_triton() without
            # limit argument, So we manually make the call.
            # Remove this once it supported.
            swiglustep_and_mul_triton(
                output,
                input,
                limit=float(clamp_limit),
            )
        super().activation(
            activation, output, input, clamp_limit=clamp_limit, alpha=alpha, beta=beta
        )

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        # FIXME (varun): We should be able to dispatch only from the leader
        # DP ranks in the case of TP > 1. At the moment, all the Ranks
        # end up sending their tokens. This needs to be fixed.
        assert self.num_dispatchers is not None
        assert self.max_num_tokens is not None
        num_dispatchers = self.num_dispatchers
        num_experts = local_num_experts
        max_num_tokens = M if self.max_num_tokens is None else self.max_num_tokens
        activation_out_dim = self.adjust_N_for_activation(N, activation)
        workspace13 = (num_experts, max_num_tokens * num_dispatchers, max(K, N))
        workspace2 = (num_experts, max_num_tokens * num_dispatchers, activation_out_dim)
        output = (num_experts, max_num_tokens * num_dispatchers, K)
        return (workspace13, workspace2, output)

    def estimate_expected_m(
        self, global_num_experts: int, max_tokens_per_expert: int, topk: int
    ) -> int:
        dp_meta = (
            get_forward_context().dp_metadata
            if is_forward_context_available()
            else None
        )
        if dp_meta is None:
            logger.warning_once(
                "DPMetadata unavailable. Defaulting expected_m to "
                f"{max_tokens_per_expert}.",
            )
            return max_tokens_per_expert

        total_num_tokens = dp_meta.num_tokens_across_dp_cpu.sum().item()
        total_num_tokens_replicated = total_num_tokens * topk

        # Assume even load balancing
        assert global_num_experts != 0
        estimate = round_up(int(total_num_tokens_replicated // global_num_experts), 16)
        # clamp estimate
        estimate = max(estimate, 16)
        estimate = min(max_tokens_per_expert, estimate)
        return estimate

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ) -> None:
        if self.is_int8:
            apply_impl = self.apply_int8
        elif self.is_fp8:
            apply_impl = self.apply_fp8
        else:
            raise NotImplementedError(
                "BatchedDeepGemmExperts requires either FP8 W8A8 or INT8 W8A8"
            )
        apply_impl(
            output,
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            activation,
            global_num_experts,
            expert_map,
            a1q_scale,
            a2_scale,
            workspace13,
            workspace2,
            expert_tokens_meta,
            apply_router_weight_on_input,
        )
        return

    def apply_fp8(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ) -> None:
        # Existing FP8 path.
        assert expert_tokens_meta is not None
        expert_num_tokens = expert_tokens_meta.expert_num_tokens

        assert hidden_states.ndim == 3
        assert self.block_shape is not None

        a1q = hidden_states
        _, N, K = w1.size()

        assert w2.size(1) == K

        E, max_num_tokens, N, K, _ = self.moe_problem_size(
            hidden_states, w1, w2, topk_ids
        )

        workspace1 = _resize_cache(workspace13, (E, max_num_tokens, N))

        expected_m = self.estimate_expected_m(
            global_num_experts=global_num_experts,
            max_tokens_per_expert=max_num_tokens,
            topk=topk_ids.size(-1),
        )

        fp8_m_grouped_gemm_nt_masked(
            (a1q, a1q_scale),
            (w1, self.w1_scale),
            workspace1,
            expert_num_tokens,
            expected_m,
        )

        quant_scale_fmt = DeepGemmQuantScaleFMT.from_oracle()
        if activation == MoEActivation.SWIGLUSTEP:
            limit = self.gemm1_clamp_limit
            assert limit is not None, "SWIGLUSTEP requires swiglu_limit in moe_config"
            a2q, a2q_scale = persistent_masked_m_swiglustep_mul_quant(
                workspace1,
                expert_num_tokens,
                limit=limit,
                quant_scale_fmt=quant_scale_fmt,
            )
        else:
            a2q, a2q_scale = persistent_masked_m_silu_mul_quant(
                workspace1,
                expert_num_tokens,
                quant_scale_fmt=quant_scale_fmt,
            )

        fp8_m_grouped_gemm_nt_masked(
            (a2q, a2q_scale),
            (w2, self.w2_scale),
            output,
            expert_num_tokens,
            expected_m,
        )

    def apply_int8(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ) -> None:
        assert expert_tokens_meta is not None
        expert_num_tokens = expert_tokens_meta.expert_num_tokens

        assert hidden_states.ndim == 3
        assert hidden_states.dtype == torch.int8
        assert w1.dtype == torch.int8 and w2.dtype == torch.int8
        assert w1.ndim == 3 and w2.ndim == 3
        assert hidden_states.is_contiguous()
        assert w1.is_contiguous() and w2.is_contiguous()
        assert w1.device == hidden_states.device and w2.device == hidden_states.device
        assert expert_num_tokens.dtype == torch.int32
        assert expert_num_tokens.ndim == 1
        assert expert_num_tokens.is_contiguous()
        assert expert_num_tokens.device == hidden_states.device
        assert a2_scale is None, "Dynamic per-token INT8 quantization has no a2 scale"

        E, max_num_tokens, N, K, _ = self.moe_problem_size(
            hidden_states, w1, w2, topk_ids
        )
        activation_out_dim = self.adjust_N_for_activation(N, activation)

        assert expert_num_tokens.shape == (E,)
        assert w1.shape == (E, N, K)
        assert w2.shape == (E, K, activation_out_dim)
        assert output.shape == (E, max_num_tokens, K)
        assert output.dtype == torch.bfloat16
        assert workspace13.dtype == torch.bfloat16
        assert workspace2.dtype == torch.bfloat16
        assert output.is_contiguous()
        assert workspace13.is_contiguous()
        assert workspace2.is_contiguous()
        assert output.device == hidden_states.device
        assert workspace13.device == hidden_states.device
        assert workspace2.device == hidden_states.device
        assert a1q_scale is not None and a1q_scale.device == hidden_states.device
        assert self.w1_scale is not None and self.w1_scale.device == w1.device
        assert self.w2_scale is not None and self.w2_scale.device == w2.device

        workspace1 = _resize_cache(workspace13, (E, max_num_tokens, N))
        activation_out = _resize_cache(
            workspace2,
            (E, max_num_tokens, activation_out_dim),
        )
        assert workspace1.is_contiguous()
        assert activation_out.is_contiguous()

        a1_s = _normalize_int8_scale_2d(a1q_scale, E, max_num_tokens, "a1_scale")
        w1_s = _normalize_int8_scale_2d(self.w1_scale, E, N, "w1_scale")

        expected_m = self.estimate_expected_m(
            global_num_experts=global_num_experts,
            max_tokens_per_expert=max_num_tokens,
            topk=topk_ids.size(-1),
        )

        # Masked grouped GEMM only writes valid token rows. Clear the shared
        # workspace so activation/quantization over the fixed graph shape sees
        # deterministic zeros in padding rows.
        workspace1.zero_()
        int8_m_grouped_gemm_nt_masked(
            (hidden_states, a1_s),
            (w1, w1_s),
            workspace1,
            expert_num_tokens,
            expected_m,
        )

        workspace1_flat = workspace1.view(-1, N)
        activation_out_flat = activation_out.view(-1, activation_out_dim)
        self.activation(
            activation,
            activation_out_flat,
            workspace1_flat,
            clamp_limit=self.gemm1_clamp_limit,
            alpha=self.gemm1_alpha,
            beta=self.gemm1_beta,
        )

        a2q, a2q_scale = moe_kernel_quantize_input(
            activation_out,
            None,
            torch.int8,
            per_act_token_quant=True,
            block_shape=None,
        )
        assert a2q_scale is not None
        assert a2q.is_contiguous()
        a2_s = _normalize_int8_scale_2d(a2q_scale, E, max_num_tokens, "a2_scale")
        w2_s = _normalize_int8_scale_2d(self.w2_scale, E, K, "w2_scale")

        # The final output aliases workspace13 in the modular kernel. GEMM1 is
        # no longer needed after activation quantization, so it is safe to clear
        # the output before GEMM2 for deterministic padding rows.
        output.zero_()
        int8_m_grouped_gemm_nt_masked(
            (a2q, a2_s),
            (w2, w2_s),
            output,
            expert_num_tokens,
            expected_m,
        )
