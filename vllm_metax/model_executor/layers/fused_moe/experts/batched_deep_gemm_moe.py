# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.utils import _resize_cache
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_fp8_min_max,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.deep_gemm import (
    DeepGemmQuantScaleFMT,
    fp8_m_grouped_gemm_nt_masked,
)
from vllm.model_executor.layers.fused_moe.experts.batched_deep_gemm_moe import (
    scales_shape_stride_dtype,
    persistent_masked_m_silu_mul_quant,
    BatchedDeepGemmExperts as vllm_BatchedDeepGemmExperts,
)


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


class BatchedDeepGemmExperts(vllm_BatchedDeepGemmExperts):
    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in [MoEActivation.SILU, MoEActivation.SWIGLUSTEP]

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
    ):
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
