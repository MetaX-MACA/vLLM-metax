# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from math import log2
from typing import Optional

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig

from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceDelegate,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import is_deep_gemm_e8m0_used
from vllm_metax.utils.deep_gemm import is_deep_gemm_supported
from vllm._custom_ops import scaled_int8_quant as vllm_scaled_int8_quant

from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
)
from vllm.model_executor.layers.fused_moe.activation import MoEActivation

from vllm.model_executor.layers.fused_moe.experts.batched_deep_gemm_moe import (
    _silu_mul_fp8_quant_deep_gemm,
)
from vllm_metax.patch.dp_opt.utils import (
    m_grouped_gemm_nt_masked,
    silu_and_mul_masked_fwd_no_pack,
    silu_and_mul_masked_fwd,
)


logger = init_logger(__name__)


def silu_mul_fp8_quant_deep_gemm_cuda(
    y: torch.Tensor,  # (E, T, 2*H)
    tokens_per_expert: torch.Tensor,  # (E,) number of valid tokens per expert
    num_parallel_tokens=16,
    group_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize silu(y[..., :H]) * y[..., H:] to FP8 with group per-token scales
    y has shape (E, T, 2*H). The first half of the last dimension is
    silu-activated, multiplied by the second half, then quantized into FP8.
    Returns `(y_q, y_s)` where
    * `y_q`: FP8 tensor, shape (E, T, H), same layout as y[..., :H]
    * `y_s`: FP32 tensor, shape (E, T, H // group_size), strides (T*G, 1, T)
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

    fp8_dtype = torch.float8_e4m3fn
    y_q = torch.empty((E, T, H), dtype=fp8_dtype, device=y.device)

    stride_ys_e = T * G
    stride_ys_t = 1
    stride_ys_g = T
    y_s = torch.empty_strided(
        (E, T, G),
        (stride_ys_e, stride_ys_t, stride_ys_g),
        dtype=torch.float32,
        device=y.device,
    )

    use_ue8m0 = is_deep_gemm_e8m0_used()

    if E <= 16:
        max_empirical_parallelism = 64
    elif E <= 32:
        max_empirical_parallelism = 16
    else:
        max_empirical_parallelism = 4

    # We never want to launch more than Tx number of threads
    # This computes the clip.
    num_parallel_tokens = max(
        1, min(max_empirical_parallelism, 2 ** int(log2(min(num_parallel_tokens, T))))
    )
    cuda_arch = current_platform.get_device_capability(
        device_id=y.device.index
    ).to_int()

    if cuda_arch >= 80:
        torch.ops._C.silu_mul_fp8_quant_deep_gemm_cuda(
            y, tokens_per_expert, y_q, y_s, group_size, use_ue8m0, num_parallel_tokens
        )
    else:
        # Default to triton if not on cuda or if arch is too old
        y_q = torch.empty((E, T, H), dtype=fp8_dtype, device=y.device)

        stride_cnt_e = tokens_per_expert.stride()[0]

        # Static grid over experts and H-groups.
        # A loop inside the kernel handles the token dim
        grid = (E * G,)
        # strides (elements)
        stride_i_e, stride_i_t, stride_i_h = y.stride()
        stride_yq_e, stride_yq_t, stride_yq_h = y_q.stride()

        # desired scale strides (elements): (T*G, 1, T)
        stride_ys_e = T * G
        stride_ys_t = 1
        stride_ys_g = T
        y_s = torch.empty_strided(
            (E, T, G),
            (stride_ys_e, stride_ys_t, stride_ys_g),
            dtype=torch.float32,
            device=y.device,
        )
        f_info = torch.finfo(fp8_dtype)
        fp8_max = f_info.max
        fp8_min = f_info.min
        eps: float = 1e-10
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
            stride_ys_e,
            stride_ys_t,
            stride_ys_g,
            stride_cnt_e,
            eps,
            fp8_min,
            fp8_max,
            is_deep_gemm_e8m0_used(),
            BLOCK=group_size,
            NUM_STAGES=4,
            num_warps=1,
        )

    return y_q, y_s


class MacaBatchedDeepGemmExperts(mk.FusedMoEExpertsModular):
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
        self.need_mul_scale = False
        self.use_fused_quant = True

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.BatchedExperts

    @staticmethod
    def _supports_current_device() -> bool:
        return is_deep_gemm_supported()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(weight_key, activation_key) -> bool:
        return True

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation == MoEActivation.SILU

    @staticmethod
    def _supports_parallel_config(moe_parallel_config) -> bool:
        return True

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        # Let PrepareAndFinalize::finalize() decide the impl.
        return TopKWeightAndReduceDelegate()

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
        K = 4096
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

    def moe_problem_size(
        self,
        a1: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[int, int, int, int, int]:
        """
        Extract the MoE problem size from the given tensor arguments:
        - a: The hidden states, input to the MoE layer.
        - w1: The first set of expert weights.
        - w2: The second set of expert weights.
        - topk_ids: The topk ids.

        Note: extracting the problem shape from the weight and activation
        tensors is not obvious.  It needs to be done this way specifically
        due to subtle issues with particular kernels, e.g. the int4 kernels
        divide the trailing dimension by two, so it's not "correct" to
        extract N or K from the trailing dimension of w1 or w2.  Similarly,
        some kernels transpose the weights, so this needs to be kept in mind.

        Note: This implementation covers most cases. However, if experts
        require a specialized implementation, like MarlinExperts, they are free
        to override this function.
        """

        # TODO Tuple K value
        assert w1.dim() == 3 and w2.dim() == 3
        E, N, _ = w1.size()
        K = a1.size(-1)

        if a1.dim() == 2:
            # Make sure we are using the correct a1 (pre-permute).
            assert topk_ids.size(0) == a1.size(0), f"{topk_ids.size(0)} != {a1.size(0)}"
            M = a1.size(0)
        else:
            assert a1.dim() == 3
            assert a1.size(0) == E, f"{a1.size(0)} == {E}"
            M = a1.size(1)  # This is max_num_tokens

        assert topk_ids.dim() == 2
        topk = topk_ids.size(1)

        return E, M, N, K, topk

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
        expert_map: Optional[torch.Tensor],
        a1q_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: Optional[mk.ExpertTokensMetadata],
        apply_router_weight_on_input: bool,
    ):
        if self.need_mul_scale:
            if w2.dtype == torch.int8:
                self.w2_scale.mul_(self.routed_scaling_factor)
            else:
                w2.mul_(self.routed_scaling_factor)
            self.need_mul_scale = False
        assert expert_tokens_meta is not None
        masked_m = expert_tokens_meta.expert_num_tokens
        expected_m = 10

        # GroupGemm-0
        num_groups, m, k = hidden_states.size()
        n = w1.size(1)
        gateup_output = torch.empty(
            (num_groups, m, n), device=hidden_states.device, dtype=torch.bfloat16
        )

        m_grouped_gemm_nt_masked(
            hidden_states,
            w1,
            gateup_output,
            masked_m,
            expected_m,
            self.w1_scale,
            # use_triton_kernel = False,
            # unpack_tensor=not self.use_fused_quant
        )
        # Act
        down_input = torch.empty(
            (
                gateup_output.shape[0],
                gateup_output.shape[1],
                gateup_output.shape[2] // 2,
            ),
            device=gateup_output.device,
            dtype=gateup_output.dtype,
        )

        down_input = silu_and_mul_masked_fwd(gateup_output, masked_m)

        # GroupGemm-1
        n = w2.size(1)
        if output is not None:
            down_output = output
        else:
            down_output = torch.empty(
                (num_groups, m, n), device=down_input.device, dtype=torch.bfloat16
            )
        m_grouped_gemm_nt_masked(
            down_input,
            w2,
            down_output,
            masked_m,
            expected_m,
            self.w2_scale,
            # use_triton_kernel = False,
            # unpack_tensor=False
        )

    def apply_ori(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: str,
        global_num_experts: int,
        expert_map: Optional[torch.Tensor],
        a1q_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: Optional[mk.ExpertTokensMetadata],
        apply_router_weight_on_input: bool,
    ):
        if self.need_mul_scale:
            if w2.dtype == torch.int8:
                self.w2_scale.mul_(self.routed_scaling_factor)
            else:
                w2.mul_(self.routed_scaling_factor)
            self.need_mul_scale = False

        def pack_tensor(hidden_states):
            num_groups, m, k = hidden_states.size()
            print(f"{hidden_states.shape=},{hidden_states.dtype=}")
            # print(f"{hidden_states.shape=},{hidden_states.dtype=},{a1q_scale.shape=},{a1q_scale.dtype=}")
            # hidden_states.shape=torch.Size([8, 1024, 7168]),hidden_states.dtype=torch.int8,a1q_scale.shape=torch.Size([8, 1024, 1]),a1q_scale.dtype=torch.float32

            k = 3840
            # a_quant_val, a_quant_scale, _ = vllm_scaled_int8_quant(hidden_states)
            pack_hidden_states = torch.empty(
                (num_groups, m, k), device=hidden_states.device, dtype=torch.bfloat16
            )
            pack_hidden_states[:, :, 0:3584] = hidden_states.view(dtype=torch.bfloat16)
            assert a1q_scale is not None
            pack_hidden_states[:, :, 3584:3586] = a1q_scale.view(dtype=torch.bfloat16)
            return pack_hidden_states

        hidden_states = pack_tensor(hidden_states)
        assert expert_tokens_meta is not None
        masked_m = expert_tokens_meta.expert_num_tokens

        expected_m = 10

        # GroupGemm-0
        num_groups, m, k = hidden_states.size()
        n = w1.size(1)

        gateup_output = torch.empty(
            (num_groups, m, n), device=hidden_states.device, dtype=torch.bfloat16
        )

        m_grouped_gemm_nt_masked(
            hidden_states,
            w1,
            gateup_output,
            masked_m,
            expected_m,
            self.w1_scale,
            use_triton_kernel=False,
            unpack_tensor=not self.use_fused_quant,
        )
        # Act
        down_input = torch.empty(
            (
                gateup_output.shape[0],
                gateup_output.shape[1],
                gateup_output.shape[2] // 2,
            ),
            device=gateup_output.device,
            dtype=gateup_output.dtype,
        )

        if self.use_fused_quant:

            def tmp_pack_tensor(a_quant_val, a_quant_scale, packed_size):
                b_bytes = (
                    a_quant_val.reshape(-1, a_quant_val.shape[-1])
                    .contiguous()
                    .view(torch.bfloat16)
                )
                c_bytes = (
                    a_quant_scale.reshape(-1, a_quant_scale.shape[-1])
                    .contiguous()
                    .view(torch.bfloat16)
                )

                random_tensor = torch.cat([b_bytes, c_bytes], dim=-1)
                random_tensor = random_tensor.contiguous()
                pad_zeros = torch.zeros(
                    (random_tensor.shape[0], packed_size - random_tensor.shape[-1]),
                    dtype=torch.bfloat16,
                    device=a_quant_val.device,
                )

                random_tensor_pad = torch.cat([random_tensor, pad_zeros], dim=-1)
                random_tensor_pad = random_tensor_pad.contiguous().view(torch.bfloat16)

                return random_tensor_pad

            silu_and_mul_masked_fwd_no_pack(gateup_output, down_input, masked_m)
            output_quant, output_scale, _ = vllm_scaled_int8_quant(down_input)
            down_input = tmp_pack_tensor(
                output_quant,
                output_scale,
                ((output_quant.shape[-1] // 2 + 2) // 256 + 1) * 256,
            )
            down_input = down_input.reshape(
                gateup_output.shape[0], gateup_output.shape[1], down_input.shape[-1]
            )
            # down_input = silu_and_mul_masked_fwd(gateup_output, masked_m)
        else:
            silu_and_mul_masked_fwd_no_pack(gateup_output, down_input, masked_m)

        # GroupGemm-1
        n = w2.size(1)
        if output is not None:
            down_output = output
        else:
            down_output = torch.empty(
                (num_groups, m, n), device=down_input.device, dtype=torch.bfloat16
            )
        m_grouped_gemm_nt_masked(
            down_input,
            w2,
            down_output,
            masked_m,
            expected_m,
            self.w2_scale,
            use_triton_kernel=False,
            unpack_tensor=False,
        )
