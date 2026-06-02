# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton-based MoE expert implementations."""

import importlib
from typing import Any

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm_metax.model_executor.layers.fused_moe.fused_moe import (
    _prepare_expert_assignment,
    invoke_fused_moe_triton_kernel,
    invoke_fused_moe_wna16_triton_kernel,
    try_get_optimal_moe_config,
)
from vllm.model_executor.layers.fused_moe.lora_experts_mixin import (
    LoRAExpertsMixin,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.fused_moe.utils import (
    _resize_cache,
    moe_kernel_quantize_input,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8Dynamic128Sym,
    kFp8DynamicTensorSym,
    kFp8DynamicTokenSym,
    kFp8Static128BlockSym,
    kFp8StaticChannelSym,
    kFp8StaticTensorSym,
    kInt8DynamicTokenSym,
    kInt8StaticChannelSym,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl
from vllm_metax.model_executor.layers.fused_moe.utils import (
    get_config_dtype_str,
    initialize_staged_config,
)
import vllm_metax.envs as mx_envs

if mx_envs.USE_PRECOMPILED_KERNEL:
    from mcoplib.triton_fused_moe import (
        fused_moe_triton_kernel,
        fused_moe_triton_kernel_gptq_awq,
    )
else:
    fused_moe_triton_kernel = None
    fused_moe_triton_kernel_gptq_awq = None

_mctlass_modname = (
    "vllm_metax.model_executor.layers.quantization._python_api_ops"
    if mx_envs.MACA_VLLM_ENABLE_MCTLASS_PYTHON_API
    else "vllm_metax.model_executor.layers.quantization._cutlass_ops"
)
mctlass_ops: Any = importlib.import_module(_mctlass_modname)


class TritonExperts(LoRAExpertsMixin, mk.FusedMoEExpertsModular):
    """Triton-based fused MoE expert implementation."""

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        # Whether quantized MOE runs natively, or through
        # higher-precision + activation QDQ.
        self.quantization_emulation = False
        super().__init__(moe_config, quant_config)

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        return current_platform.is_cuda_alike() or current_platform.is_xpu()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return True

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        device_supports_int8 = True

        supported: list[tuple[QuantKey | None, QuantKey | None]] = [(None, None)]
        if device_supports_int8:
            supported.append((kInt8StaticChannelSym, kInt8DynamicTokenSym))
        if current_platform.supports_fp8():
            supported += [
                (kFp8Static128BlockSym, kFp8Dynamic128Sym),
                (kFp8StaticChannelSym, kFp8DynamicTokenSym),
                (kFp8StaticTensorSym, kFp8DynamicTokenSym),
                (kFp8StaticTensorSym, kFp8StaticTensorSym),
                (kFp8StaticTensorSym, kFp8DynamicTensorSym),
            ]
        return (weight_key, activation_key) in supported

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in [
            MoEActivation.SILU,
            MoEActivation.GELU,
            MoEActivation.GELU_TANH,
            MoEActivation.SWIGLUOAI,
            MoEActivation.SWIGLUSTEP,
            MoEActivation.SILU_NO_MUL,
            MoEActivation.GELU_NO_MUL,
            MoEActivation.GELU_TANH_NO_MUL,
            MoEActivation.RELU2_NO_MUL,
        ]

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        return not (
            moe_parallel_config.use_fi_nvl_two_sided_kernels
            or moe_parallel_config.use_fi_nvl_one_sided_kernels
        )

    @staticmethod
    def _supports_batch_invariance():
        return True

    def supports_expert_map(self) -> bool:
        return True

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

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
        activation_out_dim = self.adjust_N_for_activation(N, activation)
        workspace1 = (M, topk, max(activation_out_dim, K))
        workspace2 = (M, topk, max(N, K))
        output = (M, K)
        return (workspace1, workspace2, output)

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
        # Check constraints.
        if self.quant_config.use_int4_w4a16:
            assert hidden_states.size(-1) // 2 == w1.size(2), "Hidden size mismatch"
        if self.quant_config.use_int4_w4a8:
            # 8bit activation and int4 packed weight
            assert hidden_states.size(-1) // 8 == w1.size(2), "Hidden size mismatch"
        else:
            assert hidden_states.size(-1) == w1.size(2), (
                f"Hidden size mismatch {hidden_states.size(-1)} != {w1.size(2)}"
            )

        assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
        assert hidden_states.dim() == 2
        assert w1.stride(-1) == 1, "Stride of last dimension must be 1"
        assert w2.stride(-1) == 1, "Stride of last dimension must be 1"
        assert hidden_states.dtype in [
            torch.float32,
            torch.float16,
            torch.bfloat16,
            torch.float8_e4m3fn,
            torch.float8_e4m3fnuz,
            torch.int8,
        ]

        E, num_tokens, N, K, top_k_num = self.moe_problem_size(
            hidden_states, w1, w2, topk_ids
        )

        if global_num_experts == -1:
            global_num_experts = E

        # ┌------------------------  Metax Modification -------------------------┐
        config_dtype = get_config_dtype_str(
            dtype=hidden_states.dtype,
            use_int4_w4a16=self.quant_config.use_int4_w4a16,
            use_int4_w4a8=self.quant_config.use_int4_w4a8,
            use_int8_w8a8=self.quant_config.use_int8_w8a8,
            use_int8_w8a16=self.quant_config.use_int8_w8a16,
            use_fp8_w8a8=self.quant_config.use_fp8_w8a8,
            ocp_mx_scheme=self.quant_config.ocp_mx_scheme,
        )
        # └------------------------- Metax Modification -------------------------┘

        config = try_get_optimal_moe_config(
            w1.size(),
            w2.size(),
            top_k_num,
            config_dtype,
            num_tokens,
            block_shape=self.block_shape,
        )
        # /------------------------- Metax Modification --------------------------\
        stage1_config, stage2_config = initialize_staged_config(config)
        # \-----------------------------------------------------------------------/

        if hidden_states.dtype == torch.bfloat16:
            compute_type = tl.bfloat16
        elif hidden_states.dtype == torch.float16:
            compute_type = tl.float16
        elif hidden_states.dtype == torch.float32:
            compute_type = tl.float32
        elif (
            hidden_states.dtype == torch.float8_e4m3fn
            or hidden_states.dtype == torch.float8_e4m3fnuz
        ) or hidden_states.dtype == torch.int8:
            compute_type = tl.bfloat16
        else:
            raise ValueError(f"Unsupported compute_type: {hidden_states.dtype}")

        # Note that the output tensor might be in workspace1
        intermediate_cache1 = _resize_cache(workspace2, (num_tokens, top_k_num, N))
        cache2_dim = self.adjust_N_for_activation(N, activation)
        intermediate_cache2 = _resize_cache(
            workspace13, (num_tokens * top_k_num, cache2_dim)
        )
        intermediate_cache3 = _resize_cache(workspace2, (num_tokens, top_k_num, K))

        def maybe_override_stage_block_size_m(block_size_m: int) -> int:
            if (
                self.quant_config.use_int8_w8a8
                and mx_envs.MACA_VLLM_ENABLE_MCTLASS_FUSED_MOE
            ):
                kernel_m = mctlass_ops.cutlass_moe_mm_w8a8_get_kernel_m(
                    hidden_states, w1, intermediate_cache1, top_k_num
                )
                assert kernel_m > 0, (
                    "cutlass_moe_w8a8 BLOCK_SIZE_M must greater than zero."
                )
                stage1_config["BLOCK_SIZE_M"] = kernel_m
                stage2_config["BLOCK_SIZE_M"] = kernel_m
                return kernel_m

            if (
                self.quant_config.use_int4_w4a8
                and mx_envs.MACA_VLLM_ENABLE_MCTLASS_PYTHON_API
            ):
                if self.block_shape is None:
                    kernel_m = mctlass_ops.cutlass_moe_mm_w4a8_get_kernel_m_per_channel(
                        a=hidden_states,
                        b=w1,
                        c=intermediate_cache1,
                        K=K,
                        num_valid_tokens=hidden_states.size(0) * top_k_num,
                        topk=top_k_num,
                    )
                else:
                    kernel_m = mctlass_ops.mctlassEx_fused_moe_w4a8_get_kernel_m(
                        hidden_states,
                        w1.view(dtype=torch.quint4x2),
                        intermediate_cache1,
                        num_experts=w1.size(0),
                        batch_size=hidden_states.size(0),
                        N=N,
                        K=K,
                        num_valid_tokens=num_tokens,
                        topk=top_k_num,
                        group_size=self.block_shape[1],
                    )
                assert kernel_m > 0, (
                    "cutlass_fused_moe_w4a8 BLOCK_SIZE_M must greater than zero."
                )
                stage1_config["BLOCK_SIZE_M"] = kernel_m
                stage2_config["BLOCK_SIZE_M"] = kernel_m
                return kernel_m

            # bf16 moe
            if (
                hidden_states.dtype == torch.bfloat16
                and not self.quant_config.use_int4_w4a8
                and not self.quant_config.use_int4_w4a16
                and not self.quant_config.use_int8_w8a8
                and not self.quant_config.use_int8_w8a16
                and mx_envs.MACA_VLLM_ENABLE_MCTLASS_FUSED_MOE
                and mx_envs.MACA_VLLM_ENABLE_MCTLASS_PYTHON_API
            ):
                kernel_m = mctlass_ops.mctlassEx_fused_moe_bf16_get_kernel_m(
                    hidden_states,                # A
                    w1,                           # B
                    intermediate_cache1,          # C
                    global_num_experts,           # num_experts
                    hidden_states.shape[0],       # batch_size
                    N,                            # N
                    hidden_states.shape[1],       # K
                    top_k_num,                    # topk
                )
                assert kernel_m > 0, (
                    "cutlass_fused_moe_bf16 BLOCK_SIZE_M must greater than zero."
                )
                stage1_config["BLOCK_SIZE_M"] = kernel_m
                stage2_config["BLOCK_SIZE_M"] = kernel_m
                return kernel_m

            return block_size_m

        # Ensure correctness when SPLIT_K>1 (atomic_add path).
        if stage1_config.get("SPLIT_K", 1) > 1:
            intermediate_cache1.zero_()

        sorted_token_ids, expert_ids, num_tokens_post_padded = (
            _prepare_expert_assignment(
                topk_ids,
                stage1_config,
                num_tokens,
                top_k_num,
                global_num_experts,
                expert_map,
                use_int8_w8a16=self.quant_config.use_int8_w8a16,
                use_int4_w4a16=self.quant_config.use_int4_w4a16,
                use_int8_w8a8=self.quant_config.use_int8_w8a8,
                use_int4_w4a8=self.quant_config.use_int4_w4a8,
                block_shape=self.block_shape,
                block_size_m_override_getter=maybe_override_stage_block_size_m,
            )
        )

        invoke_fused_moe_triton_kernel(
            hidden_states,
            w1,
            intermediate_cache1,
            a1q_scale,
            self.w1_scale,
            topk_weights,  # topk_weights mctlass bug here: must not be None
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            False,  # mul_routed_weights
            top_k_num,
            stage1_config,
            compute_type=compute_type,
            use_fp8_w8a8=self.quant_config.use_fp8_w8a8,
            use_int8_w8a8=self.quant_config.use_int8_w8a8,
            use_int8_w8a16=self.quant_config.use_int8_w8a16,
            use_int4_w4a8=self.quant_config.use_int4_w4a8,
            use_int4_w4a16=self.quant_config.use_int4_w4a16,
            per_channel_quant=self.per_act_token_quant,
            block_shape=self.block_shape,
            B_bias=self.w1_bias,
        )

        # LoRA w13: applied to intermediate_cache1 before activation, using
        # hidden_states as the lora_a input.  moe_lora_align_block_size is
        # called once here and results reused for the w2 LoRA below.
        sorted_token_ids_lora = None
        expert_ids_lora = None
        num_tokens_post_padded_lora = None
        token_lora_mapping = None
        lora_context = self._lora_context
        if lora_context is not None:
            (
                sorted_token_ids_lora,
                expert_ids_lora,
                num_tokens_post_padded_lora,
                token_lora_mapping,
            ) = self.apply_w13_lora(
                lora_context,
                y=intermediate_cache1,
                x=hidden_states,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
                expert_map=expert_map,
                w1=w1,
                w2=w2,
                num_tokens=num_tokens,
                top_k_num=top_k_num,
            )

        self.activation(
            activation, intermediate_cache2, intermediate_cache1.view(-1, N)
        )

        a2q_scale: torch.Tensor | None = None

        qintermediate_cache2, a2q_scale = moe_kernel_quantize_input(
            intermediate_cache2,
            a2_scale,
            self.quant_dtype,
            self.per_act_token_quant,
            self.block_shape,
            quantization_emulation=self.quantization_emulation,
        )

        if stage2_config["BLOCK_SIZE_M"] != stage1_config["BLOCK_SIZE_M"]:
            sorted_token_ids, expert_ids, num_tokens_post_padded = (
                _prepare_expert_assignment(
                    topk_ids,
                    stage2_config,
                    num_tokens,
                    top_k_num,
                    global_num_experts,
                    expert_map,
                    use_int8_w8a16=self.quant_config.use_int8_w8a16,
                    use_int4_w4a16=self.quant_config.use_int4_w4a16,
                    use_int8_w8a8=self.quant_config.use_int8_w8a8,
                    use_int4_w4a8=self.quant_config.use_int4_w4a8,
                    block_shape=self.block_shape,
                )
            )

        if expert_map is not None or stage2_config.get("SPLIT_K", 1) > 1:
            intermediate_cache3.zero_()

        invoke_fused_moe_triton_kernel(
            qintermediate_cache2,
            w2,
            intermediate_cache3,
            a2q_scale,
            self.w2_scale,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            not apply_router_weight_on_input,
            1,
            stage2_config,
            compute_type=compute_type,
            use_fp8_w8a8=self.quant_config.use_fp8_w8a8,
            use_int8_w8a8=self.quant_config.use_int8_w8a8,
            use_int8_w8a16=self.quant_config.use_int8_w8a16,
            use_int4_w4a8=self.quant_config.use_int4_w4a8,
            use_int4_w4a16=self.quant_config.use_int4_w4a16,
            per_channel_quant=self.per_act_token_quant,
            block_shape=self.block_shape,
            B_bias=self.w2_bias,
        )

        # LoRA w2: applied to intermediate_cache3 before moe_sum, using the
        # unquantized intermediate_cache2 as the lora_a input.  Reuses the
        # sorted_token_ids_lora computed above.
        if lora_context is not None:
            self.apply_w2_lora(
                lora_context,
                y=intermediate_cache3,
                x=intermediate_cache2,
                topk_weights=topk_weights,
                sorted_token_ids_lora=sorted_token_ids_lora,
                expert_ids_lora=expert_ids_lora,
                num_tokens_post_padded_lora=num_tokens_post_padded_lora,
                token_lora_mapping=token_lora_mapping,
                num_tokens=num_tokens,
                w1=w1,
                w2=w2,
                top_k_num=top_k_num,
            )

        # separate function is required for MoE + LoRA
        self.moe_sum(intermediate_cache3, output)

    def moe_sum(self, input: torch.Tensor, output: torch.Tensor) -> None:
        ops.moe_sum(input, output)


class TritonWNA16Experts(TritonExperts):
    @staticmethod
    def _supports_current_device() -> bool:
        raise NotImplementedError(
            "TritonWNA16Experts is not yet used by an Oracle. "
            "This method should not be called."
        )

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        raise NotImplementedError(
            "TritonWNA16Experts is not yet used by an Oracle. "
            "This method should not be called."
        )

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        raise NotImplementedError(
            "TritonWNA16Experts is not yet used by an Oracle. "
            "This method should not be called."
        )

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        raise NotImplementedError(
            "TritonWNA16Experts is not yet used by an Oracle. "
            "This method should not be called."
        )

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        raise NotImplementedError(
            "TritonWNA16Experts is not yet used by an Oracle. "
            "This method should not be called."
        )

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
        # Check constraints.
        if self.quant_config.use_int4_w4a16:
            assert hidden_states.size(-1) // 2 == w1.size(2), "Hidden size mismatch"
        else:
            assert hidden_states.size(-1) == w1.size(2), (
                f"Hidden size mismatch {hidden_states.size(-1)} != {w1.size(2)}"
            )

        assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
        assert hidden_states.dim() == 2
        assert w1.stride(-1) == 1, "Stride of last dimension must be 1"
        assert w2.stride(-1) == 1, "Stride of last dimension must be 1"
        assert hidden_states.dtype in [
            torch.float32,
            torch.float16,
            torch.bfloat16,
            torch.float8_e4m3fn,
            torch.float8_e4m3fnuz,
        ]

        E, num_tokens, N, K, top_k_num = self.moe_problem_size(
            hidden_states, w1, w2, topk_ids
        )

        if global_num_experts == -1:
            global_num_experts = E

        # ┌------------------------  Metax Modification -------------------------┐
        config_dtype = get_config_dtype_str(
            dtype=hidden_states.dtype,
            use_int4_w4a16=self.quant_config.use_int4_w4a16,
            use_int4_w4a8=self.quant_config.use_int4_w4a8,
            use_int8_w8a8=self.quant_config.use_int8_w8a8,
            use_int8_w8a16=self.quant_config.use_int8_w8a16,
            use_fp8_w8a8=self.quant_config.use_fp8_w8a8,
            ocp_mx_scheme=self.quant_config.ocp_mx_scheme,
        )
        # └------------------------- Metax Modification -------------------------┘

        config = try_get_optimal_moe_config(
            w1.size(),
            w2.size(),
            top_k_num,
            config_dtype,
            num_tokens,
            block_shape=self.block_shape,
        )

        # /------------------------- Metax Modification --------------------------\
        stage1_config, stage2_config = initialize_staged_config(config)
        # \-----------------------------------------------------------------------/

        if hidden_states.dtype == torch.bfloat16:
            compute_type = tl.bfloat16
        elif hidden_states.dtype == torch.float16:
            compute_type = tl.float16
        elif hidden_states.dtype == torch.float32:
            compute_type = tl.float32
        elif (
            hidden_states.dtype == torch.float8_e4m3fn
            or hidden_states.dtype == torch.float8_e4m3fnuz
        ):
            compute_type = tl.bfloat16
        else:
            raise ValueError(f"Unsupported compute_type: {hidden_states.dtype}")

        # Note that the output tensor might be in workspace1
        intermediate_cache1 = _resize_cache(workspace2, (num_tokens, top_k_num, N))
        activation_out_dim = self.adjust_N_for_activation(N, activation)
        intermediate_cache2 = _resize_cache(
            workspace13, (num_tokens * top_k_num, activation_out_dim)
        )
        intermediate_cache3 = _resize_cache(workspace2, (num_tokens, top_k_num, K))

        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids,
            stage1_config["BLOCK_SIZE_M"],
            global_num_experts,
            expert_map,
        )

        invoke_fused_moe_wna16_triton_kernel(
            hidden_states,
            w1,
            intermediate_cache1,
            self.w1_scale,
            self.quant_config.w1_zp,
            None,  # topk_weights
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            False,  # mul_routed_weights
            top_k_num,
            stage1_config,
            compute_type=compute_type,
            use_int8_w8a16=self.quant_config.use_int8_w8a16,
            use_int4_w4a16=self.quant_config.use_int4_w4a16,
            block_shape=self.block_shape,
        )

        self.activation(
            activation, intermediate_cache2, intermediate_cache1.view(-1, N)
        )

        a2q_scale: torch.Tensor | None = None

        qintermediate_cache2, a2q_scale = moe_kernel_quantize_input(
            intermediate_cache2,
            a2_scale,
            self.quant_dtype,
            self.per_act_token_quant,
            self.block_shape,
        )

        if stage2_config["BLOCK_SIZE_M"] != stage1_config["BLOCK_SIZE_M"]:
            sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
                topk_ids,
                stage2_config["BLOCK_SIZE_M"],
                global_num_experts,
                expert_map,
            )

        # `workspace2` is reused and WNA16 kernels may not overwrite all elements
        # when expert_map is used; clear to avoid stale contributions.
        if expert_map is not None or stage2_config.get("SPLIT_K", 1) > 1:
            intermediate_cache3.zero_()

        invoke_fused_moe_wna16_triton_kernel(
            qintermediate_cache2,
            w2,
            intermediate_cache3,
            self.w2_scale,
            self.quant_config.w2_zp,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            not apply_router_weight_on_input,
            1,
            stage2_config,
            compute_type=compute_type,
            use_int8_w8a16=self.quant_config.use_int8_w8a16,
            use_int4_w4a16=self.quant_config.use_int4_w4a16,
            block_shape=self.block_shape,
        )

        # separate function is required for MoE + LoRA
        self.moe_sum(intermediate_cache3, output)
