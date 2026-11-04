# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3 packed NVFP4 experts using vLLM 0.30's emulation kernels.

Post-load integration (before graph capture)::

    quant = build_minimax_m3_nvfp4_quant_config(
        layer,
        swiglu_alpha=config.swiglu_alpha,
        swiglu_limit=config.swiglu_limit,
    )
    kernel = make_nvfp4_moe_kernel(
        moe_quant_config=quant,
        moe_config=layer.moe_config,
        experts_cls=MiniMaxM3Nvfp4Experts,
        backend=NvFp4MoeBackend.EMULATION,
    )

``make_nvfp4_moe_kernel`` and ``NvFp4MoeBackend`` are upstream oracle.nvfp4
APIs. Pass the ORIGINAL ``layer.w13_weight_packed`` / ``layer.w2_weight_packed``
to ``kernel.apply``; no weight conversion, renaming or upstream post-load
conversion is needed. Keep the layer/quant tensors alive. Use Standard-format
prepare/finalize (TP or naive DP/EP); batched all-to-all and EPLB are unsupported.
The parent must validate finite, positive global encode scales after loading.

Checkpoint globals are ENCODE scales: W = decode(packed) * E4M3 / global.
The builder reciprocates weight globals, but leaves input globals as encode
scales. Gate/up globals, including input globals, remain independent per expert.
Do NOT pass this quant config to the ordinary upstream emulation experts.
The upstream GEMM bitcasts block scales to E4M3, so folding globals into FP32
block scales is NOT supported. Instead, gate and up use separate GEMM launches
with the original E4M3 bytes and their own contiguous per-expert descales.

All route-dependent operations have fixed shapes and execute on the device.
Alignment, scale gathers, and workspace clearing run on EVERY graph replay;
no Python dispatch by expert occupancy or cached token assignments is used.
Only activation-sized temporaries are allocated, never dense expert weights.
"""

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    nvfp4_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.experts.nvfp4_emulation_moe import (
    Nvfp4QuantizationEmulationTritonExperts,
    invoke_fused_moe_nvfp4_emulation_kernel,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.fused_moe.utils import (
    _resize_cache,
    moe_kernel_quantize_input,
)
from vllm.triton_utils import tl

__all__ = ["MiniMaxM3Nvfp4Experts", "build_minimax_m3_nvfp4_quant_config"]

# Local, conservative tiles: avoid the much larger tuned dense/FP8 MoE tiles.
# The upstream launcher only consumes these four keys, not num_stages/num_warps.
_EMULATION_CONFIG = {
    "BLOCK_SIZE_M": 16,
    "BLOCK_SIZE_N": 32,
    "BLOCK_SIZE_K": 32,
    "GROUP_SIZE_M": 1,
}


def _check_packed_weight(packed: torch.Tensor, scale: torch.Tensor, name: str) -> None:
    if packed.ndim != 3 or packed.dtype != torch.uint8:
        raise ValueError(f"{name} must be raw uint8 [E, N, K/2] NVFP4 weights")
    if scale.dtype != torch.float8_e4m3fn or scale.ndim != 3:
        raise ValueError(f"{name} requires unswizzled E4M3 [E, N, K/16] scales")
    if (
        packed.shape[:2] != scale.shape[:2]
        or packed.size(2) != scale.size(2) * 8
        or packed.device != scale.device
        or min(packed.shape) == 0
    ):
        raise ValueError(f"Invalid {name} group16 weight/scale layout")


def build_minimax_m3_nvfp4_quant_config(
    layer: torch.nn.Module,
    *,
    swiglu_alpha: float,
    swiglu_limit: float,
    swiglu_beta: float = 1.0,
) -> FusedMoEQuantConfig:
    """Build only scale metadata; do not modify the layer or its packed weights.

    Requires CT-style ``w13/w2_weight_packed``, ``w13/w2_weight_scale``,
    ``w13/w2_weight_global_scale`` and ``w13/w2_input_global_scale`` attributes.
    Both w13 globals must have shape [E, 2], ordered gate then up; w2 globals
    must have shape [E]. All four globals must be FP32 on the weights' device.
    Values must already have been validated by the loader. Call once after
    loading, not during capture; rebuilding is required after replacing scales.

    The required alpha/limit come from the model config, not guessed defaults.
    Beta defaults to MiniMax's up-branch offset of one. The activation is
    clamp(gate) * sigmoid(alpha * clamp(gate)) * (clamp(up) + beta).
    """
    w13, w2 = layer.w13_weight_packed, layer.w2_weight_packed
    s13, s2 = layer.w13_weight_scale, layer.w2_weight_scale
    _check_packed_weight(w13, s13, "w13")
    _check_packed_weight(w2, s2, "w2")
    experts, twice_intermediate, packed_hidden = w13.shape
    if (
        w2.shape[:2] != (experts, packed_hidden * 2)
        or w2.size(2) * 4 != twice_intermediate
        or w2.device != w13.device
    ):
        raise ValueError("Expected w13 [E, 2*I, H/2] and w2 [E, H, I/2]")

    globals_ = (
        (layer.w13_weight_global_scale, (experts, 2)),
        (layer.w2_weight_global_scale, (experts,)),
        (layer.w13_input_global_scale, (experts, 2)),
        (layer.w2_input_global_scale, (experts,)),
    )
    for scale, shape in globals_:
        if (
            scale.shape != shape
            or scale.dtype != torch.float32
            or scale.device != w13.device
        ):
            raise ValueError(f"Expected FP32 encode scale {shape} on {w13.device}")

    return nvfp4_moe_quant_config(
        g1_alphas=layer.w13_weight_global_scale.reciprocal().contiguous(),
        g2_alphas=layer.w2_weight_global_scale.reciprocal().contiguous(),
        a1_gscale=layer.w13_input_global_scale,
        a2_gscale=layer.w2_input_global_scale,
        w1_scale=s13.view(torch.uint8),
        w2_scale=s2.view(torch.uint8),
        is_scale_swizzled=False,
        gemm1_alpha=swiglu_alpha,
        gemm1_beta=swiglu_beta,
        gemm1_clamp_limit=swiglu_limit,
    )


class MiniMaxM3Nvfp4Experts(Nvfp4QuantizationEmulationTritonExperts):
    """Reuse upstream workspaces, activation, reduction and packed GEMM/QDQ.

    Constructor has the exact upstream ``(moe_config, quant_config)`` ABI.
    Use the builder above: g1_alphas/a1_gscale are [E, 2], g2_alphas/a2_gscale
    are [E]. A legacy SILU activation label is accepted only with explicit
    SwiGLU-OAI parameters; it NEVER selects ordinary SiLU math here.
    """

    def __init__(
        self, moe_config: FusedMoEConfig, quant_config: FusedMoEQuantConfig
    ) -> None:
        if not self._supports_parallel_config(moe_config.moe_parallel_config):
            raise ValueError("MiniMax NVFP4 requires Standard-format prepare/finalize")
        if moe_config.has_bias or moe_config.is_lora_enabled:
            raise ValueError("MiniMax NVFP4 does not support bias or LoRA")
        if not self._supports_activation(moe_config.activation):
            raise ValueError("MiniMax NVFP4 requires non-interleaved SwiGLU-OAI")
        super().__init__(moe_config, quant_config)
        if (
            self.activation_config.clamp_limit is None
            or (quant_config.gemm1_alpha is None and moe_config.swiglu_alpha is None)
            or (quant_config.gemm1_beta is None and moe_config.swiglu_beta is None)
        ):
            raise ValueError("MiniMax NVFP4 requires explicit SwiGLU alpha/beta/limit")

        g13, g2 = quant_config.g1_alphas, quant_config.g2_alphas
        a13, a2 = quant_config.a1_gscale, quant_config.a2_gscale
        assert g13 is not None and g2 is not None
        assert a13 is not None and a2 is not None
        experts = self.w1_scale_val.size(0)
        assert g13.shape == a13.shape == (experts, 2)
        assert g2.shape == a2.shape == (experts,)
        # The upstream GEMM reads global_scale[expert] without a stride argument.
        self._weight_descales = (
            g13[:, 0].contiguous(),
            g13[:, 1].contiguous(),
            g2.contiguous(),
        )
        self._unit_scale = torch.ones(1, dtype=torch.float32, device=g13.device)

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in (MoEActivation.SILU, MoEActivation.SWIGLUOAI_UNINTERLEAVE)

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_parallel_config(config: FusedMoEParallelConfig) -> bool:
        return (
            not config.use_batched_activation_format
            and Nvfp4QuantizationEmulationTritonExperts._supports_parallel_config(
                config
            )
        )

    @property
    def a1_scale(self) -> torch.Tensor:
        return self.quant_config.a1_gscale

    def _qdq(
        self, values: torch.Tensor, encode_scale: torch.Tensor, dtype: torch.dtype
    ) -> torch.Tensor:
        # QDQ(x, g) = QDQ(x*g, 1)/g. FP32 normalization avoids an extra BF16
        # rounding before QDQ and supports a different g for every routed row.
        normalized = values.float() * encode_scale
        qdq, _ = moe_kernel_quantize_input(
            A=normalized,
            A_scale=self._unit_scale,
            quant_dtype="nvfp4",
            per_act_token_quant=False,
            quantization_emulation=True,
        )
        return (qdq / encode_scale).to(dtype)

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
        assert hidden_states.ndim == 2 and hidden_states.is_contiguous()
        assert w1.dtype == w2.dtype == torch.uint8
        assert self._supports_activation(activation)
        E, M, N, K, topk = self.moe_problem_size(hidden_states, w1, w2, topk_ids)
        assert w1.size(2) * 2 == K and w2.shape == (E, K, N // 4)
        assert K % 16 == 0 and N % 32 == 0
        assert topk_weights.shape == topk_ids.shape
        if M == 0:
            output.zero_()
            return
        if global_num_experts == -1:
            global_num_experts = E
        compute_type = {
            torch.bfloat16: tl.bfloat16,
            torch.float16: tl.float16,
            torch.float32: tl.float32,
        }[hidden_states.dtype]

        fc1 = _resize_cache(workspace2, (M, topk, N))
        activated = _resize_cache(workspace13, (M * topk, N // 2))
        fc2 = _resize_cache(workspace2, (M, topk, K))
        # Invalid/nonlocal routes may not be visited by alignment. Never reuse
        # their contents from a prior replay with different expert assignments.
        fc1.zero_()
        alignment = moe_align_block_size(
            topk_ids,
            _EMULATION_CONFIG["BLOCK_SIZE_M"],
            global_num_experts,
            expert_map,
            ignore_invalid_experts=True,
        )

        ids = topk_ids.reshape(-1).long()
        valid = (ids >= 0) & (ids < global_num_experts)
        safe_ids = ids.clamp(0, global_num_experts - 1)
        local_ids = (
            expert_map.long().index_select(0, safe_ids)
            if expert_map is not None
            else safe_ids
        )
        valid = valid & (local_ids >= 0) & (local_ids < E)
        local_ids = local_ids.clamp(0, E - 1)
        a13 = self.quant_config.a1_gscale.index_select(0, local_ids)
        a2 = self.quant_config.a2_gscale.index_select(0, local_ids).unsqueeze(1)
        a13 = torch.where(valid.unsqueeze(1), a13, 1.0)
        a2 = torch.where(valid.unsqueeze(1), a2, 1.0)
        routed_input = hidden_states.unsqueeze(1).expand(M, topk, K).reshape(-1, K)

        # Each row is already a (token, top-k slot), hence GEMM top_k=1 for
        # BOTH FC1 branches and FC2. C retains its original token-slot strides.
        for branch in range(2):
            start, end = branch * (N // 2), (branch + 1) * (N // 2)
            qdq = self._qdq(
                routed_input, a13[:, branch : branch + 1], hidden_states.dtype
            )
            invoke_fused_moe_nvfp4_emulation_kernel(
                A=qdq,
                B=w1[:, start:end],
                C=fc1[:, :, start:end],
                B_scale=self.w1_scale_val[:, start:end],
                act_global_scale=self._unit_scale,
                w_global_scale=self._weight_descales[branch],
                topk_weights=None,
                sorted_token_ids=alignment[0],
                expert_ids=alignment[1],
                num_tokens_post_padded=alignment[2],
                mul_routed_weight=False,
                top_k=1,
                config=_EMULATION_CONFIG,
                compute_type=compute_type,
            )

        self.activation(
            MoEActivation.SWIGLUOAI_UNINTERLEAVE, activated, fc1.view(-1, N)
        )
        # The prepare stage already weighted x when this flag is true. Otherwise
        # apply routing AFTER SwiGLU but BEFORE FC2 QDQ, never after GEMM2.
        fc2_input = activated.float()
        if not apply_router_weight_on_input:
            fc2_input = fc2_input * topk_weights.reshape(-1, 1).float()
        fc2_qdq = self._qdq(fc2_input, a2, hidden_states.dtype)
        fc2.zero_()  # Aliases FC1; clear only after activation has consumed FC1.
        invoke_fused_moe_nvfp4_emulation_kernel(
            A=fc2_qdq,
            B=w2,
            C=fc2,
            B_scale=self.w2_scale_val,
            act_global_scale=self._unit_scale,
            w_global_scale=self._weight_descales[2],
            topk_weights=None,
            sorted_token_ids=alignment[0],
            expert_ids=alignment[1],
            num_tokens_post_padded=alignment[2],
            mul_routed_weight=False,
            top_k=1,
            config=_EMULATION_CONFIG,
            compute_type=compute_type,
        )
        self.moe_sum(fc2, output)
