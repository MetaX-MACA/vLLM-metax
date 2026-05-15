# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import torch
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
    int8_w8a8_moe_quant_config,
)

from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationStrategy,
)

from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w8a8_int8 import (
    CompressedTensorsW8A8Int8MoEMethod as vllm_ctm_w8a8_int8,
)

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kInt8DynamicTokenSym,
    kInt8StaticChannelSym,
)

import vllm.model_executor.layers.fused_moe.modular_kernel as mk

from vllm.model_executor.layers.fused_moe.oracle.int8 import (
    Int8MoeBackend,
)


def backend_to_kernel_cls(
    backend: Int8MoeBackend,
) -> type[mk.FusedMoEExperts]:
    if backend == Int8MoeBackend.TRITON:
        from vllm_metax.model_executor.layers.fused_moe.fused_moe import (
            TritonExperts,
        )

        return TritonExperts

    else:
        raise ValueError(f"Unknown Int8 MoE backend: {backend.value}")


def select_int8_moe_backend(
    config: FusedMoEConfig,
    weight_key: QuantKey | None = kInt8StaticChannelSym,
    activation_key: QuantKey | None = kInt8DynamicTokenSym,
) -> tuple[Int8MoeBackend, type[mk.FusedMoEExperts]]:
    if config.is_lora_enabled:
        return Int8MoeBackend.TRITON, backend_to_kernel_cls(Int8MoeBackend.TRITON)

    requested_backend = Int8MoeBackend.TRITON
    activation_format = (
        mk.FusedMoEActivationFormat.BatchedExperts
        if config.moe_parallel_config.use_batched_activation_format
        else mk.FusedMoEActivationFormat.Standard
    )

    k_cls = backend_to_kernel_cls(requested_backend)
    supported, reason = k_cls.is_supported_config(
        k_cls, config, weight_key, activation_key, activation_format
    )

    assert supported, (
        f"Requested Int8 MoE backend {requested_backend.value} does not support the given config. Reason: {reason}"
    )

    return requested_backend, k_cls


def make_int8_moe_quant_config(
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    a1_scale: torch.Tensor | None = None,
    a2_scale: torch.Tensor | None = None,
    per_act_token_quant: bool = False,
) -> FusedMoEQuantConfig:
    # assert (a1_scale is None and a2_scale is None) or (
    #     a1_scale is not None and a2_scale is not None
    # ), "a1_scale and a2_scale must both be provided or both be None"

    # if a1_scale is None or a2_scale is None:
    #     return int8_w8a16_moe_quant_config(
    #         w1_scale=w1_scale,
    #         w2_scale=w2_scale,
    #         w1_zp=None,
    #         w2_zp=None,
    #     )

    # remove the int8_w8a16 logic since dynamic per token int8_w8a8
    # has no a1/a2 scale, and we want to avoid confusion
    return int8_w8a8_moe_quant_config(
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        per_act_token_quant=per_act_token_quant,
    )


# -----------------------------------------------------------
# Note: We need to keep the method name **the same** as vLLM's
# -----------------------------------------------------------
class CompressedTensorsW8A8Int8MoEMethod(vllm_ctm_w8a8_int8):
    def __init__(
        self,
        weight_quant: QuantizationArgs,
        input_quant: QuantizationArgs,
        moe: FusedMoEConfig,
        layer_name: str | None = None,
    ):
        super(vllm_ctm_w8a8_int8, self).__init__(moe)
        self.weight_quant = weight_quant
        self.input_quant = input_quant

        per_channel = (
            self.weight_quant.strategy == QuantizationStrategy.CHANNEL
            and self.input_quant.strategy == QuantizationStrategy.TOKEN
        )
        if not per_channel:
            raise ValueError(
                "For INT8 Fused MoE layers, we require channelwise, "
                "dynamic per token quantization. Found "
                f"{self.weight_quant}, {self.input_quant}"
            )

        self.static_input_scales = not self.input_quant.dynamic
        if self.static_input_scales:
            raise ValueError(
                "For INT8 Fused MoE layers, we require channelwise, "
                "dynamic per token quantization. Found static input scales."
            )

        # Select Int8 MoE backend.
        self.int8_backend, self.experts_cls = select_int8_moe_backend(
            config=self.moe,
            weight_key=kInt8StaticChannelSym,
            activation_key=kInt8DynamicTokenSym,
        )

    def get_fused_moe_quant_config(self, layer: torch.nn.Module) -> FusedMoEQuantConfig:
        return make_int8_moe_quant_config(
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            a1_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
            per_act_token_quant=True,
        )

    def apply(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        assert not self.is_monolithic
        assert self.moe_kernel is not None
        return self.moe_kernel.apply(
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            shared_experts_input=shared_experts_input,
        )
