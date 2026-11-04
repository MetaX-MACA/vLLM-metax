# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

from typing import Any

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
    NvFp4MoeBackend,
    make_nvfp4_moe_kernel,
)
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a4_nvfp4 import (  # noqa: E501
    CompressedTensorsW4A4Nvfp4MoEMethod,
)
from vllm.model_executor.layers.quantization.modelopt import ModelOptMxFp8Config
from vllm_metax.model_executor.layers.fused_moe.experts.minimax_m3_nvfp4 import (
    MiniMaxM3Nvfp4Experts,
    build_minimax_m3_nvfp4_quant_config,
)

logger = init_logger(__name__)


@register_quantization_config("mxfp8")
@register_quantization_config("modelopt_mxfp8")
class MacaMiniMaxM3MxFp8Config(ModelOptMxFp8Config):
    """MXFP8 linears and packed NVFP4 experts with software emulation."""

    moe_quant_algo: str | None = None
    moe_quant_format: str | None = None
    group_size: int | None = None

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant, hf_config=None):
        if (
            hf_quant_cfg
            and hf_quant_cfg.get("quant_method") == "mxfp8"
            and hf_quant_cfg.get("moe_quant_algo") == "NVFP4"
        ):
            return "modelopt_mxfp8"
        return super().override_quantization_method(hf_quant_cfg, user_quant, hf_config)

    @classmethod
    def from_config(cls, config: dict[str, Any]):
        quant_config = super().from_config(config)
        algo = config.get("moe_quant_algo")
        fmt = config.get("moe_quant_format")
        if algo is None and fmt is None:
            return quant_config
        if algo != "NVFP4" or fmt != "nvfp4-pack-quantized":
            raise ValueError(
                "MiniMax-M3 mixed MXFP8 requires moe_quant_algo='NVFP4' and "
                "moe_quant_format='nvfp4-pack-quantized'."
            )
        if config.get("group_size") != 16:
            raise ValueError("MiniMax-M3 NVFP4 requires group_size=16.")
        if config.get("weight_block_size") != [1, 32]:
            raise ValueError("MiniMax-M3 MXFP8 requires weight_block_size=[1, 32].")
        if config.get("activation_scheme", "dynamic") != "dynamic":
            raise ValueError("MiniMax-M3 MXFP8 requires dynamic activations.")
        quant_config.moe_quant_algo = algo
        quant_config.moe_quant_format = fmt
        quant_config.group_size = 16
        return quant_config

    def get_quant_method(self, layer, prefix):
        if (
            self.moe_quant_algo == "NVFP4"
            and isinstance(layer, RoutedExperts)
            and not self.is_layer_excluded(prefix)
        ):
            return MacaMiniMaxM3NvFp4MoEMethod(layer.moe_config, self)
        return super().get_quant_method(layer, prefix)


class MacaMiniMaxM3NvFp4MoEMethod(CompressedTensorsW4A4Nvfp4MoEMethod):
    """Keep the checkpoint's canonical packed layout and all independent scales."""

    def __init__(self, moe, quant_config):
        # Reuse CT parameter creation, not its NVIDIA backend selection or
        # post-load conversion (which collapses the gate/up global scales).
        FusedMoEMethodBase.__init__(self, moe)
        self.quant_config = quant_config
        self.group_size = 16
        self.use_a16 = False
        self.use_global_sf = False

    @property
    def supports_eplb(self) -> bool:
        return False

    def create_weights(
        self,
        layer,
        num_experts,
        hidden_size,
        intermediate_size_per_partition,
        params_dtype,
        **extra_weight_attrs,
    ):
        if not self.moe.is_act_and_mul:
            raise ValueError("MiniMax-M3 NVFP4 expects gate/up activation.")
        if hidden_size % 16 or intermediate_size_per_partition % 16:
            raise ValueError("MiniMax-M3 NVFP4 weight dimensions must divide by 16.")
        super().create_weights(
            layer,
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            params_dtype,
            **extra_weight_attrs,
        )
        # Legacy checkpoints omit these activation scales. Explicit values are
        # loaded over the defaults; the strict loader reports uncovered slots.
        layer.w13_input_global_scale.data.fill_(1.0)
        layer.w2_input_global_scale.data.fill_(16.0)

    def process_weights_after_loading(self, layer):
        for projection in ("w13", "w2"):
            packed = getattr(layer, f"{projection}_weight_packed")
            scale = getattr(layer, f"{projection}_weight_scale")
            if packed.dtype != torch.uint8 or scale.dtype != torch.float8_e4m3fn:
                raise ValueError(f"Invalid MiniMax-M3 {projection} packed/scale dtype.")
            if (
                packed.shape[:-1] != scale.shape[:-1]
                or packed.shape[-1] != scale.shape[-1] * 8
            ):
                raise ValueError(f"Invalid MiniMax-M3 {projection} block16 layout.")
            if not torch.isfinite(scale.float()).all() or (scale.float() < 0).any():
                raise ValueError(f"Invalid MiniMax-M3 {projection} block scales.")
            for suffix in ("weight_global_scale", "input_global_scale"):
                value = getattr(layer, f"{projection}_{suffix}")
                if not torch.isfinite(value).all() or (value <= 0).any():
                    raise ValueError(f"Invalid MiniMax-M3 {projection}_{suffix}.")
        # Rebuild after loading/reloading: the experts keep device-side scale
        # views, while the early runner configuration may predate weight loading.
        self.moe_quant_config = None
        self.moe_kernel = make_nvfp4_moe_kernel(
            moe_quant_config=self.get_fused_moe_quant_config(layer),
            moe_config=self.moe,
            experts_cls=MiniMaxM3Nvfp4Experts,
            backend=NvFp4MoeBackend.EMULATION,
            routing_tables=layer._expert_routing_tables(),
        )
        logger.info_once(
            "MiniMax-M3 uses packed NVFP4 emulation with independent gate/up "
            "scales. CUDA Graph uses the same device-side routing path."
        )

    def get_fused_moe_quant_config(self, layer):
        if self.moe_quant_config is None:
            self.moe_quant_config = build_minimax_m3_nvfp4_quant_config(
                layer,
                swiglu_alpha=self.moe.swiglu_alpha,
                swiglu_limit=self.moe.swiglu_limit,
                swiglu_beta=self.moe.swiglu_beta,
            )
        return self.moe_quant_config

    def apply(
        self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input
    ):
        if self.moe_kernel is None:
            raise RuntimeError("MiniMax-M3 NVFP4 requires post-load kernel setup.")
        return self.moe_kernel.apply(
            x,
            layer.w13_weight_packed,
            layer.w2_weight_packed,
            topk_weights,
            topk_ids,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            shared_experts=shared_experts,
            shared_experts_input=shared_experts_input,
        )
