# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.model_executor.layers.quantization.mxfp4 import (
    Mxfp4Config,
    Mxfp4MoEMethod as _vllm_Mxfp4MoEMethod,
)
from vllm.model_executor.layers.fused_moe import (
    FusedMoEConfig,
    FusedMoEMethodBase,
    RoutedExperts,
)
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.logger import init_logger
import torch
from vllm.model_executor.layers.fused_moe import modular_kernel as mk
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
    TRITON_BACKENDS,
)
from vllm_metax.model_executor.layers.fused_moe.oracle.mxfp4 import (
    convert_weight_to_mxfp4_moe_kernel_format,
    select_deepseek_v4_mxfp4_moe_backend,
    make_mxfp4_moe_kernel,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
)
from vllm.model_executor.utils import replace_parameter

logger = init_logger(__name__)


@register_quantization_config("mxfp4")
class MacaMxfp4Config(Mxfp4Config):
    def _make_moe_method(self, moe: FusedMoEConfig) -> FusedMoEMethodBase:
        """MoE method for RoutedExperts. Subclasses override to pick a
        checkpoint-specific kernel family."""
        return Mxfp4MoEMethod(moe)


class Mxfp4MoEMethod(_vllm_Mxfp4MoEMethod):
    def __init__(self, moe: FusedMoEConfig, activation_key: QuantKey | None = None):
        super(_vllm_Mxfp4MoEMethod, self).__init__(moe)
        self.weight_dtype = "mxfp4"
        # Checkpoint-declared activation quantization. Only backends that pick
        # their activation format from the layer (Humming) consume it.
        self.activation_key = activation_key
        self.is_k3_situ_aiter = False
        self.is_k3_situ_int4_gfx942 = False
        self.mxfp4_backend, self.experts_cls = select_deepseek_v4_mxfp4_moe_backend(moe)

        self.max_capture_size = moe.max_capture_size

        self._cache_permute_indices: dict[torch.Size, torch.Tensor] = {}
        self.moe_kernel: mk.FusedMoEKernel | None = None

        # Used for triton kernel precision configs
        self.w13_precision_config = None
        self.w2_precision_config = None

    def _setup_kernel(
        self,
        layer: RoutedExperts,
        w13: torch.Tensor,
        w2: torch.Tensor,
        w13_scale: torch.Tensor,
        w2_scale: torch.Tensor,
        w13_bias: torch.Tensor | None = None,
        w2_bias: torch.Tensor | None = None,
    ) -> None:
        num_experts = self.num_experts
        intermediate_size = self.intermediate_size
        hidden_size = self.hidden_size
        sf_block_size = 32

        # Shape assertions — skipped for SITU since its kernel handles native
        # (non-256-aligned) intermediate sizes without prior round-up.
        from vllm.model_executor.layers.fused_moe.activation import MoEActivation

        if self.moe.activation != MoEActivation.SITU:
            assert (
                w13.dim() == 3
                and w13.shape[0] == num_experts
                and w13.shape[1] == intermediate_size * self.moe.w13_num_shards
                and w13.shape[2] == hidden_size // 2
            )
            assert (
                w13_scale.dim() == 3
                and w13_scale.shape[0] == num_experts
                and w13_scale.shape[1] == intermediate_size * self.moe.w13_num_shards
                and w13_scale.shape[2] == hidden_size // sf_block_size
            )
            assert (
                w2.dim() == 3
                and w2.shape[0] == num_experts
                and w2.shape[1] == hidden_size
                and w2.shape[2] == intermediate_size // 2
            )
            assert (
                w2_scale.dim() == 3
                and w2_scale.shape[1] == hidden_size
                and w2_scale.shape[2] == intermediate_size // sf_block_size
            )
            if w13_bias is not None:
                assert (
                    w13_bias.dim() == 2
                    and w13_bias.shape[0] == num_experts
                    and w13_bias.shape[1] == intermediate_size * self.moe.w13_num_shards
                )
            if w2_bias is not None:
                assert (
                    w2_bias.dim() == 2
                    and w2_bias.shape[0] == num_experts
                    and w2_bias.shape[1] == hidden_size
                )

        # Convert weights to kernel format
        w13, w2, w13_scale, w2_scale, w13_bias, w2_bias = (
            convert_weight_to_mxfp4_moe_kernel_format(
                mxfp4_backend=self.mxfp4_backend,
                layer=layer,
                w13_weight=w13,
                w2_weight=w2,
                w13_weight_scale=w13_scale,
                w2_weight_scale=w2_scale,
                w13_bias=w13_bias,
                w2_bias=w2_bias,
                _cache_permute_indices=self._cache_permute_indices,
                activation=self.moe.activation,
            )
        )

        # For TRITON backends, weights are wrapped tensors from triton_kernels
        # that don't support .detach(). Manually assign parameters.

        uses_triton_weight_format = self.mxfp4_backend in TRITON_BACKENDS
        assert uses_triton_weight_format, "Expected TRITON backend for MXFP4"

        layer.w13_weight = w13
        layer.w2_weight = w2
        self.w13_precision_config = w13_scale
        self.w2_precision_config = w2_scale

        if w13_bias is not None and w2_bias is not None:
            replace_parameter(layer, "w13_bias", w13_bias)
            replace_parameter(layer, "w2_bias", w2_bias)

        # Build quant config
        self._build_moe_kernel(layer)

    def _build_moe_kernel(self, layer: RoutedExperts) -> None:
        """Build the modular MoE kernel from the (already in-format) weights."""
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        if self.moe_quant_config is not None and self.experts_cls is not None:
            self.moe_kernel = make_mxfp4_moe_kernel(
                moe_quant_config=self.moe_quant_config,
                moe_config=self.moe,
                mxfp4_backend=self.mxfp4_backend,
                experts_cls=self.experts_cls,
                routing_tables=layer._expert_routing_tables(),
            )
            self.moe_kernel.fused_experts.process_weights_after_loading(layer)
