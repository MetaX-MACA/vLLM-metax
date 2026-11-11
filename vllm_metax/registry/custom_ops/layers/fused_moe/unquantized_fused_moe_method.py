# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import torch
from typing import TYPE_CHECKING

from vllm.model_executor.layers.fused_moe import (
    UnquantizedFusedMoEMethod as vllm_UnquantizedFusedMoEMethod,
)

from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
)

from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
    UnquantizedMoeBackend,
    convert_to_unquantized_kernel_format,
)

from vllm_metax.model_executor.layers.fused_moe.oracle.unquantized import (
    make_unquantized_moe_kernel,
    select_unquantized_moe_backend,
)
from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
    SharedExperts,
)
from vllm.model_executor.utils import replace_parameter


if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts


# -----------------------------------------------------------
# Note: We need to keep the method name **the same** as vLLM's
# -----------------------------------------------------------
@vllm_UnquantizedFusedMoEMethod.register_oot
class UnquantizedFusedMoEMethod(vllm_UnquantizedFusedMoEMethod):
    def __init__(self, moe: FusedMoEConfig):
        # NOTE(metax): Select the MACA oracle instead of upstream CUDA backends.
        super(vllm_UnquantizedFusedMoEMethod, self).__init__(moe)
        self.unquantized_backend, self.experts_cls = select_unquantized_moe_backend(
            moe_config=self.moe,
        )
        # MoonEP replaces the named parameters with its [E+B] gate/down
        # split and keeps the up projection outside the parameter registry,
        # so a pre-processed (e.g. ipc_cache) load cannot reconstruct it.
        self.supports_pre_processed_weights = (
            self.unquantized_backend != UnquantizedMoeBackend.MOONEP
        )

    def _setup_kernel(
        self,
        layer: "RoutedExperts",
        w13: torch.Tensor,
        w2: torch.Tensor,
    ) -> None:
        # Shuffle weights to runtime format.
        w13_new, w2_new = convert_to_unquantized_kernel_format(
            self.unquantized_backend,
            moe_config=layer.moe_config,
            w13_weight=w13,
            w2_weight=w2,
            layer=layer,
        )
        is_weight_update = self.moe_kernel is not None  # type: ignore[has-type]
        replace_parameter(layer, "w13_weight", w13_new)
        replace_parameter(layer, "w2_weight", w2_new)

        if not is_weight_update:
            # Setup moe kernel only on the first call. For the unquantized
            # method, moe_quant_config carries no quantized scales -- only
            # optional w{13,2}_bias references and SwiGLU gate params. Since
            # weight updates mutate those bias tensors in place, the kernel
            # does not need to be re-built.
            self._init_moe_kernel(layer)

            # No-op by default. Experts that need the layer itself for setup
            # convert_to_unquantized_kernel_format cannot express override
            # it: CPU prepacks into its grouped-gemm layout and captures the
            # router config; MoonEP picks up the [E+B] weight layout for
            # prefetch and the up projection.
            assert self.moe_kernel is not None
            self.moe_kernel.fused_experts.process_weights_after_loading(layer)

    def _init_moe_kernel(self, layer: "RoutedExperts") -> None:
        """Build the MoE kernel from the layer's current (shuffled) weights."""
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        assert self.moe_quant_config is not None
        assert self.experts_cls is not None
        self.moe_kernel = make_unquantized_moe_kernel(
            quant_config=self.moe_quant_config,
            moe_config=self.moe,
            backend=self.unquantized_backend,
            experts_cls=self.experts_cls,
            routing_tables=layer._expert_routing_tables(),
        )

    def forward_oot(
        self,
        layer: "RoutedExperts",
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        return self.forward_native(
            layer,
            x,
            topk_weights,
            topk_ids,
            shared_experts,
            shared_experts_input,
        )
