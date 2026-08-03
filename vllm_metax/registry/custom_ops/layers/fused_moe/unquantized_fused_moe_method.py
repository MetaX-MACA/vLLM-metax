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
        super(vllm_UnquantizedFusedMoEMethod, self).__init__(moe)
        # -------------------------------------------------
        # Here in maca we use Triton for Modular MoE kernel
        self.unquantized_backend, self.experts_cls = select_unquantized_moe_backend(
            moe_config=self.moe,
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
        )
        # `moe_kernel` is initialized to None in FusedMoEMethodBase.__init__;
        # On the first call we replace the parameter normally. On subsequent
        # calls (e.g. RL weight updates that re-trigger
        # process_weights_after_loading) the moe kernel has already been set
        # up and CUDA graphs may have captured the parameter addresses, so
        # we copy the shuffled data into the existing storage instead of
        # re-registering a new Parameter.
        is_weight_update = self.moe_kernel is not None  # type: ignore[has-type]
        replace_parameter(layer, "w13_weight", w13_new, prefer_copy=is_weight_update)
        replace_parameter(layer, "w2_weight", w2_new, prefer_copy=is_weight_update)

        if not is_weight_update:
            # Setup moe kernel only on the first call. For the unquantized
            # method, moe_quant_config carries no quantized scales -- only
            # optional w{13,2}_bias references and SwiGLU gate params. Since
            # weight updates mutate those bias tensors in place, the kernel
            # does not need to be re-built.
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
