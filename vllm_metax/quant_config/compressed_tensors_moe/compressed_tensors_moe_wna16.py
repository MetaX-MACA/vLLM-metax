# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import torch
from vllm.model_executor.layers.fused_moe import FusedMoE

from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_wna16 import (
    CompressedTensorsWNA16MoEMethod as vllm_ctm_wna16,
)

from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_wna16_marlin import (
    CompressedTensorsWNA16MarlinMoEMethod as vllm_ctm_wna16_marlin,  # noqa: F401
)


import vllm.model_executor.layers.fused_moe.modular_kernel as mk


# -----------------------------------------------------------
# Note: We need to keep the method name **the same** as vLLM's
# -----------------------------------------------------------
class CompressedTensorsWNA16MoEMethod(vllm_ctm_wna16):
    def select_gemm_impl(
        self,
        prepare_finalize: mk.FusedMoEPrepareAndFinalizeModular,
        layer: torch.nn.Module,
    ) -> mk.FusedMoEExpertsModular:
        if self.moe.is_lora_enabled:
            assert self.moe_quant_config is not None
            from vllm.triton_utils import HAS_TRITON

            if HAS_TRITON:
                # use Metax's TritonWNA16Experts
                from vllm_metax.utils.fused_moe import get_triton_experts_cls

                TritonWNA16Experts = get_triton_experts_cls()

                layer.w13_weight = layer.w13_weight_packed
                layer.w2_weight = layer.w2_weight_packed
                return TritonWNA16Experts(
                    moe_config=self.moe, quant_config=self.moe_quant_config
                )
            else:
                raise NotImplementedError(
                    "TritonExperts requires Triton. "
                    "Install triton or disable LoRA for MoE."
                )

        raise NotImplementedError

    def apply(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # here we use Metax's `fused_experts`
        from vllm_metax.utils.fused_moe import get_fused_experts_fn

        fused_experts = get_fused_experts_fn()
        return fused_experts(
            x,
            layer.w13_weight_packed,
            layer.w2_weight_packed,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=not self.moe.disable_inplace,
            activation=layer.activation,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            quant_config=self.moe_quant_config,
        )
