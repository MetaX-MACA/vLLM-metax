# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

"""MiniMax M3 checkpoint loading for group-32 MXFP4 routed experts."""

from types import MethodType

import torch

from vllm.config import get_current_vllm_config_or_none
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
    make_mxfp4_moe_quant_config,
)
from vllm_metax import envs
from vllm_metax.registry.quant_config.mxfp4 import Mxfp4MoEMethod


def uses_minimax_m3_mxfp4() -> bool:
    config = get_current_vllm_config_or_none()
    if config is None or config.model_config is None:
        return False
    model = config.model_config
    hf = model.hf_config
    architectures = getattr(hf, "architectures", None) or ()
    if getattr(hf, "model_type", None) != "minimax_m3_vl" and not (
        "MiniMaxM3SparseForCausalLM" in architectures
        or "MiniMaxM3SparseForConditionalGeneration" in architectures
    ):
        return False
    dtype = getattr(model.hf_text_config, "expert_dtype", "fp8")
    if dtype not in ("fp4", "fp8"):
        raise ValueError(f"MiniMax M3 expert_dtype must be fp4 or fp8, got {dtype!r}")
    return dtype == "fp4"


class MiniMaxM3Mxfp4MoEMethod(Mxfp4MoEMethod):
    """Keep packed weights and verify every local checkpoint expert shard."""

    def __init__(self, moe):
        if moe.activation != MoEActivation.SWIGLUOAI_UNINTERLEAVE:
            raise ValueError("MiniMax M3 MXFP4 requires non-interleaved SwiGLU-OAI")
        if moe.has_bias or moe.is_lora_enabled:
            raise ValueError("MiniMax M3 MXFP4 does not support expert bias or LoRA")
        if moe.in_dtype != torch.bfloat16:
            raise ValueError("MiniMax M3 MXFP4 requires bfloat16 activations")
        # The W4A16 kernel requires alignment from its native block-size query.
        if not envs.MACA_VLLM_ENABLE_MCTLASS_FUSED_MOE:
            raise ValueError(
                "MiniMax M3 MXFP4 requires MACA_VLLM_ENABLE_MCTLASS_FUSED_MOE=1"
            )
        super().__init__(moe)
        self._loaded_slots: dict[str, set[tuple[int, str]]] = {}

    def create_weights(
        self,
        layer,
        num_experts,
        hidden_size,
        intermediate_size_per_partition,
        params_dtype,
        **extra_weight_attrs,
    ):
        if params_dtype != torch.bfloat16:
            raise ValueError("MiniMax M3 MXFP4 requires --dtype bfloat16")
        if hidden_size % 32 or intermediate_size_per_partition % 32:
            raise ValueError("MiniMax M3 MXFP4 TP shards must align to group 32")
        super().create_weights(
            layer,
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            params_dtype,
            **extra_weight_attrs,
        )
        # Keep the original delegates on the layer, outside tensor reload metadata.
        layer._minimax_mxfp4_weight_loaders = {}
        for name in ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale"):
            param = getattr(layer, name)
            self._loaded_slots[name] = set()
            layer._minimax_mxfp4_weight_loaders[name] = param.weight_loader
            param.weight_loader = self._strict_loader(layer, name)

    def _strict_loader(self, layer, name):
        gate_up = name.startswith("w13_")
        is_scale = name.endswith("_scale")
        loaded_slots = self._loaded_slots[name]

        def load(
            layer,
            param,
            loaded_weight,
            weight_name,
            shard_id,
            expert_id,
            return_success=False,
        ):
            if shard_id not in (("w1", "w3") if gate_up else ("w2",)):
                raise ValueError(f"{name}: invalid expert shard {shard_id!r}")
            local_id = layer._map_global_expert_id_to_local_expert_id(expert_id)
            if local_id == -1:
                return False if return_success else None
            if is_scale:
                if loaded_weight.dtype not in (torch.float8_e8m0fnu, torch.uint8):
                    raise ValueError(f"{weight_name}: expected E8M0 scale bytes")
                # Numerical conversion would corrupt the stored E8M0 exponents.
                loaded_weight = loaded_weight.view(torch.uint8)
            elif loaded_weight.dtype != torch.uint8:
                raise ValueError(f"{weight_name}: expected packed uint8 E2M1 weights")

            expected = list(param.shape[1:])
            shard_dim = 0 if gate_up else 1
            if gate_up:
                expected[0] //= 2
            expected[shard_dim] *= layer.moe_config.moe_parallel_config.tp_size
            if tuple(loaded_weight.shape) != tuple(expected):
                raise ValueError(
                    f"{weight_name}: expected checkpoint shape {tuple(expected)}, "
                    f"got {tuple(loaded_weight.shape)}"
                )
            loaded = layer._minimax_mxfp4_weight_loaders[name](
                param,
                loaded_weight,
                weight_name,
                shard_id=shard_id,
                expert_id=expert_id,
                return_success=True,
            )
            if loaded:
                loaded_slots.add((local_id, shard_id))
            return loaded if return_success else None

        # vLLM reload metadata sanitizes bound layer methods.
        return MethodType(load, layer)

    def process_weights_after_loading(self, layer):
        for name, seen in self._loaded_slots.items():
            param = getattr(layer, name)
            shards = ("w1", "w3") if name.startswith("w13_") else ("w2",)
            missing = {
                (expert, shard) for expert in range(param.shape[0]) for shard in shards
            } - seen
            if missing:
                raise ValueError(
                    f"MiniMax M3 MXFP4 incomplete {name}: "
                    f"{len(missing)} missing slots, examples={sorted(missing)[:8]}"
                )
        super().process_weights_after_loading(layer)
        for seen in self._loaded_slots.values():
            seen.clear()

    def get_fused_moe_quant_config(self, layer):
        if self.w13_precision_config is None or self.w2_precision_config is None:
            return None
        return make_mxfp4_moe_quant_config(
            mxfp4_backend=self.mxfp4_backend,
            w1_scale=self.w13_precision_config,
            w2_scale=self.w2_precision_config,
            gemm1_alpha=self.moe.swiglu_alpha,
            gemm1_beta=self.moe.swiglu_beta,
            swiglu_limit=self.moe.swiglu_limit,
            layer=layer,
        )
