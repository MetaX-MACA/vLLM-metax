# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import torch

from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import (
    RoutedExperts,
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
    Fp8MoeBackend,
    convert_to_fp8_moe_kernel_format,
)
from vllm_metax.model_executor.layers.fused_moe.oracle.fp8 import (
    make_fp8_moe_kernel,
    select_fp8_moe_backend,
)
from vllm.model_executor.layers.linear import (
    LinearBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    get_marlin_input_dtype,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped,
    kFp8Dynamic128Sym,
    kFp8DynamicTensorSym,
    kFp8Static128BlockSym,
    kFp8StaticTensorSym,
)
from vllm.model_executor.utils import replace_parameter

from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.fp8 import (
    Fp8Config,
    Fp8KVCacheMethod,
    Fp8LinearMethod,
    Fp8MoEMethod as vllm_Fp8MoEMethod,
)


@register_quantization_config("fp8")
class MacaFp8Config(Fp8Config):
    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> "QuantizeMethodBase | None":
        if isinstance(layer, LinearBase):
            if is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
            ):
                return UnquantizedLinearMethod()
            if not self.is_checkpoint_fp8_serialized:
                from vllm.model_executor.layers.quantization.online.fp8 import (
                    Fp8PerTensorOnlineLinearMethod,
                )

                online_method = Fp8PerTensorOnlineLinearMethod()
                online_method.marlin_input_dtype = get_marlin_input_dtype(prefix)
                return online_method
            else:
                offline_method = Fp8LinearMethod(self)
                offline_method.marlin_input_dtype = get_marlin_input_dtype(prefix)
                return offline_method
        elif isinstance(layer, RoutedExperts):
            if is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
            ):
                return UnquantizedFusedMoEMethod(layer.moe_config)
            if self.store_dtype == "mxfp4":
                from vllm.model_executor.layers.quantization.mxfp4 import (
                    Mxfp4MoEMethod,
                )

                return Mxfp4MoEMethod(layer.moe_config)
            if self.is_checkpoint_fp8_serialized:
                moe_quant_method = Fp8MoEMethod(self, layer)
            else:
                from vllm.model_executor.layers.quantization.online.fp8 import (
                    Fp8PerTensorOnlineMoEMethod,
                )

                moe_quant_method = Fp8PerTensorOnlineMoEMethod(self, layer)
            return moe_quant_method
        elif isinstance(layer, Attention):
            return Fp8KVCacheMethod(self)
        return None


class Fp8MoEMethod(vllm_Fp8MoEMethod):
    def __init__(self, quant_config: Fp8Config, layer: RoutedExperts):
        super(vllm_Fp8MoEMethod, self).__init__(layer.moe_config)
        self.quant_config = quant_config
        self.weight_block_size = self.quant_config.weight_block_size
        self.block_quant: bool = self.weight_block_size is not None
        self.weight_scale_name = (
            "weight_scale_inv" if self.block_quant else "weight_scale"
        )

        # Set weight key and activation key for kernel compatibility
        if self.block_quant:
            weight_key = kFp8Static128BlockSym
            activation_key = kFp8Dynamic128Sym
        else:
            weight_key = kFp8StaticTensorSym
            activation_key = (
                kFp8StaticTensorSym
                if self.quant_config.activation_scheme == "static"
                else kFp8DynamicTensorSym
            )

        # Select Fp8 MoE backend
        self.fp8_backend, self.experts_cls = select_fp8_moe_backend(
            config=self.moe,
            weight_key=weight_key,
            activation_key=activation_key,
            allow_vllm_cutlass=False,
        )

    def _setup_kernel(
        self,
        layer: RoutedExperts,
        w13: torch.Tensor,
        w2: torch.Tensor,
        w13_scale: torch.Tensor,
        w2_scale: torch.Tensor,
        w13_input_scale: torch.Tensor | None,
        w2_input_scale: torch.Tensor | None,
    ) -> None:
        # Shuffle weights to runtime format.
        w13, w2, w13_scale, w2_scale = convert_to_fp8_moe_kernel_format(
            fp8_backend=self.fp8_backend,
            layer=layer,
            w13=w13,
            w2=w2,
            w13_scale=w13_scale,
            w2_scale=w2_scale,
            w13_input_scale=w13_input_scale,
            w2_input_scale=w2_input_scale,
        )

        # Replace parameters with updated versions. Note that this helper
        # function ensures the replacement is compatible with RL weight reloads.
        replace_parameter(layer, "w13_weight", w13)
        replace_parameter(layer, "w2_weight", w2)
        replace_parameter(layer, f"w13_{self.weight_scale_name}", w13_scale)
        replace_parameter(layer, f"w2_{self.weight_scale_name}", w2_scale)

        # AITER backend requires weights to be marked as shuffled.
        if self.fp8_backend == Fp8MoeBackend.AITER:
            layer.w13_weight.is_shuffled = True
            layer.w2_weight.is_shuffled = True

        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        if self.moe_quant_config:
            assert self.experts_cls is not None
            self.moe_kernel = make_fp8_moe_kernel(
                moe_quant_config=self.moe_quant_config,
                moe_config=self.moe,
                fp8_backend=self.fp8_backend,
                experts_cls=self.experts_cls,
                routing_tables=layer._expert_routing_tables(),
            )
