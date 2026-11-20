# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import torch
from vllm.logger import init_logger
import vllm.envs as envs
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import (
    RoutedExperts,
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
    convert_to_fp8_moe_kernel_format,
)
from vllm_metax.model_executor.layers.fused_moe.oracle.fp8 import (
    make_fp8_moe_kernel,
    resolve_fp8_moe_weight_block_shape,
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
    GroupShape,
    create_fp8_quant_key,
    is_layer_skipped,
    kFp8Dynamic128Sym,
    kFp8DynamicTensorSym,
    kFp8DynamicTokenSym,
    kFp8StaticTensorSym,
)
from vllm.model_executor.utils import replace_parameter

from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.fp8 import (
    Fp8Config,
    Fp8KVCacheMethod,
    Fp8LinearMethod as vllm_Fp8LinearMethod,
    Fp8MoEMethod as vllm_Fp8MoEMethod,
)
from vllm.model_executor.layers.linear import register_weight_loader_v2_supported_method

logger = init_logger(__name__)


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
                match_mode=self.ignored_layers_match_mode,
            ):
                return UnquantizedLinearMethod()
            offline_method = vllm_Fp8LinearMethod(self)
            offline_method.marlin_input_dtype = get_marlin_input_dtype(prefix)
            return offline_method
        elif isinstance(layer, RoutedExperts):
            if is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
                match_mode=self.ignored_layers_match_mode,
            ):
                return UnquantizedFusedMoEMethod(layer.moe_config)
            from vllm_metax.registry.quant_config.minimax_m3_mxfp4 import (
                MiniMaxM3Mxfp4MoEMethod,
                uses_minimax_m3_mxfp4,
            )

            if uses_minimax_m3_mxfp4():
                return MiniMaxM3Mxfp4MoEMethod(layer.moe_config)
            if self.store_dtype == "mxfp4":
                from vllm.model_executor.layers.quantization.mxfp4 import (
                    Mxfp4MoEMethod,
                )

                return Mxfp4MoEMethod(layer.moe_config)
            return Fp8MoEMethod(self, layer)
        elif isinstance(layer, Attention):
            return Fp8KVCacheMethod(self)
        return None


class Fp8MoEMethod(vllm_Fp8MoEMethod):
    def __init__(self, quant_config: Fp8Config, layer: RoutedExperts):
        """the same as upstream __init__ and maca select its own fp8 moe_backend"""
        super(vllm_Fp8MoEMethod, self).__init__(layer.moe_config)
        self.quant_config = quant_config
        self.weight_block_size = self.quant_config.weight_block_size
        self.block_quant: bool = self.weight_block_size is not None
        self.weight_scale_name = (
            "weight_scale_inv" if self.block_quant else "weight_scale"
        )

        self.weight_scale_refine: tuple[int, int] | None = None
        self.moe_block_shape = self.weight_block_size

        # Set weight key and activation key for kernel compatibility
        if self.block_quant:
            assert self.weight_block_size is not None
            # Adapt the checkpoint block shape to TP sharding before
            # building the weight key.
            self.moe_block_shape, self.weight_scale_refine = (
                resolve_fp8_moe_weight_block_shape(
                    self.moe,
                    self.weight_block_size,
                    kFp8Dynamic128Sym,
                    self.quant_config.is_checkpoint_fp8_serialized,
                )
            )
            weight_key = create_fp8_quant_key(
                static=True, group_shape=GroupShape(*self.moe_block_shape)
            )
            activation_key = kFp8Dynamic128Sym
        else:
            weight_key = kFp8StaticTensorSym
            if self.quant_config.activation_scheme == "static":
                activation_key = kFp8StaticTensorSym
            elif envs.VLLM_BATCH_INVARIANT:
                # A per-tensor dynamic scale is the max over every token in
                # the batch, so a request's quantized activations would depend
                # on the requests it is batched with.
                activation_key = kFp8DynamicTokenSym
            else:
                activation_key = kFp8DynamicTensorSym
        self.per_act_token_quant = activation_key == kFp8DynamicTokenSym

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

        self._init_moe_kernel(layer)

    def _init_moe_kernel(self, layer: RoutedExperts) -> None:
        """Build the MoE kernel from the layer's current (converted) weights."""
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        assert self.moe_quant_config is not None
        assert self.experts_cls is not None
        self.moe_kernel = make_fp8_moe_kernel(
            moe_quant_config=self.moe_quant_config,
            moe_config=self.moe,
            fp8_backend=self.fp8_backend,
            experts_cls=self.experts_cls,
            routing_tables=layer._expert_routing_tables(),
        )


@register_weight_loader_v2_supported_method
class MctlassFp8LinearMethod(vllm_Fp8LinearMethod):
    """Keep V4.1's weight_scale name with the standard block-FP8 loader."""

    def create_weights(self, layer, *args, **kwargs):
        weight_loader = kwargs["weight_loader"]

        def load_weight(param, loaded_weight, *loader_args, **loader_kwargs):
            if param.dtype == torch.float8_e8m0fnu:
                if loaded_weight.dtype not in (torch.uint8, torch.float8_e8m0fnu):
                    raise ValueError("Expected UE8M0 checkpoint scales")
                # Both checkpoint representations contain the same exponent bytes.
                loaded_weight = loaded_weight.view(param.dtype)
            return weight_loader(param, loaded_weight, *loader_args, **loader_kwargs)

        kwargs["weight_loader"] = load_weight
        super().create_weights(layer, *args, **kwargs)
        scale = layer.weight_scale_inv
        del layer.weight_scale_inv
        layer.register_parameter("weight_scale", scale)
