# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import torch

from fractions import Fraction
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_wna16 import (
    CompressedTensorsWNA16MoEMethod as vllm_ctm_wna16,
)
from compressed_tensors.quantization import (
    ActivationOrdering,
    QuantizationArgs,
    QuantizationStrategy,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
)
from vllm_metax.model_executor.layers.fused_moe.oracle.int_wna16 import (
    WNA16MoEBackend,
    select_wna16_moe_backend,
    make_wna16_moe_kernel,
    convert_to_wna16_moe_kernel_format,
)
from vllm.model_executor.layers.fused_moe import (
    RoutedExperts,
)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_wNa16 import (  # noqa
    WNA16_SUPPORTED_TYPES_MAP,
    WNA16_ZP_SUPPORTED_TYPES_MAP,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    QuantKey,
    ScaleDesc,
    kInt4Static32GroupScale,
    kInt4StaticGroupScale,
    kInt8StaticGroupScale,
)

from vllm.model_executor.utils import replace_parameter


# -----------------------------------------------------------
# Note: We need to keep the method name **the same** as vLLM's
#       and on Maca we don't support marlin.
# -----------------------------------------------------------
class CompressedTensorsWNA16MoEMethod(vllm_ctm_wna16):
    def __init__(
        self,
        weight_quant: QuantizationArgs,
        input_quant: QuantizationArgs | None,
        moe: FusedMoEConfig,
        layer_name: str | None = None,
    ):
        super(vllm_ctm_wna16, self).__init__(moe)
        self.weight_quant = weight_quant
        self.input_quant = input_quant
        # Extract properties from weight_quant
        self.symmetric = weight_quant.symmetric
        self.num_bits = weight_quant.num_bits
        self.packed_factor = Fraction(32, weight_quant.num_bits)
        self.strategy = weight_quant.strategy
        self.group_size = weight_quant.group_size
        self.actorder = weight_quant.actorder

        # Extract quant_type and create weight key for oracle selection
        self.quant_type = (
            WNA16_SUPPORTED_TYPES_MAP[self.num_bits]
            if self.symmetric
            else WNA16_ZP_SUPPORTED_TYPES_MAP[self.num_bits]
        )

        if self.num_bits == 4:
            if self.group_size == 32:
                scale = kInt4Static32GroupScale
            else:
                scale = kInt4StaticGroupScale
        elif self.num_bits == 8:
            scale = kInt8StaticGroupScale
        else:
            scale = ScaleDesc(
                dtype=torch.float16,
                static=True,
                group_shape=(
                    GroupShape.PER_CHANNEL
                    if self.group_size == -1
                    else GroupShape(row=1, col=self.group_size)
                ),
            )

        weight_key = QuantKey(self.quant_type, scale, symmetric=self.symmetric)

        is_actorder = self.strategy == QuantizationStrategy.GROUP and self.actorder in (
            ActivationOrdering.GROUP,
            ActivationOrdering.DYNAMIC,
        )

        # Select WNA16 MoE backend via oracle.
        self.wna16_backend, self.experts_cls = select_wna16_moe_backend(
            config=self.moe,
            weight_key=weight_key,
            quant_config=self.weight_quant,
            may_have_zp=not self.symmetric,
            may_have_bias=False,
            allow_tile_padding=not is_actorder,
        )

        assert self.wna16_backend == WNA16MoEBackend.TRITON, (
            "only support triton on wna16 moe"
        )

        # Inherited weight allocation/loading uses K-first checkpoint tensors.
        # The Triton converter transposes them to N-first after loading; this
        # checkpoint layout does not mean that the runtime uses Marlin.
        self.is_transposed = True
        self.is_marlin = False

        # channelwise is not supported by this kernel
        assert weight_quant.strategy == "group"
        # grouped actorder isn't supported by this kernel
        assert weight_quant.actorder != "group"

        # Non-Marlin WNA16 always uses bf16/fp16 inputs
        self.input_dtype = torch.bfloat16

    def _setup_kernel(self, layer: RoutedExperts):
        assert self.experts_cls is not None
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        assert self.moe_quant_config is not None

        self.moe_kernel = make_wna16_moe_kernel(
            moe_quant_config=self.moe_quant_config,
            moe_config=self.moe,
            experts_cls=self.experts_cls,
            routing_tables=layer._expert_routing_tables(),
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Process weights using the shared oracle infrastructure
        converted = convert_to_wna16_moe_kernel_format(
            backend=self.wna16_backend,
            layer=layer,
            quant_config=self.weight_quant,
            input_dtype=self.input_dtype,
            w13=layer.w13_weight_packed,
            w2=layer.w2_weight_packed,
            w13_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            w13_g_idx=layer.w13_weight_g_idx,
            w2_g_idx=layer.w2_weight_g_idx,
            w13_qzeros=getattr(layer, "w13_weight_zero_point", None),
            w2_qzeros=getattr(layer, "w2_weight_zero_point", None),
        )

        if converted is None:
            self._setup_kernel(layer)
            return

        (
            w13_qweight,
            w2_qweight,
            w13_scales,
            w2_scales,
            w13_g_idx_processed,
            w2_g_idx_processed,
            w13_g_idx_sort_indices,
            w2_g_idx_sort_indices,
            w13_qzeros,
            w2_qzeros,
            w13_input_global_scale,
            w2_input_global_scale,
            _,  # w13_bias
            _,  # w2_bias
        ) = converted

        # Replace common parameters
        replace_parameter(layer, "w13_weight_packed", w13_qweight)
        replace_parameter(layer, "w2_weight_packed", w2_qweight)
        replace_parameter(layer, "w13_weight_scale", w13_scales)
        replace_parameter(layer, "w2_weight_scale", w2_scales)

        # The quant config reads zero points from the layer, so scatter the
        # converted N-first uint8 tensors together with weights and scales.
        if not self.symmetric:
            assert w13_qzeros is not None and w2_qzeros is not None
            replace_parameter(layer, "w13_weight_zero_point", w13_qzeros)
            replace_parameter(layer, "w2_weight_zero_point", w2_qzeros)

        # Alias packed weights to w13_weight/w2_weight for the modular kernel interface
        layer.w13_weight = layer.w13_weight_packed
        layer.w2_weight = layer.w2_weight_packed

        self._setup_kernel(layer)
