# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

# -----------------------------------------------
# Note: Add plugin-aware scaled-MM kernel dispatch for MetaX.
#
# Affected versions: v0.21.0
# -----------------------------------------------

import torch
from vllm.platforms import current_platform

from vllm.model_executor.kernels.linear.scaled_mm.cutlass import (
    CutlassFp8BlockScaledMMKernel,
)

from vllm.model_executor.kernels.linear import register_linear_kernel  # noqa: F401

from vllm_metax.model_executor.layers.quantization import (
    _python_api_ops as mctlass_ops,
)

from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    _upcast_e8m0_to_fp32,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape


class MctlassFp8BlockScaledMMKernel(CutlassFp8BlockScaledMMKernel):
    def __init__(self, config):
        super().__init__(config)
        self.group_size = config.weight_quant_key.scale.group_shape.col
        if self.group_size == 32:
            # Same quantizer as the block-128 path: dynamic per-token-group
            # quantization with `use_ue8m0=False`, i.e. plain FP32 scales.
            # Scales stay row-major so grouped callers can view them per group
            # without a copy.
            self.quant_fp8 = QuantFP8(
                static=False,
                group_shape=GroupShape(1, 32),
                use_ue8m0=False,
            )

    @classmethod
    def can_implement(cls, config):
        weight_group = config.weight_quant_key.scale.group_shape
        if weight_group == GroupShape(32, 32):
            act_scale = config.activation_quant_key.scale
            if act_scale.static or act_scale.group_shape != GroupShape(1, 32):
                return False, "Block-32 requires dynamic 1x32 activation groups"
            if config.weight_shape[1] % 32:
                return False, "Block-32 requires K divisible by 32"
            return True, None
        return super().can_implement(config)

    @classmethod
    def is_supported(cls, compute_capability=None):
        if not current_platform.supports_fp8():
            return (
                False,
                "MCTlass block-FP8 requires MetaX ISA >= 16 (capability >= 8.9)",
            )
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)
        # The kernel takes FP32 block scales on both operands. Block-32
        # checkpoints store them UE8M0-encoded, so decode the exponent bytes
        # once here, the same way the DeepGEMM backend post-processes its block
        # scales. FP32 scales (block-128 checkpoints) are left untouched.
        params = self._get_layer_params(layer)
        weight_scale = (
            params.weight_scale
            if params.weight_scale_inv is None
            else params.weight_scale_inv
        )
        if weight_scale.dtype in (torch.uint8, torch.float8_e8m0fnu):
            weight_scale = _upcast_e8m0_to_fp32(weight_scale.data.contiguous())
            replace_parameter(
                layer,
                params.WEIGHT_SCALE
                if params.weight_scale_inv is None
                else params.WEIGHT_SCALE_INV,
                weight_scale,
            )

        # Refer to: `deepgemm_post_process_fp8_weight_block`
        # Reshape 2D weight/scale to 3D for grouped BMM (einsum):
        # wq: (g*r, d) -> (g, r, d)
        # Scale dimensions follow quant_block_shape.
        if (
            getattr(layer, "is_bmm", False)
            and layer.weight_block_size == [128, 128]
            and params.weight.ndim == 2
        ):
            groups = layer.bmm_batch_size
            rows, cols = params.weight.shape
            if groups <= 0 or rows % groups or (rows // groups) % 128:
                raise ValueError("Grouped FP8 weight rows must align to 128 per group")
            expected_shape = (rows // 128, (cols + 127) // 128)
            if weight_scale.shape != expected_shape:
                raise ValueError(
                    f"Expected grouped FP8 scale shape {expected_shape}, "
                    f"got {tuple(weight_scale.shape)}"
                )
            replace_parameter(
                layer,
                params.WEIGHT,
                params.weight.reshape(groups, rows // groups, cols),
            )
            replace_parameter(
                layer,
                params.block_scale_attr,
                weight_scale.reshape(groups, rows // groups // 128, -1),
            )

    def apply_block_scaled_mm(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        As: torch.Tensor,
        Bs: torch.Tensor,
    ) -> torch.Tensor:
        out_dtype = self.config.out_dtype
        # Both block sizes consume FP32 scales: the weight side is decoded in
        # `process_weights_after_loading` and the activation quantizer uses
        # `use_ue8m0=False`. `_python_api_ops` only enforces this for
        # group_size 32, so check both here to keep a mismatched scale from
        # being silently read as FP32.
        if As.dtype != torch.float32 or Bs.dtype != torch.float32:
            raise ValueError(
                f"Expected FP32 block scales, got {As.dtype} and {Bs.dtype}"
            )
        return mctlass_ops.cutlass_fp8_block_scaled_mm(
            A,
            B,
            As,
            Bs,
            out_dtype,
            group_size=self.group_size,
        )


# register_linear_kernel(
#     kernel_class=MctlassFp8BlockScaledMMKernel,
#     platform=PlatformEnum.OOT,
#     kernel_type="fp8_block"
# )
