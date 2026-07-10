# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# -----------------------------------------------
# Note: Override FP8 MoE kernel dispatch to use MetaX Triton experts.
#
# Affected versions: v0.21.0
# -----------------------------------------------

# import torch
# from enum import Enum
import logging
import torch
from typing import ClassVar
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
import vllm.model_executor.layers.fused_moe.oracle.fp8 as vllm_fp8
from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
    Fp8MoeBackend,
    backend_to_kernel_cls,
)
from vllm.model_executor.kernels.linear.scaled_mm import (
    Fp8BlockScaledMMLinearKernel,
)

from vllm.model_executor.kernels.linear.scaled_mm import (
    Fp8BlockScaledMMLinearKernel,
)
from vllm.model_executor.kernels import linear
from vllm.model_executor.kernels.linear import (
    _POSSIBLE_KERNELS,
    _POSSIBLE_INT8_KERNELS,
    _POSSIBLE_FP8_KERNELS,
    _POSSIBLE_FP8_BLOCK_KERNELS,
    _POSSIBLE_MXFP8_KERNELS,
    _POSSIBLE_NVFP4_KERNELS,
)
from vllm.model_executor.kernels.linear.scaled_mm.triton import (
    TritonFp8BlockScaledMMKernel,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    process_fp8_weight_block_strategy,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import PlatformEnum
from vllm.utils.torch_utils import direct_register_custom_op
import mctlassEx
from mctlassEx import ScaledGEMM


def _mctlass_scaled_gemm_impl(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    M = A.shape[0]
    N = B.shape[0]
    K = B.shape[1]
    C = torch.zeros((M, N), dtype=out_dtype, device=A.device)
    scale_a_trans = As.T.contiguous()
    scale_b_trans = Bs.T.contiguous()
    gemm = ScaledGEMM()
    gemm(
        [M, N, K],
        A,
        B,
        C,
        scale_a_trans,
        scale_b_trans,
        None,
        is_blockwise=True,
        use_fp8=True,
        is_scale_a_1d=True,
        is_scale_b_1d=False,
        scale_a_layout="m-major",
        scale_b_layout="n-major",
    )
    return C


def _mctlass_scaled_gemm_fake(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    M = A.shape[0]
    N = B.shape[0]
    return torch.empty((M, N), dtype=out_dtype, device=A.device)


direct_register_custom_op(
    op_name="mctlass_scaled_gemm",
    op_func=_mctlass_scaled_gemm_impl,
    fake_impl=_mctlass_scaled_gemm_fake,
)


def maca_backend_to_kernel_cls(
    backend: Fp8MoeBackend,
) -> list[type[mk.FusedMoEExperts]]:
    kernels = backend_to_kernel_cls(backend)
    if backend == Fp8MoeBackend.TRITON:
        # ┌------------------------  Metax Modification -------------------------┐
        from vllm_metax.model_executor.layers.fused_moe.experts.triton_moe import (
            TritonExperts as mx_TritonExperts,
        )

        kernels = [mx_TritonExperts]
        # └------------------------- Metax Modification -------------------------┘
    return kernels


vllm_fp8.backend_to_kernel_cls = maca_backend_to_kernel_cls


class MACAFp8BlockScaledMMKernel(Fp8BlockScaledMMLinearKernel):
    # apply_input_quant: ClassVar[bool] = True
    def is_supported(cls, compute_capability=None):
        return True, None

    def apply_block_scaled_mm(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        As: torch.Tensor,
        Bs: torch.Tensor,
    ) -> torch.Tensor:
        return torch.ops.vllm.w8a8_triton_block_scaled_mm_func(
            A,
            B,
            As,
            Bs,
            list(self.weight_group_shape),
            self.config.out_dtype,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module):
        params = self._get_layer_params(layer)
        # Fp8LinearMethod registered weight scale
        # buffer as weight_scale_inv unlike compressed tensors.
        weight_scale = (
            params.weight_scale
            if params.weight_scale_inv is None
            else params.weight_scale_inv
        )
        scale_attr_name = (
            params.WEIGHT_SCALE
            if params.weight_scale_inv is None
            else params.WEIGHT_SCALE_INV
        )
        new_weight, new_weight_scale = process_fp8_weight_block_strategy(
            params.weight,
            weight_scale,
        )

        replace_parameter(layer, params.WEIGHT, new_weight.data)
        replace_parameter(layer, scale_attr_name, new_weight_scale.data)
        # Just placeholder for workaround some bug for CG dyno, cause FP8Params will getattr all the parameters
        # No one use those
        replace_parameter(layer, params.WEIGHT_SCALE, None)
        replace_parameter(layer, params.INPUT_SCALE_UB, None)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        out_dtype = self.config.out_dtype
        params = self._get_layer_params(layer)
        weight = params.weight
        weight_scale = params.weight_scale_inv
        input_scale = params.input_scale
        scale_up = None  # params.input_scale_ub

        # View input as 2D matrix for fp8 methods
        input_2d = x.view(-1, x.shape[-1])
        output_shape = [*x.shape[:-1], weight.shape[0]]

        if self.apply_input_quant:
            q_input, input_scale = self.quant_fp8(
                input_2d, input_scale, scale_up, use_triton=self.use_triton
            )
        else:
            q_input = input_2d
            # Provide a concrete placeholder so apply_block_scaled_mm args are
            # always Tensors. Subclasses with apply_input_quant=False must not
            # use As in apply_block_scaled_mm.
            input_scale = (
                input_scale if input_scale is not None else input_2d.new_ones(1)
            )

        output = self.apply_block_scaled_mm(
            A=q_input,
            B=weight,
            As=input_scale,
            Bs=weight_scale,
        )

        if bias is not None:
            output = output + bias
        return output.to(dtype=out_dtype).view(*output_shape)


def maca_register_linear_kernel(
    kernel_class: type,
    platform: PlatformEnum,
    kernel_type: str = "mp",
) -> None:
    """
    Register a new linear kernel class to be considered in kernel selection.

    Args:
        kernel_class (type): The kernel class to register.
        platform (PlatformEnum): The platform for which this kernel is applicable.
        kernel_type (str): The type of the kernel, either "mp", "int8", or "fp8".
            Defaults to "mp".

    Raises:
        ValueError: If the kernel_type is not recognized.
    """
    if kernel_type == "mp":
        if platform not in _POSSIBLE_KERNELS:
            _POSSIBLE_KERNELS[platform] = []
        _POSSIBLE_KERNELS[platform].append(kernel_class)
    elif kernel_type == "int8":
        if platform not in _POSSIBLE_INT8_KERNELS:
            _POSSIBLE_INT8_KERNELS[platform] = []
        _POSSIBLE_INT8_KERNELS[platform].append(kernel_class)
    elif kernel_type == "fp8":
        if platform not in _POSSIBLE_FP8_KERNELS:
            _POSSIBLE_FP8_KERNELS[platform] = []
        _POSSIBLE_FP8_KERNELS[platform].append(kernel_class)
    # ┌------------------------  Metax Modification -------------------------┐
    elif kernel_type == "fp8_block":
        if platform not in _POSSIBLE_FP8_BLOCK_KERNELS:
            _POSSIBLE_FP8_BLOCK_KERNELS[platform] = []
        _POSSIBLE_FP8_BLOCK_KERNELS[platform].append(kernel_class)
    # └------------------------- Metax Modification -------------------------┘
    elif kernel_type == "mxfp8":
        if platform not in _POSSIBLE_MXFP8_KERNELS:
            _POSSIBLE_MXFP8_KERNELS[platform] = []
        _POSSIBLE_MXFP8_KERNELS[platform].append(kernel_class)
    elif kernel_type == "nvfp4":
        if platform not in _POSSIBLE_NVFP4_KERNELS:
            _POSSIBLE_NVFP4_KERNELS[platform] = []
        _POSSIBLE_NVFP4_KERNELS[platform].append(kernel_class)
    else:
        raise ValueError(f"Unrecognized kernel type: {kernel_type}")


from vllm.model_executor.kernels.linear.scaled_mm.BlockScaledMMLinearKernel import (
    Fp8BlockScaledMMLinearKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import (
    FP8ScaledMMLinearLayerConfig,
)
from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    CUTLASS_BLOCK_FP8_SUPPORTED,
)


class MACACutlassFp8BlockScaledMMKernel(Fp8BlockScaledMMLinearKernel):
    def __init__(self, config: FP8ScaledMMLinearLayerConfig) -> None:
        super().__init__(config)
        act_scale_descriptor = config.activation_quant_key.scale
        self.quant_fp8 = QuantFP8(
            static=act_scale_descriptor.static,
            group_shape=act_scale_descriptor.group_shape,
            num_token_padding=self.get_output_padding(),
            use_ue8m0=False,
            column_major_scales=True,
        )

    @classmethod
    def is_supported(cls, compute_capability=None):
        return True, None

    @classmethod
    def can_implement(cls, config: FP8ScaledMMLinearLayerConfig):
        can_implement_base, reason = super().can_implement(config)
        if not can_implement_base:
            return can_implement_base, reason

        act_quant_desc = config.activation_quant_key.scale
        if act_quant_desc.group_shape != GroupShape(1, 128):
            return (
                False,
                "Supports only dynamic per token group activation "
                "quantization with group_shape=(1,128).",
            )
        return True, None

    def apply_block_scaled_mm(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        As: torch.Tensor,
        Bs: torch.Tensor,
    ) -> torch.Tensor:
        out_dtype = self.config.out_dtype
        return torch.ops.vllm.mctlass_scaled_gemm(A, B, As, Bs, out_dtype)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        out_dtype = self.config.out_dtype
        params = self._get_layer_params(layer)
        weight = params.weight
        weight_scale = params.weight_scale_inv
        input_scale = params.input_scale
        scale_up = None  # params.input_scale_ub

        # View input as 2D matrix for fp8 methods
        input_2d = x.view(-1, x.shape[-1])
        output_shape = [*x.shape[:-1], weight.shape[0]]

        if self.apply_input_quant:
            q_input, input_scale = self.quant_fp8(
                input_2d, input_scale, scale_up, use_triton=self.use_triton
            )
        else:
            q_input = input_2d
            # Provide a concrete placeholder so apply_block_scaled_mm args are
            # always Tensors. Subclasses with apply_input_quant=False must not
            # use As in apply_block_scaled_mm.
            input_scale = (
                input_scale if input_scale is not None else input_2d.new_ones(1)
            )

        output = self.apply_block_scaled_mm(
            A=q_input,
            B=weight,
            As=input_scale,
            Bs=weight_scale,
        )

        if bias is not None:
            output = output + bias
        return output.to(dtype=out_dtype).view(*output_shape)

    def process_weights_after_loading(self, layer: torch.nn.Module):
        params = self._get_layer_params(layer)
        # Fp8LinearMethod registered weight scale
        # buffer as weight_scale_inv unlike compressed tensors.
        weight_scale = (
            params.weight_scale
            if params.weight_scale_inv is None
            else params.weight_scale_inv
        )
        scale_attr_name = (
            params.WEIGHT_SCALE
            if params.weight_scale_inv is None
            else params.WEIGHT_SCALE_INV
        )
        new_weight, new_weight_scale = process_fp8_weight_block_strategy(
            params.weight,
            weight_scale,
        )

        replace_parameter(layer, params.WEIGHT, new_weight.data)
        replace_parameter(layer, scale_attr_name, new_weight_scale.data)
        # Just placeholder for workaround some bug for CG dyno, cause FP8Params will getattr all the parameters
        # No one use those
        replace_parameter(layer, params.WEIGHT_SCALE, None)
        replace_parameter(layer, params.INPUT_SCALE_UB, None)


maca_register_linear_kernel(
    kernel_class=MACACutlassFp8BlockScaledMMKernel,
    platform=PlatformEnum.OOT,
    kernel_type="fp8_block",
)
maca_register_linear_kernel(
    kernel_class=MACAFp8BlockScaledMMKernel,
    platform=PlatformEnum.OOT,
    kernel_type="fp8_block",
)
maca_register_linear_kernel(
    kernel_class=TritonFp8BlockScaledMMKernel,
    platform=PlatformEnum.OOT,
    kernel_type="fp8_block",
)

linear.register_linear_kernel = maca_register_linear_kernel
