# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Union

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.utils import replace_parameter

from vllm.model_executor.layers.fused_moe.activation import MoEActivation

from vllm.utils.import_utils import has_triton_kernels

from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
    Mxfp4MoeBackend,
    mxfp4_round_up_hidden_size_and_intermediate_size as vllm_mxfp4_round_up_hidden_size_and_intermediate_size,
)

logger = init_logger(__name__)

if has_triton_kernels():
    try:
        from triton_kernels.matmul_ogs import PrecisionConfig
    except (ImportError, AttributeError) as e:
        logger.error(
            "Failed to import Triton kernels. Please make sure your triton "
            "version is compatible. Error: %s",
            e,
        )


def backend_to_kernel_cls(
    backend: Mxfp4MoeBackend,
) -> list[type[mk.FusedMoEExperts]]:
    if backend == Mxfp4MoeBackend.TRITON:
        from vllm_metax.model_executor.layers.fused_moe.experts.triton_moe import (
            TritonExperts,
        )

        return [TritonExperts]

    else:
        raise ValueError(f"Not supported MXFP4 MoE backend: {backend.value}")


def convert_weight_to_mxfp4_moe_kernel_format(
    mxfp4_backend: Mxfp4MoeBackend,
    layer: torch.nn.Module,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    w13_weight_scale: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    w13_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
    _cache_permute_indices: dict[torch.Size, torch.Tensor] | None = None,
    activation: MoEActivation | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    Union[torch.Tensor, "PrecisionConfig"],
    Union[torch.Tensor, "PrecisionConfig"],
    torch.Tensor | None,
    torch.Tensor | None,
]:
    if mxfp4_backend == Mxfp4MoeBackend.TRITON:
        # The mctlass W4A16 GEMM consumes N-major block scales
        # (``[experts, K / 32, N]``), so transpose them here, as part of the
        # kernel-format conversion, instead of on every forward inside
        # ``mctlassEx_fused_moe_mxfp4_w4a16_gemm``. They are registered on the
        # layer so that the quantization config the experts read keeps aliasing
        # the layer's parameters.
        replace_parameter(
            layer,
            "w13_weight_scale",
            w13_weight_scale.transpose(1, 2).contiguous(),
        )
        replace_parameter(
            layer,
            "w2_weight_scale",
            w2_weight_scale.transpose(1, 2).contiguous(),
        )
        w13_weight_scale = layer.w13_weight_scale
        w2_weight_scale = layer.w2_weight_scale
        return (
            w13_weight,
            w2_weight,
            w13_weight_scale,
            w2_weight_scale,
            w13_bias,
            w2_bias,
        )

    else:
        raise ValueError(
            f"Unsupported mxfp4_backend for Mxfp4MoEMethod: {mxfp4_backend}. "
            f"Expected TRTLLM, Triton, AITER, XPU, or emulation backend."
        )


def mxfp4_round_up_hidden_size_and_intermediate_size(
    backend: Mxfp4MoeBackend,
    hidden_size: int,
    intermediate_size: int,
    activation: MoEActivation | None = None,
) -> tuple[int, int]:
    """Round up hidden_size and intermediate_size based on backend requirements."""
    if backend == Mxfp4MoeBackend.TRITON:
        return hidden_size, intermediate_size
    else:
        return vllm_mxfp4_round_up_hidden_size_and_intermediate_size(
            backend, hidden_size, intermediate_size, activation
        )
