# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

# -----------------------------------------------
# Note: Add plugin-aware scaled-MM kernel dispatch for MetaX.
#
# Affected versions: v0.21.0
# -----------------------------------------------

import importlib
from typing import Any
import torch

from vllm.model_executor.kernels.linear.scaled_mm.cutlass import (
    CutlassInt8ScaledMMLinearKernel,
    CutlassFp8BlockScaledMMKernel,
)

from vllm.platforms import PlatformEnum

from vllm import _custom_ops as ops
import vllm_metax.envs as mx_envs

from vllm.model_executor.kernels.linear import register_linear_kernel

_mctlass_modname = (
    "vllm_metax.model_executor.layers.quantization._python_api_ops"
    if mx_envs.MACA_VLLM_ENABLE_MCTLASS_PYTHON_API
    else "vllm_metax.model_executor.layers.quantization._cutlass_ops"
)
mctlass_ops: Any = importlib.import_module(_mctlass_modname)


class MctlassScaledMMLinearKernel(CutlassInt8ScaledMMLinearKernel):
    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        return True, None

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        w_q, w_s, i_s, i_zp, azp_adj = self._get_layer_params(layer)

        # ops.scaled_int8_quant supports both dynamic and static quant:
        # * dynamic, i_s is None and x_s computed from x.
        # * static, i_s is scalar and x_s is i_s.
        symmetric = azp_adj is None
        x_q, x_s, x_zp = ops.scaled_int8_quant(
            x.contiguous(), i_s, i_zp, symmetric=symmetric
        )

        if x_zp is not None:
            # Currently, static is always per-tensor and dynamic is per-token
            static = i_zp is not None
            azp = None if static else x_zp
            return mctlass_ops.cutlass_scaled_mm_azp(
                x_q,
                w_q,
                scale_a=x_s,
                scale_b=w_s,
                out_dtype=x.dtype,
                azp_adj=azp_adj,
                azp=azp,
                bias=bias,
            )
        return mctlass_ops.cutlass_scaled_mm(
            x_q, w_q, scale_a=x_s, scale_b=w_s, out_dtype=x.dtype, bias=bias
        )


register_linear_kernel(
    kernel_class=MctlassScaledMMLinearKernel,
    platform=PlatformEnum.OOT,
    kernel_type="int8",
)


class MctlassFp8BlockScaledMMKernel(CutlassFp8BlockScaledMMKernel):
    @classmethod
    def is_supported(cls, compute_capability=None):
        return True, None

    def apply_block_scaled_mm(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        As: torch.Tensor,
        Bs: torch.Tensor,
    ) -> torch.Tensor:
        out_dtype = self.config.out_dtype
        return mctlass_ops.cutlass_fp8_block_scaled_mm(A, B, As, Bs, out_dtype)


# register_linear_kernel(
#     kernel_class=MctlassFp8BlockScaledMMKernel,
#     platform=PlatformEnum.OOT,
#     kernel_type="fp8_block"
# )
