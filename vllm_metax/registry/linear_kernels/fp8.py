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
    CutlassFp8BlockScaledMMKernel,
)
import vllm_metax.envs as mx_envs

from vllm.model_executor.kernels.linear import register_linear_kernel  # noqa: F401

_mctlass_modname = (
    "vllm_metax.model_executor.layers.quantization._python_api_ops"
    if mx_envs.MACA_VLLM_ENABLE_MCTLASS_PYTHON_API
    else "vllm_metax.model_executor.layers.quantization._cutlass_ops"
)
mctlass_ops: Any = importlib.import_module(_mctlass_modname)


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
