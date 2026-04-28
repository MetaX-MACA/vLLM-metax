# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import torch
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.int8_utils import (
    per_token_group_quant_int8,
)
from vllm.utils.math_utils import cdiv

from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe import utils
from dataclasses import dataclass
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig

logger = init_logger(__name__)


def _int8_quantize(
    A: torch.Tensor,
    A_scale: torch.Tensor | None,
    per_act_token: bool,
    block_shape: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Perform int8 quantization on the inputs.  If a block_shape
    is provided, the output will be blocked.
    """

    # If weights are per-channel (per_channel_quant=True), then
    # activations apply per-token quantization. Otherwise, assume
    # activation tensor-wise fp8/int8 quantization, dynamic or static
    if block_shape is None:
        if per_act_token:
            # ┌------------------------  Metax Modification -------------------------┐
            # A, A_scale = per_token_quant_int8(A)
            A, A_scale, _ = ops.scaled_int8_quant(A, A_scale)
            # └------------------------- Metax Modification -------------------------┘
        elif A_scale is not None:
            # Static per-tensor: use the optimized CUDA kernel
            A, A_scale, _ = ops.scaled_int8_quant(A, scale=A_scale)
        elif A_scale is None:
            # Dynamic per-tensor: compute scale then quantize via kernel
            A_scale = torch.clamp(A.abs().max() / 127.0, min=1e-10)
            A, A_scale, _ = ops.scaled_int8_quant(A, scale=A_scale)

    else:
        assert not per_act_token
        assert len(block_shape) == 2
        _, block_k = block_shape[0], block_shape[1]
        A, A_scale = per_token_group_quant_int8(A, block_k)
        assert cdiv(A.size(-1), block_k) == A_scale.size(-1)

    return A, A_scale


@dataclass
class MacaFusedMoEQuantConfig(FusedMoEQuantConfig):
    @property
    def use_int4_w4a8(self):
        return self._a1.dtype == "int8" and self._w1.dtype == "int4"


utils._int8_quantize = _int8_quantize
FusedMoEQuantConfig.use_int4_w4a8 = MacaFusedMoEQuantConfig.use_int4_w4a8
