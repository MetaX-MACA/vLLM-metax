# SPDX-License-Identifier: Apache-2.0
from typing import Optional

from vllm.model_executor.layers.quantization.kernels.scaled_mm.cutlass import (
    CutlassScaledMMLinearKernel,
)
from vllm.model_executor.layers.quantization.kernels.scaled_mm.ScaledMMLinearKernel import (  # noqa: E501
    ScaledMMLinearKernel,
    ScaledMMLinearLayerConfig,
)
from vllm.platforms import PlatformEnum


class MctlassScaledMMLinearKernel(CutlassScaledMMLinearKernel):
    @classmethod
    def can_implement(cls, c: ScaledMMLinearLayerConfig) -> tuple[bool, Optional[str]]:
        return True, None

# /------------------------  Metax Modification ----------------------------\
#缩放矩阵乘内核，增添上对OOT平台的支持

_POSSIBLE_KERNELS: dict[PlatformEnum, list[type[ScaledMMLinearKernel]]] = {
    PlatformEnum.OOT: [MctlassScaledMMLinearKernel]
}
# \------------------------- Metax Modification ----------------------------/

import vllm.model_executor.layers.quantization.kernels.scaled_mm

vllm.model_executor.layers.quantization.kernels.scaled_mm._POSSIBLE_KERNELS = (
    _POSSIBLE_KERNELS
)
