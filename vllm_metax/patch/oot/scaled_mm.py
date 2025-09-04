from typing import Optional

from vllm.platforms import PlatformEnum
from vllm.model_executor.layers.quantization.kernels.scaled_mm.ScaledMMLinearKernel import (  # noqa: E501
    ScaledMMLinearKernel, ScaledMMLinearLayerConfig)
from vllm.model_executor.layers.quantization.kernels.scaled_mm.cutlass import (
    CutlassScaledMMLinearKernel)

class MctlassScaledMMLinearKernel(CutlassScaledMMLinearKernel):
    @classmethod
    def can_implement(
            cls, c: ScaledMMLinearLayerConfig) -> tuple[bool, Optional[str]]:
        return True, None

_POSSIBLE_KERNELS: dict[PlatformEnum, list[type[ScaledMMLinearKernel]]] = {
    PlatformEnum.OOT: [MctlassScaledMMLinearKernel]
}

import vllm.model_executor.layers.quantization.kernels.scaled_mm
vllm.model_executor.layers.quantization.kernels.scaled_mm._POSSIBLE_KERNELS = _POSSIBLE_KERNELS