# SPDX-License-Identifier: Apache-2.0
from vllm.model_executor.layers.quantization.kernels.mixed_precision import (
    ExllamaLinearKernel as vllm_ExllamaLinearKernel,
    MPLinearLayerConfig,
)
from vllm.platforms import current_platform

import torch


class MacaExllamaLinearKernel(vllm_ExllamaLinearKernel):
    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        if not current_platform.is_cuda_alike():
            return (
                False,
                "Exllama is only supported on CUDA and ROCm",
            )

        if c.has_g_idx and c.partition_weight_shape[0] != c.full_weight_shape[0]:
            return (
                False,
                "Act reordering currently not supported by Exllama, "
                "when the input features are partitioned across "
                "devices",
            )

        if c.partition_weight_shape[1] % (32 // c.weight_type.size_bits) != 0:
            return (
                False,
                "Output features must be a multiple of the pack "
                "factor (32 / num_bits) so that we can correctly "
                "pack the zero points",
            )

        if c.act_type not in (torch.float16, torch.bfloat16):
            return False, "Exllama only supports float16 and bfloat16 activations"

        if c.weight_type not in cls.SUPPORTED_QUANT_TYPES:
            return (
                False,
                f"Quant type ({c.weight_type}) not supported by "
                "Exllama, supported types are: "
                f"{cls.SUPPORTED_QUANT_TYPES}",
            )

        if c.full_weight_shape[0] % c.group_size != 0:
            return (
                False,
                f"Group size ({c.group_size}) does not evenly divide"
                " the number of input features "
                f"({c.full_weight_shape[0]})",
            )

        return True, None


import vllm.model_executor.layers.quantization.kernels.mixed_precision

vllm.model_executor.layers.quantization.kernels.mixed_precision._POSSIBLE_KERNELS = [
    MacaExllamaLinearKernel
]
