# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
from typing import Optional

from vllm.model_executor.layers.quantization.kernels.mixed_precision.MPLinearKernel import (  # noqa: E501
    MPLinearKernel,
)

from vllm_metax.quant_config.exllama import MacaExllamaLinearKernel

import vllm.model_executor.layers.quantization.kernels.mixed_precision

# in priority/performance order (when available)
vllm.model_executor.layers.quantization.kernels.mixed_precision._POSSIBLE_KERNELS = [
    MacaExllamaLinearKernel,
]
