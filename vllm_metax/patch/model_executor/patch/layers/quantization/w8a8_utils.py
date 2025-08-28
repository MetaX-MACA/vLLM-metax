# SPDX-License-Identifier: Apache-2.0

import vllm



def cutlass_fp8_supported() -> bool:
    return False

_CUTLASS_FP8_SUPPORTED = cutlass_fp8_supported()

from vllm.model_executor.layers.quantization.utils import w8a8_utils

vllm.model_executor.layers.quantization.utils.w8a8_utils.cutlass_fp8_supported = cutlass_fp8_supported
vllm.model_executor.layers.quantization.utils.w8a8_utils.CUTLASS_FP8_SUPPORTED = _CUTLASS_FP8_SUPPORTED




