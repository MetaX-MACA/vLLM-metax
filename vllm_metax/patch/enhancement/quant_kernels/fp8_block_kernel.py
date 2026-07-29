# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

# -----------------------------------------------
# Note: Add plugin-aware scaled-MM kernel dispatch for MetaX.
#
# Affected versions: v0.24.0
# -----------------------------------------------

from vllm_metax.customized.kernels.scaled_mm import (
    MctlassFp8BlockScaledMMKernel,
)
from vllm.platforms import PlatformEnum


import vllm.model_executor.kernels.linear

vllm.model_executor.kernels.linear._POSSIBLE_FP8_BLOCK_KERNELS = {
    PlatformEnum.OOT: [MctlassFp8BlockScaledMMKernel]
}
