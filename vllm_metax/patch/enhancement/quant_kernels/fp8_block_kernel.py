# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#
# -----------------------------------------------------------------------------
# Note: Add plugin-aware scaled-MM kernel dispatch for MetaX.
#
# Affected versions: vLLM 0.29.1.dev0 (98dff2a81d), verified 2026-09-17.
#
# Remove at: Upstream register_linear_kernel supports fp8_block, allowing this entry to
#     move to the normal MetaX kernel registry.
# -----------------------------------------------------------------------------

from vllm_metax.registry.linear_kernels.fp8 import (
    MctlassFp8BlockScaledMMKernel,
)
from vllm.platforms import PlatformEnum


import vllm.model_executor.kernels.linear

vllm.model_executor.kernels.linear._POSSIBLE_FP8_BLOCK_KERNELS.setdefault(
    PlatformEnum.OOT, []
).append(MctlassFp8BlockScaledMMKernel)
