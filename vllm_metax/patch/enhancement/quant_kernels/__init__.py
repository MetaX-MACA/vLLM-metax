# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#
# -----------------------------------------------------------------------------
# Note: Load MetaX FP8 block-kernel registration at plugin initialization. This
#     initializer applies the patches by importing the child modules.
#
# Affected versions: vLLM 0.29.1.dev0 (98dff2a81d), verified 2026-09-17.
#
# Remove at: All quantized-kernel patches imported here have been removed or migrated to
#     native extension points.
# -----------------------------------------------------------------------------

from . import fp8_block_kernel  # noqa: F401
