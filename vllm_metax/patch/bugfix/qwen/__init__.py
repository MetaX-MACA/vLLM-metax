# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#
# -----------------------------------------------------------------------------
# Note: Activate the Qwen3.5 MTP weight loading adjustment.
#
# Affected versions: vLLM 0.32.0 (5df63a7203), reviewed 2026-11-03.
#
# Remove at: The child patches migrate to supported upstream hooks or their
#     platform/checkpoint restrictions are resolved.
# -----------------------------------------------------------------------------

from . import qwen3_5_moe_loading  # noqa: F401
