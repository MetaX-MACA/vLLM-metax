# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#
# -----------------------------------------------------------------------------
# Note: Activate the Kimi-K3 CUDA-only low-latency GEMM dispatch workaround.
#
# Affected versions: MetaX patch activation with vLLM 0.32.0 (5df63a7203cd),
#     reviewed 2026-11-04.
#
# Remove at: The child patch is removed or migrates to a native platform hook.
# -----------------------------------------------------------------------------

from . import disable_low_latency  # noqa: F401
