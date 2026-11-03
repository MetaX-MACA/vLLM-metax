# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#
# -----------------------------------------------------------------------------
# Note: Load MACA Triton autotuning compatibility at plugin initialization. This
#     initializer applies the patches by importing the child modules.
#
# Affected versions: vLLM 0.30.0 (ced6857afa0e), reviewed 2026-10-12.
#
# Remove at: All Triton compatibility patches imported here have been removed or
#     migrated to native extension points.
# -----------------------------------------------------------------------------

from . import kda  # noqa: F401
from . import chunk_delta_h  # noqa: F401
from . import fused_moe_lora_shmem  # noqa: F401
from . import rejection_sample  # noqa: F401
