# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#
# -----------------------------------------------------------------------------
# Note: Load MetaX device-visibility, benchmark-default and FlashAttention compatibility
#     at plugin initialization. This initializer applies the patches by importing the
#     child modules.
#
# Affected versions: vLLM 0.29.1.dev0 (98dff2a81d), verified 2026-09-17.
#
# Remove at: All miscellaneous compatibility patches imported here have been removed or
#     migrated to native extension points.
# -----------------------------------------------------------------------------

from . import maca_visible_device  # noqa: F401
from . import bench_serve_args  # noqa: F401
from . import remove_error_log  # noqa: F401
