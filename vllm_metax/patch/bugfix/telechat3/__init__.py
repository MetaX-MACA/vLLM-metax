# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#
# -----------------------------------------------------------------------------
# Note: Load TeleChat3 YaRN maximum-length compatibility at plugin initialization. This
#     initializer applies the patches by importing the child modules.
#
# Affected versions: vLLM 0.29.1.dev0 (98dff2a81d), verified 2026-09-17.
#
# Remove at: All TeleChat3 patches imported here have been removed or migrated to native
#     extension points.
# -----------------------------------------------------------------------------

from . import telechat3_yarn  # noqa: F401
