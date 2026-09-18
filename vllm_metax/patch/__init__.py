# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#
# -----------------------------------------------------------------------------
# Note: Expose the shared patch decorator for MetaX runtime compatibility modules. This
#     package initializer is infrastructure and does not install patches itself.
#
# Affected versions: MetaX patch infrastructure with vLLM 0.29.1.dev0
#     (98dff2a81d), verified 2026-09-17.
#
# Remove at: No MetaX runtime patch modules import the package-level patch decorator.
# -----------------------------------------------------------------------------

from .utils import patch

__all__ = [
    "patch",
]
