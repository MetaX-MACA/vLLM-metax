# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#
# -----------------------------------------------------------------------------
# Note: Activate static EPLB hooks once through the normal MetaX patch imports.
#
# Affected versions: Reviewed vLLM v0.33.0 (4ba787ec9959).
#
# Remove at: Static EPLB uses native upstream lifecycle hooks.
# -----------------------------------------------------------------------------
from . import static_eplb  # noqa: F401
