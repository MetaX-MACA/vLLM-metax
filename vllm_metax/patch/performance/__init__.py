# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# -----------------------------------------------
# Note: Load performance-oriented MetaX patch modules.
#
# Affected versions: v0.21.0
# -----------------------------------------------
from . import grouped_topk_router  # noqa: F401
from . import speculative_decode_perf  # noqa: F401
from . import modular_moe_output_alias  # noqa: F401
