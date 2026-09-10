# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# -----------------------------------------------
# Note: Aggregate bugfix patches for MetaX compatibility.
#
# Affected versions: v0.21.0
# -----------------------------------------------
# from . import dp_fix  # noqa: F401
from . import triton_support  # noqa: F401
from . import deepseek_v4  # noqa: F401
from . import int8_w8a8  # noqa: F401
from . import tokenizer  # noqa: F401
from . import draft_config_overrides  # noqa: F401
from . import minimax_m3  # noqa: F401
from . import qwen3_5_moe_loading  # noqa: F401
from . import telechat3  # noqa: F401
from . import bytes_to_unicode  # noqa: F401
