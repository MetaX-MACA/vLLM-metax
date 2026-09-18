# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#
# -----------------------------------------------------------------------------
# Note: Load bugfix compatibility at plugin initialization. This initializer applies the
#     patches by importing the child modules.
#
# Affected versions: vLLM 0.29.1.dev0 (98dff2a81d), verified 2026-09-17.
#
# Remove at: All bugfix patches imported here have been removed or migrated to native
#     extension points.
# -----------------------------------------------------------------------------

from . import triton_support  # noqa: F401
from . import deepseek_v4  # noqa: F401
from . import int8_w8a8  # noqa: F401
from . import tokenizer  # noqa: F401
from . import draft_config_overrides  # noqa: F401
from . import minimax_m3  # noqa: F401
from . import qwen3_5_moe_loading  # noqa: F401
from . import telechat3  # noqa: F401
from . import bytes_to_unicode  # noqa: F401
