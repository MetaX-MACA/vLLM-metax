# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#
# -----------------------------------------------------------------------------
# Note: Load bugfix compatibility at plugin initialization. This initializer applies the
#     patches by importing the child modules.
#
# Affected versions: vLLM 0.30.0 (ced6857afa0e), reviewed 2026-10-12.
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
from . import telechat3  # noqa: F401
from . import bytes_to_unicode  # noqa: F401

# from . import qwen  # noqa: F401
from . import routed_experts_fused_weight_fix  # noqa: F401
from . import rejection_sampler  # noqa: F401
