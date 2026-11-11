# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#
# -----------------------------------------------------------------------------
# Note: Activate model-specific compatibility patches. Directory ownership follows
#     the affected model; shared runtime fixes stay in bugfix or enhancement.
#
# Affected versions: MetaX patch activation with vLLM 0.32.0 (5df63a7203cd),
#     reviewed 2026-11-04. Child headers describe their own restrictions.
#
# Remove at: All child patches migrate to native extension points or their
#     model-specific restrictions are resolved.
# -----------------------------------------------------------------------------

from . import joyai  # noqa: F401
from . import kimi_k3  # noqa: F401
from . import qwen  # noqa: F401
from . import telechat3  # noqa: F401
