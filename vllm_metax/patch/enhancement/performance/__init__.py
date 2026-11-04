# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#
# -----------------------------------------------------------------------------
# Note: Load MetaX grouped-topk dispatch and speculative-decode batch ordering at plugin
#     initialization. This initializer applies the patches by importing the child
#     modules.
#
# Affected versions: vLLM 0.30.0 (ced6857afa0e), reviewed 2026-10-12.
#
# Remove at: All performance patches imported here have been removed or migrated to
#     native extension points.
# -----------------------------------------------------------------------------

from . import grouped_topk_router  # noqa: F401
from . import speculative_decode_perf  # noqa: F401
from . import gumbel_sample  # noqa: F401
