# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#
# -----------------------------------------------------------------------------
# Note: Load MiniMax-M3 index-score, sparse-attention and weight-loading compatibility
#     at plugin initialization. This initializer applies the patches by importing the
#     child modules.
#
# Affected versions: vLLM 0.29.1.dev0 (98dff2a81d), verified 2026-09-17.
#
# Remove at: All MiniMax-M3 patches imported here have been removed or migrated to
#     native extension points.
# -----------------------------------------------------------------------------

from . import index_topk  # noqa: F401
from . import sparse_attn  # noqa: F401
from . import load_weights  # noqa: F401
