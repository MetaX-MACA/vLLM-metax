# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
from vllm_metax.envs import VLLM_METAX_PATCH_KV_BLOCK_ZEROER
from . import parallel_state  # noqa: F401
from . import topk_global_indices_oob  # noqa: F401

if VLLM_METAX_PATCH_KV_BLOCK_ZEROER:
    from . import utils  # noqa: F401
# ---------------------------
# Note: enable for dsv4 branch with torch2.8
# from . import torch_fix
