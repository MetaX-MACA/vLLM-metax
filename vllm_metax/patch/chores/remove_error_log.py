# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

# ------------------------------------------------------------------------
# Note: This file contains non-functional code changes (chores) for vLLM
#       to support the Metax platform.
#
# Remove the wrong error log for Maca when checking the flash attention version.
# ------------------------------------------------------------------------

import vllm.v1.attention.backends.fa_utils
from vllm_metax.v1.attention.backends.fa_utils import get_flash_attn_version

vllm.v1.attention.backends.fa_utils.get_flash_attn_version = get_flash_attn_version
