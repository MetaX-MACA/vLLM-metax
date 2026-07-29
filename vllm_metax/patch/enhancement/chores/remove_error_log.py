# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

# -----------------------------------------------
# Note: Suppress incorrect flash-attention version error logs on MACA.
#
# Affected versions: v0.21.0
# -----------------------------------------------

import vllm.v1.attention.backends.fa_utils
from vllm_metax.v1.attention.backends.fa_utils import get_flash_attn_version

vllm.v1.attention.backends.fa_utils.get_flash_attn_version = get_flash_attn_version
