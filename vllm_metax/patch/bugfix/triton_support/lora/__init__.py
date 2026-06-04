# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# -----------------------------------------------
# Note: Load LoRA-related Triton compatibility patches.
#
# Affected versions: v0.21.0
# -----------------------------------------------
from . import fused_moe_lora_op
from . import lora_shrink_op
from . import kernel_utils
from . import lora_split_3d_moe
