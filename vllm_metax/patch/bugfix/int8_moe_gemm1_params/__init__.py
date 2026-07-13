# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# -----------------------------------------------
# Note: Load int8 MoE gemm1_alpha/beta/clamp_limit forwarding patches.
#
# Affected versions: v0.24.0
# -----------------------------------------------
from . import fused_moe_config
from . import oracle_int8

