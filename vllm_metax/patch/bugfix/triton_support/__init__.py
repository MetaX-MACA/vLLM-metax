# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# -----------------------------------------------
# Note: Load Triton compatibility bugfix patches for MetaX.
#
# Affected versions: v0.21.0
# -----------------------------------------------
from . import custom_op_schemas
from . import kda
from . import lora
from . import chunk_delta_h
from . import rejection_sampler
from . import eagle
from . import topk_topp_sampler
from . import mrv2
from . import jit_monitor
