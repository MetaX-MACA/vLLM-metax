# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# -----------------------------------------------
# Note: Redirect NCCL and device-management utilities to MetaX equivalents.
#
# Affected versions: v0.21.0
# -----------------------------------------------

from vllm_metax.utils.mccl import find_mccl_library
from vllm_metax.utils import import_pymxsml
from vllm_metax.patch.utils import patch

find_mccl_library = patch("vllm.utils.nccl", "find_nccl_library")(find_mccl_library)
import_pymxsml = patch("vllm.utils.import_utils", "import_pynvml")(import_pymxsml)
