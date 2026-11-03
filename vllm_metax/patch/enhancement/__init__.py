# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#
# -----------------------------------------------------------------------------
# Note: Load MetaX enhancement patches and registration hooks at plugin initialization.
#     This initializer applies the patches by importing the child modules.
#
# Affected versions: vLLM 0.30.0 (ced6857afa0e), reviewed 2026-10-12.
#
# Remove at: All enhancement patches imported here have been removed or migrated to
#     native extension points.
# -----------------------------------------------------------------------------

# module level imports

from . import device_allocator  # noqa: F401
from . import chores  # noqa: F401
from . import distributed  # noqa: F401
from . import quant_kernels  # noqa: F401
from . import joyai_support  # noqa: F401
from . import MRV2  # noqa: F401

# single files
from . import utils  # noqa: F401
from . import dbo  # noqa: F401
from . import indexer_kv_dtype  # noqa: F401
