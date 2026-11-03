# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#
# -----------------------------------------------------------------------------
# Note: Redirect vLLM's lazy cumem imports to the MACA allocator. Replacing only
#     the parent package attribute does not redirect submodule imports, and
#     importing the upstream allocator first can initialize CUDA-only bindings.
#
# Affected versions: vLLM 0.32.0 (5df63a7203), reviewed 2026-11-03.
#
# Remove at: vLLM's allocator dispatch selects the MACA implementation through a
#     supported platform hook for all allocation and worker-shutdown callers.
# -----------------------------------------------------------------------------

from vllm_metax.device_allocator import cumem
from vllm_metax.patch import patch

patch("vllm.device_allocator.cumem", replace_module=True)(cumem)
