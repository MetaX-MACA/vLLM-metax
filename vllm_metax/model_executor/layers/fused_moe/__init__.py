# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional

from vllm.triton_utils import HAS_TRITON

_config: Optional[dict[str, Any]] = None


def get_config() -> Optional[dict[str, Any]]:
    return _config


if HAS_TRITON:
    # import to register the custom ops
    from vllm_metax.model_executor.layers.fused_moe.fused_moe import (
        fused_experts)
    __all__ = ["fused_experts"]
