# SPDX-License-Identifier: Apache-2.0
from functools import cache

from vllm.config import ModelConfig

_MOE_CONFIG_PREFIX: str | None = None


@cache
def infer_hidden_size_from_model_config(model_config: ModelConfig) -> int:
    """Best-effort hidden_size resolution (supports multimodal configs)."""
    # 1) Prefer vLLM's resolved hidden size (MM-safe).
    get_hidden_size = model_config.get_hidden_size()
    if get_hidden_size > 0:
        return get_hidden_size

    return 0


def set_moe_config_prefix(hidden_size: int) -> None:
    global _MOE_CONFIG_PREFIX
    if hidden_size <= 0:
        return

    if _MOE_CONFIG_PREFIX is None:
        _MOE_CONFIG_PREFIX = f"H={hidden_size}"
