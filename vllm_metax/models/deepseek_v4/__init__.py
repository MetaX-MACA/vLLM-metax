# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

from .metax.model import DeepseekV4ForCausalLM  # type: ignore[assignment]
from .metax.mtp import DeepSeekV4MTP  # type: ignore[assignment]
from .metax.dspark import DSparkDeepseekV4ForCausalLM

__all__ = [
    "DeepSeekV4MTP",
    "DeepseekV4ForCausalLM",
    "DSparkDeepseekV4ForCausalLM",
]
