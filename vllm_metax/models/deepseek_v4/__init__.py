# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

from vllm_metax.models.deepseek_v4.metax.model import DeepseekV4ForCausalLM  # type: ignore[assignment]
from vllm_metax.models.deepseek_v4.metax.mtp import DeepSeekV4MTP  # type: ignore[assignment]
from vllm_metax.models.deepseek_v4.metax.dspark import DSparkDeepseekV4ForCausalLM

__all__ = [
    "DeepSeekV4MTP",
    "DeepseekV4ForCausalLM",
    "DSparkDeepseekV4ForCausalLM",
]
