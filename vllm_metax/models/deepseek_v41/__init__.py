# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 hardware-isolated model entry point."""

from vllm_metax.registry.quant_config.deepseek_fp8 import (
    MacaDeepseekV41FP8Config as DeepseekV4FP8Config,
)

from vllm_metax.models.deepseek_v41.metax.dspark import DSparkDeepseekV4ForCausalLM
from vllm_metax.models.deepseek_v41.metax.vl_model import DeepseekV41ForCausalLM

__all__ = [
    "DSparkDeepseekV4ForCausalLM",
    "DeepseekV4FP8Config",
    "DeepseekV41ForCausalLM",
]
