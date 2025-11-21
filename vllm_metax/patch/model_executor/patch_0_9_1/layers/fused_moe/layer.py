# 2025 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import vllm

import torch
from typing import Callable, Optional

from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm_metax.model_executor.layers.fused_moe.fused_moe import metax_fused_experts
from vllm.model_executor.layers.fused_moe.layer import UnquantizedFusedMoEMethod, MoEConfig

original_unquantized_fused_moe_init_func = UnquantizedFusedMoEMethod.__init__

def unquantized_fused_moe_init_func(self, moe: MoEConfig):
    original_unquantized_fused_moe_init_func(self, moe)
    self.fused_experts = metax_fused_experts  # type: ignore

UnquantizedFusedMoEMethod.__init__ = unquantized_fused_moe_init_func