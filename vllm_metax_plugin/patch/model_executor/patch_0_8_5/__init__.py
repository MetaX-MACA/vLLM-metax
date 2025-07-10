# SPDX-License-Identifier: Apache-2.0

import vllm_metax_plugin.patch.model_executor.patch_0_8_5.w8a8_utils
import vllm_metax_plugin.patch.model_executor.patch_0_8_5.quantization
import vllm_metax_plugin.patch.model_executor.patch_0_8_5.awq
import vllm_metax_plugin.patch.model_executor.patch_0_8_5.base_config
import vllm_metax_plugin.patch.model_executor.patch_0_8_5.fused_moe
import vllm_metax_plugin.patch.model_executor.patch_0_8_5.gptq
import vllm_metax_plugin.patch.model_executor.patch_0_8_5.linear
import vllm_metax_plugin.patch.model_executor.patch_0_8_5.moe_wna16
import vllm_metax_plugin.patch.model_executor.patch_0_8_5.rotary_embedding
import vllm_metax_plugin.patch.model_executor.patch_0_8_5.vocab_parallel_embedding
from . import compressed_tensors
from . import compressed_tensors_moe
from . import cutlass
