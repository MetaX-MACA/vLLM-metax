# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

from vllm_metax.models.deepseek_v4.common.ops.cache_utils import (
    build_flashinfer_mixed_sparse_indices,
    combine_topk_swa_indices,
    compute_global_topk_indices_and_lens,
    dequantize_and_gather_k_cache,
    quantize_and_insert_k_cache,
)
from vllm_metax.models.deepseek_v4.common.ops.fused_indexer_q import (
    MXFP4_BLOCK_SIZE,
    fused_indexer_q_rope_quant,
)
from vllm_metax.models.deepseek_v4.common.ops.fused_indexer_q_metax.int8 import (
    fused_indexer_q_rope_int8_quant,
)
from vllm_metax.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import (
    fused_inv_rope_fp8_quant,
)
from vllm_metax.models.deepseek_v4.metax.cache_utils import gather_k_cache
from vllm_metax.models.deepseek_v4.common.ops.fused_inv_rope_quant import inv_rope

__all__ = [
    "MXFP4_BLOCK_SIZE",
    "build_flashinfer_mixed_sparse_indices",
    "combine_topk_swa_indices",
    "compute_global_topk_indices_and_lens",
    "dequantize_and_gather_k_cache",
    "fused_indexer_q_rope_int8_quant",
    "fused_indexer_q_rope_quant",
    "fused_inv_rope_fp8_quant",
    "gather_k_cache",
    "inv_rope",
    "quantize_and_insert_k_cache",
]
