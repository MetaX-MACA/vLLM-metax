# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
from vllm_metax.models.deepseek_v4.common.ops.fused_indexer_q import (
    MXFP4_BLOCK_SIZE,
    _FUSED_INDEXER_Q_ROPE_MXFP4_TRITON_KERNEL,
    _FUSED_INDEXER_Q_ROPE_QUANT_TRITON_KERNEL,
    _fp32x2_to_fp4x2,
    fused_indexer_q_rope_quant,
)
from vllm_metax.models.deepseek_v4.common.ops.fused_indexer_q_metax.fp4 import (
    fused_indexer_q_rope_mxfp4_quant,
)
from vllm_metax.models.deepseek_v4.common.ops.fused_indexer_q_metax.int8 import (
    fused_indexer_q_rope_int8_quant,
)

__all__ = [
    "MXFP4_BLOCK_SIZE",
    "_FUSED_INDEXER_Q_ROPE_MXFP4_TRITON_KERNEL",
    "_FUSED_INDEXER_Q_ROPE_QUANT_TRITON_KERNEL",
    "fused_indexer_q_rope_quant",
    "fused_indexer_q_rope_int8_quant",
    "fused_indexer_q_rope_mxfp4_quant",
]
