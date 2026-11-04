# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
from vllm.v1.attention.backends.mla.prefill.registry import (
    register_mla_prefill_backend,
    MLAPrefillBackendEnum,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend

# Pre-register all attention backends
# NOTE(metax): Resolve explicit MiniMax backend selection to the local model too.
register_backend(
    backend=AttentionBackendEnum.MINIMAX_M3_SPARSE,
    class_path="vllm_metax.models.minimax_m3.common.sparse_attention.MiniMaxM3SparseBackend",
)
# NOTE(metax): Legacy MSA aliases also resolve to portable Triton, never CuTeDSL.
register_backend(
    backend=AttentionBackendEnum.CUTLASS_MSA,
    class_path="vllm_metax.models.minimax_m3.common.sparse_attention.MiniMaxM3SparseBackend",
)
register_backend(
    backend=AttentionBackendEnum.TRITON_MSA,
    class_path="vllm_metax.models.minimax_m3.common.sparse_attention.MiniMaxM3SparseBackend",
)
register_backend(
    backend=AttentionBackendEnum.FLASHMLA,
    class_path="vllm_metax.v1.attention.backends.mla.flashmla.MacaFlashMLABackend",
)
register_backend(
    backend=AttentionBackendEnum.FLASHMLA_SPARSE,
    class_path="vllm_metax.v1.attention.backends.mla.flashmla_sparse.MacaFlashMLASparseBackend",
)
register_backend(
    backend=AttentionBackendEnum.TRITON_MLA,
    class_path="vllm_metax.v1.attention.backends.mla.triton_mla.MacaTritonMLABackend",
)
register_backend(
    AttentionBackendEnum.FLASH_ATTN,
    class_path="vllm_metax.v1.attention.backends.flash_attn.MacaFlashAttentionBackend",
)
register_backend(
    backend=AttentionBackendEnum.FLASH_ATTN_DIFFKV,
    class_path="vllm_metax.v1.attention.backends.flash_attn_diffkv.FlashAttentionDiffKVBackend",
)
register_backend(
    backend=AttentionBackendEnum.FLASHINFER,
    class_path="vllm_metax.v1.attention.backends.flashinfer.MacaFlashInferBackend",
)
register_backend(
    backend=AttentionBackendEnum.TRITON_ATTN,
    class_path="vllm_metax.v1.attention.backends.triton_attn.MacaTritonAttentionBackend",
)
register_backend(
    backend=AttentionBackendEnum.FLEX_ATTENTION,
    class_path="vllm_metax.v1.attention.backends.flex_attention.MacaFlexAttentionBackend",
)
register_backend(
    backend=AttentionBackendEnum.TURBOQUANT,
    class_path="vllm_metax.v1.attention.backends.turboquant_attn.MacaTurboQuantAttentionBackend",
)
register_mla_prefill_backend(
    backend=MLAPrefillBackendEnum.FLASH_ATTN,
    class_path="vllm_metax.v1.attention.backends.mla.prefill.flash_attn.MacaFlashAttnPrefillBackend",
)
