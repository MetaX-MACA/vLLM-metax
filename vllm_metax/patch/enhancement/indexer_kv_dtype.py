# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#
# -----------------------------------------------------------------------------
# Note: This patch is to add int8 to vllm.config.attention.IndexerKVDType
#
# Affected versions: vLLM 0.33.0
#     (4ba787ec9959e566d3b587a796dff07673324da5).
#
# Remove at: Upstream includes int8 in vllm.config.attention.IndexerKVDType
# -----------------------------------------------------------------------------

from typing import Literal, get_args
from pydantic.dataclasses import rebuild_dataclass

from vllm.config import attention as attention_config_module
from vllm.config.attention import AttentionConfig

_METAX_INDEXER_KV_DTYPE = Literal["auto", "bf16", "fp8", "int8", "mxfp4", "nvfp4"]


def _enable_int8_indexer_kv_dtype() -> None:
    """Extend vLLM's AttentionConfig with MetaX's INT8 Indexer cache type.

    vLLM does not currently include ``int8`` in ``IndexerKVDType``. Updating
    only ``__annotations__`` is insufficient because Pydantic dataclasses keep
    separate dataclass-field and validator metadata. Keep all three views in
    sync and rebuild the validator before command-line config construction.
    """
    dataclass_field = AttentionConfig.__dataclass_fields__["indexer_kv_dtype"]
    if "int8" in get_args(dataclass_field.type):
        return

    attention_config_module.IndexerKVDType = _METAX_INDEXER_KV_DTYPE
    AttentionConfig.__annotations__["indexer_kv_dtype"] = _METAX_INDEXER_KV_DTYPE
    dataclass_field.type = _METAX_INDEXER_KV_DTYPE
    AttentionConfig.__pydantic_fields__[
        "indexer_kv_dtype"
    ].annotation = _METAX_INDEXER_KV_DTYPE
    rebuild_dataclass(AttentionConfig, force=True)


_enable_int8_indexer_kv_dtype()
