# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import pytest
import torch

import vllm_metax.v1.attention.registry  # noqa: F401
from vllm.v1.attention.backends.registry import AttentionBackendEnum


EXPECTED_BACKENDS = {
    "FLASH_ATTN": "vllm_metax.v1.attention.backends.flash_attn.MacaFlashAttentionBackend",
    "FLASH_ATTN_DIFFKV": "vllm_metax.v1.attention.backends.flash_attn_diffkv.FlashAttentionDiffKVBackend",
    "FLASHINFER": "vllm_metax.v1.attention.backends.flashinfer.MacaFlashInferBackend",
    "TRITON_ATTN": "vllm_metax.v1.attention.backends.triton_attn.MacaTritonAttentionBackend",
    "FLEX_ATTENTION": "vllm_metax.v1.attention.backends.flex_attention.MacaFlexAttentionBackend",
    "TURBOQUANT": "vllm_metax.v1.attention.backends.turboquant_attn.MacaTurboQuantAttentionBackend",
    "FLASHMLA": "vllm_metax.v1.attention.backends.mla.flashmla.MacaFlashMLABackend",
    "FLASHMLA_SPARSE": "vllm_metax.v1.attention.backends.mla.flashmla_sparse.MacaFlashMLASparseBackend",
    "TRITON_MLA": "vllm_metax.v1.attention.backends.mla.triton_mla.MacaTritonMLABackend",
}


@pytest.mark.parametrize(("name", "qualified_name"), EXPECTED_BACKENDS.items())
def test_plugin_owns_attention_backend_class(name, qualified_name):
    backend_class = AttentionBackendEnum[name].get_class()
    assert f"{backend_class.__module__}.{backend_class.__name__}" == qualified_name


@pytest.mark.requires_maca
def test_plugin_backend_selection_returns_plugin_classes(device):
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
    from vllm.v1.attention.selector import _cached_get_attn_backend, get_attn_backend

    for backend in (
        AttentionBackendEnum.FLASH_ATTN,
        AttentionBackendEnum.TRITON_ATTN,
        AttentionBackendEnum.FLEX_ATTENTION,
    ):
        _cached_get_attn_backend.cache_clear()
        with set_current_vllm_config(VllmConfig()):
            selected = get_attn_backend(
                head_size=64,
                dtype=torch.float16,
                kv_cache_dtype=None,
                attn_type=None,
            )
        assert selected.get_name()


@pytest.mark.requires_maca
def test_maca_mla_priority_contains_only_plugin_supported_families(platform):
    import vllm_metax.platform as plugin_platform

    priorities = plugin_platform._get_backend_priorities(True, None)
    assert [item.name for item in priorities] == [
        "FLASHMLA",
        "TRITON_MLA",
        "FLASHMLA_SPARSE",
    ]


def test_mla_prefill_registry_is_plugin_owned():
    from vllm.v1.attention.backends.mla.prefill.registry import (
        MLAPrefillBackendEnum,
    )

    backend_class = MLAPrefillBackendEnum.FLASH_ATTN.get_class()
    assert backend_class.__module__.startswith("vllm_metax.")
