# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
from unittest.mock import patch

import pytest
import torch

from vllm.config import AttentionConfig, CacheConfig, ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.selector import get_attn_backend

from vllm_metax.platform import MacaPlatform


def _config(backend, *, block_size=64, dcp=False):
    return VllmConfig(
        attention_config=AttentionConfig(backend=AttentionBackendEnum[backend]),
        cache_config=CacheConfig(block_size=block_size),
        parallel_config=ParallelConfig(
            distributed_executor_backend="mp" if dcp else None,
            tensor_parallel_size=2 if dcp else 1,
            decode_context_parallel_size=2 if dcp else 1,
        ),
    )


@pytest.mark.requires_maca
@pytest.mark.parametrize(
    ("backend_name", "head_size", "block_size"),
    [
        ("FLASH_ATTN", 64, 16),
        ("TRITON_ATTN", 64, 16),
        ("FLEX_ATTENTION", 64, 16),
        ("TRITON_MLA", 576, 16),
        ("TRITON_MLA", 576, 64),
        ("FLASHMLA", 576, 64),
    ],
)
def test_selector_returns_plugin_backend_class(backend_name, head_size, block_size):
    config = _config(backend_name, block_size=block_size)
    with (
        set_current_vllm_config(config),
        patch("vllm.platforms.current_platform", MacaPlatform()),
    ):
        backend = get_attn_backend(
            head_size,
            torch.float16,
            None,
            use_mla=backend_name.endswith("MLA") or backend_name == "FLASHMLA",
        )
    assert backend.__module__.startswith("vllm_metax.")


@pytest.mark.requires_maca
@pytest.mark.parametrize(
    ("backend_name", "block_size"),
    [("TRITON_ATTN", 16), ("FLEX_ATTENTION", 16)],
)
def test_selector_rejects_unsupported_dcp_backends(backend_name, block_size):
    config = _config(backend_name, block_size=block_size, dcp=True)
    with (
        set_current_vllm_config(config),
        patch("vllm.platforms.current_platform", MacaPlatform()),
        pytest.raises(ValueError, match="DCP not supported"),
    ):
        get_attn_backend(64, torch.float16, None)


@pytest.mark.requires_maca
@pytest.mark.parametrize(
    ("backend_name", "head_size"),
    [("FLASH_ATTN", 64), ("TRITON_MLA", 576), ("FLASHMLA", 576)],
)
def test_selector_accepts_dcp_for_plugin_supported_backends(backend_name, head_size):
    config = _config(backend_name, block_size=64, dcp=True)
    with (
        set_current_vllm_config(config),
        patch("vllm.platforms.current_platform", MacaPlatform()),
    ):
        backend = get_attn_backend(
            head_size,
            torch.float16,
            None,
            use_mla=backend_name != "FLASH_ATTN",
        )
    assert backend.__module__.startswith("vllm_metax.")


def test_selector_uses_maca_fa2_for_supported_fp16_head_sizes():
    from vllm_metax.v1.attention.backends import fa_utils

    with patch(
        "vllm_metax.v1.attention.backends.fa_utils.get_current_vllm_config_or_none",
        return_value=None,
    ):
        assert fa_utils.get_flash_attn_version(head_size=128, kv_cache_dtype="auto") == 2
        assert fa_utils.get_flash_attn_version(head_size=96, kv_cache_dtype="auto") == 2
        assert fa_utils.get_flash_attn_version(head_size=80, kv_cache_dtype="auto") == 2
        assert fa_utils.get_flash_attn_version(head_size=72) == 2
        assert MacaPlatform.get_vit_attn_backend(72, torch.float16) == AttentionBackendEnum.FLASH_ATTN
        assert fa_utils.get_flash_attn_version(head_size=73, kv_cache_dtype="auto") is None
