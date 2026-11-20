# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import pytest
import torch

from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding

from vllm_metax.registry.custom_ops.layers.rotary_embedding.rotary_embedding import (
    MacaRotaryEmbedding,
)

pytestmark = pytest.mark.requires_maca

# From the upstream kernel tests' allclose_default.py.
ATOL = {torch.float16: 1e-3, torch.bfloat16: 1e-3, torch.float32: 1e-5}
RTOL = {torch.float16: 1e-3, torch.bfloat16: 1.6e-2, torch.float32: 1.3e-6}


@pytest.fixture(autouse=True)
def enable_custom_ops():
    config = VllmConfig(compilation_config=CompilationConfig(custom_ops=["all"]))
    with set_current_vllm_config(config):
        yield


# Adapted from vLLM v0.33.0 tests/kernels/core/test_pos_encoding.py.
@pytest.mark.parametrize("is_neox", [True, False])
@pytest.mark.parametrize("with_key", [True, False])
@pytest.mark.parametrize("rotary_dim", [32, 64])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("padding", [0, 8])
@torch.inference_mode()
def test_registered_rope_matches_reference(
    device, is_neox, with_key, rotary_dim, dtype, padding
):
    torch.manual_seed(0)
    rotary = RotaryEmbedding(64, rotary_dim, 64, 10000, is_neox, dtype).to(device)
    assert type(rotary) is MacaRotaryEmbedding
    assert rotary._forward_method == rotary.forward_oot
    positions = torch.tensor([0, 3, 5, 19, 63], device=device)
    query = torch.randn(5, 4, 64 + padding, dtype=dtype, device=device)[..., :64]
    key = None
    if with_key:
        key = torch.randn(5, 2, 64 + padding, dtype=dtype, device=device)[..., :64]
    query_tail = query[..., rotary_dim:].clone()
    # The native kernel accumulates in FP32 and rounds once on writeback.
    expected_query, expected_key = rotary.forward_static(
        positions,
        query.float(),
        key.float() if key is not None else None,
        64,
        rotary_dim,
        rotary.cos_sin_cache.float(),
        is_neox,
    )
    actual_query, actual_key = rotary(positions, query, key)
    assert actual_query is query
    assert actual_key is key
    torch.testing.assert_close(query[..., rotary_dim:], query_tail, atol=0, rtol=0)
    torch.testing.assert_close(
        actual_query, expected_query.to(dtype), atol=ATOL[dtype], rtol=RTOL[dtype]
    )
    if with_key:
        torch.testing.assert_close(
            actual_key, expected_key.to(dtype), atol=ATOL[dtype], rtol=RTOL[dtype]
        )
