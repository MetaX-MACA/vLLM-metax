# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import pytest
import torch

from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding

from vllm_metax.registry.custom_ops.layers.rotary_embedding.rotary_embedding import (
    MacaMRotaryEmbedding,
)

pytestmark = pytest.mark.requires_maca


@pytest.mark.parametrize("is_neox_style", [True, False])
@pytest.mark.parametrize("mrope_interleaved", [True, False])
@pytest.mark.parametrize("packed", [True, False])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_mrope(device, is_neox_style, mrope_interleaved, packed, dtype):
    config = VllmConfig(compilation_config=CompilationConfig(custom_ops=["all"]))
    with set_current_vllm_config(config):
        layer = MRotaryEmbedding(
            head_size=64,
            rotary_dim=48,
            max_position_embeddings=256,
            base=10000,
            is_neox_style=is_neox_style,
            dtype=dtype,
            mrope_section=[8, 8, 8],
            mrope_interleaved=mrope_interleaved,
        ).to(device)
        assert type(layer) is MacaMRotaryEmbedding
        assert layer._forward_method == layer.forward_oot
        rng = torch.Generator(device=device).manual_seed(42)
        positions = torch.randint(256, (3, 11), device=device, generator=rng)
        qkv = torch.randn(11, 64 * 7, dtype=dtype, device=device, generator=rng)
        query, key, untouched = qkv.split([64 * 3, 64 * 2, 64 * 2], dim=-1)
        if not packed:
            query, key = query.contiguous(), key.contiguous()
        expected_q, expected_k = layer.forward_native(
            positions, query.clone(), key.clone()
        )
        original = untouched.clone()

        actual_q, actual_k = layer(positions, query, key)

    torch.testing.assert_close(actual_q, expected_q, atol=1e-3, rtol=1.6e-2)
    torch.testing.assert_close(actual_k, expected_k, atol=1e-3, rtol=1.6e-2)
    torch.testing.assert_close(untouched, original, atol=0, rtol=0)
