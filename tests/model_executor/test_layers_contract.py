# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import torch

from vllm_metax.model_executor.layers.fused_moe.oracle import int8, mxfp4, unquantized
from vllm_metax.model_executor.layers.indexer_topk import (
    SparseIndexerTopk,
    get_indexer_topk,
)
from vllm_metax.model_executor.layers.online_rotation import (
    CompositeHadamardRotation,
    build_rotation,
)


def test_int8_backend_priority_and_kernel_classes_are_plugin_owned():
    assert [backend.value for backend in int8._get_priority_backends(None)] == [
        "TRITON",
        "BATCHED_TRITON",
        "BATCHED_DEEPGEMM",
    ]
    for backend in int8._get_priority_backends(None):
        kernels = int8.backend_to_kernel_cls(backend)
        assert kernels
        assert all(kernel.__module__.startswith("vllm_metax.") for kernel in kernels)


def test_mxfp4_backend_priority_and_roundup_contract():
    assert [backend.value for backend in mxfp4._get_priority_backends()] == ["TRITON"]
    triton = mxfp4._get_priority_backends()[0]
    assert mxfp4.backend_to_kernel_cls(triton)[0].__module__.startswith("vllm_metax.")
    assert mxfp4.mxfp4_round_up_hidden_size_and_intermediate_size(
        triton, 4097, 8193
    ) == (4097, 8193)


def test_unquantized_priority_and_kernel_classes_are_plugin_owned():
    priorities = unquantized._get_priority_backends(None)
    assert priorities
    for backend in priorities:
        kernels = unquantized.backend_to_kernel_cls(backend)
        assert kernels
        assert all(kernel.__module__.startswith("vllm_metax.") for kernel in kernels)


def test_indexer_topk_factory_returns_plugin_module():
    topk = get_indexer_topk("torch")
    assert isinstance(topk, SparseIndexerTopk)
    assert type(topk).__module__.startswith("vllm_metax.")


def test_indexer_topk_debug_backend_masks_past_row_end(device):
    topk = SparseIndexerTopk("torch")
    logits = torch.tensor(
        [[5.0, 4.0, 3.0, 2.0, 1.0, 100.0], [9.0, 8.0, 7.0, 100.0, 100.0, 100.0]],
        device=device,
    )
    seq_lens = torch.tensor([5, 3], dtype=torch.int32, device=device)
    indices = torch.full((2, 3), -9, dtype=torch.int32, device=device)
    topk(logits, seq_lens, next_n=1, topk_indices=indices, topk_tokens=3, max_seq_len=6)
    assert indices[0].tolist() == [0, 1, 2]
    assert indices[1].tolist() == [0, 1, 2]


def test_online_rotation_builds_from_plugin_spec(device):
    hidden_size = 64
    sign = torch.ones(hidden_size, dtype=torch.int8, device=device)
    hk = torch.ones(1, 1, dtype=torch.float32, device=device)
    rotation = build_rotation(
        {
            "type": "composite_hadamard",
            "hidden_size": hidden_size,
            "K": 1,
            "P": 64,
        }
    )
    assert isinstance(rotation, CompositeHadamardRotation)
    actual = CompositeHadamardRotation(hidden_size, sign=sign, hk=hk, k=1, p=64)
    x = torch.randn(3, hidden_size, dtype=torch.bfloat16, device=device)
    transformed = actual(x)
    restored = actual(transformed, inverse=True)
    assert transformed.shape == x.shape
    torch.testing.assert_close(restored, x, atol=3e-2, rtol=3e-2)
