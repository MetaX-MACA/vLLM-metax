# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Lossless packing contracts; GPU cases use the production kernel module."""

import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist

from vllm_metax import envs
from vllm_metax.distributed.device_communicators import byte_pack


def test_opt_in_default(monkeypatch):
    key = "VLLM_METAX_EXPERIMENTAL_MOE_BYTE_PACK"
    monkeypatch.delenv(key, raising=False)
    assert not getattr(envs, key)
    monkeypatch.setenv(key, "1")
    assert getattr(envs, key)


@pytest.mark.parametrize(
    "sizes,expected",
    [
        ([0, 0, 0, 8], True),
        ([0, 1, 2, 3], True),
        ([0, 8], False),
        ([1, 1, 1, 1], False),
        ([0, 0, 0, 9], False),
        ([-1, 0, 0, 8], False),
    ],
)
def test_measured_envelope(sizes, expected):
    tensors = [torch.empty(1, 4096, dtype=torch.float16), torch.empty(1, 17)]
    assert byte_pack.should_pack(tensors, sizes) == expected
    assert not byte_pack.should_pack(tensors + [torch.empty(1, 1)], sizes)


def test_routing_does_not_depend_on_local_rows_or_strides():
    for rows in (0, 1, 8):
        contiguous = torch.empty(rows, 7)
        strided = torch.empty(rows, 14)[:, ::2]
        assert byte_pack.should_pack([contiguous], [0, 1, 2, 8])
        assert byte_pack.should_pack([strided], [0, 1, 2, 8])


@pytest.mark.parametrize(
    "templates", [[], [torch.empty(1, 1)] * 9, [torch.empty(1, 0)]]
)
def test_invalid_templates(templates):
    assert not byte_pack.metadata_supported(templates)
    with pytest.raises(ValueError, match="template"):
        byte_pack.unpack_rows(torch.empty(1, 4, dtype=torch.uint8), templates)


def test_invalid_storage_and_wire_shape():
    tensors = [torch.empty(2, 1)]
    with pytest.raises(ValueError, match="Unsupported"):
        byte_pack.pack_rows(tensors)
    with pytest.raises(ValueError, match="rank-2 uint8"):
        byte_pack.unpack_rows(torch.empty(2, 4), tensors)
    with pytest.raises(ValueError, match="row width"):
        byte_pack.unpack_rows(torch.empty(2, 3, dtype=torch.uint8), tensors)


def assert_bits_equal(actual, expected):
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    if not actual.numel():
        return
    assert torch.equal(
        actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
    )


def inputs(rows, device):
    return [
        torch.arange(rows * 12, device=device, dtype=torch.float32).reshape(rows, 12)[
            :, 1::2
        ],
        torch.full((rows, 2), 2**53 + 1, dtype=torch.int64, device=device),
        torch.full((rows, 1), -(2**31), dtype=torch.int32, device=device),
        torch.tensor([float("nan"), -0.0], device=device).repeat(rows, 1),
        torch.ones(rows, 2, device=device, dtype=torch.bool),
        torch.ones(rows, 2, device=device, dtype=torch.float16),
        torch.ones(rows, 2, device=device, dtype=torch.bfloat16),
        torch.ones(rows, 2, device=device, dtype=torch.float64),
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU runtime required")
@pytest.mark.parametrize("rows", [0, 1, 8])
def test_bitwise_roundtrip_and_fresh_outputs(rows):
    tensors = inputs(rows, "cuda")
    packed = byte_pack.pack_rows(tensors)
    first = byte_pack.unpack_rows(packed, tensors)
    second = byte_pack.unpack_rows(packed, tensors)
    for original, a, b in zip(tensors, first, second, strict=True):
        assert_bits_equal(a, original)
        assert_bits_equal(b, original)
        if rows:
            assert a.data_ptr() != b.data_ptr()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU runtime required")
def test_current_stream_and_changed_graph_inputs():
    tensors = inputs(3, "cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            byte_pack.unpack_rows(byte_pack.pack_rows(tensors), tensors)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            result = byte_pack.unpack_rows(byte_pack.pack_rows(tensors), tensors)
        for value in range(5):
            tensors[0].fill_(value)
            graph.replay()
            for a, b in zip(result, tensors, strict=True):
                assert_bits_equal(a, b)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph.reset()


@pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) not in (2, 4),
    reason="launch this test with torchrun --nproc-per-node=2 or 4",
)
def test_distributed_uneven_gather():
    """Real collective composition, not a MacaAgRsAll2AllManager closure."""
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(seconds=90))
    try:
        rank, world = dist.get_rank(), dist.get_world_size()
        for sizes in ([0] * world, [0] * (world - 1) + [8], list(range(world))):
            tensors = inputs(sizes[rank], "cuda")
            calls = []

            def gather(values, *, dim, sizes):
                assert dim == 0 and len(values) == 1
                calls.append(1)
                output = torch.empty(
                    sum(sizes), values[0].shape[1], dtype=torch.uint8, device="cuda"
                )
                offset = 0
                for source, size in enumerate(sizes):
                    if size:
                        view = output[offset : offset + size]
                        if rank == source:
                            view.copy_(values[0])
                        dist.broadcast(view, source)
                    offset += size
                return [output]

            result = byte_pack.gather_packed(tensors, sizes, gather)
            expected = [
                torch.cat(parts)
                for parts in zip(*(inputs(n, "cuda") for n in sizes), strict=True)
            ]
            assert calls == [1]
            for a, b in zip(result, expected, strict=True):
                assert_bits_equal(a, b)
        torch.cuda.synchronize()
    finally:
        dist.destroy_process_group()
