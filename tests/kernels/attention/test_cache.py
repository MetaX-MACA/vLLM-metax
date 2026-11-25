# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import pytest
import torch

from vllm_metax import envs
from vllm_metax.platform import MacaPlatform

pytestmark = pytest.mark.requires_maca


# Adapted from vLLM v0.33.0 tests/kernels/attention/test_cache.py.
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("layout", ["NHD", "HND"])
@pytest.mark.parametrize("padded_rows", [False, True])
@pytest.mark.parametrize("num_heads,head_size", [(3, 64), (1, 8), (1, 16), (1, 24), (2, 24)])
def test_reshape_and_cache_flash_writes_only_selected_slots(
    device, dtype, layout, padded_rows, num_heads, head_size
):
    if layout == "HND" and head_size < 32 and envs.USE_PRECOMPILED_KERNEL:
        pytest.skip("This case requires the updated source kernel.")
    MacaPlatform.import_kernels()
    torch.manual_seed(0)
    num_blocks, block_size = 4, 16
    key = torch.randn(8, num_heads, head_size, device=device, dtype=dtype)
    value = torch.randn_like(key)
    if padded_rows:
        # Interleave unused rows so the token stride exceeds the logical row size.
        key, value = key[::2], value[::2]
    cache_shape = (num_blocks, block_size, num_heads, head_size)
    if layout == "HND":
        cache_shape = (num_blocks, num_heads, block_size, head_size)
    key_cache = torch.full(cache_shape, -100, device=device, dtype=dtype)
    value_cache = key_cache.clone()
    if layout == "HND":
        key_cache, value_cache = key_cache.transpose(1, 2), value_cache.transpose(1, 2)

    slots = torch.tensor([35, -1, 17], device=device, dtype=torch.int64)
    scale = torch.ones(1, device=device, dtype=torch.float32)
    torch.ops._C_cache_ops.reshape_and_cache_flash(
        key,
        value,
        key_cache,
        value_cache,
        slots,
        "auto",
        scale,
        scale,
    )

    expected_key = torch.full_like(key_cache, -100)
    expected_value = torch.full_like(value_cache, -100)
    expected_key[2, 3], expected_key[1, 1] = key[0], key[2]
    expected_value[2, 3], expected_value[1, 1] = value[0], value[2]
    torch.testing.assert_close(key_cache, expected_key, atol=0, rtol=0)
    torch.testing.assert_close(value_cache, expected_value, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("interleaved_v", [False, True])
def test_reshape_and_cache(device, dtype, interleaved_v):
    MacaPlatform.import_kernels()
    blocks, page, heads, dim, pack = 4, 16, 2, 64, 8
    key = torch.randn(10, heads, dim, device=device, dtype=dtype)[::2]
    value = torch.randn_like(key)
    key_cache = torch.full(
        (blocks, heads, dim // pack, page, pack), -99, device=device, dtype=dtype
    )
    value_shape = (
        (blocks, heads, page // pack, dim, pack)
        if interleaved_v
        else (blocks, heads, dim, page)
    )
    value_cache = torch.full(value_shape, -98, device=device, dtype=dtype)
    slot_ids = [47, 5, -1, 16, 63]
    slots = torch.tensor(slot_ids, device=device, dtype=torch.int64)
    scale = torch.ones((), device=device)
    expected_k, expected_v = key_cache.clone(), value_cache.clone()
    for token, slot in enumerate(slot_ids):
        if slot < 0:
            continue
        page_id, offset = divmod(slot, page)
        expected_k[page_id, :, :, offset, :] = key[token].unflatten(-1, (-1, pack))
        if interleaved_v:
            expected_v[page_id, :, offset // pack, :, offset % pack] = value[token]
        else:
            expected_v[page_id, :, :, offset] = value[token]

    torch.ops._C_cache_ops.reshape_and_cache(
        key, value, key_cache, value_cache, slots, "auto", scale, scale, interleaved_v
    )

    torch.testing.assert_close(key_cache, expected_k, atol=0, rtol=0)
    torch.testing.assert_close(value_cache, expected_v, atol=0, rtol=0)


@pytest.mark.parametrize(
    "direction", [("device", "device"), ("cpu", "device"), ("device", "cpu")]
)
def test_swap_blocks(device, direction):
    MacaPlatform.import_kernels()
    devices = {"cpu": torch.device("cpu"), "device": device}
    src_device, dst_device = (devices[part] for part in direction)
    src = torch.arange(8 * 96, dtype=torch.int32).view(8, 96).to(src_device)
    dst = torch.full_like(src, -1, device=dst_device)
    src_rows, dst_rows = [0, 6, 4], [3, 0, 7]
    mapping = torch.tensor([src_rows, dst_rows], dtype=torch.int64).t()
    expected = dst.clone()
    expected[dst_rows] = src[src_rows].to(dst_device)

    torch.ops._C_cache_ops.swap_blocks(
        src, dst, src.stride(0) * src.element_size(), mapping
    )
    torch.cuda.synchronize(device)

    torch.testing.assert_close(dst, expected)


@pytest.mark.parametrize("shared_slot_mapping", [False, True])
def test_concat_and_cache_mla_grouped(device, shared_slot_mapping):
    MacaPlatform.import_kernels()
    layers, tokens, blocks, block_size, rank, rope = 3, 5, 6, 16, 512, 64
    key = torch.randn(layers, tokens, rank, dtype=torch.bfloat16, device=device)
    pos = torch.randn(layers, tokens, rope, dtype=key.dtype, device=device)
    cache = torch.full(
        (layers, blocks, block_size, rank + rope), -3, device=device, dtype=key.dtype
    )
    ids = [[17, 4, -1, 80, 21], [2, -1, 45, 5, 57], [96 - 1, 80, 3, -1, 5]]
    if shared_slot_mapping:
        ids = [ids[0]] * layers
    slots = torch.tensor(ids, dtype=torch.int64, device=device)
    if shared_slot_mapping:
        slots = slots[:1].expand(layers, -1)
    cache_ptrs = torch.tensor([c.data_ptr() for c in cache], device=device)
    expected = torch.full_like(cache, -3)
    for layer, layer_slots in enumerate(ids):
        for token, slot in enumerate(layer_slots):
            if slot >= 0:
                expected[layer, slot // block_size, slot % block_size] = torch.cat(
                    (key[layer, token], pos[layer, token])
                )

    torch.ops._C_cache_ops.concat_and_cache_mla_grouped(
        key,
        pos,
        cache_ptrs,
        slots,
        block_size,
        cache.stride(1),
        cache.stride(2),
        None,
        "auto",
    )

    torch.testing.assert_close(cache, expected, atol=0, rtol=0)


@pytest.mark.parametrize("head_size", [320, 512, 576])
@pytest.mark.parametrize("with_seq_starts", [False, True])
@pytest.mark.parametrize("context_parallel", [False, True])
def test_gather_and_maybe_dequant_cache_mla(
    device, head_size, with_seq_starts, context_parallel
):
    MacaPlatform.import_kernels()
    block_size = 16
    cache = torch.randn(8, block_size, head_size, device=device, dtype=torch.bfloat16)
    pages = [[6, 2, 5], [7, 0, 4], [3, 1, 2]]
    block_table = torch.tensor(pages, device=device, dtype=torch.int32)
    lengths = [19, 0, 9]
    offsets = [11, 0, 17] if with_seq_starts else [0] * len(lengths)
    seq_starts = (
        torch.tensor(offsets, dtype=torch.int32, device=device)
        if with_seq_starts
        else None
    )
    cu_lens = torch.tensor([0, 19, 19, 28], device=device, dtype=torch.int32)
    num_tokens = sum(lengths)
    workspace = torch.full(
        (num_tokens, head_size + 16), -3, dtype=cache.dtype, device=device
    )
    output = workspace[:, :head_size]
    if context_parallel:
        torch.ops._C_cache_ops.cp_gather_cache(
            cache, output, block_table, cu_lens, len(lengths), seq_starts
        )
    else:
        token_to_seq = torch.arange(len(lengths), device=device, dtype=torch.int32)
        token_to_seq = token_to_seq.repeat_interleave(
            torch.tensor(lengths, device=device)
        )
        torch.ops._C_cache_ops.gather_and_maybe_dequant_cache(
            cache,
            output,
            block_table,
            cu_lens,
            token_to_seq,
            num_tokens,
            "auto",
            torch.ones((), device=device),
            seq_starts,
        )

    expected = torch.cat(
        [
            cache[page_list].flatten(0, 1)[offset : offset + length]
            for page_list, offset, length in zip(pages, offsets, lengths, strict=True)
        ]
    )
    torch.testing.assert_close(output, expected, atol=0, rtol=0)
    assert (workspace[:, head_size:] == -3).all()


@pytest.mark.parametrize("on_dedicated_stream", [False, True])
def test_swap_blocks_batch(device, on_dedicated_stream):
    MacaPlatform.import_kernels()
    src = torch.arange(5 * 48, dtype=torch.int32).view(5, 48).pin_memory()
    dst = torch.full(src.shape, -1, dtype=src.dtype, device=device)
    src_rows, dst_rows, counts = [1, 4, 0], [4, 2, 3], [48, 19, 32]
    src_ptrs = torch.tensor([src[i].data_ptr() for i in src_rows], dtype=torch.int64)
    dst_ptrs = torch.tensor([dst[i].data_ptr() for i in dst_rows], dtype=torch.int64)
    sizes = torch.tensor(counts, dtype=torch.int64) * src.element_size()
    expected = torch.full_like(src, -1)
    for i, j, count in zip(src_rows, dst_rows, counts, strict=True):
        expected[j, :count] = src[i, :count]

    stream = (
        torch.cuda.Stream(device=device)
        if on_dedicated_stream
        else torch.cuda.current_stream(device)
    )
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        torch.ops._C_cache_ops.swap_blocks_batch(src_ptrs, dst_ptrs, sizes)
    stream.synchronize()

    torch.testing.assert_close(dst.cpu(), expected, atol=0, rtol=0)
