# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

# ------------------------------------------------------------
# Note: This is an out-of-bounds access issue in the dsv4_zero_kv_blocks_kernel
#       operator caused by an upstream bug in vllm, which has been resolved
#       in vLLM v0.28.0 via PR #52058.
#
# Affected versions: v0.27.0
#
# Remove at: Remove after v0.28.0
# ------------------------------------------------------------

from vllm_metax.patch.utils import patch
from vllm.triton_utils import tl, triton
import torch
from collections.abc import Iterable
from typing import Any

from itertools import product as iprod
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.worker.utils import AttentionGroup


from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
)


@patch("vllm.v1.worker.utils", "_zero_kv_blocks_kernel")
@triton.jit
def _zero_kv_blocks_kernel(
    seg_addrs_ptr,
    seg_block_strides_ptr,
    seg_page_sizes_ptr,
    block_ids_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    """Zero KV cache blocks across all segments in a single launch.

    Each segment is a contiguous region of one block's data.  For backends
    where blocks are outermost (block_dim=0) there is one segment per
    buffer.  For backends where K/V is outermost (block_dim=1) there are
    two segments per buffer (one for K, one for V).

    Segments may have different block strides and page sizes (e.g. packed
    KV views or models with multiple KV cache groups like MLA + DSA
    indexer). Each segment's block stride determines where a logical block
    begins, while its page size determines how many elements are cleared.

    seg_addrs_ptr holds absolute byte addresses (int64) for each segment,
    allowing segments to live in different CUDA allocations.

    Programs are mapped directly onto a 3-D grid as
    (block_index, seg_index, chunk_index).
    """
    block_index = tl.program_id(0)
    seg_index = tl.program_id(1)
    chunk_index = tl.program_id(2)
    block_stride_el = tl.load(seg_block_strides_ptr + seg_index)
    page_size_el = tl.load(seg_page_sizes_ptr + seg_index)
    chunk_offset = chunk_index.to(tl.int64) * BLOCK_SIZE
    if chunk_offset >= page_size_el:
        return
    block_id = tl.load(block_ids_ptr + block_index)
    seg_addr = tl.load(seg_addrs_ptr + seg_index)
    ptr = tl.cast(seg_addr, tl.pointer_type(tl.int32))
    block_offset = block_id.to(tl.int64) * block_stride_el.to(tl.int64)
    cols = chunk_offset + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    tl.store(
        ptr + block_offset + cols,
        tl.zeros([BLOCK_SIZE], dtype=tl.int32),
        mask=cols < page_size_el,
    )


@patch("vllm.v1.worker.utils", "KVBlockZeroer.__init__")
def __init__(
    self,
    device: torch.device,
    attn_groups_iter: Iterable["AttentionGroup"],
    kernel_block_sizes: list[int],
    cache_dtype: str,
    static_forward_context: dict[str, Any],
    runner_only_attn_layers: set[str] | None = None,
) -> None:
    """Precompute the absolute-address table for the Triton zeroing kernel.

    Each entry is the absolute byte address of a segment start on the
    GPU, so segments in different CUDA allocations work correctly.

    Block IDs from the scheduler reference logical blocks whose size
    may differ from the kernel block size (virtual block splitting).
    Each virtual block is represented as an independent segment so its
    physical block stride and zeroed page span remain independent.

    Only AttentionSpec layers are processed; Mamba layers are skipped.
    """
    self.device = device
    self._meta = None

    if runner_only_attn_layers is None:
        runner_only_attn_layers = set()
    seen_ptrs: set[int] = set()
    seg_addrs: list[int] = []
    seg_block_strides: list[int] = []
    seg_page_sizes: list[int] = []

    for group in attn_groups_iter:
        spec = group.kv_cache_spec
        if not isinstance(spec, FullAttentionSpec):
            continue
        if group.kv_cache_group_id >= len(kernel_block_sizes):
            continue
        kernel_bs = kernel_block_sizes[group.kv_cache_group_id]
        assert spec.block_size % kernel_bs == 0
        ratio = spec.block_size // kernel_bs
        block_dim = group.backend.get_kv_cache_block_dim(
            kernel_bs,
            spec.num_kv_heads,
            spec.head_size,
            cache_dtype_str=cache_dtype,
        )

        for layer_name in group.layer_names:
            if layer_name in runner_only_attn_layers:
                continue
            kv = static_forward_context[layer_name].kv_cache
            if not isinstance(kv, torch.Tensor):
                continue
            dp = kv.data_ptr()
            if dp in seen_ptrs:
                continue
            seen_ptrs.add(dp)

            el = kv.element_size()
            block_stride_bytes = kv.stride(block_dim) * el
            assert block_stride_bytes % 4 == 0
            assert kv.shape[block_dim] % ratio == 0
            outer_dims = [
                d for d in range(block_dim) if kv.stride(d) * el > block_stride_bytes
            ]
            outer_strides = [kv.stride(d) * el for d in outer_dims]
            inner_dims = [
                d for d in range(kv.ndim) if d != block_dim and d not in outer_dims
            ]
            kernel_page_bytes = el + sum(
                (kv.shape[d] - 1) * kv.stride(d) * el for d in inner_dims
            )
            assert kernel_page_bytes % 4 == 0
            logical_block_stride_bytes = block_stride_bytes * ratio
            for outer in iprod(*(range(kv.shape[d]) for d in outer_dims)):
                off_bytes = sum(i * s for i, s in zip(outer, outer_strides))
                assert (dp + off_bytes) % 4 == 0
                for virtual_index in range(ratio):
                    seg_addrs.append(
                        dp + off_bytes + virtual_index * block_stride_bytes
                    )
                    seg_block_strides.append(logical_block_stride_bytes // 4)
                    seg_page_sizes.append(kernel_page_bytes // 4)

    if not seg_addrs:
        self._meta = None
        return

    max_page_size_el = max(seg_page_sizes)
    blk_size = min(1 << (max_page_size_el - 1).bit_length(), 1024)
    self._meta = (
        torch.tensor(seg_addrs, dtype=torch.uint64, device=self.device),
        torch.tensor(seg_block_strides, dtype=torch.int64, device=self.device),
        torch.tensor(seg_page_sizes, dtype=torch.int64, device=self.device),
        (max_page_size_el + blk_size - 1) // blk_size,
        blk_size,
        len(seg_addrs),
    )


@patch("vllm.v1.worker.utils", "KVBlockZeroer.zero_block_ids")
def zero_block_ids(self, block_ids: list[int]) -> None:
    """Zero the KV cache memory for the given block IDs."""
    if not block_ids or self._meta is None:
        return
    (
        seg_addrs,
        seg_block_strides,
        seg_page_sizes,
        max_chunks,
        blk_size,
        n_segs,
    ) = self._meta
    n_blocks = len(block_ids)
    idx = async_tensor_h2d(block_ids, device=self.device, dtype=torch.int64)
    grid = (n_blocks, n_segs, max_chunks)
    _zero_kv_blocks_kernel[grid](
        seg_addrs,
        seg_block_strides,
        seg_page_sizes,
        idx,
        BLOCK_SIZE=blk_size,
    )
