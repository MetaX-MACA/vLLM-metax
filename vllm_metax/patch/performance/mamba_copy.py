# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd.
# All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# -----------------------------------------------
# Note: Build Mamba copy metadata per request and move hidden states with one
#       device-side kernel. No hidden-state payload traverses PCIe.
#
# Affected versions: v0.22.0
# -----------------------------------------------

import dataclasses
import os
from typing import Any

import torch

from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    get_conv_copy_spec,
    get_temporal_copy_spec,
    is_conv_state_dim_first,
)
from vllm.triton_utils import tl, triton
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker import mamba_utils
from vllm.v1.worker.gpu_input_batch import CachedRequestState


@triton.jit
def mamba_copy_blocks_kernel(
    packed_src_block_ids_ptr,
    dst_block_ids_ptr,
    accept_token_bias_ptr,
    state_base_addrs_ptr,
    state_block_strides_ptr,
    state_elem_sizes_ptr,
    state_inner_sizes_ptr,
    state_conv_widths_ptr,
    state_group_indices_ptr,
    NUM_GROUPS: tl.constexpr,
    COPY_BLOCK_SIZE: tl.constexpr,
):
    """Copy all Mamba layer states for one request with device addresses."""
    copy_req_idx = tl.program_id(0)
    state_idx = tl.program_id(1)

    group_idx = tl.load(state_group_indices_ptr + state_idx).to(tl.int64)
    copy_idx = copy_req_idx * NUM_GROUPS + group_idx

    packed_src_ids = tl.load(packed_src_block_ids_ptr + copy_idx)
    conv_src_block_id = (packed_src_ids & 0xFFFFFFFF).to(tl.int64)
    temporal_src_block_id = (packed_src_ids >> 32).to(tl.int64)
    dst_block_id = tl.load(dst_block_ids_ptr + copy_idx).to(tl.int64)
    accept_token_bias = tl.load(accept_token_bias_ptr + copy_idx).to(tl.int64)

    state_base_addr = tl.load(state_base_addrs_ptr + state_idx)
    state_block_stride = tl.load(state_block_strides_ptr + state_idx)
    state_elem_size = tl.load(state_elem_sizes_ptr + state_idx)
    state_inner_size = tl.load(state_inner_sizes_ptr + state_idx)
    conv_width = tl.load(state_conv_widths_ptr + state_idx)

    if conv_width > 0:
        src_offset = accept_token_bias * state_inner_size * state_elem_size
        src_addr = state_base_addr + conv_src_block_id * state_block_stride + src_offset
        copy_width = tl.maximum(conv_width - accept_token_bias, 0)
        copy_size = copy_width * state_inner_size * state_elem_size
    else:
        src_addr = state_base_addr + temporal_src_block_id * state_block_stride
        copy_size = state_inner_size * state_elem_size

    dst_addr = state_base_addr + dst_block_id * state_block_stride
    offsets = tl.arange(0, COPY_BLOCK_SIZE)
    for i in range(0, copy_size, COPY_BLOCK_SIZE):
        mask = i + offsets < copy_size
        src = (src_addr + i + offsets).to(tl.pointer_type(tl.uint8))
        dst = (dst_addr + i + offsets).to(tl.pointer_type(tl.uint8))
        data = tl.load(src, mask=mask)
        tl.store(dst, data, mask=mask)


@dataclasses.dataclass
class _MambaCopyKernelContext:
    """Static device metadata shared by every Mamba preprocess invocation."""

    state_base_addrs: torch.Tensor
    state_block_strides: torch.Tensor
    state_elem_sizes: torch.Tensor
    state_inner_sizes: torch.Tensor
    state_conv_widths: torch.Tensor
    state_group_indices: torch.Tensor
    num_groups: int
    total_states: int

    @classmethod
    def create(
        cls,
        kv_cache_config: KVCacheConfig,
        mamba_state_copy_funcs: tuple[MambaStateCopyFunc, ...],
        mamba_group_ids: list[int],
        forward_context: dict[str, Any],
    ) -> "_MambaCopyKernelContext":
        base_addrs: list[int] = []
        block_strides: list[int] = []
        elem_sizes: list[int] = []
        inner_sizes: list[int] = []
        conv_widths: list[int] = []
        group_indices: list[int] = []
        device: torch.device | None = None

        for group_idx, mamba_group_id in enumerate(mamba_group_ids):
            layer_names = kv_cache_config.kv_cache_groups[mamba_group_id].layer_names
            for layer_name in layer_names:
                states: list[torch.Tensor] = forward_context[layer_name].kv_cache
                assert len(states) == len(mamba_state_copy_funcs)
                for state, copy_func in zip(states, mamba_state_copy_funcs):
                    assert copy_func in (
                        get_conv_copy_spec,
                        get_temporal_copy_spec,
                    ), f"unexpected Mamba state copy function: {copy_func}"
                    device = state.device
                    elem_size = state.element_size()
                    base_addrs.append(state.data_ptr())
                    block_strides.append(state.stride(0) * elem_size)
                    elem_sizes.append(elem_size)
                    group_indices.append(group_idx)

                    if copy_func is get_conv_copy_spec:
                        # For SD this is state_len * dim. For DS, nonzero
                        # bias is rejected below and this still gives the full
                        # state size when bias is zero.
                        conv_widths.append(state.size(1))
                        inner_sizes.append(state.stride(1))
                    else:
                        conv_widths.append(0)
                        inner_sizes.append(state[0].numel())

        assert device is not None, "Mamba copy kernel has no states"
        tensor = lambda values, dtype: torch.tensor(  # noqa: E731
            values, dtype=dtype, device=device
        )
        total_states = len(base_addrs)
        return cls(
            state_base_addrs=tensor(base_addrs, torch.int64),
            state_block_strides=tensor(block_strides, torch.int64),
            state_elem_sizes=tensor(elem_sizes, torch.int32),
            state_inner_sizes=tensor(inner_sizes, torch.int64),
            state_conv_widths=tensor(conv_widths, torch.int32),
            state_group_indices=tensor(group_indices, torch.int32),
            num_groups=len(mamba_group_ids),
            total_states=total_states,
        )

    def run(
        self,
        num_copy_reqs: int,
        packed_src_block_ids: torch.Tensor,
        dst_block_ids: torch.Tensor,
        accept_token_bias: torch.Tensor,
    ) -> None:
        grid = (num_copy_reqs, self.total_states)
        mamba_copy_blocks_kernel[grid](
            packed_src_block_ids,
            dst_block_ids,
            accept_token_bias,
            self.state_base_addrs,
            self.state_block_strides,
            self.state_elem_sizes,
            self.state_inner_sizes,
            self.state_conv_widths,
            self.state_group_indices,
            NUM_GROUPS=self.num_groups,
            COPY_BLOCK_SIZE=1024,
        )


def collect_mamba_copy_meta(
    copy_bufs: mamba_utils.MambaCopyBuffers,
    kv_cache_config: KVCacheConfig,
    mamba_state_copy_funcs: tuple[MambaStateCopyFunc, ...],
    mamba_group_ids: list[int],
    src_block_idx: int,
    dest_block_idx: int,
    accept_token_bias: int,
    req_state: CachedRequestState,
    forward_context: dict[str, Any],
) -> None:
    """Collect one compact descriptor per request/group, not per layer/state."""
    if src_block_idx == dest_block_idx and accept_token_bias == 0:
        return

    accept_token_bias = int(accept_token_bias)
    has_conv = get_conv_copy_spec in mamba_state_copy_funcs
    has_temporal = get_temporal_copy_spec in mamba_state_copy_funcs
    if accept_token_bias > 0 and has_conv and is_conv_state_dim_first():
        raise NotImplementedError(
            "DS conv state layout does not yet support speculative decoding "
            "with mamba_cache_mode='align' (num_accepted_tokens > 1)."
        )

    context = getattr(copy_bufs, "_metax_mamba_copy_kernel_context", None)
    if context is None:
        context = _MambaCopyKernelContext.create(
            kv_cache_config,
            mamba_state_copy_funcs,
            mamba_group_ids,
            forward_context,
        )
        setattr(copy_bufs, "_metax_mamba_copy_kernel_context", context)

    src_ids = copy_bufs.src_ptrs.np
    dst_ids = copy_bufs.dst_ptrs.np
    biases = copy_bufs.sizes.np
    offset = copy_bufs.offset

    for mamba_group_id in mamba_group_ids:
        block_ids = req_state.block_ids[mamba_group_id]
        conv_src_id = block_ids[src_block_idx]
        temporal_src_id = (
            block_ids[src_block_idx + accept_token_bias]
            if has_temporal
            else conv_src_id
        )
        if not has_conv:
            conv_src_id = temporal_src_id

        # Two non-negative int32 physical block IDs fit in one int64 entry.
        src_ids[offset] = (int(temporal_src_id) << 32) | int(conv_src_id)
        dst_ids[offset] = block_ids[dest_block_idx]
        biases[offset] = accept_token_bias
        offset += 1

    copy_bufs.offset = offset


def do_mamba_copy_block_kernel(
    copy_bufs: mamba_utils.MambaCopyBuffers,
) -> None:
    """Launch one D2D kernel for every pending Mamba state relocation."""
    n = copy_bufs.offset
    if n == 0:
        return

    context = getattr(copy_bufs, "_metax_mamba_copy_kernel_context")
    assert n % context.num_groups == 0
    context.run(
        n // context.num_groups,
        copy_bufs.src_ptrs.copy_to_gpu(n),
        copy_bufs.dst_ptrs.copy_to_gpu(n),
        copy_bufs.sizes.copy_to_gpu(n),
    )


_ENABLE_METAX_MAMBA_COPY_KERNEL = os.getenv(
    "VLLM_USE_MACA_MAMBA_COPY_KERNEL", "0"
).strip().lower() in {"1", "true", "yes", "on"}

if _ENABLE_METAX_MAMBA_COPY_KERNEL:
    mamba_utils.collect_mamba_copy_meta = collect_mamba_copy_meta
    mamba_utils.do_mamba_copy_block = do_mamba_copy_block_kernel
