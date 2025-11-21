# 2025 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import vllm
from vllm_metax_plugin.patch.hook_registry import register_patch
from vllm.logger import init_logger

logger = init_logger(__name__)

import torch
from typing import Tuple, Optional
from vllm_metax_plugin import _custom_ops as ops
from vllm.attention.ops.paged_attn import _PARTITION_SIZE

def split_kv_cache(
    kv_cache: torch.Tensor,
    num_kv_heads: int,
    head_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    logger.info(f"[Plugin] Hooked split_kv_cache -> {split_kv_cache}")

    # Support page attention backend
    x = 32 // kv_cache.element_size()
    num_blocks = kv_cache.shape[1]

    key_cache = kv_cache[0]
    key_cache = key_cache.view(num_blocks, num_kv_heads, head_size // x,
                                -1, x)
    value_cache = kv_cache[1]
    # Support page attention backend
    value_cache = value_cache.view(num_blocks, num_kv_heads, -1, head_size)
    return key_cache, value_cache

def write_to_paged_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> None:
    # Support page attention backend
    ops.reshape_and_cache_new(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping.flatten(),
        kv_cache_dtype,
        k_scale,
        v_scale,
    )
    
def forward_decode(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    kv_cache_dtype: str,
    num_kv_heads: int,
    scale: float,
    alibi_slopes: Optional[torch.Tensor],
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    tp_rank: int = 0,
    blocksparse_local_blocks: int = 0,
    blocksparse_vert_stride: int = 0,
    blocksparse_block_size: int = 64,
    blocksparse_head_sliding_step: int = 0,
) -> torch.Tensor:
    if blocksparse_vert_stride is not None and blocksparse_vert_stride > 1:
        # use blocksparse paged attention
        block_size = value_cache.shape[2] # # Support page attention backend
        assert (blocksparse_block_size > 0 and
                blocksparse_block_size % block_size == 0), \
            (f"{blocksparse_block_size=} needs to be a multiple of"
                f"{block_size=} used in block_tables.")

    output = torch.empty_like(query)
    block_size = value_cache.shape[2]
    num_seqs, num_heads, head_size = query.shape
    max_num_partitions = ((max_seq_len + _PARTITION_SIZE - 1) //
                            _PARTITION_SIZE)
    # NOTE(woosuk): We use a simple heuristic to decide whether to use
    # PagedAttention V1 or V2. If the number of partitions is 1, we use
    # V1 to avoid the overhead of reduction. Also, if the number of
    # sequences or heads is large, we use V1 since there is enough work
    # to parallelize.
    # TODO(woosuk): Tune this heuristic.
    # For context len > 8192, use V2 kernel to avoid shared memory shortage.
    use_v1 = (max_seq_len <= 8192
                and (max_num_partitions == 1 or num_seqs * num_heads > 512))

    if use_v1:
        # Run PagedAttention V1.
        ops.paged_attention_v1(
            output,
            query,
            key_cache,
            value_cache,
            num_kv_heads,
            scale,
            block_tables,
            seq_lens,
            block_size,
            max_seq_len,
            alibi_slopes,
            kv_cache_dtype,
            k_scale,
            v_scale,
            tp_rank,
            blocksparse_local_blocks,
            blocksparse_vert_stride,
            blocksparse_block_size,
            blocksparse_head_sliding_step,
        )
    else:
        # Run PagedAttention V2.
        assert _PARTITION_SIZE % block_size == 0
        tmp_output = torch.empty(
            size=(num_seqs, num_heads, max_num_partitions, head_size),
            dtype=output.dtype,
            device=output.device,
        )
        block_count = torch.zeros(
            size=(num_seqs, num_heads),
            dtype=torch.int,
            device=output.device,
        )
        exp_sums = torch.empty(
            size=(num_seqs, num_heads, max_num_partitions),
            dtype=torch.float32,
            device=output.device,
        )
        max_logits = torch.empty_like(exp_sums)
        block_count_init_once = False
        ops.paged_attention_v2(
            output,
            exp_sums,
            max_logits,
            tmp_output,
            block_count,
            query,
            key_cache,
            value_cache,
            num_kv_heads,
            scale,
            block_tables,
            seq_lens,
            block_size,
            max_seq_len,
            alibi_slopes,
            kv_cache_dtype,
            k_scale,
            v_scale,
            tp_rank,
            blocksparse_local_blocks,
            blocksparse_vert_stride,
            blocksparse_block_size,
            blocksparse_head_sliding_step,
            block_count_init_once
        )
    return output

vllm.attention.ops.paged_attn.PagedAttention.split_kv_cache = staticmethod(split_kv_cache)
vllm.attention.ops.paged_attn.PagedAttention.write_to_paged_cache = staticmethod(write_to_paged_cache)
vllm.attention.ops.paged_attn.PagedAttention.forward_decode = staticmethod(forward_decode)

register_patch("vllm.attention.ops.paged_attn", "PagedAttention.split_kv_cache", staticmethod(split_kv_cache))
register_patch("vllm.attention.ops.paged_attn", "PagedAttention.write_to_paged_cache", staticmethod(write_to_paged_cache))
register_patch("vllm.attention.ops.paged_attn", "PagedAttention.forward_decode", staticmethod(forward_decode))