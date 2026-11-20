# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Normal startup compilation for the kernels actually used by MetaX QSA."""

import sys
import time

from vllm.logger import logger


def warmup_loaded_qsa(worker) -> bool:
    """Return False for other models so their stock warmup remains available.

    Sparse attention owns a compile-only warmup for its reachable dispatch
    configurations. The torch.topk-based index selector has dynamic shapes and
    continues to JIT on demand; the unused NVIDIA-style selector is not warmed.
    """
    attn_module = sys.modules.get("vllm_metax.models.qwen4_exp.metax.qsa")
    if attn_module is None:
        return False
    owners = [
        layer
        for layer in worker.get_model().modules()
        if isinstance(layer, attn_module.Qwen4ExpQSAAttention)
    ]
    if not owners:
        return False
    from .ops.qsa import warmup_qsa_sparse_paged_attention

    runner = worker.model_runner
    start = time.perf_counter()
    signatures = set()
    profiles = []
    for owner in owners:
        group_id = next(
            i
            for i, group in enumerate(runner.kv_cache_config.kv_cache_groups)
            if owner.layer_name in group.layer_names
        )
        if worker.use_v2_model_runner:
            table = runner.block_tables.input_block_tables[group_id]
        else:
            table = runner.input_batch.block_table[group_id].get_device_tensor(
                runner.max_num_reqs
            )
        cache = owner.kv_cache
        width = owner.indexer.output_width
        signature = (
            tuple(cache.shape),
            tuple(cache.stride()),
            cache.dtype,
            tuple(table.shape),
            tuple(table.stride()),
            table.dtype,
            owner.num_heads,
            width,
        )
        if signature in signatures:
            continue
        signatures.add(signature)
        profiles.append(
            warmup_qsa_sparse_paged_attention(
                cache, table, num_query_heads=owner.num_heads, selection_width=width
            )
        )
    logger.info(
        "MetaX QSA sparse attention startup warmup: %d signatures, profiles=%s, "
        "elapsed=%.2fs. Production topk selector retains shape-dependent JIT.",
        len(signatures),
        profiles,
        time.perf_counter() - start,
    )
    return True
