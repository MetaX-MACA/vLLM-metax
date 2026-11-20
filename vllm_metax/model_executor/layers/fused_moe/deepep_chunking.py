# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Bounded DeepEP dispatch/combine scheduling shared by modular MoE kernels."""

import torch

from vllm_metax.model_executor.layers.fused_moe import source_packed_dispatch


def apply_chunked_deepep(
    kernel, x, topk_weights, topk_ids, max_tokens, chunk_size, kwargs
):
    """All EP ranks issue the same number of bounded dispatch/combine calls.

    Attention and the scheduler retain the complete multimodal prefix; only
    token-independent MoE compute is chunked. Shared experts run once on the
    original input, preserving their output cache for the outer MoE runner.
    """
    if max_tokens <= chunk_size:
        return kernel.apply(x, topk_weights=topk_weights, topk_ids=topk_ids, **kwargs)
    result = torch.empty_like(x)
    shared = kwargs.get("shared_experts")
    shared_input = kwargs.get("shared_experts_input")
    if shared_input is None:
        shared_input = x
    local = x.shape[0]
    for start in range(0, max_tokens, chunk_size):
        end = min(start + chunk_size, local)
        if start < local:
            chunk = x[start:end]
            weights = topk_weights[start:end]
            ids = topk_ids[start:end]
        else:
            # Empty ranks must still participate in this EP collective.
            chunk = x.new_zeros((1, x.shape[-1]))
            weights = topk_weights.new_zeros((1, topk_weights.shape[-1]))
            ids = topk_ids.new_full((1, topk_ids.shape[-1]), -1)
        options = dict(kwargs)
        options["shared_experts"] = shared if start == 0 else None
        options["shared_experts_input"] = shared_input if start == 0 else None
        if start < local:
            value = kernel.apply(chunk, topk_weights=weights, topk_ids=ids, **options)
        else:
            carrier = source_packed_dispatch.dummy_carrier_for_active_scope()
            if carrier is None:
                value = kernel.apply(
                    chunk, topk_weights=weights, topk_ids=ids, **options
                )
            else:
                with source_packed_dispatch.transport_override(carrier):
                    value = kernel.apply(
                        chunk, topk_weights=weights, topk_ids=ids, **options
                    )
        if start < local:
            result[start:end].copy_(value)
    return result
