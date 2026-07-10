# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# ---------------------------------------------------------------------------
# Note: Work around CUDA stream priority assertions (least_priority == 0) in
#       mcPytorch and argument signature mismatches in vLLM's KV cache
#       offloading backend (truncating swap_blocks_batch to 3 arguments).
#
# Affected versions: vLLM v1 engine / mcPytorch current release
# Remove at: after Metax C++ backend supports 4-argument swap_blocks_batch
#            and flexible CUDA stream priority ranges.
# ---------------------------------------------------------------------------


import torch
import logging

logger = logging.getLogger(__name__)


def apply_all_vllm_patches():
    _patch_cuda_stream_priority()
    _patch_swap_blocks_batch()


def _patch_cuda_stream_priority():
    """
    Patch 1: bypass mcPytorch Stream Priority assertion crash (least_priority == 0)
    """
    if not hasattr(torch.cuda.Stream, "priority_range"):
        return

    original_priority_range = torch.cuda.Stream.priority_range

    def _mock_priority_range(*args, **kwargs):
        return (0, 0)

    torch.cuda.Stream.priority_range = _mock_priority_range
    logger.info(
        "[Metax Patch] Applied patch: torch.cuda.Stream.priority_range -> (0, 0)"
    )


def _patch_swap_blocks_batch():
    """
    Patch 2: fix C++ Mismatched operator parameter counts
    """
    try:
        import vllm._custom_ops
    except ImportError:
        logger.warning(
            "[Metax Patch] Could not import vllm._custom_ops. Skip patching swap_blocks_batch."
        )
        return

    if not hasattr(vllm._custom_ops, "swap_blocks_batch"):
        return

    original_swap = vllm._custom_ops.swap_blocks_batch

    def _patched_swap_blocks_batch(*args, **kwargs):
        safe_args = args[:3]
        return torch.ops._C_cache_ops.swap_blocks_batch(*safe_args)

    vllm._custom_ops.swap_blocks_batch = _patched_swap_blocks_batch
    logger.info(
        "[Metax Patch] Applied patch: vllm._custom_ops.swap_blocks_batch (truncated to 3 args)"
    )


apply_all_vllm_patches()
