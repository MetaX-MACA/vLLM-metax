# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#
# -----------------------------------------------------------------------------
# Note: Use mcoplib's optimized Gumbel sampler and its all-greedy argmax path
#     through _sample_random, preserving target sequence-length handling and
#     fused-sampler eligibility. For all-zero temperatures, pass all_greedy=True.
#
# Affected versions: Reviewed vLLM 0.32.0 (5df63a7203cd), MCOPLIB 0.4.14.
#     The former Sampler.sample override omits the new sequence-length argument.
#
# Remove at: vLLM provides an equivalent optimized sampler or a sampling-kernel
#     extension point for MetaX.
# -----------------------------------------------------------------------------
import numpy as np
import torch
from vllm.v1.sample.ops.topk_topp_sampler import (
    apply_top_k_top_p,
    flashinfer_sample,
    xpu_sample,
)

from vllm_metax.patch.utils import patch

try:
    from mcoplib.triton_gumbel_sample import gumbel_sample

    use_optimized_gumbel_sample = True
except ImportError:
    from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample

    use_optimized_gumbel_sample = False


def _sample_random(
    self,
    processed_logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    idx_mapping_np: np.ndarray,
    pos: torch.Tensor,
    top_k: torch.Tensor | None,
    top_p: torch.Tensor | None,
    use_fused_sampler: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if use_fused_sampler:
        if self.use_flashinfer:
            sampled = flashinfer_sample(processed_logits, top_k, top_p).to(torch.int64)
        else:  # Use XPU sampler
            sampled, _ = xpu_sample(processed_logits, top_k, top_p)
    else:
        processed_logits = apply_top_k_top_p(processed_logits, top_k, top_p)
        # NOTE(metax): MCOPLIB accepts an all-greedy argmax fast path.
        sampled = gumbel_sample(
            processed_logits,
            expanded_idx_mapping,
            self.sampling_states.temperature.gpu,
            self.sampling_states.seeds.gpu,
            pos,
            apply_temperature=False,
            is_drafting=False,
            use_fp64=self.use_fp64_gumbel,
            all_greedy=bool(
                np.all(self.sampling_states.temperature.np[idx_mapping_np] == 0.0)
            ),
        )
    return sampled, processed_logits


if use_optimized_gumbel_sample:
    patch("vllm.v1.worker.gpu.sample.sampler", "Sampler._sample_random")(_sample_random)
