# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#
# -----------------------------------------------------------------------------
# Note: Use mcoplib's optimized Gumbel sampler and its all-greedy argmax path
#     without editing the installed vLLM package.
#
# Affected versions: vLLM 0.29.0 and 0.29.1.dev0 with the GPU Sampler.sample
#     signature below.
#
# Remove at: vLLM provides an equivalent optimized sampler or a sampling-kernel
#     extension point for MetaX.
# -----------------------------------------------------------------------------

import numpy as np
import torch

from mcoplib.triton_gumbel_sample import gumbel_sample as mcop_gumbel_sample
from vllm.config.model import PROCESSED_LOGPROBS_MODES
from vllm.v1.sample.ops.topk_topp_sampler import (
    apply_top_k_top_p,
    flashinfer_sample,
)
from vllm_metax.patch.utils import patch


@patch("vllm.v1.worker.gpu.sample.sampler", "Sampler.sample")
def sample(
    self,
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    idx_mapping: torch.Tensor,
    idx_mapping_np: np.ndarray,
    pos: torch.Tensor,
    input_ids: torch.Tensor,
    expanded_local_pos: torch.Tensor,
    return_logprobs: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    processed_logits = self.apply_sampling_params(
        logits,
        expanded_idx_mapping,
        idx_mapping,
        idx_mapping_np,
        pos,
        input_ids,
        expanded_local_pos,
        skip_top_k_top_p=True,
    )
    top_k, top_p = self.sampling_states.get_top_k_top_p(
        expanded_idx_mapping, idx_mapping_np
    )
    use_flashinfer = self.use_flashinfer and not (
        (top_k is None and top_p is None)
        or (return_logprobs and self.logprobs_mode in PROCESSED_LOGPROBS_MODES)
        or self.sampling_states.any_greedy(idx_mapping_np)
        or self.sampling_states.any_explicit_seed(idx_mapping_np)
    )

    if use_flashinfer:
        sampled = flashinfer_sample(processed_logits, top_k, top_p).to(torch.int64)
    else:
        processed_logits = apply_top_k_top_p(processed_logits, top_k, top_p)
        # /-------------------- MetaX Modification --------------------\
        sampled = mcop_gumbel_sample(
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
        # \-------------------- MetaX Modification --------------------/
    return sampled, processed_logits
