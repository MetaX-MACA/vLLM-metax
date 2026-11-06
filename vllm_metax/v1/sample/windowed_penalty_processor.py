# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Opt-in vLLM V1 adapter for sparse, modality-scoped windowed penalties."""

from typing import TYPE_CHECKING

import torch
from vllm.v1.sample.logits_processor.interface import (
    BatchUpdate,
    LogitsProcessor,
    MoveDirectionality,
)

from .windowed_penalties import (
    EXTRA_ARG,
    SparsePenaltyBatch,
    WindowedPenaltyConfig,
    WindowedPenaltyState,
)

if TYPE_CHECKING:
    from vllm import SamplingParams
    from vllm.config import VllmConfig


class WindowedPenaltyLogitsProcessor(LogitsProcessor):
    """Load explicitly with ``--logits-processors module:class``.

    The extra-argument policy applies only to generated tokens. Standard vLLM
    penalties must remain neutral to avoid applying two different history scopes.
    """

    def __init__(
        self, vllm_config: "VllmConfig", device: torch.device, is_pin_memory: bool
    ):
        self.requests: dict[int, WindowedPenaltyState] = {}
        self.batch = SparsePenaltyBatch(device, is_pin_memory)

    @classmethod
    def validate_params(cls, sampling_params: "SamplingParams") -> None:
        extra = sampling_params.extra_args or {}
        if EXTRA_ARG not in extra:
            return
        WindowedPenaltyConfig.parse(extra[EXTRA_ARG])
        if (
            sampling_params.repetition_penalty != 1.0
            or sampling_params.frequency_penalty != 0.0
            or sampling_params.presence_penalty != 0.0
        ):
            raise ValueError(
                f"{EXTRA_ARG} requires neutral standard repetition/frequency/presence penalties"
            )

    def is_argmax_invariant(self) -> bool:
        return False

    def update_state(self, batch_update: BatchUpdate | None) -> None:
        if batch_update is not None:
            # Follow the interface's removed -> added -> moved lifecycle. A new
            # request can reuse a removed slot in the same scheduler update.
            for index in batch_update.removed:
                self.requests.pop(index, None)
            for index, params, _, output_token_ids in batch_update.added:
                self.requests.pop(index, None)
                extra = params.extra_args or {}
                if EXTRA_ARG in extra:
                    self.validate_params(params)
                    config = WindowedPenaltyConfig.parse(extra[EXTRA_ARG])
                    if config.active:
                        self.requests[index] = WindowedPenaltyState(
                            config, output_token_ids
                        )
            for source, target, direction in batch_update.moved:
                if source == target:
                    continue
                old_source = self.requests.pop(source, None)
                old_target = self.requests.pop(target, None)
                if old_source is not None:
                    self.requests[target] = old_source
                if direction == MoveDirectionality.SWAP and old_target is not None:
                    self.requests[source] = old_target
        if not self.requests and self.batch.num_updates:
            self.batch.update({})

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        # Async scheduling replaces history placeholders after update_state(),
        # immediately before sampling. Read the live lists only at this point.
        if self.requests:
            self.batch.update(self.requests)
        return self.batch.apply(logits)
