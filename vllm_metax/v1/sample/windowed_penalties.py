# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Bounded, sparse output-token penalties, independent of engine initialization."""

from collections import Counter
from dataclasses import dataclass
import json
import math
from typing import Any

import torch

EXTRA_ARG = "metax_windowed_penalty"
MAX_WINDOW = 65536
MAX_RANGES = 256
MAX_EXEMPT_TOKENS = 4096
MAX_POLICY_CHARS = 131072


@dataclass(frozen=True)
class WindowedPenaltyConfig:
    window: int
    repetition: float = 1.0
    frequency: float = 0.0
    presence: float = 0.0
    token_ranges: tuple[tuple[int, int], ...] = ()
    exempt_token_ids: frozenset[int] = frozenset()

    @classmethod
    def parse(cls, value: Any) -> "WindowedPenaltyConfig":
        # OpenAI's vllm_xargs admits scalar strings, but not nested objects.
        if isinstance(value, str):
            if len(value) > MAX_POLICY_CHARS:
                raise ValueError(
                    f"{EXTRA_ARG} JSON exceeds {MAX_POLICY_CHARS} characters"
                )
            try:
                value = json.loads(value)
            except (ValueError, RecursionError) as exc:
                raise ValueError(f"{EXTRA_ARG} must be a valid JSON object") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{EXTRA_ARG} must be an object or JSON object string")
        if any(not isinstance(key, str) for key in value):
            raise ValueError(f"{EXTRA_ARG} field names must be strings")
        allowed = {
            "window",
            "repetition",
            "frequency",
            "presence",
            "token_ranges",
            "exempt_token_ids",
        }
        if unknown := set(value) - allowed:
            raise ValueError(f"Unknown {EXTRA_ARG} fields: {sorted(unknown)}")
        window = value.get("window")
        if type(window) is not int or not 1 <= window <= MAX_WINDOW:
            raise ValueError(f"window must be an integer in [1, {MAX_WINDOW}]")
        penalties = {}
        for name, default in (
            ("repetition", 1.0),
            ("frequency", 0.0),
            ("presence", 0.0),
        ):
            number = value.get(name, default)
            if isinstance(number, bool) or not isinstance(number, (float, int)):
                raise ValueError(f"{name} must be a finite number")
            try:
                number = float(number)
            except OverflowError as exc:
                raise ValueError(f"{name} must be a finite number") from exc
            if not math.isfinite(number):
                raise ValueError(f"{name} must be a finite number")
            if name == "repetition" and not 1e-6 <= number <= 1e6:
                raise ValueError("repetition must be in [1e-6, 1e6]")
            if name != "repetition" and not -2 <= number <= 2:
                raise ValueError(f"{name} must be in [-2, 2]")
            penalties[name] = number
        ranges = value.get("token_ranges", [])
        if not isinstance(ranges, (list, tuple)) or len(ranges) > MAX_RANGES:
            raise ValueError(f"token_ranges must contain at most {MAX_RANGES} ranges")
        normalized = []
        for pair in ranges:
            if (
                not isinstance(pair, (list, tuple))
                or len(pair) != 2
                or any(type(x) is not int for x in pair)
                or not 0 <= pair[0] < pair[1]
            ):
                raise ValueError(
                    "token_ranges entries must be [start, end) with 0 <= start < end"
                )
            normalized.append(tuple(pair))
        normalized.sort()
        if any(a[1] > b[0] for a, b in zip(normalized, normalized[1:], strict=False)):
            raise ValueError("token_ranges must not overlap")
        exempt = value.get("exempt_token_ids", [])
        if (
            not isinstance(exempt, (list, tuple))
            or len(exempt) > MAX_EXEMPT_TOKENS
            or any(type(x) is not int or x < 0 for x in exempt)
        ):
            raise ValueError(
                f"exempt_token_ids must contain at most {MAX_EXEMPT_TOKENS} nonnegative integers"
            )
        return cls(
            window,
            **penalties,
            token_ranges=tuple(normalized),
            exempt_token_ids=frozenset(exempt),
        )

    @property
    def active(self) -> bool:
        return self.repetition != 1 or self.frequency != 0 or self.presence != 0

    def includes(self, token: int) -> bool:
        return token not in self.exempt_token_ids and (
            not self.token_ranges
            or any(start <= token < end for start, end in self.token_ranges)
        )


@dataclass
class WindowedPenaltyState:
    config: WindowedPenaltyConfig
    output_token_ids: list[int]


class SparsePenaltyBatch:
    """Pack unique (row, token) updates once, without a batch-by-vocab buffer.

    History is re-read from the bounded suffix on every update, including when
    batch membership is unchanged. This also handles rollback and same-length
    history replacement without retaining stale device counters.
    """

    def __init__(self, device: torch.device, is_pin_memory: bool = False):
        self.device = torch.device(device)
        self.is_pin_memory = is_pin_memory and self.device.type != "cpu"
        self.indices = torch.empty((2, 0), dtype=torch.int64, device=self.device)
        self.parameters = torch.empty((2, 0), dtype=torch.float32, device=self.device)

    @property
    def num_updates(self) -> int:
        return self.indices.shape[1]

    def update(self, requests: dict[int, WindowedPenaltyState]) -> None:
        rows, tokens, repetition, offsets = [], [], [], []
        for row, state in requests.items():
            config = state.config
            if not config.active:
                continue
            counts = Counter(state.output_token_ids[-config.window :])
            for token, count in counts.items():
                if token < 0 or not config.includes(token):
                    continue
                rows.append(row)
                tokens.append(token)
                repetition.append(config.repetition)
                offsets.append(config.frequency * count + config.presence)
        self.indices = self._tensor([rows, tokens], torch.int64)
        self.parameters = self._tensor([repetition, offsets], torch.float32)

    def _tensor(self, data: list[list], dtype: torch.dtype) -> torch.Tensor:
        cpu = torch.tensor(data, dtype=dtype, pin_memory=self.is_pin_memory)
        return cpu.to(device=self.device, non_blocking=self.is_pin_memory)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.num_updates:
            return logits
        if logits.ndim != 2 or logits.dtype not in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
        ):
            raise ValueError("logits must be a 2D float16, bfloat16 or float32 tensor")
        if logits.device != self.indices.device:
            raise ValueError("logits and penalty state must be on the same device")
        if logits.device.type == "cuda":
            from .windowed_penalties_triton import apply_sparse_penalties

            return apply_sparse_penalties(logits, self.indices, self.parameters)
        return self.apply_torch(logits)

    def apply_torch(self, logits: torch.Tensor) -> torch.Tensor:
        """Portable sparse reference, also useful for GPU differential tests."""
        rows, tokens = self.indices
        valid = (
            (rows >= 0)
            & (rows < logits.shape[0])
            & (tokens >= 0)
            & (tokens < logits.shape[1])
        )
        rows, tokens = rows[valid], tokens[valid]
        penalties, offsets = self.parameters[:, valid]
        original = logits[rows, tokens].float()
        updated = (
            torch.where(original > 0, original / penalties, original * penalties)
            - offsets
        )
        logits[rows, tokens] = updated.to(logits.dtype)
        return logits
