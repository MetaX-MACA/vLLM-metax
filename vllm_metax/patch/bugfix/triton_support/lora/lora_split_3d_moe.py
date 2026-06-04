# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#
# Patch split 3D MoE LoRA checkpoints to vLLM-compatible 2D expert LoRA keys.
#
# Input checkpoint format:
#   base_model.model.model.layers.0.mlp.experts.w1.lora_A.weight
#   base_model.model.model.layers.0.mlp.experts.w1.lora_B.weight
#   base_model.model.model.layers.0.mlp.experts.w2.lora_A.weight
#   base_model.model.model.layers.0.mlp.experts.w2.lora_B.weight
#   base_model.model.model.layers.0.mlp.experts.w3.lora_A.weight
#   base_model.model.model.layers.0.mlp.experts.w3.lora_B.weight
#
# vLLM-compatible output format:
#   base_model.model.model.layers.0.mlp.experts.0.gate_proj.lora_A.weight
#   base_model.model.model.layers.0.mlp.experts.0.gate_proj.lora_B.weight
#   base_model.model.model.layers.0.mlp.experts.0.down_proj.lora_A.weight
#   base_model.model.model.layers.0.mlp.experts.0.down_proj.lora_B.weight
#   base_model.model.model.layers.0.mlp.experts.0.up_proj.lora_A.weight
#   base_model.model.model.layers.0.mlp.experts.0.up_proj.lora_B.weight

from __future__ import annotations

import os
import regex as re
from typing import Any

import safetensors
import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

_ORIG_SAFE_OPEN = safetensors.safe_open

_SPLIT_3D_MOE_RE = re.compile(
    r"^(?P<prefix>.*\.mlp\.experts)\."
    r"(?P<part>w[123])\."
    r"(?P<ab>lora_[AB])\.weight$"
)

_UNEMBED_TOKENS_RE = re.compile(
    r"^(?P<prefix>base_model\.model\.)"
    r"model\.unembed_tokens\."
    r"(?P<ab>lora_[AB])\.weight$"
)

_UNEMBED_POLICY = os.getenv("VLLM_METAX_UNEMBED_TOKENS_LORA", "filter").strip().lower()

_WARNED_UNEMBED = False


class _Split3DMoESafeOpen:
    def __init__(self, inner_cm: Any, filename: str):
        self._inner_cm = inner_cm
        self._filename = filename
        self._reader: Any = None

        self._enabled = False
        self._keys: list[str] = []
        self._mapping: dict[str, str | tuple[str, int]] = {}
        self._source_tensors: dict[str, torch.Tensor] = {}

    def __enter__(self):
        self._reader = self._inner_cm.__enter__()
        self._prepare()
        return self

    def __exit__(self, exc_type, exc, tb):
        return self._inner_cm.__exit__(exc_type, exc, tb)

    def keys(self):
        if self._enabled:
            return self._keys
        return self._reader.keys()

    def get_tensor(self, key: str):
        if not self._enabled:
            return self._reader.get_tensor(key)

        ref = self._mapping[key]

        if isinstance(ref, tuple):
            source_key, expert_id = ref
            tensor = self._source_tensors[source_key]

            # Some tensors are shared across all experts:
            #   (1, rank, hidden)
            #   (1, hidden, rank)
            # Treat first dim == 1 as broadcast-to-all-experts.
            if tensor.shape[0] == 1:
                return tensor[0].contiguous()

            return tensor[expert_id].contiguous()

        return self._reader.get_tensor(ref)

    def metadata(self):
        if hasattr(self._reader, "metadata"):
            return self._reader.metadata()
        return None

    def _prepare(self) -> None:
        original_keys = list(self._reader.keys())

        split_3d_keys = [key for key in original_keys if _SPLIT_3D_MOE_RE.match(key)]
        if not split_3d_keys:
            self._enabled = False
            return

        self._enabled = True

        groups: dict[str, dict[tuple[str, str], str]] = {}

        for key in original_keys:
            if ".modules_to_save." in key:
                continue

            m = _SPLIT_3D_MOE_RE.match(key)
            if m:
                prefix = m.group("prefix")
                part = m.group("part")
                ab = m.group("ab")
                groups.setdefault(prefix, {})[(part, ab)] = key
                continue

            mapped_key = self._map_non_expert_key(key)
            if mapped_key is None:
                continue

            self._mapping[mapped_key] = key

        for prefix, entries in groups.items():
            self._expand_split_3d_group(prefix, entries)

        self._keys = sorted(self._mapping.keys())

        logger.info(
            "MetaX split 3D MoE LoRA patch enabled for %s: "
            "%d split expert keys expanded to %d compatible keys.",
            self._filename,
            len(split_3d_keys),
            len(self._keys),
        )

    def _map_non_expert_key(self, key: str) -> str | None:
        global _WARNED_UNEMBED

        m = _UNEMBED_TOKENS_RE.match(key)
        if not m:
            return key

        if _UNEMBED_POLICY == "filter":
            if not _WARNED_UNEMBED:
                logger.warning(
                    "MetaX split 3D MoE LoRA patch drops unembed_tokens LoRA "
                    "weights by default. Set "
                    "VLLM_METAX_UNEMBED_TOKENS_LORA=lm_head to map them to "
                    "lm_head instead."
                )
                _WARNED_UNEMBED = True
            return None

        if _UNEMBED_POLICY == "lm_head":
            ab = m.group("ab")
            return f"base_model.model.lm_head.{ab}.weight"

        return key

    def _expand_split_3d_group(
        self,
        prefix: str,
        entries: dict[tuple[str, str], str],
    ) -> None:
        required = (
            ("w1", "lora_A"),
            ("w1", "lora_B"),
            ("w2", "lora_A"),
            ("w2", "lora_B"),
            ("w3", "lora_A"),
            ("w3", "lora_B"),
        )

        missing = [item for item in required if item not in entries]
        if missing:
            raise ValueError(
                f"Split 3D MoE LoRA group {prefix} is incomplete. "
                f"Missing tensors: {missing}"
            )

        for item in required:
            source_key = entries[item]
            if source_key not in self._source_tensors:
                self._source_tensors[source_key] = self._reader.get_tensor(source_key)

        self._validate_shapes(prefix, entries)

        num_experts = self._infer_num_experts(entries)

        part_to_proj = {
            "w1": "gate_proj",
            "w2": "down_proj",
            "w3": "up_proj",
        }

        for expert_id in range(num_experts):
            for part, proj in part_to_proj.items():
                for ab in ("lora_A", "lora_B"):
                    source_key = entries[(part, ab)]
                    target_key = f"{prefix}.{expert_id}.{proj}.{ab}.weight"
                    self._mapping[target_key] = (source_key, expert_id)

    def _infer_num_experts(
        self,
        entries: dict[tuple[str, str], str],
    ) -> int:
        return max(
            self._source_tensors[source_key].shape[0] for source_key in entries.values()
        )

    def _validate_shapes(
        self,
        prefix: str,
        entries: dict[tuple[str, str], str],
    ) -> None:
        tensors = {
            item: self._source_tensors[source_key]
            for item, source_key in entries.items()
        }

        for item, tensor in tensors.items():
            if tensor.ndim != 3:
                raise ValueError(
                    f"Expected split 3D MoE LoRA tensor {entries[item]} "
                    f"under {prefix} to be 3D, got shape {tuple(tensor.shape)}."
                )

        num_experts = max(t.shape[0] for t in tensors.values())
        for item, tensor in tensors.items():
            if tensor.shape[0] not in (1, num_experts):
                raise ValueError(
                    f"Inconsistent expert dimension for {entries[item]}: "
                    f"shape={tuple(tensor.shape)}, expected first dim 1 "
                    f"or {num_experts}."
                )

        w1_a = tensors[("w1", "lora_A")]
        w1_b = tensors[("w1", "lora_B")]
        w2_a = tensors[("w2", "lora_A")]
        w2_b = tensors[("w2", "lora_B")]
        w3_a = tensors[("w3", "lora_A")]
        w3_b = tensors[("w3", "lora_B")]

        rank = w1_a.shape[1]
        hidden = w1_a.shape[2]
        intermediate = w1_b.shape[1]

        checks = [
            ("w1_B rank", w1_b.shape[2], rank),
            ("w2_A rank", w2_a.shape[1], rank),
            ("w2_A intermediate", w2_a.shape[2], intermediate),
            ("w2_B hidden", w2_b.shape[1], hidden),
            ("w2_B rank", w2_b.shape[2], rank),
            ("w3_A rank", w3_a.shape[1], rank),
            ("w3_A hidden", w3_a.shape[2], hidden),
            ("w3_B intermediate", w3_b.shape[1], intermediate),
            ("w3_B rank", w3_b.shape[2], rank),
        ]

        for name, got, expected in checks:
            if got != expected:
                raise ValueError(
                    f"Invalid split 3D MoE LoRA shape under {prefix}: "
                    f"{name} got {got}, expected {expected}."
                )


def _patched_safe_open(filename: Any, *args: Any, **kwargs: Any):
    inner_cm = _ORIG_SAFE_OPEN(filename, *args, **kwargs)
    path = os.fspath(filename)

    if os.path.basename(path) != "adapter_model.safetensors":
        return inner_cm

    return _Split3DMoESafeOpen(inner_cm, path)


def _install_patch() -> None:
    global _ORIG_SAFE_OPEN

    current_safe_open = safetensors.safe_open
    if getattr(current_safe_open, "_metax_split_3d_moe_patch", False):
        return

    _ORIG_SAFE_OPEN = current_safe_open

    setattr(_patched_safe_open, "_metax_split_3d_moe_patch", True)
    safetensors.safe_open = _patched_safe_open

    logger.info("Installed MetaX split 3D MoE LoRA compatibility patch.")


_install_patch()
