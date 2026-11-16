# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#
# -----------------------------------------------------------------------------
# Note: Static EPLB placement file formats and validation shared by the recorder.
#     This module does not patch upstream symbols or mutate live expert state.
#
# Affected versions: Reviewed vLLM v0.33.0 (4ba787ec9959).
#
# Remove at: Native EPLB provides equivalent static placement file I/O.
# -----------------------------------------------------------------------------
"""Read, validate and write static expert-placement payloads."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch


def read_eplb_file(path: str | Path) -> dict[str, Any]:
    """Read a JSON or tensor-only PyTorch EPLB file."""
    source = Path(path)
    if source.suffix == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
    elif source.suffix == ".pt":
        payload = torch.load(source, map_location="cpu", weights_only=True)
    else:
        raise ValueError("EPLB file must end in .json or .pt")
    if not isinstance(payload, dict):
        raise ValueError("EPLB file must contain an object")
    return payload


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.cpu().tolist()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def write_eplb_file(path: str | Path, payload: dict[str, Any]) -> None:
    """Atomically write a map that can be reloaded by read_eplb_file."""
    target = Path(path)
    if target.suffix not in (".json", ".pt"):
        raise ValueError("EPLB file must end in .json or .pt")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        if target.suffix == ".json":
            temporary.write_text(
                json.dumps(_jsonable(payload), indent=2) + "\n",
                encoding="utf-8",
            )
        else:
            torch.save(_jsonable(payload), temporary)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def _integer_tensor(value: Any, name: str) -> torch.Tensor:
    try:
        tensor = torch.as_tensor(value, device="cpu")
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"{name} must be a rectangular integer array") from exc
    if (
        tensor.dtype == torch.bool
        or torch.is_floating_point(tensor)
        or tensor.is_complex()
    ):
        raise ValueError(f"{name} must contain integers")
    return tensor.to(dtype=torch.long)


def validate_physical_to_logical_map(
    value: Any,
    *,
    num_layers: int,
    num_physical_experts: int,
    num_logical_experts: int,
) -> torch.Tensor:
    """Check dimensions, expert IDs, and coverage before moving any weights."""
    mapping = _integer_tensor(value, "physical_to_logical_map")
    expected = (num_layers, num_physical_experts)
    if tuple(mapping.shape) != expected:
        raise ValueError(
            f"physical_to_logical_map shape {tuple(mapping.shape)} != {expected}"
        )
    if bool(((mapping < 0) | (mapping >= num_logical_experts)).any()):
        raise ValueError("physical_to_logical_map contains an invalid expert ID")
    for layer in mapping:
        counts = torch.bincount(layer, minlength=num_logical_experts)
        if bool((counts == 0).any()):
            raise ValueError("Every logical expert must have a physical slot")
        if num_physical_experts == num_logical_experts and bool((counts != 1).any()):
            raise ValueError("Without redundancy, every expert must occur once")
    return mapping


def _legacy_layer_list(
    payload: dict[str, Any], num_layers: int, ep_size: int
) -> list[list[int]]:
    layers = payload["layer_list"]
    if not isinstance(layers, list) or len(layers) != num_layers:
        raise ValueError("layer_list length does not match MoE layer count")
    ordered = sorted(
        layers,
        key=lambda layer: int(
            layer["moe_layer_idx"] if "moe_layer_idx" in layer else layer["layer_id"]
        ),
    )
    positions = [
        int(layer.get("moe_layer_idx", index)) for index, layer in enumerate(ordered)
    ]
    if positions != list(range(num_layers)):
        raise ValueError("moe_layer_idx must cover all MoE layers once")
    result = []
    for layer in ordered:
        devices = sorted(layer["device_list"], key=lambda item: item["device_id"])
        if len(devices) != ep_size:
            raise ValueError("device_count does not match EP size")
        if [item["device_id"] for item in devices] != list(range(ep_size)):
            raise ValueError("device_id must cover all EP ranks once")
        result.append(
            [expert for device in devices for expert in device["device_expert"]]
        )
    return result


def select_model_payload(
    payload: dict[str, Any],
    *,
    model_name: str,
    model_index: int,
    model_count: int,
) -> dict[str, Any]:
    """Choose the matching model section, if the file contains several."""
    if "models" not in payload:
        if model_count != 1:
            raise ValueError("A multi-model EPLB file requires a models list")
        return payload
    models = payload["models"]
    if not isinstance(models, list) or len(models) != model_count:
        raise ValueError("models length does not match loaded MoE models")
    selected = models[model_index]
    if not isinstance(selected, dict):
        raise ValueError("Each models entry must be an object")
    if selected.get("model_name") != model_name:
        raise ValueError(
            f"Model {model_index} is {model_name!r}, not {selected.get('model_name')!r}"
        )
    return selected


def map_from_payload(
    payload: dict[str, Any],
    *,
    num_layers: int,
    num_physical_experts: int,
    num_logical_experts: int,
    ep_size: int,
    num_groups: int,
    num_nodes: int,
    old_map: torch.Tensor,
    policy: Any,
) -> torch.Tensor:
    """Resolve a static map or balance an exported logical-count tensor."""
    if "ep_size" in payload and int(payload["ep_size"]) != ep_size:
        raise ValueError("EPLB file EP size does not match the server")
    if "physical_to_logical_map" in payload:
        raw_map = payload["physical_to_logical_map"]
    elif "layer_list" in payload:
        raw_map = _legacy_layer_list(payload, num_layers, ep_size)
    elif "logical_count" in payload:
        counts = torch.as_tensor(payload["logical_count"], device="cpu")
        if counts.ndim == 3:
            counts = counts.sum(dim=0)
        if tuple(counts.shape) != (num_layers, num_logical_experts):
            raise ValueError("logical_count shape does not match the model")
        if bool((~torch.isfinite(counts) | (counts < 0)).any()):
            raise ValueError("logical_count must be finite and nonnegative")
        raw_map = policy.rebalance_experts(
            counts,
            num_physical_experts,
            num_groups,
            num_nodes,
            ep_size,
            old_map.cpu(),
        )
    else:
        raise ValueError(
            "EPLB file needs physical_to_logical_map, layer_list, or logical_count"
        )
    return validate_physical_to_logical_map(
        raw_map,
        num_layers=num_layers,
        num_physical_experts=num_physical_experts,
        num_logical_experts=num_logical_experts,
    )
