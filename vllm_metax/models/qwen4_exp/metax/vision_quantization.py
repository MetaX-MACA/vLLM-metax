# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Keep floating-point Qwen vision checkpoints out of text W8A8 schemes."""

import json
from pathlib import Path

from safetensors import safe_open

from vllm.logger import logger
from vllm.model_executor.layers.quantization import QuantizationConfig


def vision_quant_config(
    quant_config: QuantizationConfig | None, model_path: str
) -> QuantizationConfig | None:
    """Use checkpoint headers, not a global layer-prefix exception.

    This is a model-owned compatibility rule for local W8A8 checkpoints whose
    broad Linear scheme also matches their unquantized vision tower. Unknown
    formats, remote repositories and quantized vision weights retain the normal
    quantization configuration. Only metadata is read; tensors are never loaded.
    """
    if quant_config is None or quant_config.get_name() != "compressed-tensors":
        return quant_config
    scheme = quant_config.target_scheme_map.get("Linear")
    if not scheme or quant_config.transform_config is not None:
        return quant_config
    for name in ("weights", "input_activations"):
        args = scheme.get(name)
        if args is None or args.num_bits != 8 or args.type != "int":
            return quant_config

    root = Path(model_path)
    index = root / "model.safetensors.index.json"
    if not index.is_file():
        return quant_config
    weight_map = json.loads(index.read_text())["weight_map"]
    vision_weights = {
        name: filename
        for name, filename in weight_map.items()
        if name.startswith(("visual.", "model.visual.")) and name.endswith(".weight")
    }
    if not vision_weights:
        return quant_config
    by_file: dict[str, list[str]] = {}
    for name, filename in vision_weights.items():
        by_file.setdefault(filename, []).append(name)
    checked = 0
    for filename, names in by_file.items():
        with safe_open(root / filename, framework="pt", device="cpu") as handle:
            for name in names:
                if handle.get_slice(name).get_dtype() not in {"BF16", "F16", "F32"}:
                    return quant_config
                checked += 1
    logger.info(
        "Qwen vision tower: %d floating-point weight headers verified; "
        "text and MTP retain compressed-tensors W8A8 quantization.",
        checked,
    )
    return None
