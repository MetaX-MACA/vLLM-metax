#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Reproduce the GLM-5.3-Flash INT8 W8A8 dummy model; never copy weights."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import regex as re
import shutil


TEXT_LAYERS = [0, 3, 44]
VISION_LAYERS = [0]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if source == output:
        raise ValueError("Source and output directories must differ")
    assets = json.loads((source / "assets_manifest.json").read_text())
    for item in assets["assets"]:
        if sha256(source / item["path"]) != item["sha256"]:
            raise ValueError(f"Source asset changed: {item['path']}")

    original = json.loads((source / "config.json").read_text())
    config = copy.deepcopy(original)
    text, vision = config["text_config"], config["vision_config"]
    original_text = original["text_config"]
    original_depth = original_text["num_hidden_layers"]
    assert original_depth == 45
    assert original_text["num_nextn_predict_layers"] == 1
    mapping = {old: new for new, old in enumerate(TEXT_LAYERS)}
    mapping[original_depth] = len(TEXT_LAYERS)
    text["num_hidden_layers"] = len(TEXT_LAYERS)
    for field in ("layer_types", "mlp_layer_types", "indexer_types"):
        text[field] = [original_text[field][old] for old in TEXT_LAYERS]
    text["first_k_dense_replace"] = sum(x == "dense" for x in text["mlp_layer_types"])
    for field in ("kda_layers", "full_attn_layers"):
        old_layers = original_text["linear_attn_config"][field]
        text["linear_attn_config"][field] = [
            new for new, old in enumerate(TEXT_LAYERS) if old in old_layers
        ]
    vision["depth"] = len(VISION_LAYERS)

    text_pattern = re.compile(r"^(model\.layers\.)(\d+)(\..*)$")
    vision_pattern = re.compile(r"^(visual\.blocks\.)(\d+)(\..*)$")
    vision_mapping = {old: new for new, old in enumerate(VISION_LAYERS)}
    excluded = []
    for name in original["quantization_config"]["modules_to_not_convert"]:
        match = text_pattern.fullmatch(name)
        if match:
            old = int(match[2])
            if old in mapping:
                excluded.append(f"{match[1]}{mapping[old]}{match[3]}")
            continue
        match = vision_pattern.fullmatch(name)
        if match:
            old = int(match[2])
            if old in vision_mapping:
                excluded.append(f"{match[1]}{vision_mapping[old]}{match[3]}")
            continue
        excluded.append(name)
    # User explicitly requested replacing FP8 metadata with MetaX INT8 W8A8.
    # This changes dummy parameter construction; it does not convert a checkpoint.
    config.pop("quantization_config")
    config["compression_config"] = {
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "input_activations": {
                    "dynamic": True,
                    "group_size": None,
                    "num_bits": 8,
                    "observer": "minmax",
                    "observer_kwargs": {},
                    "strategy": "token",
                    "symmetric": True,
                    "type": "int",
                },
                "weights": {
                    "dynamic": False,
                    "group_size": None,
                    "num_bits": 8,
                    "observer": "minmax",
                    "observer_kwargs": {},
                    "strategy": "channel",
                    "symmetric": True,
                    "type": "int",
                },
            }
        },
        "format": "int-quantized",
        "ignore": excluded,
        "quant_method": "compressed-tensors",
    }

    assert text["layer_types"] == [
        "linear_attention",
        "deepseek_sparse_attention",
        "linear_attention",
    ]
    assert text["mlp_layer_types"] == ["dense", "sparse", "sparse"]
    assert text["linear_attn_config"]["kda_layers"] == [0, 2]
    assert text["linear_attn_config"]["full_attn_layers"] == [1]
    assert len(excluded) == len(set(excluded))

    output.mkdir(parents=True, exist_ok=True)
    for item in assets["assets"]:
        name = item["path"]
        if name == "config.json":
            continue
        destination_name = "SOURCE_MODEL_CARD.md" if name == "README.md" else name
        shutil.copyfile(source / name, output / destination_name)
    shutil.copyfile(
        source / "assets_manifest.json", output / "source_assets_manifest.json"
    )
    (output / "source_fp8_quantization_config.json").write_text(
        json.dumps(original["quantization_config"], indent=2) + "\n"
    )
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    manifest = {
        "mode": "dummy",
        "source": assets["source_page"],
        "source_config_sha256": sha256(source / "config.json"),
        "output_config_sha256": sha256(output / "config.json"),
        "text_original_to_new": {str(k): v for k, v in mapping.items()},
        "text_backbone_original_layers": TEXT_LAYERS,
        "mtp_original_to_new": {str(original_depth): len(TEXT_LAYERS)},
        "vision_original_to_new": {str(k): v for k, v in vision_mapping.items()},
        "changes": {
            "text_config.num_hidden_layers": [original_depth, len(TEXT_LAYERS)],
            "text_config.first_k_dense_replace": [
                original_text["first_k_dense_replace"],
                text["first_k_dense_replace"],
            ],
            "text_config.layer_types": [
                original_text["layer_types"],
                text["layer_types"],
            ],
            "text_config.mlp_layer_types": [
                original_text["mlp_layer_types"],
                text["mlp_layer_types"],
            ],
            "text_config.indexer_types": [
                original_text["indexer_types"],
                text["indexer_types"],
            ],
            "text_config.linear_attn_config.kda_layers": [
                original_text["linear_attn_config"]["kda_layers"],
                text["linear_attn_config"]["kda_layers"],
            ],
            "text_config.linear_attn_config.full_attn_layers": [
                original_text["linear_attn_config"]["full_attn_layers"],
                text["linear_attn_config"]["full_attn_layers"],
            ],
            "vision_config.depth": [
                original["vision_config"]["depth"],
                vision["depth"],
            ],
            "quantization": [
                "FP8 e4m3 block [128,128]",
                "compressed-tensors INT8 W8A8: dynamic per-token activation; static per-channel weights",
            ],
            "quantization_excluded_modules_count": [
                len(original["quantization_config"]["modules_to_not_convert"]),
                len(excluded),
            ],
        },
        "quantization_conversion": {
            "authorization": "模型config改一下，改成vllm-metax支持的 int8_w8a8的量化的config",
            "scope": "dummy configuration only; no real checkpoint weights converted",
            "source_quantization_metadata": "source_fp8_quantization_config.json",
            "active_config_key": "compression_config",
            "method": "compressed-tensors",
            "format": "int-quantized",
            "weight": "symmetric INT8 static per-channel",
            "activation": "symmetric INT8 dynamic per-token",
            "original_exclusions": "Retained after exact text/MTP/vision layer remapping",
        },
        "retained": [
            "KDA+Dense, DSA/kpool+MoE and KDA+MoE backbone combinations",
            "mHC expansion, inter-layer fusion and final contraction",
            "index_kpool=4, topk=2048, compressed cache and incomplete tail",
            "288 routed experts, 1 shared expert, top-8 routing",
            "one configured MTP layer with DSA+MoE",
            "image/video processor, vision attention/MLP/downsample/merger",
            "all original widths, head/LoRA dimensions and vocabulary",
            "user-authorized INT8 W8A8 with original remapped BF16 exclusions",
        ],
        "limitations": [
            "Only repeated layers removed; this is not an accuracy/performance checkpoint",
            "No weights or weight index; --load-format dummy is mandatory",
            "MTP configuration retained; MetaX MTP registration/execution must be verified separately",
            "DCP remains unsupported by the reviewed MetaX adaptation",
            "Long input must exceed topk=2048 (and effective pool selection range) to prove sparse selection",
        ],
        "assets": [
            {"path": p.name, "bytes": p.stat().st_size, "sha256": sha256(p)}
            for p in sorted(output.iterdir())
            if p.is_file()
            and p.name
            in {
                "config.json",
                "configuration.json",
                "generation_config.json",
                "processor_config.json",
                "tokenizer_config.json",
                "tokenizer.json",
                "chat_template.jinja",
                "SOURCE_MODEL_CARD.md",
                "LICENSE",
            }
        ],
    }
    (output / "trim_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "text_layers": len(TEXT_LAYERS),
                "vision_layers": len(VISION_LAYERS),
                "excluded_modules": len(excluded),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
