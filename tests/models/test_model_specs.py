# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import re
from pathlib import Path

import pytest

from vllm import ModelRegistry

import vllm_metax.models


MODEL_ROOT = Path(vllm_metax.models.__file__).parent
SPEC_PATHS = sorted(MODEL_ROOT.glob("*/SPEC.md"))


def _registered_architectures(text: str) -> set[str]:
    match = re.search(
        r"Registered architectures?:\s*(.*?)(?:\n- |\n\n)",
        text,
        flags=re.S,
    )
    if match:
        return set(re.findall(r"`([A-Za-z0-9_]+)`", match.group(1)))
    return set(re.findall(r"Registered architecture[^\n]*", text))


@pytest.mark.parametrize("spec_path", SPEC_PATHS, ids=lambda path: path.parent.name)
def test_model_spec_targets_plugin_runtime_and_maca(spec_path: Path):
    text = spec_path.read_text()
    assert "vLLM `v0.33.0`" in text or "vLLM v0.33.0" in text
    assert "MACA" in text
    assert "Difference matrix" in text
    architectures = _registered_architectures(text)
    assert architectures
    assert architectures <= set(ModelRegistry.models), (
        f"{spec_path.parent.name} lists unregistered architectures: "
        f"{architectures - set(ModelRegistry.models)}"
    )
    for arch in architectures:
        assert ModelRegistry.models[arch].module_name.startswith("vllm_metax.")


def test_all_architecture_specs_are_present():
    expected = {"deepseek_v4", "deepseek_v41", "glm5next", "minimax_m3", "qwen4_exp"}
    assert {path.parent.name for path in SPEC_PATHS} == expected
