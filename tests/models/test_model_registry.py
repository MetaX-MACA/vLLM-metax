# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import importlib

import pytest

from vllm import ModelRegistry

import vllm_metax.models
from vllm_metax.models import register_model


EXPECTED_REGISTRATIONS = {
    "CosyVoice2ForCausalLM": ("vllm_metax.models.cosyvoice2", "CosyVoice2ForCausalLM"),
    "DeepSeekMTPModel": ("vllm_metax.models.deepseek_mtp", "DeepSeekMTP"),
    "DeepseekV2ForCausalLM": ("vllm_metax.models.deepseek_v2", "DeepseekV2ForCausalLM"),
    "DeepseekV3ForCausalLM": ("vllm_metax.models.deepseek_v2", "DeepseekV3ForCausalLM"),
    "DeepseekV32ForCausalLM": ("vllm_metax.models.deepseek_v2", "DeepseekV3ForCausalLM"),
    "DeepseekV32MTPModel": ("vllm_metax.models.deepseek_mtp", "DeepSeekMTP"),
    "GlmMoeDsaForCausalLM": ("vllm_metax.models.deepseek_v2", "GlmMoeDsaForCausalLM"),
    "DeepseekV4ForCausalLM": ("vllm_metax.models.deepseek_v4", "DeepseekV4ForCausalLM"),
    "DeepSeekV4MTPModel": ("vllm_metax.models.deepseek_v4", "DeepSeekV4MTP"),
    "DSparkDraftModel": (
        "vllm_metax.models.deepseek_v4",
        "DSparkDeepseekV4ForCausalLM",
    ),
    "Step3p5MTP": ("vllm_metax.models.step3p5_mtp", "Step3p5MTP"),
    "Step3p5ForCausalLM": ("vllm_metax.models.step3p5", "Step3p5ForCausalLM"),
    "Qwen4ExpForCausalLM": ("vllm_metax.models.qwen4_exp", "Qwen4ExpForCausalLM"),
    "Qwen4ExpForConditionalGeneration": (
        "vllm_metax.models.qwen4_exp",
        "Qwen4ExpForConditionalGeneration",
    ),
    "Qwen4ExpMTP": ("vllm_metax.models.qwen4_exp", "Qwen4ExpMTP"),
    "DeepseekV41ForCausalLM": (
        "vllm_metax.models.deepseek_v41",
        "DeepseekV41ForCausalLM",
    ),
    "DSparkDeepseekV4ForCausalLM": (
        "vllm_metax.models.deepseek_v41",
        "DSparkDeepseekV4ForCausalLM",
    ),
    "DSparkV41DraftModel": (
        "vllm_metax.models.deepseek_v41",
        "DSparkDeepseekV4ForCausalLM",
    ),
    "Glm5NextForCausalLM": (
        "vllm_metax.models.glm5next",
        "Glm5NextForCausalLM",
    ),
    "Glm5NextForConditionalGeneration": (
        "vllm_metax.models.glm5next",
        "Glm5NextForConditionalGeneration",
    ),
    "MiniMaxM3SparseForCausalLM": (
        "vllm_metax.models.minimax_m3",
        "MiniMaxM3SparseForCausalLM",
    ),
    "MiniMaxM3SparseForConditionalGeneration": (
        "vllm_metax.models.minimax_m3",
        "MiniMaxM3SparseForConditionalGeneration",
    ),
    "MiniMaxM3MTP": ("vllm_metax.models.minimax_m3", "MiniMaxM3MTP"),
    "DSparkMiniMaxDraftModel": (
        "vllm_metax.models.minimax_m3",
        "MiniMaxM3DSparkForCausalLM",
    ),
}


@pytest.mark.parametrize(("arch", "expected"), EXPECTED_REGISTRATIONS.items())
def test_plugin_model_registration_points_to_plugin(arch, expected):
    registered = ModelRegistry.models[arch]
    assert (registered.module_name, registered.class_name) == expected


@pytest.mark.parametrize(("arch", "expected"), EXPECTED_REGISTRATIONS.items())
def test_registered_plugin_model_class_loads(arch, expected):
    registered = ModelRegistry.models[arch]
    model_class = registered.load_model_cls()
    assert model_class.__module__.startswith("vllm_metax.")
    assert model_class.__module__ == expected[0] or model_class.__module__.startswith(
        expected[0] + "."
    )
    assert model_class.__name__ == registered.class_name


def test_registration_function_is_idempotent_for_plugin_paths():
    before = {
        arch: (info.module_name, info.class_name)
        for arch, info in ModelRegistry.models.items()
        if info.module_name.startswith("vllm_metax.")
    }
    register_model()
    after = {
        arch: (info.module_name, info.class_name)
        for arch, info in ModelRegistry.models.items()
        if info.module_name.startswith("vllm_metax.")
    }
    assert after == before


@pytest.mark.parametrize(
    "module_name",
    [
        "vllm_metax.models.cosyvoice2",
        "vllm_metax.models.deepseek_mtp",
        "vllm_metax.models.deepseek_v2",
        "vllm_metax.models.step3p5",
        "vllm_metax.models.step3p5_mtp",
        "vllm_metax.models.minimax_m3_dspark",
        "vllm_metax.models.deepseek_v4",
        "vllm_metax.models.deepseek_v41",
        "vllm_metax.models.glm5next",
        "vllm_metax.models.minimax_m3",
        "vllm_metax.models.qwen4_exp",
    ],
)
def test_plugin_model_package_imports(module_name):
    module = importlib.import_module(module_name)
    assert module.__file__.startswith(str(vllm_metax.models.__file__).rsplit("/", 1)[0])
