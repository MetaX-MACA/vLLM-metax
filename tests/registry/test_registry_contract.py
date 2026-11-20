# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import pytest

from vllm.model_executor.layers.quantization import get_quantization_config

import vllm_metax.registry  # noqa: F401
import vllm_metax.registry.custom_ops.activation as activation_ops
import vllm_metax.registry.custom_ops.layernorm as layernorm_ops


QUANTIZATION_CLASSES = {
    "awq": "vllm_metax.registry.quant_config.auto_awq.MacaAutoAWQConfig",
    "auto_awq": "vllm_metax.registry.quant_config.auto_awq.MacaAutoAWQConfig",
    "gptq": "vllm_metax.registry.quant_config.auto_gptq.MacaAutoGPTQConfig",
    "auto_gptq": "vllm_metax.registry.quant_config.auto_gptq.MacaAutoGPTQConfig",
    "compressed-tensors": "vllm_metax.registry.quant_config.compressed_tensors.MacaCompressedTensorsConfig",
    "fp8": "vllm_metax.registry.quant_config.fp8.MacaFp8Config",
    "mxfp4": "vllm_metax.registry.quant_config.mxfp4.MacaMxfp4Config",
    "mxfp8": "vllm_metax.registry.quant_config.minimax_m3.MacaMiniMaxM3MxFp8Config",
    "modelopt_mxfp8": "vllm_metax.registry.quant_config.minimax_m3.MacaMiniMaxM3MxFp8Config",
    "deepseek_v4_fp8": "vllm_metax.registry.quant_config.deepseek_fp8.MacaDeepseekV41FP8Config",
    "moe_wna16": "vllm_metax.registry.quant_config.moe_wna16.MacaMoeWNA16Config",
}


@pytest.mark.parametrize(("method", "qualified_name"), QUANTIZATION_CLASSES.items())
def test_metax_quantization_config_registration(method, qualified_name):
    config_class = get_quantization_config(method)
    assert f"{config_class.__module__}.{config_class.__name__}" == qualified_name


OOT_LAYER_CONTRACTS = [
    ("RMSNorm", layernorm_ops.MacaRMSNorm),
    ("GemmaRMSNorm", layernorm_ops.MacaGemmaRMSNorm),
    ("RMSNormGated", layernorm_ops.MacaRMSNormGated),
    ("SiluAndMul", activation_ops.MacaSiluAndMul),
    ("GeluAndMul", activation_ops.MacaGeluAndMul),
    ("FastGELU", activation_ops.MacaFastGELU),
]


@pytest.mark.parametrize(("class_name", "replacement"), OOT_LAYER_CONTRACTS)
def test_out_of_tree_layer_registration_is_plugin_owned(class_name, replacement):
    from vllm.model_executor.custom_op import op_registry_oot

    assert op_registry_oot[class_name] is replacement


def test_registry_modules_import_without_precompiled_kernel_fallback():
    from vllm_metax.registry.linear_kernels import fp8, int8, mp, mxfp8
    from vllm_metax.registry.quant_config import (
        auto_awq,
        auto_gptq,
        compressed_tensors,
        deepseek_fp8,
        fp8 as quant_fp8,
        minimax_m3,
        moe_wna16,
        mxfp4,
    )

    modules = [
        fp8,
        int8,
        mp,
        mxfp8,
        auto_awq,
        auto_gptq,
        compressed_tensors,
        deepseek_fp8,
        quant_fp8,
        minimax_m3,
        moe_wna16,
        mxfp4,
    ]
    assert all(module.__file__ for module in modules)


# The upstream FP8 quantization tests exercise the registered quantizer.
# This configuration test needs no FP8 arithmetic.
@pytest.mark.parametrize(
    ("custom_ops", "enabled"),
    [
        (["all"], True),
        (["none", "+quant_fp8"], True),
        (["none"], False),
        (["all", "-quant_fp8"], False),
    ],
)
def test_registered_fp8_quantizer_respects_configuration(custom_ops, enabled):
    from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
    from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape

    from vllm_metax.registry.custom_ops.layers.quantization.input_quant_fp8 import (
        QuantFP8 as MacaQuantFP8,
    )

    config = VllmConfig(compilation_config=CompilationConfig(custom_ops=custom_ops))
    with set_current_vllm_config(config):
        quantizer = QuantFP8(
            False, GroupShape.PER_TOKEN, use_ue8m0=False, compile_native=False
        )
    assert type(quantizer) is MacaQuantFP8
    expected = quantizer.forward_oot if enabled else quantizer.forward_native
    assert quantizer._forward_method == expected
