# SPDX-License-Identifier: Apache-2.0

import vllm

from vllm.logger import init_logger

logger = init_logger(__name__)

from typing import Literal, Type, get_args
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)

QuantizationMethods = Literal[
    "aqlm",
    "awq",
    # ┌------------------------  Metax Modification -------------------------┐
    "gptq",
    # └------------------------- Metax Modification -------------------------┘
    "deepspeedfp",
    "tpu_int8",
    "fp8",
    "ptpc_fp8",
    "fbgemm_fp8",
    "modelopt",
    "nvfp4",
    "marlin",
    "bitblas",
    "gguf",
    "gptq_marlin_24",
    "gptq_marlin",
    "gptq_bitblas",
    "awq_marlin",
    # ┌------------------------  Metax Modification -------------------------┐
    # "gptq",
    # └------------------------- Metax Modification -------------------------┘
    "compressed-tensors",
    "bitsandbytes",
    "qqq",
    "hqq",
    "experts_int8",
    "neuron_quant",
    "ipex",
    "quark",
    "moe_wna16",
    "torchao",
]
QUANTIZATION_METHODS: list[str] = list(get_args(QuantizationMethods))

_CUSTOMIZED_METHOD_TO_QUANT_CONFIG = {}

def get_quantization_config(quantization: str) -> Type[QuantizationConfig]:
    if quantization not in QUANTIZATION_METHODS:
        raise ValueError(f"Invalid quantization method: {quantization}")

    # lazy import to avoid triggering `torch.compile` too early
    from vllm.model_executor.layers.quantization.quark.quark import QuarkConfig

    from vllm.model_executor.layers.quantization.aqlm import AQLMConfig
    from vllm.model_executor.layers.quantization.awq import AWQConfig
    from vllm.model_executor.layers.quantization.awq_marlin import AWQMarlinConfig
    from vllm.model_executor.layers.quantization.bitblas import BitBLASConfig
    from vllm.model_executor.layers.quantization.bitsandbytes import BitsAndBytesConfig
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import CompressedTensorsConfig
    from vllm.model_executor.layers.quantization.deepspeedfp import DeepSpeedFPConfig
    from vllm.model_executor.layers.quantization.experts_int8 import ExpertsInt8Config
    from vllm.model_executor.layers.quantization.fbgemm_fp8 import FBGEMMFp8Config
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config
    from vllm.model_executor.layers.quantization.gguf import GGUFConfig
    from vllm.model_executor.layers.quantization.gptq import GPTQConfig
    from vllm.model_executor.layers.quantization.gptq_bitblas import GPTQBitBLASConfig
    from vllm.model_executor.layers.quantization.gptq_marlin import GPTQMarlinConfig
    from vllm.model_executor.layers.quantization.gptq_marlin_24 import GPTQMarlin24Config
    from vllm.model_executor.layers.quantization.hqq_marlin import HQQMarlinConfig
    from vllm.model_executor.layers.quantization.ipex_quant import IPEXConfig
    from vllm.model_executor.layers.quantization.marlin import MarlinConfig
    from vllm.model_executor.layers.quantization.modelopt import ModelOptFp8Config, ModelOptNvFp4Config
    from vllm.model_executor.layers.quantization.moe_wna16 import MoeWNA16Config
    from vllm.model_executor.layers.quantization.neuron_quant import NeuronQuantConfig
    from vllm.model_executor.layers.quantization.ptpc_fp8 import PTPCFp8Config
    from vllm.model_executor.layers.quantization.qqq import QQQConfig
    from vllm.model_executor.layers.quantization.torchao import TorchAOConfig
    from vllm.model_executor.layers.quantization.tpu_int8 import Int8TpuConfig

    method_to_config: dict[str, Type[QuantizationConfig]] = {
        "aqlm": AQLMConfig,
        "awq": AWQConfig,
        "deepspeedfp": DeepSpeedFPConfig,
        "tpu_int8": Int8TpuConfig,
        "fp8": Fp8Config,
        "fbgemm_fp8": FBGEMMFp8Config,
        "modelopt": ModelOptFp8Config,
        "nvfp4": ModelOptNvFp4Config,
        "marlin": MarlinConfig,
        "bitblas": BitBLASConfig,
        "gguf": GGUFConfig,
        "gptq_marlin_24": GPTQMarlin24Config,
    # ┌------------------------  Metax Modification -------------------------┐
        "gptq_marlin": GPTQConfig,
    # └------------------------- Metax Modification -------------------------┘
        "gptq_bitblas": GPTQBitBLASConfig,
    # ┌------------------------  Metax Modification -------------------------┐
        "awq_marlin": AWQConfig,
    # └------------------------- Metax Modification -------------------------┘
        "gptq": GPTQConfig,
        "compressed-tensors": CompressedTensorsConfig,
        "bitsandbytes": BitsAndBytesConfig,
        "ptpc_fp8": PTPCFp8Config,
        "qqq": QQQConfig,
        "hqq": HQQMarlinConfig,
        "experts_int8": ExpertsInt8Config,
        "neuron_quant": NeuronQuantConfig,
        "ipex": IPEXConfig,
        "quark": QuarkConfig,
        "moe_wna16": MoeWNA16Config,
        "torchao": TorchAOConfig,
    }
    # Update the `method_to_config` with customized quantization methods.
    method_to_config.update(_CUSTOMIZED_METHOD_TO_QUANT_CONFIG)

    return method_to_config[quantization]

vllm.model_executor.layers.quantization.QuantizationMethods = QuantizationMethods
vllm.model_executor.layers.quantization.QUANTIZATION_METHODS = QUANTIZATION_METHODS
vllm.model_executor.layers.quantization.get_quantization_config = get_quantization_config




