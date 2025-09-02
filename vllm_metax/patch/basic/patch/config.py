# SPDX-License-Identifier: Apache-2.0

import vllm
import hashlib

from vllm import envs
from vllm_metax import envs as mx_envs

from vllm.config import logger
from vllm_metax.patch.model_executor.patch.layers.quantization.quantization_init import (QUANTIZATION_METHODS, 
                                                                             get_quantization_config,
                                                                             QuantizationMethods)

from typing import Any, cast

def _verify_quantization(self) -> None:
    supported_quantization = QUANTIZATION_METHODS
    optimized_quantization_methods = [
        "fp8", "marlin", "modelopt", "gptq_marlin_24", "gptq_marlin",
        "awq_marlin", "fbgemm_fp8", "compressed-tensors", "experts_int8",
        "quark", "modelopt_fp4", "bitblas", "gptq_bitblas"
    ]
    if self.quantization is not None:
        self.quantization = cast(QuantizationMethods, self.quantization)

    # Parse quantization method from the HF model config, if available.
    quant_cfg = self._parse_quant_hf_config()

    if quant_cfg is not None:
        quant_method = quant_cfg.get("quant_method", "").lower()
        quant_method = quant_method.replace("compressed_tensors",
                                            "compressed-tensors")
        quant_cfg["quant_method"] = quant_method

        # Quantization methods which are overrides (i.e. they have a
        # `override_quantization_method` method) must be checked in order
        # of preference (this is particularly important for GPTQ).
        overrides = [
            "marlin",
            "bitblas",
            "gptq_marlin_24",
            "gptq_marlin",
            "gptq_bitblas",
            "awq_marlin",
            "ipex",
            "moe_wna16",
        ]
        quantization_methods = [
            q for q in supported_quantization if q not in overrides
        ]
        # Any custom overrides will be in quantization_methods so we place
        # them at the start of the list so custom overrides have preference
        # over the built in ones.
        quantization_methods = quantization_methods + overrides

        # Detect which checkpoint is it
        for name in quantization_methods:
            method = get_quantization_config(name)
            quantization_override = method.override_quantization_method(
                quant_cfg, self.quantization)
            if quantization_override is not None:
                # ┌------------------------  Metax Modification -------------------------┐
                # Raise error if the override is not custom (custom would
                # be in QUANTIZATION_METHODS but not QuantizationMethods)
                # and hasn't been added to the overrides list.
                # if (name in get_args(QuantizationMethods)
                #         and name not in overrides):
                #     raise ValueError(
                #         f"Quantization method {name} is an override but "
                #         "is has not been added to the `overrides` list "
                #         "above. This is necessary to ensure that the "
                #         "overrides are checked in order of preference.")
                # └------------------------- Metax Modification -------------------------┘
                quant_method = quantization_override
                self.quantization = quantization_override
                break

        # Verify quantization configurations.
        if self.quantization is None:
            self.quantization = quant_method
        elif self.quantization != quant_method:
            raise ValueError(
                "Quantization method specified in the model config "
                f"({quant_method}) does not match the quantization "
                f"method specified in the `quantization` argument "
                f"({self.quantization}).")

    if self.quantization is not None:
        if self.quantization not in supported_quantization:
            raise ValueError(
                f"Unknown quantization method: {self.quantization}. Must "
                f"be one of {supported_quantization}.")
        from vllm.platforms import current_platform
        current_platform.verify_quantization(self.quantization)
        if self.quantization not in optimized_quantization_methods:
            logger.warning(
                "%s quantization is not fully "
                "optimized yet. The speed can be slower than "
                "non-quantized models.", self.quantization)


vllm.config.QUANTIZATION_METHODS = QUANTIZATION_METHODS
vllm.config.QuantizationMethods = QuantizationMethods
vllm.config.get_quantization_config = get_quantization_config

vllm.config.ModelConfig._verify_quantization = _verify_quantization



