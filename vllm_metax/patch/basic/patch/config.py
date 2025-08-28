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

def metax_compute_hash(self) -> str:
    """
    WARNING: Whenever a new field is added to this config,
    ensure that it is included in the factors list if
    it affects the computation graph.

    Provide a hash that uniquely identifies all the configs
    that affect the structure of the computation
    graph from input ids/embeddings to the final hidden states,
    excluding anything before input ids/embeddings and after
    the final hidden states.
    """
    factors: list[Any] = []

    # summarize vllm config
    vllm_factors: list[Any] = []
    from vllm import __version__
    vllm_factors.append(__version__)
    vllm_factors.append(envs.VLLM_USE_V1)
    vllm_factors.append(mx_envs.MACA_VLLM_USE_TN_2_NN)

    logger.info(f"[Plugin] Hooked compute_hash -> {metax_compute_hash}")
    
    if self.model_config:
        vllm_factors.append(self.model_config.compute_hash())
    else:
        vllm_factors.append("None")
    if self.cache_config:
        vllm_factors.append(self.cache_config.compute_hash())
    else:
        vllm_factors.append("None")
    if self.parallel_config:
        vllm_factors.append(self.parallel_config.compute_hash())
    else:
        vllm_factors.append("None")
    if self.scheduler_config:
        vllm_factors.append(self.scheduler_config.compute_hash())
    else:
        vllm_factors.append("None")
    if self.device_config:
        vllm_factors.append(self.device_config.compute_hash())
    else:
        vllm_factors.append("None")
    if self.load_config:
        vllm_factors.append(self.load_config.compute_hash())
    else:
        vllm_factors.append("None")
    if self.lora_config:
        vllm_factors.append(self.lora_config.compute_hash())
        # LoRA creates static buffers based on max_num_batched_tokens.
        # The tensor sizes and strides get captured in the torch.compile
        # graph explicitly.
        vllm_factors.append(
            str(self.scheduler_config.max_num_batched_tokens))
    else:
        vllm_factors.append("None")
    if self.speculative_config:
        vllm_factors.append(self.speculative_config.compute_hash())
    else:
        vllm_factors.append("None")
    if self.decoding_config:
        vllm_factors.append(self.decoding_config.compute_hash())
    else:
        vllm_factors.append("None")
    if self.observability_config:
        vllm_factors.append(self.observability_config.compute_hash())
    else:
        vllm_factors.append("None")
    if self.prompt_adapter_config:
        vllm_factors.append(self.prompt_adapter_config.compute_hash())
    else:
        vllm_factors.append("None")
    if self.quant_config:
        pass  # should be captured by model_config.quantization
    if self.compilation_config:
        vllm_factors.append(self.compilation_config.compute_hash())
    else:
        vllm_factors.append("None")
    if self.kv_transfer_config:
        vllm_factors.append(self.kv_transfer_config.compute_hash())
    else:
        vllm_factors.append("None")
    if self.additional_config:
        vllm_factors.append(self.additional_config.compute_hash())
    else:
        vllm_factors.append("None")
    factors.append(vllm_factors)

    hash_str = hashlib.md5(str(factors).encode(),
                            usedforsecurity=False).hexdigest()[:10]
    return hash_str

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

vllm.config.VllmConfig.compute_hash = metax_compute_hash
vllm.config.ModelConfig._verify_quantization = _verify_quantization



