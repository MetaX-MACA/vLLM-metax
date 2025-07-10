# SPDX-License-Identifier: Apache-2.0

import vllm
import hashlib

from vllm import envs
from vllm_metax_plugin.patch.hook_registry import register_patch

from vllm.config import (VllmConfig, logger, ModelConfig, SpeculativeConfig,
                         ParallelConfig, PretrainedConfig)
from vllm_metax_plugin.patch.model_executor.patch_0_8_5.quantization import (QUANTIZATION_METHODS, 
                                                                             get_quantization_config,
                                                                             QuantizationMethods)

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
    vllm_factors.append(envs.MACA_VLLM_USE_TN_2_NN)

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
    logger.info(f"[Plugin] Hooked _verify_quantization -> {_verify_quantization}")
    supported_quantization = QUANTIZATION_METHODS
    optimized_quantization_methods = [
        "fp8", "marlin", "modelopt", "gptq_marlin_24", "gptq_marlin",
        "awq_marlin", "fbgemm_fp8", "compressed-tensors", "experts_int8",
        "quark", "nvfp4", "bitblas", "gptq_bitblas"
    ]
    if self.quantization is not None:
        self.quantization = self.quantization.lower()

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

def get_layers_start_end_indices(
        self, parallel_config: "ParallelConfig") -> tuple[int, int]:
    from vllm.distributed.utils import get_pp_indices
    if (self.hf_text_config.model_type == "deepseek_mtp"
            or self.hf_config.model_type == "mimo_mtp"):
        total_num_hidden_layers = getattr(self.hf_text_config,
                                            "num_nextn_predict_layers", 0)
    else:
        total_num_hidden_layers = getattr(self.hf_text_config,
                                            "num_hidden_layers", 0)
    # the layout order is: DP x PP x TP
    pp_rank = (parallel_config.rank // parallel_config.tensor_parallel_size
                ) % parallel_config.pipeline_parallel_size
    pp_size = parallel_config.pipeline_parallel_size
    start, end = get_pp_indices(total_num_hidden_layers, pp_rank, pp_size)
    return start, end

def hf_config_override(hf_config: PretrainedConfig) -> PretrainedConfig:
    if hf_config.model_type == "deepseek_v3":
        hf_config.model_type = "deepseek_mtp"
    if hf_config.model_type == "deepseek_mtp":
        n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
        hf_config.update({
            "n_predict": n_predict,
            "architectures": ["DeepSeekMTPModel"]
        })

    if hf_config.architectures[0] == "MiMoForCausalLM":
        hf_config.model_type = "mimo_mtp"
        n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
        hf_config.update({
            "num_hidden_layers": 0,
            "n_predict": n_predict,
            "architectures": ["MiMoMTPModel"]
        })
        return hf_config

    return hf_config

def __post_init__(self):

    # Note: "method" is a new parameter that helps to extend the
    # configuration of non-model-based proposers, and the "model" parameter
    # will be used to set the draft model, eagle head, or additional weight
    # when needed. If users do not specify "method", the speculative method
    # will be detected automatically if possible. If the speculative method
    # can not be detected, it will be considered as the "draft_model" by
    # default.

    if self.model is None and self.num_speculative_tokens is not None:
        # TODO(Shangming): Refactor mtp configuration logic when supporting
        # mtp acceleration for more models besides deepseek_v3
        if self.target_model_config and \
            (self.target_model_config.hf_text_config.model_type \
                    == "deepseek_v3" or
                self.target_model_config.hf_text_config.model_type \
                    == "mimo"):
            # use the draft model from the same model:
            self.model = self.target_model_config.model
        elif self.method in ("ngram", "[ngram]"):
            self.model = "ngram"
        else:
            raise ValueError("num_speculative_tokens was provided without "
                                "speculative model.")

    # Automatically configure the method for ngram when "model" is used
    # instead of "method"
    if self.method is None and (self.model is not None
                                and self.model in ("ngram", "[ngram]")):
        self.method = "ngram"

    if self.method in ("ngram", "[ngram]"):
        # Unified to "ngram" internally
        self.method = "ngram"
        # Set default values if not provided
        if (self.prompt_lookup_min is None
                and self.prompt_lookup_max is None):
            # TODO(woosuk): Tune these values. They are arbitrarily chosen.
            self.prompt_lookup_min = 5
            self.prompt_lookup_max = 5
        elif self.prompt_lookup_min is None:
            assert self.prompt_lookup_max is not None
            self.prompt_lookup_min = self.prompt_lookup_max
        elif self.prompt_lookup_max is None:
            assert self.prompt_lookup_min is not None
            self.prompt_lookup_max = self.prompt_lookup_min

        # Validate values
        if self.prompt_lookup_min < 1:
            raise ValueError(
                f"prompt_lookup_min={self.prompt_lookup_min} must be > 0")
        if self.prompt_lookup_max < 1:
            raise ValueError(
                f"prompt_lookup_max={self.prompt_lookup_max} must be > 0")
        if self.prompt_lookup_min > self.prompt_lookup_max:
            raise ValueError(
                f"prompt_lookup_min={self.prompt_lookup_min} must "
                f"be <= prompt_lookup_max={self.prompt_lookup_max}")

        # TODO: current we still need extract vocab_size from target model
        # config, in future, we may try refactor it out, and set
        # draft related config as None here.
        self.draft_model_config = self.target_model_config
        self.draft_parallel_config = self.target_parallel_config
    else:
        self.prompt_lookup_max = 0
        self.prompt_lookup_min = 0

        if self.model is not None:
            self.draft_model_config = ModelConfig(
                model=self.model,
                task="draft",
                tokenizer=self.target_model_config.tokenizer,
                tokenizer_mode=self.target_model_config.tokenizer_mode,
                trust_remote_code=self.target_model_config.
                trust_remote_code,
                allowed_local_media_path=self.target_model_config.
                allowed_local_media_path,
                dtype=self.target_model_config.dtype,
                seed=self.target_model_config.seed,
                revision=self.revision,
                code_revision=self.code_revision,
                tokenizer_revision=self.target_model_config.
                tokenizer_revision,
                max_model_len=None,
                spec_target_max_model_len=self.target_model_config.
                max_model_len,
                quantization=self.quantization,
                enforce_eager=self.target_model_config.enforce_eager,
                max_seq_len_to_capture=self.target_model_config.
                max_seq_len_to_capture,
                max_logprobs=self.target_model_config.max_logprobs,
                hf_overrides=SpeculativeConfig.hf_config_override,
            )

            # Automatically detect the method
            if self.method in ('eagle', 'eagle3'):
                pass
            elif "eagle-" in self.draft_model_config.model.lower() or \
                    "eagle3-" in self.draft_model_config.model.lower():
                self.method = "eagle"
            elif self.draft_model_config.hf_config.model_type == "medusa":
                self.method = "medusa"
            elif (self.draft_model_config.hf_config.model_type ==
                    "mlp_speculator"):
                self.method = "mlp_speculator"
            else:
                self.method = "draft_model"

            # Replace hf_config for EAGLE draft_model
            if self.method in ("eagle", "eagle3"):
                if self.enable_chunked_prefill and not envs.VLLM_USE_V1:
                    raise ValueError(
                        "Chunked prefill and EAGLE are not compatible "
                        "when using V0.")

                from vllm.transformers_utils.configs.eagle import (
                    EAGLEConfig)
                if isinstance(self.draft_model_config.hf_config,
                                EAGLEConfig):
                    pass
                else:
                    eagle_config = EAGLEConfig(
                        self.draft_model_config.hf_config)
                    self.draft_model_config.hf_config = eagle_config

            if (self.num_speculative_tokens is not None
                    and hasattr(self.draft_model_config.hf_config,
                                "num_lookahead_tokens")):
                self.draft_model_config.hf_config.num_lookahead_tokens = \
                self.num_speculative_tokens

            n_predict = getattr(self.draft_model_config.hf_config,
                                "n_predict", None)
            if n_predict is not None:
                if self.num_speculative_tokens is None:
                    # Default to max value defined in draft model config.
                    self.num_speculative_tokens = n_predict
                elif self.num_speculative_tokens > n_predict and \
                        self.num_speculative_tokens % n_predict != 0:
                    # Ensure divisibility for MTP module reuse.
                    raise ValueError(
                        f"num_speculative_tokens:{self.num_speculative_tokens}"
                        f" must be divisible by {n_predict=}")

            self.draft_tensor_parallel_size = \
                SpeculativeConfig._verify_and_get_draft_tp(
                    self.target_parallel_config,
                    self.draft_tensor_parallel_size,
                    self.draft_model_config.hf_config
            )

            self.draft_model_config.max_model_len = (
                SpeculativeConfig._maybe_override_draft_max_model_len(
                    self.max_model_len,
                    self.draft_model_config.max_model_len,
                    self.target_model_config.max_model_len,
                ))

            self.draft_parallel_config = (
                SpeculativeConfig.create_draft_parallel_config(
                    self.target_parallel_config,
                    self.draft_tensor_parallel_size))

    if self.acceptance_method == "typical_acceptance_sampler":
        if self.posterior_threshold is None:
            self.posterior_threshold = 0.09
        if self.posterior_alpha is None:
            self.posterior_alpha = 0.3

    self._verify_args()


vllm.config.QUANTIZATION_METHODS = QUANTIZATION_METHODS
vllm.config.QuantizationMethods = QuantizationMethods
vllm.config.get_quantization_config = get_quantization_config

vllm.config.VllmConfig.compute_hash = metax_compute_hash
vllm.config.ModelConfig._verify_quantization = _verify_quantization
vllm.config.ModelConfig.get_layers_start_end_indices = get_layers_start_end_indices

vllm.config.SpeculativeConfig.hf_config_override = staticmethod(hf_config_override)
vllm.config.SpeculativeConfig.__post_init__ = __post_init__

register_patch("vllm.config", "VllmConfig.compute_hash", metax_compute_hash)
register_patch("vllm.config", "ModelConfig._verify_quantization", _verify_quantization)
register_patch("vllm.config", "ModelConfig.get_layers_start_end_indices", get_layers_start_end_indices)
register_patch("vllm.config", "SpeculativeConfig.hf_config_override", staticmethod(hf_config_override))
register_patch("vllm.config", "SpeculativeConfig.__post_init__", __post_init__)