# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#
# -----------------------------------------------------------------------------
# Note: When vLLM modifies the target model config via hf-overrides,
# the draft model config is generated only from the base model and does not apply hf-overrides.
#
# Affected versions: 0.26.0
#
# Remove at: Can be removed after the vLLM PR #37443 is merged.
# -----------------------------------------------------------------------------

import copy
from typing import TYPE_CHECKING, Any, Literal, get_args
from collections.abc import Mapping
from copy import deepcopy


from vllm.config.speculative import SpeculativeConfig, MTPModelTypes
from vllm.config.model import ModelConfig
from vllm.logger import init_logger

if TYPE_CHECKING:
    from transformers import PretrainedConfig
else:
    PretrainedConfig = Any

logger = init_logger(__name__)

from vllm_metax.patch.utils import patch


class _DraftHfOverrides:
    """Compose target-model HF overrides with draft-model overrides."""

    def __init__(self, target_hf_overrides: Mapping[str, Any]) -> None:
        self.target_hf_overrides = target_hf_overrides

    def __call__(self, hf_config: PretrainedConfig) -> PretrainedConfig:
        SpeculativeConfig._apply_hf_overrides_dict(
            hf_config, dict(self.target_hf_overrides)
        )
        return SpeculativeConfig.hf_config_override(hf_config)

    def get(self, key: str, default: Any = None) -> Any:
        if isinstance(self.target_hf_overrides, Mapping):
            return self.target_hf_overrides.get(key, default)
        return default


@patch(
    "vllm.config.speculative",
    "SpeculativeConfig._update_nested_hf_config",
    allow_missing=True,
)
@staticmethod
def _update_nested_hf_config(
    target: PretrainedConfig | dict[str, Any],
    updates: dict[str, Any],
) -> None:
    for key, value in updates.items():
        if isinstance(value, dict):
            if isinstance(target, dict):
                nested_target = target.get(key)
            else:
                nested_target = getattr(target, key, None)

            if nested_target is not None and (
                isinstance(nested_target, dict) or hasattr(nested_target, "__dict__")
            ):
                SpeculativeConfig._update_nested_hf_config(nested_target, value)
                continue

        if isinstance(target, dict):
            target[key] = value
        else:
            setattr(target, key, value)
    raise NotImplementedError("Complete the class method patch first")


@patch(
    "vllm.config.speculative",
    "SpeculativeConfig._apply_hf_overrides_dict",
    allow_missing=True,
)
@staticmethod
def _apply_hf_overrides_dict(
    config: PretrainedConfig,
    overrides: dict[str, Any],
) -> None:
    from transformers import PretrainedConfig

    for key, value in overrides.items():
        attr = getattr(config, key, None)
        if (
            attr is not None
            and isinstance(attr, PretrainedConfig)
            and isinstance(value, dict)
        ):
            SpeculativeConfig._update_nested_hf_config(attr, value)
        else:
            setattr(config, key, value)


@patch(
    "vllm.config.speculative",
    "SpeculativeConfig._get_draft_hf_overrides",
    allow_missing=True,
)
@staticmethod
def _get_draft_hf_overrides(target_hf_overrides: Any) -> Any:
    if isinstance(target_hf_overrides, Mapping):
        if not target_hf_overrides:
            return SpeculativeConfig.hf_config_override
        return _DraftHfOverrides(copy.deepcopy(dict(target_hf_overrides)))

    # Arbitrary callable hf_overrides can encode target-only config mutations
    # (e.g. layer-count or architecture changes) that should not leak into
    # the derived draft model. Preserve only the draft-specific rewrite here.
    return SpeculativeConfig.hf_config_override


@patch("vllm.config.speculative", "SpeculativeConfig.__post_init__")
def __post_init__(self):
    # Note: "method" is a new parameter that helps to extend the
    # configuration of non-model-based proposers, and the "model" parameter
    # will be used to set the draft model, eagle head, or additional weight
    # when needed. If users do not specify "method", the speculative method
    # will be detected automatically if possible. If the speculative method
    # can not be detected, it will be considered as the "draft_model" by
    # default.

    # infer method from user args
    # Check if the model field contains a custom module path (e.g., 'pkg.Mod')
    if (
        self.model is not None
        and "." in self.model
        and not self.model.startswith(("http://", "https://", "file://"))
        and "/" not in self.model  # not a HuggingFace repo (org/model)
    ):
        # Treat as a custom class path
        self.method = "custom_class"
    elif self.method is None:
        if self.model in ("ngram", "[ngram]"):
            self.method = "ngram"
        else:
            self.method = "draft_model"

    if self.method in get_args(MTPModelTypes) and self.method != "mtp":
        logger.warning("method `%s` is deprecated and replaced with mtp.", self.method)
        self.method = "mtp"

    if self.model is None and self.num_speculative_tokens is not None:
        if self.method == "mtp":
            if self.target_model_config is None:
                raise ValueError("target_model_config must be present for mtp")
            if self.target_model_config.hf_text_config.model_type == "deepseek_v32":
                # FIXME(luccafong): cudagraph with v32 MTP is not supported,
                # remove this when the issue is fixed.
                self.enforce_eager = True
            # use the draft model from the same model:
            self.model = self.target_model_config.model
            # Align the quantization of draft model for cases such as
            # --quantization fp8 with a bf16 checkpoint.
            if not self.quantization:
                self.quantization = self.target_model_config.quantization
        elif self.method in ("ngram", "[ngram]"):
            self.model = "ngram"
        elif self.method == "ngram_gpu":
            self.model = "ngram_gpu"
        elif self.method == "suffix":
            self.model = "suffix"
        elif self.method == "extract_hidden_states":
            self.model = "extract_hidden_states"
        elif self.method == "custom_class":
            # method was set explicitly, but model should already contain the
            # custom module path. If not, this is a configuration error.
            if self.model is None:
                raise ValueError(
                    "method='custom_class' requires 'model' to contain the "
                    "custom proposer module path (e.g., 'my_module.MyProposer')."
                )
        else:
            raise ValueError(
                "num_speculative_tokens was provided but without speculative model."
            )

    if self.method in ("ngram", "[ngram]"):
        self.method = "ngram"

    if self.method in ("ngram", "ngram_gpu"):
        # Set default values if not provided
        if self.prompt_lookup_min is None and self.prompt_lookup_max is None:
            # TODO(woosuk): Tune these values. They are arbitrarily chosen.
            self.prompt_lookup_min = 5
            self.prompt_lookup_max = 5
        elif self.prompt_lookup_min is None:
            if self.prompt_lookup_max is None:
                raise ValueError(
                    "Either prompt_lookup_max or prompt_lookup_min must be "
                    "provided when using the ngram method."
                )
            self.prompt_lookup_min = self.prompt_lookup_max
        elif self.prompt_lookup_max is None:
            if self.prompt_lookup_min is None:
                raise ValueError(
                    "Either prompt_lookup_max or prompt_lookup_min must be "
                    "provided when using the ngram method."
                )
            self.prompt_lookup_max = self.prompt_lookup_min

        # Validate values
        if self.prompt_lookup_min > self.prompt_lookup_max:
            raise ValueError(
                f"prompt_lookup_min={self.prompt_lookup_min} must "
                f"be <= prompt_lookup_max={self.prompt_lookup_max}"
            )

        # TODO: current we still need extract vocab_size from target model
        # config, in future, we may try refactor it out, and set
        # draft related config as None here.
        self.draft_model_config = self.target_model_config
        self.draft_parallel_config = self.target_parallel_config
    elif self.method == "suffix":
        self._validate_suffix_decoding()
    elif self.method == "custom_class":
        # Custom class proposer does not need a draft model.
        # It will dynamically load the user-provided class at runtime.
        logger.warning_once(
            "Using a custom class-based proposer backend. This is an "
            "experimental feature and the proposer interface is subject to "
            "breaking changes in future vLLM releases."
        )
        self.prompt_lookup_max = 0
        self.prompt_lookup_min = 0
        self.draft_model_config = self.target_model_config
        self.draft_parallel_config = self.target_parallel_config
    elif self.method == "extract_hidden_states":
        from vllm.transformers_utils.configs.extract_hidden_states import (
            ExtractHiddenStatesConfig,
        )

        # ExtractHiddenStatesModel is instantiated manually in load_model()
        # We just need to store the target model config for KV cache shape info
        self.model = "extract_hidden_states"
        self.prompt_lookup_max = 0
        self.prompt_lookup_min = 0

        if hasattr(self.draft_model_config, "hf_config"):
            hf_config = self.draft_model_config.hf_config.to_dict()
        elif (
            isinstance(self.draft_model_config, dict)
            and "hf_config" in self.draft_model_config
        ):
            hf_config = self.draft_model_config["hf_config"]
        else:
            hf_config = {}

        self.draft_model_config = copy.copy(self.target_model_config)
        self.draft_model_config.hf_config = ExtractHiddenStatesConfig(
            self.draft_model_config.hf_config, **hf_config
        )
        self.update_arch_()
        self.draft_parallel_config = self.target_parallel_config

    else:
        self.prompt_lookup_max = 0
        self.prompt_lookup_min = 0

        if self.model is not None:
            self.draft_model_config = ModelConfig(
                model=self.model,
                runner="draft",
                tokenizer=self.target_model_config.tokenizer,
                tokenizer_mode=self.target_model_config.tokenizer_mode,
                trust_remote_code=self.target_model_config.trust_remote_code,
                allowed_local_media_path=self.target_model_config.allowed_local_media_path,
                allowed_media_domains=self.target_model_config.allowed_media_domains,
                dtype=self.target_model_config.dtype,
                seed=self.target_model_config.seed,
                revision=self.revision,
                code_revision=self.code_revision,
                tokenizer_revision=self.target_model_config.tokenizer_revision,
                max_model_len=self.max_model_len,  # type: ignore[arg-type]
                spec_target_max_model_len=self.target_model_config.max_model_len,
                quantization=self.quantization,
                enforce_eager=self.target_model_config.enforce_eager,
                max_logprobs=self.target_model_config.max_logprobs,
                # /-------------------- MetaX Modification --------------------\
                # Apply the configuration passed by the user via --hf-overrides
                hf_overrides=SpeculativeConfig._get_draft_hf_overrides(
                    self.target_model_config.hf_overrides
                ),
                # \-------------------- MetaX Modification --------------------/
                config_format=self.target_model_config.config_format,
            )
            # Automatically detect the method
            if self.method in ("eagle", "eagle3", "dflash"):
                pass
            # examples:
            # yuhuili/EAGLE-LLaMA3-Instruct-8B
            # yuhuili/EAGLE3-LLaMA3.1-Instruct-8B
            # AngelSlim/Qwen3-8B_eagle3
            elif "eagle-" in self.draft_model_config.model.lower():
                self.method = "eagle"
            elif "eagle3" in self.draft_model_config.model.lower():
                self.method = "eagle3"
            elif "dflash" in self.draft_model_config.model.lower():
                self.method = "dflash"
            elif self.draft_model_config.hf_config.model_type == "medusa":
                self.method = "medusa"
            elif self.draft_model_config.hf_config.model_type == "mlp_speculator":
                self.method = "mlp_speculator"
            elif self.draft_model_config.hf_config.model_type in get_args(
                MTPModelTypes
            ):
                self.method = "mtp"
                if self.num_speculative_tokens > 1:
                    logger.warning(
                        "Enabling num_speculative_tokens > 1 will run "
                        "multiple times of forward on same MTP layer"
                        ",which may result in lower acceptance rate"
                    )
            elif self.draft_model_config.hf_config.model_type in ("longcat_flash_mtp"):
                self.method = "longcat_flash_mtp"
                if self.num_speculative_tokens > 1:
                    logger.warning(
                        "LongCat MTP models only have "
                        "one layer. Might need some code changes "
                        "to support multiple layers."
                    )
            elif self.method == "draft_model":
                pass
            else:
                raise NotImplementedError(
                    f"Unsupported speculative method: '{self.method}'"
                )

            # Replace hf_config for EAGLE draft_model
            if self.method in ("eagle", "eagle3", "dflash"):
                from vllm.transformers_utils.configs.eagle import EAGLEConfig
                from vllm.transformers_utils.configs.speculators import (
                    SpeculatorsConfig,
                )

                if isinstance(
                    self.draft_model_config.hf_config,
                    (EAGLEConfig, SpeculatorsConfig),
                ):
                    pass
                else:
                    eagle_config = EAGLEConfig(
                        self.draft_model_config.hf_config,
                        method=self.method,
                        model_type="eagle",
                    )
                    self.draft_model_config.hf_config = eagle_config
                    self.update_arch_()

            if self.method == "dflash":
                self.parallel_drafting = True

            if self.num_speculative_tokens is not None and hasattr(
                self.draft_model_config.hf_config, "num_lookahead_tokens"
            ):
                self.draft_model_config.hf_config.num_lookahead_tokens = (
                    self.num_speculative_tokens
                )

            n_predict = getattr(self.draft_model_config.hf_config, "n_predict", None)
            if n_predict is not None:
                if self.num_speculative_tokens is None:
                    # Default to max value defined in draft model config.
                    self.num_speculative_tokens = n_predict
                elif (
                    self.num_speculative_tokens > n_predict
                    and self.num_speculative_tokens % n_predict != 0
                ):
                    # Ensure divisibility for MTP module reuse.
                    raise ValueError(
                        f"num_speculative_tokens:{self.num_speculative_tokens}"
                        f" must be divisible by {n_predict=}"
                    )

            if self.num_speculative_tokens is None:
                raise ValueError(
                    "A speculative model was provided, but "
                    "`num_speculative_tokens` was not provided"
                )

            self.draft_tensor_parallel_size = (
                SpeculativeConfig._verify_and_get_draft_tp(
                    self.target_parallel_config,
                    self.draft_tensor_parallel_size,
                    self.draft_model_config.hf_config,
                )
            )

            self.draft_model_config.max_model_len = (
                SpeculativeConfig._maybe_override_draft_max_model_len(
                    self.max_model_len,
                    self.draft_model_config.max_model_len,
                    self.target_model_config.max_model_len,
                )
            )

            self.draft_parallel_config = SpeculativeConfig.create_draft_parallel_config(
                self.target_parallel_config, self.draft_tensor_parallel_size
            )
    return self


@patch(
    "vllm.model_executor.models.config",
    "NomicBertModelConfig.verify_and_update_model_config",
)
@staticmethod
def verify_and_update_model_config(model_config: "ModelConfig") -> None:
    config = model_config.hf_config

    assert config.__class__.__name__ == "NomicBertConfig"
    assert config.activation_function in ["swiglu", "gelu"]
    config.position_embedding_type = getattr(config, "position_embedding_type", "rope")

    if config.activation_function == "swiglu":
        config.hidden_act = "silu"
    else:
        config.hidden_act = config.activation_function

    assert config.mlp_fc1_bias == config.mlp_fc2_bias == config.qkv_proj_bias
    config.bias = config.qkv_proj_bias

    assert config.rotary_emb_scale_base is None
    assert not config.rotary_emb_interleaved

    config.layer_norm_eps = config.layer_norm_epsilon
    config.intermediate_size = config.n_inner
    config.hidden_size = config.n_embd
    config.num_hidden_layers = config.n_layer
    model_config.model_arch_config.hidden_size = config.hidden_size
    model_config.model_arch_config.total_num_hidden_layers = config.num_hidden_layers

    head_dim = config.hidden_size // config.num_attention_heads
    max_position_embeddings = getattr(config, "max_position_embeddings", 2048)
    max_trained_positions = getattr(
        config, "max_trained_positions", max_position_embeddings
    )

    rope_parameters = {
        "max_trained_positions": max_trained_positions,
        **(config.rope_parameters or {}),
    }

    config.rotary_kwargs = {
        "head_size": head_dim,
        "max_position": model_config.max_model_len,
        "rope_parameters": rope_parameters,
    }

    # /-------------------- MetaX Modification --------------------\
    # ignore config.rotary_scaling_factor so that for datasets shorter
    # than max_trained_positions 2048, the results are consistent
    # with SentenceTransformer.
    # The context extension uses vllm style rope_theta and rope_parameters.
    # See #17785 #18755
    if not model_config.hf_overrides and model_config.original_max_model_len is None:
        # Default
        # Reset max_model_len to max_trained_positions.
        # nomic-embed-text-v2-moe the length is set to 512
        # by sentence_bert_config.json.
        max_model_len_before = model_config.max_model_len
        max_model_len = min(model_config.max_model_len, max_trained_positions)
        model_config.max_model_len = model_config.get_and_verify_max_len(max_model_len)
        if model_config.max_model_len != max_model_len_before:
            logger.warning(
                "Nomic context extension is disabled. "
                "Changing max_model_len from %s to %s. "
                "To enable context extension, see: "
                "https://github.com/vllm-project/vllm/tree/main/examples/offline_inference/context_extension.py",
                max_model_len_before,
                model_config.max_model_len,
            )
    else:
        # We need to re-verify max_model_len to avoid lengths
        # greater than position_embedding.
        hf_text_config = model_config.hf_text_config

        if hasattr(model_config.hf_overrides, "get"):
            # Mapping-style hf_overrides_kw (including draft wrappers).
            max_model_len = model_config.hf_overrides.get(
                "max_model_len", model_config.max_model_len
            )
        else:
            # hf_overrides_fn
            # This might be overridden by sentence_bert_config.json.
            max_model_len = model_config.max_model_len
        # reset hf_text_config for recalculate_max_model_len.
        if hasattr(hf_text_config, "max_model_len"):
            delattr(hf_text_config, "max_model_len")
        hf_text_config.max_position_embeddings = max_trained_positions
        hf_text_config.rope_parameters = config.rotary_kwargs["rope_parameters"]
        # Update the cached derived_max_model_len to enforce the limit
        model_config.model_arch_config.derived_max_model_len_and_key = (
            float(max_trained_positions),
            "max_position_embeddings",
        )
        # The priority of sentence_bert_config.json is higher
        # than max_position_embeddings
        encoder_config = deepcopy(model_config.encoder_config)
        encoder_config.pop("max_seq_length", None)
        model_config.encoder_config = encoder_config
        model_config.max_model_len = model_config.get_and_verify_max_len(max_model_len)
    # \-------------------- MetaX Modification --------------------/
