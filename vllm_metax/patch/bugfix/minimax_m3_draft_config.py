# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# -----------------------------------------------------------------------------
# Note: MiniMax DSpark ships a separate checkpoint with its own quantization.
#     Initialize it explicitly: v0.33's generic DSpark branch otherwise selects
#     the target checkpoint and rewrites unknown draft architectures to DeepSeek.
#     Other models and speculative methods retain the original __post_init__.
#
# Affected versions: vLLM 0.33.0
#     (4ba787ec9959e566d3b587a796dff07673324da5).
#
# Remove at: SpeculativeConfig recognizes DSparkMiniMaxDraftModel and resolves
#     its bundled draft directory without inheriting target quantization.
# -----------------------------------------------------------------------------

import os

from vllm.config.model import ModelConfig
from vllm.config.speculative import SpeculativeConfig

from vllm_metax.patch.utils import patch

_ARCH = "DSparkMiniMaxDraftModel"
_original_post_init = SpeculativeConfig.__post_init__


def _init_minimax_dspark(self: SpeculativeConfig) -> SpeculativeConfig:
    """The model-based v0.32 initializer, restricted to standalone MiniMax DSpark.

    Keep model construction and DSpark validation explicit. Reuse upstream's TP,
    length and parallel-config helpers; pydantic still runs _verify_args after
    __post_init__ (including positive token counts and parallel compatibility).
    """
    target = self.target_model_config
    assert target is not None
    if self.model is None:
        self.model = os.path.join(target.model, "dspark")
        if not os.path.isfile(os.path.join(self.model, "config.json")):
            raise ValueError(
                f"MiniMax DSpark checkpoint not found in {self.model}; "
                "set speculative_config.model to a standalone DSpark checkpoint"
            )

    self.prompt_lookup_max = 0
    self.prompt_lookup_min = 0
    # Do not inherit target.quantization: this checkpoint can have a different
    # quantization format. ModelConfig resolves it from the draft's own config.
    draft = ModelConfig(
        model=self.model,
        runner="draft",
        tokenizer=self.model if self.use_heterogeneous_vocab else target.tokenizer,
        tokenizer_mode=target.tokenizer_mode,
        trust_remote_code=target.trust_remote_code,
        allowed_local_media_path=target.allowed_local_media_path,
        allowed_media_domains=target.allowed_media_domains,
        dtype=target.dtype,
        seed=target.seed,
        revision=self.revision,
        code_revision=self.code_revision,
        tokenizer_revision=target.tokenizer_revision,
        max_model_len=self.max_model_len,  # type: ignore[arg-type]
        spec_target_max_model_len=target.max_model_len,
        quantization=self.quantization,
        enforce_eager=target.enforce_eager,
        max_logprobs=target.max_logprobs,
        hf_overrides=SpeculativeConfig.compose_draft_hf_overrides(
            target.hf_overrides, None
        ),
        config_format=target.config_format,
    )
    if _ARCH not in draft.architectures:
        raise ValueError(
            f"MiniMax DSpark requires a {_ARCH} checkpoint; "
            f"got {draft.architectures} from {self.model}"
        )
    self.draft_model_config = draft
    self.parallel_drafting = True
    hf_config = draft.hf_config
    if self.num_speculative_tokens is not None and hasattr(
        hf_config, "num_lookahead_tokens"
    ):
        hf_config.num_lookahead_tokens = self.num_speculative_tokens
    if self.num_speculative_tokens is None:
        # DSpark width is independent of any MTP n_predict setting.
        block_size = getattr(hf_config, "block_size", None)
        if block_size is None:
            block_size = getattr(hf_config, "dspark_block_size", None)
        self.num_speculative_tokens = block_size
    if self.num_speculative_tokens is None:
        raise ValueError(
            "A speculative model was provided, but "
            "`num_speculative_tokens` was not provided"
        )

    dspark_draft_topk = self.dspark_draft_topk
    if dspark_draft_topk is None:
        dspark_draft_topk = getattr(hf_config, "dspark_draft_topk", None)
    if dspark_draft_topk is not None:
        draft_vocab_size = (
            getattr(hf_config, "draft_vocab_size", None) or hf_config.vocab_size
        )
        if not 1 <= dspark_draft_topk <= draft_vocab_size:
            raise ValueError(
                "dspark_draft_topk must be between 1 and the "
                f"draft vocabulary size ({draft_vocab_size})"
            )
        raise ValueError(
            "dspark_draft_topk is only supported by "
            "Qwen3DSparkModel and Qwen3OmniDSparkModel"
        )

    self.draft_tensor_parallel_size = SpeculativeConfig._verify_and_get_draft_tp(
        self.target_parallel_config, self.draft_tensor_parallel_size
    )
    SpeculativeConfig._maybe_override_draft_max_position_embeddings(
        draft.hf_config, target.max_model_len
    )
    draft.max_model_len = SpeculativeConfig._maybe_override_draft_max_model_len(
        self.max_model_len, draft.max_model_len, target.max_model_len
    )
    self.draft_parallel_config = SpeculativeConfig.create_draft_parallel_config(
        self.target_parallel_config,
        self.draft_tensor_parallel_size,
        draft_model_config=draft,
    )
    if self.index_share_for_mtp_iteration is not None:
        raise ValueError(
            "index_share_for_mtp_iteration is only supported with method='mtp'"
        )
    return self


@patch("vllm.config.speculative", "SpeculativeConfig.__post_init__")
def __post_init__(self):
    target = self.target_model_config
    if (
        self.method == "dspark"
        and target is not None
        and target.hf_config.model_type in ("minimax_m3_vl", "minimax_m3_text")
    ):
        return _init_minimax_dspark(self)
    return _original_post_init(self)
