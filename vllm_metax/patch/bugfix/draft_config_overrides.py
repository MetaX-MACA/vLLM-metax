# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#
# -----------------------------------------------------------------------------
# Note: Compose dictionary HF overrides with upstream draft-model rewrites. Upstream
#     already composes callable overrides but deliberately leaves dictionaries target-
#     only; MetaX checkpoints need dictionary inheritance.
#     Recognize JoyAI draft configs while preserving joyai_dense_mtp so the
#     decoder selects its dense MTP layer after the model-type rewrite.
#
# Affected versions: vLLM 0.33.0
#     (4ba787ec9959e566d3b587a796dff07673324da5).
#
# Remove at: Upstream supports inheriting dictionary HF overrides when constructing
#     draft model configs and preserves JoyAI's dense MTP identity.
# -----------------------------------------------------------------------------

"""Propagate dictionary HF overrides to MetaX draft models.

Affected: vLLM 0.30.0 (ced6857afa0e). Upstream composes callable overrides,
but deliberately leaves dictionaries target-only. MetaX checkpoints still need
the dictionary overrides. Remove when upstream supports opting into inheritance.
"""

import copy
from collections.abc import Mapping
from typing import Any

from transformers import PretrainedConfig
from vllm.config.speculative import SpeculativeConfig

from vllm_metax.patch.utils import patch

_original_compose = SpeculativeConfig.compose_draft_hf_overrides
_original_hf_config_override = SpeculativeConfig.hf_config_override


def _update_nested_hf_config(target, updates: Mapping[str, Any]) -> None:
    for key, value in updates.items():
        nested = (
            target.get(key) if isinstance(target, dict) else getattr(target, key, None)
        )
        if isinstance(value, dict) and isinstance(nested, (dict, PretrainedConfig)):
            _update_nested_hf_config(nested, value)
        elif isinstance(target, dict):
            target[key] = copy.deepcopy(value)
        else:
            setattr(target, key, copy.deepcopy(value))


@patch("vllm.config.speculative", "SpeculativeConfig.hf_config_override")
def hf_config_override(
    hf_config: PretrainedConfig,
    transformers_mtp_overrides: dict[str, Any] | None = None,
) -> PretrainedConfig:
    """Preserve standalone DSpark and JoyAI dense-MTP checkpoint identity."""
    if transformers_mtp_overrides is not None:
        return _original_hf_config_override(hf_config, transformers_mtp_overrides)
    if "DSparkMiniMaxDraftModel" in (getattr(hf_config, "architectures", None) or []):
        quantization = getattr(hf_config, "quantization_config", None)
        hf_config = getattr(hf_config, "text_config", hf_config)
        if (
            quantization is not None
            and getattr(hf_config, "quantization_config", None) is None
        ):
            hf_config.quantization_config = copy.deepcopy(quantization)
        dflash = dict(getattr(hf_config, "dflash_config", None) or {})
        dflash.setdefault("causal", False)
        hf_config.model_type = "minimax_m3_dspark"
        hf_config.update(
            {
                "architectures": ["DSparkMiniMaxDraftModel"],
                "dflash_config": dflash,
                "markov_rank": hf_config.dspark_markov_rank,
            }
        )
        return hf_config
    if hf_config.model_type == "joyai_llm_flash":
        # The MACA decoder needs this identity after the upstream type rewrite
        # to distinguish JoyAI's dense MTP layer from its routed backbone.
        hf_config.joyai_dense_mtp = True
        hf_config.model_type = "deepseek_mtp"
    return _original_hf_config_override(hf_config)


class _DraftHfOverrides:
    """Picklable dictionary overrides followed by upstream's draft rewrite."""

    def __init__(
        self,
        target_hf_overrides: Mapping[str, Any],
        transformers_mtp_overrides: dict[str, Any] | None = None,
    ) -> None:
        self.target_hf_overrides = copy.deepcopy(dict(target_hf_overrides))
        self.transformers_mtp_overrides = copy.deepcopy(transformers_mtp_overrides)

    def __call__(self, hf_config: PretrainedConfig) -> PretrainedConfig:
        for key, value in self.target_hf_overrides.items():
            nested = getattr(hf_config, key, None)
            if isinstance(nested, PretrainedConfig) and isinstance(value, dict):
                _update_nested_hf_config(nested, value)
            else:
                setattr(hf_config, key, copy.deepcopy(value))
        return SpeculativeConfig.hf_config_override(
            hf_config, self.transformers_mtp_overrides
        )


@patch("vllm.config.speculative", "SpeculativeConfig.compose_draft_hf_overrides")
def compose_draft_hf_overrides(
    target_hf_overrides,
    transformers_mtp_overrides: dict[str, Any] | None = None,
):
    # /-------------------- MetaX Modification --------------------\
    # The patched static method's module export is a staticmethod descriptor,
    # so returning its underlying function is not pickle-safe. Use the existing
    # callable object for empty overrides too (the default for draft configs).
    if target_hf_overrides is None or isinstance(target_hf_overrides, Mapping):
        return _DraftHfOverrides(
            target_hf_overrides or {},
            transformers_mtp_overrides,
        )
    # \-------------------- MetaX Modification --------------------/
    return _original_compose(target_hf_overrides, transformers_mtp_overrides)
