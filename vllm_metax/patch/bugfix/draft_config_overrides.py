# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#
# -----------------------------------------------------------------------------
# Note: Compose dictionary HF overrides with upstream draft-model rewrites. Upstream
#     already composes callable overrides but deliberately leaves dictionaries target-
#     only; MetaX checkpoints need dictionary inheritance.
#
# Affected versions: vLLM 0.29.1.dev0 (98dff2a81d), verified 2026-09-17.
#
# Remove at: Upstream supports inheriting dictionary HF overrides when constructing
#     draft model configs.
# -----------------------------------------------------------------------------

"""Propagate dictionary HF overrides to MetaX draft models.

Affected: vLLM 0.29.1.dev0 (98dff2a81d). Upstream composes callable overrides,
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


class _DraftHfOverrides:
    """Picklable dictionary overrides followed by upstream's draft rewrite."""

    def __init__(self, target_hf_overrides: Mapping[str, Any]) -> None:
        self.target_hf_overrides = copy.deepcopy(dict(target_hf_overrides))

    def __call__(self, hf_config: PretrainedConfig) -> PretrainedConfig:
        for key, value in self.target_hf_overrides.items():
            nested = getattr(hf_config, key, None)
            if isinstance(nested, PretrainedConfig) and isinstance(value, dict):
                _update_nested_hf_config(nested, value)
            else:
                setattr(hf_config, key, copy.deepcopy(value))
        return SpeculativeConfig.hf_config_override(hf_config)


@patch("vllm.config.speculative", "SpeculativeConfig.compose_draft_hf_overrides")
def compose_draft_hf_overrides(target_hf_overrides):
    # /-------------------- MetaX Modification --------------------\
    if isinstance(target_hf_overrides, Mapping) and target_hf_overrides:
        return _DraftHfOverrides(target_hf_overrides)
    # \-------------------- MetaX Modification --------------------/
    return _original_compose(target_hf_overrides)
