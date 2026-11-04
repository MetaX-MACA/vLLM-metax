# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#
# -----------------------------------------------------------------------------
# Note: Treat TeleChat3-36B-Thinking's telechat3-yarn as upstream YaRN during
#       length validation, preserving its already-scaled maximum and actual
#       rotary type. Preserve None entries for layers without RoPE. A stored
#       length of 32768 with factor=4 remains 32768, rather than 131072.
#
# Affected versions: vLLM 0.30.0 (ced6857afa0e), reviewed 2026-10-12.
#
# Remove at: Upstream maximum-length derivation recognizes telechat3-yarn as YaRN.
# -----------------------------------------------------------------------------

from copy import copy

from vllm.transformers_utils.config import is_rope_parameters_nested
from vllm.config.model import _get_and_verify_max_len as _orig_get_and_verify_max_len

from vllm_metax.patch.utils import patch


@patch("vllm.config.model", "_get_and_verify_max_len")
def _get_and_verify_max_len(
    hf_config,
    model_arch_config,
    tokenizer_config,
    max_model_len,
    disable_sliding_window,
    sliding_window,
    spec_target_max_model_len=None,
    encoder_config=None,
):
    # /-------------------- MetaX Modification --------------------\
    # TeleChat3 stores an already-scaled length, like upstream YaRN. Normalize only the
    # temporary config used for length validation; the rotary implementation
    # must still see telechat3-yarn on the actual model config.
    rope_parameters = getattr(hf_config, "rope_parameters", None)
    if rope_parameters:
        nested = is_rope_parameters_nested(rope_parameters)
        parameters = rope_parameters if nested else {"": rope_parameters}
        if any(
            rp is not None and rp.get("rope_type") == "telechat3-yarn"
            for rp in parameters.values()
        ):
            hf_config = copy(hf_config)
            parameters = {
                key: {**rp, "rope_type": "yarn"}
                if rp is not None and rp.get("rope_type") == "telechat3-yarn"
                else dict(rp)
                if rp is not None
                else None
                for key, rp in parameters.items()
            }
            hf_config.rope_parameters = parameters if nested else parameters[""]
    # \-------------------- MetaX Modification --------------------/
    return _orig_get_and_verify_max_len(
        hf_config=hf_config,
        model_arch_config=model_arch_config,
        tokenizer_config=tokenizer_config,
        max_model_len=max_model_len,
        disable_sliding_window=disable_sliding_window,
        sliding_window=sliding_window,
        spec_target_max_model_len=spec_target_max_model_len,
        encoder_config=encoder_config,
    )
