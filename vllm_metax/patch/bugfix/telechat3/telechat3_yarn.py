# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#
# -----------------------------------------------------------------------------
# Note: This patch is fix max_model_len over-derivation for
#       TeleChat3-36B-Thinking (rope_type == "telechat3-yarn"),
#       remove this when upstream merge PR for the derivation fix
#
# Affected versions: vLLM 0.29.1.dev0 (98dff2a81d), verified 2026-09-17.
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
    # TeleChat3 uses YaRN's original-length * factor rule. Normalize only the
    # temporary config used for length validation; the rotary implementation
    # must still see telechat3-yarn on the actual model config.
    rope_parameters = getattr(hf_config, "rope_parameters", None)
    if rope_parameters:
        nested = is_rope_parameters_nested(rope_parameters)
        parameters = rope_parameters if nested else {"": rope_parameters}
        if any(rp.get("rope_type") == "telechat3-yarn" for rp in parameters.values()):
            hf_config = copy(hf_config)
            parameters = {
                key: {**rp, "rope_type": "yarn"}
                if rp.get("rope_type") == "telechat3-yarn"
                else dict(rp)
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
