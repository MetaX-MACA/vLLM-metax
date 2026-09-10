# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# -------------------------------------------------------
# Note: This patch is fix max_model_len over-derivation for
#       TeleChat3-36B-Thinking (rope_type == "telechat3-yarn"),
#       remove this when upstream merge PR for the derivation fix
#
# Affected versions: v0.25.0+
# -------------------------------------------------------
from vllm.logger import init_logger
from vllm.transformers_utils.config import is_rope_parameters_nested
from vllm.config.model import _get_and_verify_max_len as _orig_get_and_verify_max_len

from vllm_metax.patch.utils import patch

logger = init_logger(__name__)


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
    """Cap max_model_len for telechat3-yarn.
    TeleChat3-36B-Thinking declares max_position_embeddings=32768 with
    rope_scaling.factor=4.0 and original_max_position_embeddings=8192, so
    its real context length is 8192 * 4 = 32768 (consistent with the
    official modeling_telechat3.py). vLLM derives 32768 * 4 = 131072
    because _get_and_verify_max_len only special-cases rope_type == "yarn";
    sequences longer than 32768 then index past the RoPE cos/sin cache
    (rotary_embedding/base.py: index_select(0, positions)) and trap on
    device with "vectorized gather kernel index out of bounds".
    """
    result = _orig_get_and_verify_max_len(
        hf_config=hf_config,
        model_arch_config=model_arch_config,
        tokenizer_config=tokenizer_config,
        max_model_len=max_model_len,
        disable_sliding_window=disable_sliding_window,
        sliding_window=sliding_window,
        spec_target_max_model_len=spec_target_max_model_len,
        encoder_config=encoder_config,
    )
    rope_parameters = getattr(hf_config, "rope_parameters", None)
    if not rope_parameters:
        return result
    rp_list = (
        rope_parameters.values()
        if is_rope_parameters_nested(rope_parameters)
        else [rope_parameters]
    )
    for rp in rp_list:
        if rp.get("rope_type") != "telechat3-yarn":
            continue
        original = rp.get("original_max_position_embeddings")
        factor = rp.get("factor")
        if not original or not factor:
            break
        cap = int(original * factor)
        if result > cap:
            logger.warning_once(
                "telechat3-yarn: capping max_model_len %d -> %d "
                "(original_max_position_embeddings %d * factor %s). "
                "Longer sequences would overrun the RoPE cos/sin cache.",
                result,
                cap,
                original,
                factor,
            )
            result = cap
        break
    return result
