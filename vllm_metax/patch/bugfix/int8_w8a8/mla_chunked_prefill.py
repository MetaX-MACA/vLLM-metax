# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#
# -----------------------------------------------------------------------------
# Note: On the chunked-context MLA prefill path, upstream copies the gathered
# kv_c_normed tensor to kv_b_proj.weight.dtype before the projection. For
# compressed-tensors W8A8-int8 checkpoints kv_b_proj.weight is stored as
# int8, so this turns the model-dtype activations into int8 first; the int8
# scaled-MM kernel then quantizes them again with dynamic_scaled_int8_quant,
# which has no kernel overload for int8 ('Char') input and raises
# NotImplementedError. The fp8/uint8 cases are already excluded upstream;
# int8 needs the same exclusion because the int8 kernel quantizes internally.
#
# Mirrors the v0.23 fix (e587de89 "[Bugfix][MLA] Fix int8_w8a8 mla crash while
# chunk_prefill") that exists in vllm_metax's own MLA copy but was lost on the
# v0.26 base-vLLM path used by MultiHeadLatentAttentionWrapper.
#
# Affected versions: v0.26.0
#
# Remove at: Once upstream keeps model-dtype input for int8 W8A8 kv_b_proj
#            in MLACommonBaseImpl._compute_prefill_context.
# -----------------------------------------------------------------------------
from vllm_metax.model_executor.layers.attention.mla_attention import MLACommonBaseImpl
from vllm_metax.patch.utils import patch

patch(
    "vllm.model_executor.layers.attention.mla_attention",
    "MLACommonBaseImpl._compute_prefill_context",
)(MLACommonBaseImpl._compute_prefill_context)
