# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#
# -----------------------------------------------------------------------------
# Note: Preserve the complete serialized Llama tokenizer backend. Transformers 5.17.0
#     rebuilds it from vocab/merges and can lose the decoder and pre-tokenizer stored in
#     tokenizer.json. Keep native initialization when no serialized file is supplied.
#
# Affected versions: Transformers 5.17.0, verified 2026-09-17.
#
# Remove at: Transformers Llama native-format conversion preserves the complete
#     serialized tokenizer backend, including its decoder and pre-tokenizer.
# -----------------------------------------------------------------------------

"""Preserve serialized Llama tokenizer backends, including ByteLevel decoders.

Affected: Transformers 5.17.0. Its native-format conversion reconstructs Llama
backends from vocab/merges and loses tokenizer.json's decoder/pre-tokenizer.
Remove when that conversion preserves the complete serialized backend.
"""

import os
from transformers import LlamaTokenizerFast

from vllm_metax.patch.utils import patch

_original_convert = LlamaTokenizerFast.convert_to_native_format.__func__


@patch("transformers", "LlamaTokenizerFast.convert_to_native_format")
def convert_to_native_format(cls, trust_remote_code=False, **kwargs):
    # /-------------------- MetaX Modification --------------------\
    tokenizer_file = kwargs.get("tokenizer_file")
    if tokenizer_file is not None and os.path.isfile(tokenizer_file):
        # The upstream full-backend loading branch preserves all serialization
        # settings. This flag only selects that branch; it executes no remote code.
        return _original_convert(cls, trust_remote_code=True, **kwargs)
    # \-------------------- MetaX Modification --------------------/
    return _original_convert(cls, trust_remote_code=trust_remote_code, **kwargs)
