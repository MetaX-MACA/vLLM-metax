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
import json
from pathlib import Path
from tokenizers import Tokenizer
from transformers import AutoTokenizer
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


def _restore_joyai_backend(tokenizer, tokenizer_name, kwargs):
    model_dir = Path(tokenizer_name) / kwargs.get("subfolder", "")
    if not model_dir.is_dir():
        return tokenizer
    model_type = getattr(kwargs.get("config"), "model_type", None)
    if model_type is None:
        config_file = model_dir / "config.json"
        if not config_file.is_file():
            return tokenizer
        with config_file.open(encoding="utf-8") as stream:
            model_type = json.load(stream).get("model_type")
    if model_type != "joyai_llm_flash":
        return tokenizer

    backend = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    # Retain HF-added tokens and their flags; never silently change token IDs.
    for token_id, token in sorted(tokenizer.added_tokens_decoder.items()):
        backend.add_tokens([token])
        if backend.token_to_id(str(token)) != token_id:
            raise ValueError(f"JoyAI added token ID mismatch for {token!s}")
    tokenizer._tokenizer = backend
    return tokenizer


_original_from_pretrained = AutoTokenizer.from_pretrained


@patch("transformers", "AutoTokenizer.from_pretrained")
def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
    tokenizer = _original_from_pretrained(
        pretrained_model_name_or_path, *args, **kwargs
    )
    return _restore_joyai_backend(tokenizer, pretrained_model_name_or_path, kwargs)
