# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Restore the local JoyAI backend before vLLM caches the HF wrapper."""

# -------------------------------------------------------
# Note: transformers v5 中 LlamaTokenizerFast 解码空格/换行乱码。
#       替换 __init__ 后空格出现但前面多了 'G'，还需修正 decoder。
#       纯内存操作，不动任何模型文件。
# -------------------------------------------------------

import json
from pathlib import Path
from tokenizers import Tokenizer
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from vllm_metax.patch.utils import patch


# TODO(Hank): This is dangerous, remove ASAP.
@patch("transformers", "LlamaTokenizerFast.__init__")
def __init__(self, *args, **kwargs):
    PreTrainedTokenizerFast.__init__(self, *args, **kwargs)
    # 修正 decoder：Llama tokenizer 使用 ByteLevel decoder，
    # 但 transformers v5 的 LlamaTokenizerFast 配置了错误的 decoder，
    # 导致 Ġ (space prefix) 被解码为 "G" 或 "G "。
    # 重新设置为标准 ByteLevel decoder。
    from tokenizers import decoders as _decoders

    self._tokenizer.decoder = _decoders.ByteLevel(
        add_prefix_space=False, trim_offsets=True
    )


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
