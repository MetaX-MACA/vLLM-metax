# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

# -------------------------------------------------------
# Note: transformers v5 中 LlamaTokenizerFast 解码空格/换行乱码。
#       替换 __init__ 后空格出现但前面多了 'G'，还需修正 decoder。
#       纯内存操作，不动任何模型文件。
# -------------------------------------------------------

import logging

from vllm.tokenizers.hf import CachedHfTokenizer

_logger = logging.getLogger("vllm_metax.tokenizer")

_original = CachedHfTokenizer.from_pretrained.__func__

_patch_applied = False


def _apply_patch():
    global _patch_applied
    if _patch_applied:
        return
    _patch_applied = True

    from transformers import LlamaTokenizerFast, PreTrainedTokenizerFast

    def _new_init(self, *args, **kwargs):
        PreTrainedTokenizerFast.__init__(self, *args, **kwargs)
        # 修正 decoder：Llama tokenizer 使用 ByteLevel decoder，
        # 但 transformers v5 的 LlamaTokenizerFast 配置了错误的 decoder，
        # 导致 Ġ (space prefix) 被解码为 "G" 或 "G "。
        # 重新设置为标准 ByteLevel decoder。
        from tokenizers import decoders as _decoders
        self._tokenizer.decoder = _decoders.ByteLevel(
            add_prefix_space=False, trim_offsets=True
        )

    LlamaTokenizerFast.__init__ = _new_init
    _logger.warning(
        "vllm_metax: LlamaTokenizerFast.__init__ + decoder fix applied "
        "(transformers v5 space/garbled fix)"
    )


def _patched_from_pretrained(cls, path_or_repo_id, *args, **kwargs):
    _apply_patch()
    return _original(cls, path_or_repo_id, *args, **kwargs)


CachedHfTokenizer.from_pretrained = classmethod(_patched_from_pretrained)
