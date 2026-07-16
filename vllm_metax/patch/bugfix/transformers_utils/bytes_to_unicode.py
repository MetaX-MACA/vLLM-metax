# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

# ------------------------------------------------------------
# Note: transformers v5 从 gpt2.tokenization_gpt2 移除了
#       bytes_to_unicode。Kimi-Linear 等模型的自定义 tokenizer
#       仍依赖该函数。在此恢复。
# ------------------------------------------------------------

from functools import lru_cache


@lru_cache()
def bytes_to_unicode():
    """GPT-2 byte-to-unicode mapping, identical to the original in
    transformers v4 tokenization_gpt2."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


import transformers.models.gpt2.tokenization_gpt2 as _gpt2_tok
_gpt2_tok.bytes_to_unicode = bytes_to_unicode
