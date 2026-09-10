# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# ---------------------------------------------------------------------
# Note: transformers v5 removed ``bytes_to_unicode`` from
#       ``transformers.models.gpt2.tokenization_gpt2`` (HF #40936).
#       Moonshot Moonlight-16B-A3B (and other GPT-2-style remote
#       tokenizers, e.g. Kimi-Linear) still import it from the old
#       location via trust_remote_code. Re-inject the symbol so these
#       custom tokenizers can be loaded.
#
# Affected versions: transformers >= 5.0
#
# Remove at: Once the affected model repos stop importing from the old
#            path.
# ---------------------------------------------------------------------
from functools import lru_cache

from vllm_metax.patch import patch


@patch(
    "transformers.models.gpt2.tokenization_gpt2", "bytes_to_unicode", allow_missing=True
)
@lru_cache
def _bytes_to_unicode():
    """GPT-2 byte-to-unicode mapping.
    Identical to the original helper shipped in transformers v4
    ``transformers.models.gpt2.tokenization_gpt2``.
    """
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
    return {b: chr(c) for b, c in zip(bs, cs)}
