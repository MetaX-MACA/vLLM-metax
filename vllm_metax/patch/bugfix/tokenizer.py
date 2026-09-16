# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

# -------------------------------------------------------
# Note: transformers v5 中 LlamaTokenizerFast 解码空格/换行乱码。
#       替换 __init__ 后空格出现但前面多了 'G'，还需修正 decoder。
#       纯内存操作，不动任何模型文件。
# -------------------------------------------------------

from transformers import LlamaTokenizerFast, PreTrainedTokenizerFast

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
