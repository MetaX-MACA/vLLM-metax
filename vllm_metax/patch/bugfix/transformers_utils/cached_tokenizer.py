# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

# ------------------------------------------------------------
# Note: get_cached_tokenizer 创建的 CachedTokenizer 代理类没有
#       透传 batch_encode_plus 方法。GLM-4 等模型的自定义
#       tokenization_chatglm.py 在 apply_chat_template 中调用了
#       batch_encode_plus。此 patch 在 get_cached_tokenizer 包装
#       tokenizer 后补充 batch_encode_plus 方法。
# ------------------------------------------------------------

from vllm.tokenizers.hf import get_cached_tokenizer as _original_get_cached
from functools import wraps


def get_cached_tokenizer(tokenizer):
    cached = _original_get_cached(tokenizer)

    # GLM-4 / ChatGLM 自定义 tokenizer 需要 batch_encode_plus
    if not hasattr(cached, "batch_encode_plus"):

        def _batch_encode_plus(self, *args, **kwargs):
            # batch_encode_plus 等同于调用 tokenizer 本身
            return self(*args, **kwargs)

        import types
        cached.batch_encode_plus = types.MethodType(_batch_encode_plus, cached)

    return cached


import vllm.tokenizers.hf as _hf
_hf.get_cached_tokenizer = get_cached_tokenizer