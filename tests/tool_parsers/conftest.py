# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from transformers import AutoTokenizer

from vllm.tokenizers import TokenizerLike


@pytest.fixture(scope="module")
def default_tokenizer() -> TokenizerLike:
    return AutoTokenizer.from_pretrained("openai-community/gpt2")
