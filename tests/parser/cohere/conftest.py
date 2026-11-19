# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

from .utils import MockCohereTokenizer


@pytest.fixture(scope="package")
def tokenizer() -> MockCohereTokenizer:
    return MockCohereTokenizer()


@pytest.fixture
def request_obj() -> ChatCompletionRequest:
    return ChatCompletionRequest(messages=[], model="test-model")
