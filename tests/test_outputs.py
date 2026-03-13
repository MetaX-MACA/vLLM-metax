# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.outputs import RequestOutput


def test_request_output_forward_compatible():
    output = RequestOutput(request_id="test_request_id",
                           prompt="test prompt",
                           prompt_token_ids=[1, 2, 3],
                           prompt_logprobs=None,
                           outputs=[],
                           finished=False,
                           example_arg_added_in_new_version="some_value")
    assert output is not None
