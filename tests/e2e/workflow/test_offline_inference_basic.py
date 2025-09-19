# SPDX-License-Identifier: Apache-2.0
import pytest

from tests.conftest import VllmRunner

MODELS = ["qwen3-4b", "deepseek-v2-lite"]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("enforce_eager", [True, False])
@pytest.mark.parametrize("max_tokens", [20])
def test_models_basic(model: str, max_tokens: int,
                      enforce_eager: bool) -> None:
    example_prompts = [
        "Hello, my name is",
        "The future of AI is",
    ]

    with VllmRunner(model,
                    tensor_parallel_size=1,
                    max_model_len=2048,
                    enforce_eager=enforce_eager) as vllm_model:
        vllm_model.generate_greedy(example_prompts, max_tokens)
