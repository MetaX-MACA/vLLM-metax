# SPDX-License-Identifier: Apache-2.0
import pytest

from tests.conftest import VllmRunner

MODELS = ["qwen3-30b-a3b", "deepseek-v2-lite"]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["auto"])
@pytest.mark.parametrize("max_tokens", [50])
@pytest.mark.parametrize("enforce_eager", [True])
@pytest.mark.parametrize("distributed_executor_backend", ["mp", "ray"])
def test_models_parallel(model: str, dtype: str, max_tokens: int,
                         enforce_eager: bool,
                         distributed_executor_backend: str) -> None:
    example_prompts = [
        "Hello, my name is",
        "The future of AI is",
    ]

    with VllmRunner(
            model,
            tensor_parallel_size=2,
            dtype=dtype,
            max_model_len=2048,
            enforce_eager=enforce_eager,
            distributed_executor_backend=distributed_executor_backend,
    ) as vllm_model:
        vllm_model.generate_greedy(example_prompts, max_tokens)
