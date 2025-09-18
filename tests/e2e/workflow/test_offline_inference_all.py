# SPDX-License-Identifier: Apache-2.0
import pytest

from tests.conftest import VllmRunner

MLA_MODELS = ["deepseek-v2-lite"]
NON_MLA_MODELS = ["qwen3-4b"]

MLA_BACKENDS = ["FLASHMLA_VLLM_V1", "TRITON_MLA"]
NON_MLA_BACKENDS = ["FLASH_ATTN", "FLASHINFER"]


@pytest.mark.parametrize("model", MLA_MODELS)
@pytest.mark.parametrize("backend", MLA_BACKENDS)
@pytest.mark.parametrize("enforce_eager", [True, False])
@pytest.mark.parametrize("distributed_executor_backend", ["mp", "ray"])
def test_mla_attn_backends(monkeypatch, model: str, backend: str,
                           enforce_eager: bool,
                           distributed_executor_backend: str) -> None:
    example_prompts = [
        "Hello, my name is",
        "The future of AI is",
    ]
    monkeypatch.setenv("VLLM_ATTENTION_BACKEND", backend)
    with VllmRunner(model,
                    tensor_parallel_size=1,
                    max_model_len=4096,
                    enforce_eager=enforce_eager,
                    distributed_executor_backend=distributed_executor_backend
                    ) as vllm_model:
        vllm_model.generate_greedy(example_prompts, 1024)


@pytest.mark.parametrize("model", NON_MLA_MODELS)
@pytest.mark.parametrize("backend", NON_MLA_BACKENDS)
@pytest.mark.parametrize("enforce_eager", [True, False])
@pytest.mark.parametrize("distributed_executor_backend", ["mp", "ray"])
def test_non_mla_attn_backends(monkeypatch, model: str, backend: str,
                               enforce_eager: bool,
                               distributed_executor_backend: str) -> None:
    example_prompts = [
        "Hello, my name is",
        "The future of AI is",
    ]
    monkeypatch.setenv("VLLM_ATTENTION_BACKEND", backend)
    with VllmRunner(model,
                    tensor_parallel_size=1,
                    max_model_len=4096,
                    enforce_eager=enforce_eager,
                    distributed_executor_backend=distributed_executor_backend
                    ) as vllm_model:
        vllm_model.generate_greedy(example_prompts, 1024)
