# SPDX-License-Identifier: Apache-2.0
import pytest

from tests.conftest import VllmRunner

MLA_MODELS = ["deepseek-v2-lite"]
NON_MLA_MODELS = ["qwen3-4b"]

MLA_BACKENDS = ["FLASHMLA_VLLM_V1", "TRITON_MLA"]
NON_MLA_BACKENDS = ["FLASH_ATTN", "FLASHINFER"]


@pytest.mark.parametrize("model", MLA_MODELS)
@pytest.mark.parametrize("backend", MLA_BACKENDS)
@pytest.mark.parametrize("max_tokens", [20])
def test_mla_attn_backends(monkeypatch, model: str, max_tokens: int,
                           backend: str) -> None:
    example_prompts = [
        "Hello, my name is",
        "The future of AI is",
    ]
    monkeypatch.setenv("VLLM_ATTENTION_BACKEND", backend)
    with VllmRunner(model,
                    tensor_parallel_size=1,
                    max_model_len=2048,
                    enforce_eager=True) as vllm_model:
        vllm_model.generate_greedy(example_prompts, max_tokens)


@pytest.mark.parametrize("model", NON_MLA_MODELS)
@pytest.mark.parametrize("backend", NON_MLA_BACKENDS)
@pytest.mark.parametrize("max_tokens", [20])
def test_non_mla_attn_backends(monkeypatch, model: str, max_tokens: int,
                               backend: str) -> None:
    example_prompts = [
        "Hello, my name is",
        "The future of AI is",
    ]
    monkeypatch.setenv("VLLM_ATTENTION_BACKEND", backend)
    with VllmRunner(model,
                    tensor_parallel_size=1,
                    max_model_len=2048,
                    enforce_eager=True,
                    gpu_memory_utilization=0.92) as vllm_model:
        vllm_model.generate_greedy(example_prompts, max_tokens)
