# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import pytest
import torch
import torch.multiprocessing as mp

from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import cleanup_dist_env_and_memory
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.attention.mla_attention import MLAAttention

import vllm_metax
from vllm_metax.models.deepseek_v2 import DeepseekV2ForCausalLM
from vllm_metax.v1.attention.backends.flash_attn import MacaFlashAttentionBackend
from vllm_metax.v1.attention.backends.mla.flashmla import MacaFlashMLABackend
from vllm_metax.v1.attention.backends.mla.triton_mla import MacaTritonMLABackend
from vllm_metax.v1.attention.backends.triton_attn import MacaTritonAttentionBackend


def _save_tiny_checkpoint(checkpoint, architecture):
    from transformers import (
        DeepseekV2Config,
        DeepseekV2ForCausalLM as HFDeepseekV2ForCausalLM,
        Qwen2Config,
        Qwen2ForCausalLM,
    )

    torch.manual_seed(0)
    common_args = dict(
        vocab_size=64,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=2,
        max_position_embeddings=128,
        bos_token_id=1,
        eos_token_id=None,
    )
    if architecture == "qwen2":
        model = Qwen2ForCausalLM(Qwen2Config(num_key_value_heads=1, **common_args))
    else:
        model = HFDeepseekV2ForCausalLM(
            DeepseekV2Config(
                first_k_dense_replace=1,
                q_lora_rank=64,
                **common_args,
            )
        )
    model = model.to(torch.bfloat16).eval()
    model.save_pretrained(checkpoint)
    return model


def _attention_backends(model):
    backends = {
        f"{module.attn_backend.__module__}.{module.attn_backend.__name__}"
        for module in model.modules()
        if isinstance(module, (Attention, MLAAttention))
    }
    return backends, type(model) is DeepseekV2ForCausalLM


@torch.inference_mode()
def _check_batched_prefill_and_decode(
    _rank, checkpoint, architecture, backend, backend_class
):
    vllm_metax.register_out_of_tree()
    vllm_metax.register_attn_backends()
    vllm_metax.register_model()
    reference_model = _save_tiny_checkpoint(checkpoint, architecture)
    prompts = [[1, 7, 12, 20, 6, 23], [1, 11, 6]]
    num_steps = 3
    # Use the same on-disk weights for the independently executed reference.
    expected = [
        reference_model.generate(
            torch.tensor([prompt]),
            max_new_tokens=num_steps,
            do_sample=False,
            eos_token_id=None,
            pad_token_id=0,
        )[0, len(prompt) :].tolist()
        for prompt in prompts
    ]
    llm = LLM(
        model=str(checkpoint),
        dtype="bfloat16",
        skip_tokenizer_init=True,
        max_model_len=64,
        max_num_batched_tokens=64,
        max_num_seqs=2,
        block_size=64 if "MLA" in backend else 16,
        enforce_eager=True,
        kv_cache_memory_bytes=16 << 20,
        attention_config={"backend": backend},
        enable_prefix_caching=False,
    )
    try:
        qualified_name = f"{backend_class.__module__}.{backend_class.__name__}"
        assert llm.apply_model(_attention_backends) == [
            ({qualified_name}, "MLA" in backend)
        ]
        outputs = llm.generate(
            [{"prompt_token_ids": prompt} for prompt in prompts],
            SamplingParams(
                temperature=0.0,
                max_tokens=num_steps,
                ignore_eos=True,
                detokenize=False,
            ),
            use_tqdm=False,
        )
        assert [list(out.outputs[0].token_ids) for out in outputs] == expected
    finally:
        llm.llm_engine.engine_core.shutdown()
        del llm
        cleanup_dist_env_and_memory()


@pytest.mark.requires_maca
@pytest.mark.parametrize(
    ("architecture", "backend", "backend_class"),
    [
        ("qwen2", "FLASH_ATTN", MacaFlashAttentionBackend),
        ("qwen2", "TRITON_ATTN", MacaTritonAttentionBackend),
        ("deepseek", "FLASHMLA", MacaFlashMLABackend),
        ("deepseek", "TRITON_MLA", MacaTritonMLABackend),
    ],
)
def test_batched_prefill_and_decode_uses_plugin_backend(
    tmp_path, architecture, backend, backend_class, monkeypatch
):
    # Release the engine's device resources when the child exits.
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    context = mp.spawn(
        _check_batched_prefill_and_decode,
        args=(tmp_path, architecture, backend, backend_class),
        nprocs=1,
        join=False,
    )
    process = context.processes[0]
    try:
        if not context.join(timeout=180):
            pytest.fail("model generation exceeded 180 seconds")
    finally:
        if process.is_alive():
            process.kill()
        process.join()
