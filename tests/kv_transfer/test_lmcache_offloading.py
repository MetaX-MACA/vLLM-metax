# SPDX-License-Identifier: Apache-2.0
# SPDx-FileCopyrightText: Copyright contributors to the vLLM project

import os
import time

import pytest
import torch
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

no_lmcache_import = False
try:
    from lmcache.v1.cache_engine import LMCacheEngineBuilder
    from lmcache.integration.vllm.utils import ENGINE_NAME
except ImportError:
    no_lmcache_import = True


def setup_lmcache_environment(num_prompts, num_tokens):
    """
    Configure LMCache environment variables.
    Args:
        num_prompts: Number of prompts to process
        num_tokens: Number of tokens per prompt
    """
    cpu_size = num_prompts * num_tokens * 1.5 / 10000  # 1.5GB per 10000 tokens
    cpu_size = min(30, cpu_size)

    env_vars = {
        "LMCACHE_CHUNK_SIZE": "256",  # Set tokens per chunk
        "LMCACHE_LOCAL_CPU": "True",  # Enable local CPU backend
        "LMCACHE_MAX_LOCAL_CPU_SIZE":
        str(cpu_size)  # Dynamic CPU memory limit (GB)
    }
    print(f"env_vars: {env_vars}")
    for key, value in env_vars.items():
        os.environ[key] = value
    return cpu_size


def calculate_gpu_utilization(target_memory_gb=24):
    """
    Calculate GPU memory utilization to use exactly target_memory_gb of GPU memory.
    Args:
        target_memory_gb: Target GPU memory usage in gigabytes
    Returns:
        float: GPU memory utilization ratio (0.0 to 1.0)
    Raises:
        RuntimeError: If GPU memory is less than target_memory_gb
    """
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU available")

    total_memory = torch.cuda.get_device_properties(0).total_memory / (
        1024**3)  # Convert to GB
    if total_memory < target_memory_gb:
        raise RuntimeError(
            f"GPU memory ({total_memory:.1f}GB) is less than required memory ({target_memory_gb}GB)"
        )

    return target_memory_gb / total_memory


def create_test_prompts(num_prompts=10, num_tokens=1000):
    """
    Create test prompts with index prefix and dummy body.
    Args:
        num_prompts: Number of prompts to generate
        num_tokens: Approximate number of tokens per prompt (using 'Hi ' as token unit)
    Returns:
        list: List of prompts with format '[index] Hi Hi Hi...'
    """
    prompts = []
    dummy_text = "Hi " * num_tokens

    for i in range(num_prompts):
        prompt = f"[Prompt {i}] {dummy_text} how are you?"
        prompts.append(prompt)

    return prompts


def initialize_llm(model_name="Qwen/Qwen3-0.6B",
                   max_len=20480,
                   enable_lmcache=True):
    """
    Initialize the LLM with appropriate configurations.
    Args:
        model_name: Name of the model to load
        max_len: Maximum sequence length
    Returns:
        LLM: Configured LLM instance
    """
    ktc = KVTransferConfig(
        kv_connector="LMCacheConnectorV1",
        kv_role="kv_both",
    ) if enable_lmcache else None

    return LLM(model=model_name,
               kv_transfer_config=ktc,
               max_model_len=max_len,
               enable_prefix_caching=False,
               gpu_memory_utilization=calculate_gpu_utilization(),
               disable_log_stats=False)


def generate_and_print_output(llm, prompts, sampling_params):
    """
    Generate text and print the results.
    Args:
        llm: LLM instance
        prompts: List of input prompts
        sampling_params: Configured sampling parameters
    Returns:
        float: Time taken for generation in seconds
    """
    start_time = time.time()
    _ = llm.generate(prompts, sampling_params)
    end_time = time.time()

    return end_time - start_time


def _run(num_prompts, num_tokens):
    # Setup environment if LMCache is enable
    cpu_size = setup_lmcache_environment(num_prompts, num_tokens)

    # Create prompts and sampling parameters
    prompts = create_test_prompts(num_prompts=num_prompts,
                                  num_tokens=num_tokens)
    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=1)

    # Initialize model
    llm = initialize_llm(max_len=32 * 1024, enable_lmcache=True)

    # First run
    first_run_time = generate_and_print_output(llm, prompts, sampling_params)

    # Second run
    second_run_time = generate_and_print_output(llm, prompts, sampling_params)

    # Print speedup
    if first_run_time > 0:
        speedup = first_run_time / second_run_time

    # print(f"Batch, Input, Output, CPU Memory(GB), First Run Time(s), Second Run Time(s), Speedup")
    print(
        f"{num_prompts}, {num_tokens}, {1}, {cpu_size}, {first_run_time:.2f}, {second_run_time:.2f}, {speedup:.2f}x"
    )

    # Cleanup if LMCache was enabled
    LMCacheEngineBuilder.destroy(ENGINE_NAME)

    assert speedup > 1


# Test function to send curl requests and validate responses
@pytest.mark.parametrize("num_prompts,num_tokens", [(1, 128), (1, 1024),
                                                    (1, 10000), (10, 128),
                                                    (10, 1024), (10, 10000),
                                                    (32, 128), (32, 1024)])
def test_cpu_offloading(num_prompts, num_tokens):
    # Check the number of GPUs
    if no_lmcache_import:
        pytest.skip(
            "LMCache are notavailable, please run test after try install LMCache."
        )
    # Run the test if there are at least 2 GPUs
    _run(num_prompts, num_tokens)
