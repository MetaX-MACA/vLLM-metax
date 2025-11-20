# 2025 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import vllm
from vllm import envs, logger
from vllm.logger import init_logger
import torch
from typing import (Optional, Union)
from vllm.utils import _generate_random_fp8, get_kv_cache_torch_dtype

from vllm_metax_plugin.patch.hook_registry import register_patch

logger = init_logger(__name__)

def find_nccl_library() -> str:
    logger.info(f"[Plugin] Hooked find_nccl_library -> {find_nccl_library}")

    """
    We either use the library file specified by the `VLLM_NCCL_SO_PATH`
    environment variable, or we find the library file brought by PyTorch.
    After importing `torch`, `libnccl.so.2` or `librccl.so.1` can be
    found by `ctypes` automatically.
    """
    so_file = envs.VLLM_NCCL_SO_PATH

    # manually load the nccl library
    if so_file:
        logger.info(
            "Found nccl from environment variable VLLM_NCCL_SO_PATH=%s",
            so_file)
    else:
        if torch.version.cuda is not None:
            so_file = "libmccl.so"
        elif torch.version.hip is not None:
            so_file = "librccl.so.1"
        else:
            raise ValueError("NCCL only supports CUDA and ROCm backends.")
        logger.info("Found nccl from library %s", so_file)
    return so_file

def import_pynvml():
    logger.info(f"[Plugin] Hooked import_pynvml -> {import_pynvml}")

    """
    Historical comments:

    libnvml.so is the library behind nvidia-smi, and
    pynvml is a Python wrapper around it. We use it to get GPU
    status without initializing CUDA context in the current process.
    Historically, there are two packages that provide pynvml:
    - `nvidia-ml-py` (https://pypi.org/project/nvidia-ml-py/): The official
        wrapper. It is a dependency of vLLM, and is installed when users
        install vLLM. It provides a Python module named `pynvml`.
    - `pynvml` (https://pypi.org/project/pynvml/): An unofficial wrapper.
        Prior to version 12.0, it also provides a Python module `pynvml`,
        and therefore conflicts with the official one. What's worse,
        the module is a Python package, and has higher priority than
        the official one which is a standalone Python file.
        This causes errors when both of them are installed.
        Starting from version 12.0, it migrates to a new module
        named `pynvml_utils` to avoid the conflict.
    It is so confusing that many packages in the community use the
    unofficial one by mistake, and we have to handle this case.
    For example, `nvcr.io/nvidia/pytorch:24.12-py3` uses the unofficial
    one, and it will cause errors, see the issue
    https://github.com/vllm-project/vllm/issues/12847 for example.
    After all the troubles, we decide to copy the official `pynvml`
    module to our codebase, and use it directly.
    """
    import vllm_metax_plugin.third_party.pymcml as pynvml
    return pynvml

def create_kv_caches_with_random(
    num_blocks: int,
    block_size: int,
    num_layers: int,
    num_heads: int,
    head_size: int,
    cache_dtype: Optional[Union[str, torch.dtype]],
    model_dtype: Optional[Union[str, torch.dtype]] = None,
    seed: Optional[int] = None,
    device: Optional[str] = "cuda",
    new_layerout:Optional[bool] = False,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    logger.info(f"[Plugin] Hooked create_kv_caches_with_random -> {create_kv_caches_with_random}")

    if cache_dtype == "fp8" and head_size % 16:
        raise ValueError(
            f"Does not support key cache of type fp8 with head_size {head_size}"
        )
    from vllm.platforms import current_platform
    current_platform.seed_everything(seed)

    torch_dtype = get_kv_cache_torch_dtype(cache_dtype, model_dtype)

    scale = head_size**-0.5
    x = 16 // torch.tensor([], dtype=torch_dtype).element_size()
    key_cache_shape = (num_blocks, num_heads, head_size // x, block_size, x)
    key_caches: list[torch.Tensor] = []
    for _ in range(num_layers):
        key_cache = torch.empty(size=key_cache_shape,
                                dtype=torch_dtype,
                                device=device)
        if cache_dtype in ["auto", "half", "bfloat16", "float"]:
            key_cache.uniform_(-scale, scale)
        elif cache_dtype == 'fp8':
            _generate_random_fp8(key_cache, -scale, scale)
        else:
            raise ValueError(
                f"Does not support key cache of type {cache_dtype}")
        key_caches.append(key_cache)
        if new_layerout:
            key_cache_new = torch.empty(size=key_cache_shape,
                                        dtype=torch_dtype,
                                        device=device)
            key_caches.append(key_cache_new)
            
    value_cache_shape = (num_blocks, num_heads, head_size, block_size)
    value_caches: list[torch.Tensor] = []
    for _ in range(num_layers):
        value_cache = torch.empty(size=value_cache_shape,
                                  dtype=torch_dtype,
                                  device=device)
        if cache_dtype in ["auto", "half", "bfloat16", "float"]:
            value_cache.uniform_(-scale, scale)
        elif cache_dtype == 'fp8':
            _generate_random_fp8(value_cache, -scale, scale)
        else:
            raise ValueError(
                f"Does not support value cache of type {cache_dtype}")
        value_caches.append(value_cache)
        if new_layerout:
            key_cache_new = torch.empty(size=key_cache_shape,
                                        dtype=torch_dtype,
                                        device=device)
            key_caches.append(key_cache_new)
    return key_caches, value_caches


vllm.utils.find_nccl_library = find_nccl_library
vllm.utils.import_pynvml = import_pynvml
vllm.utils.create_kv_caches_with_random = create_kv_caches_with_random

register_patch("vllm.utils", "find_nccl_library", find_nccl_library);
register_patch("vllm.utils", "import_pynvml", import_pynvml);
register_patch("vllm.utils", "create_kv_caches_with_random", create_kv_caches_with_random);

