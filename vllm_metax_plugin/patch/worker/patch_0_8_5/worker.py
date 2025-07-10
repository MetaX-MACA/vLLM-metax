# SPDX-License-Identifier: Apache-2.0

import vllm
from vllm_metax_plugin.patch.hook_registry import register_patch
from vllm.logger import init_logger

logger = init_logger(__name__)

from typing import Dict, List, Optional, Set, Tuple, Type, Union

import torch
import torch.distributed

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.sequence import (ExecuteModelRequest, IntermediateTensors,
                           SequenceGroupMetadata, SequenceGroupMetadataDelta)
from vllm.worker.cache_engine import CacheEngine
from vllm.worker.enc_dec_model_runner import EncoderDecoderModelRunner
from vllm.worker.model_runner import GPUModelRunnerBase, ModelRunner
from vllm.worker.pooling_model_runner import PoolingModelRunner
from vllm.worker.worker_base import (WorkerBase)

def __init__(
    self,
    vllm_config: VllmConfig,
    local_rank: int,
    rank: int,
    distributed_init_method: str,
    is_driver_worker: bool = False,
    model_runner_cls: Optional[Type[GPUModelRunnerBase]] = None,
) -> None:
    WorkerBase.__init__(self, vllm_config)
    self.parallel_config.rank = rank
    self.local_rank = local_rank
    self.rank = rank
    self.distributed_init_method = distributed_init_method
    self.is_driver_worker = is_driver_worker
    if self.model_config.trust_remote_code:
        # note: lazy import to avoid importing torch before initializing
        from vllm.utils import init_cached_hf_modules
        init_cached_hf_modules()

    # Return hidden states from target model if the draft model is an
    # mlp_speculator
    speculative_config = self.speculative_config
    model_config = self.model_config
    speculative_args = {} if speculative_config is None \
        or (speculative_config.draft_model_config.hf_config.model_type ==
            model_config.hf_config.model_type) \
        or (speculative_config.draft_model_config.hf_config.model_type
            not in ("medusa",
                    "mlp_speculator",
                    "eagle",
                    "deepseek_mtp",
                        "mimo_mtp")) \
                else {"return_hidden_states": True}

    ModelRunnerClass: Type[GPUModelRunnerBase] = ModelRunner
    if model_config.runner_type == "pooling":
        ModelRunnerClass = PoolingModelRunner
    elif self.model_config.is_encoder_decoder:
        ModelRunnerClass = EncoderDecoderModelRunner
    self.model_runner: GPUModelRunnerBase = ModelRunnerClass(
        vllm_config=self.vllm_config,
        kv_cache_dtype=self.cache_config.cache_dtype,
        is_driver_worker=is_driver_worker,
        **speculative_args,
    )
    if model_runner_cls is not None:
        self.model_runner = model_runner_cls(self.model_runner)

    # Uninitialized cache engine. Will be initialized by
    # initialize_cache.
    self.cache_engine: List[CacheEngine]
    # Initialize gpu_cache as pooling models don't initialize kv_caches
    self.gpu_cache: Optional[List[List[torch.Tensor]]] = None
    self._seq_group_metadata_cache: Dict[str, SequenceGroupMetadata] = {}

    # Buffers saved before sleep
    self._sleep_saved_buffers: Dict[str, torch.Tensor] = {}

    # Torch profiler. Enabled and configured through env vars:
    # VLLM_TORCH_PROFILER_DIR=/path/to/save/trace
    if envs.VLLM_TORCH_PROFILER_DIR:
        torch_profiler_trace_dir = envs.VLLM_TORCH_PROFILER_DIR
        logger.info("Profiling enabled. Traces will be saved to: %s",
                    torch_profiler_trace_dir)
        self.profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            with_stack=True,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                torch_profiler_trace_dir, use_gzip=True))
    else:
        self.profiler = None

from vllm.worker import worker
vllm.worker.worker.Worker.__init__ = __init__

register_patch("vllm.worker.worker", "Worker.__init__", __init__)
