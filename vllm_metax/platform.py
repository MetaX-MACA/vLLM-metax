# SPDX-License-Identifier: Apache-2.0
"""Code inside this file can safely assume cuda platform, e.g. importing
pynvml. However, it should not initialize cuda context.
"""

import os
from datetime import timedelta
from functools import wraps
from typing import (TYPE_CHECKING, Callable, List, Optional, TypeVar,
                    Union)

import torch
from torch.distributed import PrefixStore, ProcessGroup
from torch.distributed.distributed_c10d import is_nccl_available
from typing_extensions import ParamSpec

# import custom ops, trigger op registration
import vllm._C  # noqa
import vllm.envs as envs
from vllm.logger import logger
from vllm_metax.utils import import_pymxml
from vllm.utils import cuda_device_count_stateless

from vllm.platforms.interface import DeviceCapability, Platform, PlatformEnum, _Backend, FlexibleArgumentParser

if TYPE_CHECKING:
    from vllm.config import ModelConfig, VllmConfig

_P = ParamSpec("_P")
_R = TypeVar("_R")

pymxml = import_pymxml()

# pytorch 2.5 uses cudnn sdpa by default, which will cause crash on some models
# see https://github.com/huggingface/diffusers/issues/9704 for details
# torch.backends.cuda.enable_cudnn_sdp(False)
torch.backends.cuda.enable_cudnn_sdp(False)

def with_mcml_context(fn: Callable[_P, _R]) -> Callable[_P, _R]:

    @wraps(fn)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        pymxml.nvmlInit()
        try:
            return fn(*args, **kwargs)
        finally:
            pymxml.nvmlShutdown()

    return wrapper


class MacaPlatformBase(Platform):
    _enum = PlatformEnum.OOT
    device_name: str = "Metax"
    device_type: str = "cuda"
    dispatch_key: str = "CUDA"
    ray_device_key: str = "GPU"
    dist_backend: str = "mccl"
    device_control_env_var: str = "CUDA_VISIBLE_DEVICES"

    supported_quantization:list[str] = [
        "awq", "gptq", "compressed-tensors", "compressed_tensors",
        "moe_wna16", "gguf"
    ]

    @classmethod
    def set_device(cls, device: torch.device) -> None:
        """
        Set the device for the current platform.
        """
        torch.cuda.set_device(device)
        # With this trick we can force the device to be set eagerly
        # see https://github.com/pytorch/pytorch/issues/155668
        # for why and when it is needed
        _ = torch.zeros(1, device=device)

    @classmethod
    def get_device_capability(cls,
                              device_id: int = 0
                              ) -> Optional[DeviceCapability]:
        raise NotImplementedError

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        raise NotImplementedError

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        raise NotImplementedError

    @classmethod
    def is_cuda_alike(cls) -> bool:
        return True

    @classmethod
    def is_async_output_supported(cls, enforce_eager: Optional[bool]) -> bool:
        if enforce_eager and not envs.VLLM_USE_V1:
            logger.warning(
                "To see benefits of async output processing, enable CUDA "
                "graph. Since, enforce-eager is enabled, async output "
                "processor cannot be used")
            return False
        return True

    @classmethod
    def is_fully_connected(cls, device_ids: list[int]) -> bool:
        raise NotImplementedError

    @classmethod
    def log_warnings(cls):
        pass

    @classmethod
    def check_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        # Env Override
        envs.VLLM_USE_FLASHINFER_SAMPLER = False

        # Config Override
        parallel_config = vllm_config.parallel_config
        compilation_config = vllm_config.compilation_config
        model_config = vllm_config.model_config

        if parallel_config.worker_cls == "auto":
            if vllm_config.speculative_config:
                if not envs.VLLM_USE_V1:
                    raise NotImplementedError(
                        "Speculative decoding is not supported on vLLM V0.")
                parallel_config.worker_cls = "vllm.v1.worker.gpu_worker.Worker"
            else:
                if envs.VLLM_USE_V1:
                    parallel_config.worker_cls = \
                        "vllm.v1.worker.gpu_worker.Worker"
                else:
                    parallel_config.worker_cls = "vllm.worker.worker.Worker"

        cache_config = vllm_config.cache_config
        if cache_config and cache_config.block_size is None:
            cache_config.block_size = 16

        # TODO(lucas): handle this more gracefully
        # Note: model_config may be None during testing
        if model_config is not None and model_config.use_mla:
            # If `VLLM_ATTENTION_BACKEND` is not set and we are using MLA,
            # then we default to FlashMLA backend for non-blackwell GPUs,
            # else we default to CutlassMLA. For each case, we force the
            # required block_size.
            use_flashmla = False
            use_cutlass_mla = False

            if envs.VLLM_ATTENTION_BACKEND is None:
                # Default case
                if cls.is_device_capability(100):
                    # Blackwell => Force CutlassMLA.
                    use_cutlass_mla = True
                    # TODO: This does not work, because the
                    # global_force_attn_backend_context_manager is not set.
                    # See vllm/attention/selector.py:_cached_get_attn_backend
                    envs.VLLM_ATTENTION_BACKEND = "CUTLASS_MLA"
                else:
                    # Not Blackwell
                    use_flashmla = True
            else:
                # Forced case
                use_flashmla = (envs.VLLM_ATTENTION_BACKEND == "FLASHMLA")
                use_cutlass_mla = (
                    envs.VLLM_ATTENTION_BACKEND == "CUTLASS_MLA")

            from vllm_metax.attention.ops.flashmla import is_flashmla_supported
            if use_flashmla and is_flashmla_supported()[0] \
                and cache_config.block_size != 64:
                cache_config.block_size = 64
                logger.info(
                    "Forcing kv cache block size to 64 for FlashMLA backend.")

            if use_cutlass_mla and cache_config.block_size != 128:
                cache_config.block_size = 128
                logger.info("Forcing kv cache block size to 128 for "
                            "CUTLASS_MLA backend.")

        # lazy import to avoid circular import
        from vllm.config import CUDAGraphMode

        compilation_config = vllm_config.compilation_config
        if (envs.VLLM_ALL2ALL_BACKEND == "deepep_high_throughput"
                and parallel_config.data_parallel_size > 1
                and compilation_config.cudagraph_mode != CUDAGraphMode.NONE):
            logger.info(
                "Data Parallel: disabling cudagraphs since DP "
                "with DeepEP high-throughput kernels are not CUDA Graph "
                "compatible. The DeepEP low-latency kernels are CUDA Graph "
                "compatible. Set the all_to_all backend to deepep_low_latency "
                "to use those kernels instead.")
            compilation_config.cudagraph_mode = CUDAGraphMode.NONE
            if model_config is not None:
                model_config.enforce_eager = True

        if vllm_config.model_config is not None and \
            not vllm_config.model_config.enforce_eager and \
            compilation_config.cudagraph_capture_sizes is not None:
            batch_size_capture_list = [size for size in compilation_config.cudagraph_capture_sizes if size < 257]
            compilation_config.cudagraph_capture_sizes = None
            compilation_config.init_with_cudagraph_sizes(batch_size_capture_list)

        if vllm_config.model_config is not None:
            model_config.disable_cascade_attn = True

    @classmethod
    def get_current_memory_usage(cls,
                                 device: Optional[torch.types.Device] = None
                                 ) -> float:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        return torch.cuda.max_memory_allocated(device)

    @classmethod
    def get_attn_backend_cls(cls, selected_backend, head_size, dtype,
                             kv_cache_dtype, block_size, use_v1, use_mla,
                             has_sink) -> str:
        if use_mla:
            TRITON_MLA_V1 = "vllm_metax.v1.attention.backends.mla.triton_mla." \
                            "MacaTritonMLABackend"  # noqa: E501
            FLASHMLA_V1 = "vllm_metax.v1.attention.backends.mla.flashmla." \
                            "MacaFlashMLABackend"  # noqa: E501
            # TODO(lucas): refactor to be more concise
            #  we should probably consider factoring out V1 here
            if selected_backend == _Backend.CUTLASS_MLA or (
                    cls.is_device_capability(100) and selected_backend is None
                    and block_size == 128):
                logger.warning(
                    "Cutlass MLA backend is not supported on Maca")
            if selected_backend == _Backend.TRITON_MLA or block_size != 64:
                if use_v1:
                    logger.info_once("Using Triton MLA backend on V1 engine.")
                    return TRITON_MLA_V1
                else:
                    logger.warning("Triton MLA backend is only supported on V1 engine")
            else:
                from vllm_metax.attention.backends.flashmla import (
                    is_flashmla_supported)
                if not is_flashmla_supported()[0]:
                    logger.warning(
                        "FlashMLA backend is not supported due to %s",
                        is_flashmla_supported()[1])
                elif block_size != 64:
                    logger.warning(
                        "FlashMLA backend is not supported for block size %d"
                        " (currently only supports block size 64).",
                        block_size)
                else:
                    if use_v1:
                        logger.info_once(
                            "Using FlashMLA backend on V1 engine.")
                        return FLASHMLA_V1
                    else:
                        logger.warning("FlashMLA backend is only supported on V1 engine")
        if use_v1:
            FLASHINFER_V1 = "vllm_metax.v1.attention.backends.flashinfer.MacaFlashInferBackend"  # noqa: E501
            FLASH_ATTN_V1 = "vllm_metax.v1.attention.backends.flash_attn.MacaFlashAttentionBackend"  # noqa: E501
    
            if selected_backend == _Backend.FLASHINFER:
                logger.info_once("Using FlashInfer backend on V1 engine.")
                if cls.has_device_capability(100):
                    from vllm.v1.attention.backends.utils import (
                        set_kv_cache_layout)
                    set_kv_cache_layout("HND")
                return FLASHINFER_V1
            elif selected_backend == _Backend.FLASH_ATTN:
                logger.info_once("Using Flash Attention backend on V1 engine.")
                return FLASH_ATTN_V1

            from vllm.attention.selector import is_attn_backend_supported

            # Default backends for V1 engine
            # Prefer FlashInfer for Blackwell GPUs if installed
            if cls.is_device_capability(100):
                if is_default_backend_supported := is_attn_backend_supported(
                        FLASHINFER_V1, head_size, dtype):
                    from vllm.v1.attention.backends.utils import (
                        set_kv_cache_layout)

                    logger.info_once(
                        "Using FlashInfer backend with HND KV cache layout on "
                        "V1 engine by default for Blackwell (SM 10.0) GPUs.")
                    set_kv_cache_layout("HND")

                    return FLASHINFER_V1

                if not is_default_backend_supported.can_import:
                    logger.warning_once(
                        "FlashInfer failed to import for V1 engine on "
                        "Blackwell (SM 10.0) GPUs; it is recommended to "
                        "install FlashInfer for better performance.")

            # FlashAttention is the default for SM 8.0+ GPUs
            if cls.has_device_capability(80):
                if is_default_backend_supported := is_attn_backend_supported(
                        FLASH_ATTN_V1, head_size, dtype,
                        allow_import_error=False):
                    logger.info_once("Using Flash Attention backend on "
                                     "V1 engine.")
                    return FLASH_ATTN_V1

            assert not is_default_backend_supported

            return FLASH_ATTN_V1

        # Backends for V0 engine
        else:
            logger.warning_once("V0 engine is deprecated on Maca. Please switch to V1.")

    @classmethod
    def get_punica_wrapper(cls) -> str:
        return "vllm.lora.punica_wrapper.punica_gpu.PunicaWrapperGPU"

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        return "vllm.distributed.device_communicators.cuda_communicator.CudaCommunicator"  # noqa

    @classmethod
    def supports_fp8(cls) -> bool:
        return False

    @classmethod
    def supports_v1(cls, model_config: "ModelConfig") -> bool:
        return True

    @classmethod
    def use_custom_allreduce(cls) -> bool:
        return False
    
    @classmethod
    def get_static_graph_wrapper_cls(cls) -> str:
        return "vllm.compilation.cuda_graph.CUDAGraphWrapper"

    @classmethod
    def stateless_init_device_torch_dist_pg(
        cls,
        backend: str,
        prefix_store: PrefixStore,
        group_rank: int,
        group_size: int,
        timeout: timedelta,
    ) -> ProcessGroup:
        assert is_nccl_available()
        pg: ProcessGroup = ProcessGroup(
            prefix_store,
            group_rank,
            group_size,
        )
        from torch.distributed.distributed_c10d import ProcessGroupNCCL

        backend_options = ProcessGroupNCCL.Options()
        backend_options._timeout = timeout

        backend_class = ProcessGroupNCCL(prefix_store, group_rank, group_size,
                                         backend_options)
        backend_type = ProcessGroup.BackendType.NCCL
        device = torch.device("cuda")
        pg._set_default_backend(backend_type)
        backend_class._set_sequence_number_for_group()

        pg._register_backend(device, backend_type, backend_class)
        return pg

    @classmethod
    def device_count(cls) -> int:
        return cuda_device_count_stateless()

    @classmethod
    def is_kv_cache_dtype_supported(cls, kv_cache_dtype: str) -> bool:
        fp8_attention = kv_cache_dtype.startswith("fp8")
        will_use_fa = (not envs.is_set("VLLM_ATTENTION_BACKEND")
                       ) or envs.VLLM_ATTENTION_BACKEND == "FLASH_ATTN_VLLM_V1"
        supported = False
        if cls.is_device_capability(100):
            supported = True
        elif fp8_attention and will_use_fa:
            from vllm.attention.utils.fa_utils import flash_attn_supports_fp8
            supported = flash_attn_supports_fp8()
        return supported

    @classmethod
    def pre_register_and_update(cls,
                                parser: Optional[FlexibleArgumentParser] = None
                                ) -> None:
        logger.info("[hook] platform:pre_register_and_update...")
        import vllm_metax.patch


# NVML utils
# Note that NVML is not affected by `CUDA_VISIBLE_DEVICES`,
# all the related functions work on real physical device ids.
# the major benefit of using NVML is that it will not initialize CUDA
class MxmlPlatform(MacaPlatformBase):

    @classmethod
    @with_mcml_context
    def get_device_capability(cls,
                              device_id: int = 0
                              ) -> Optional[DeviceCapability]:
        try:
            physical_device_id = cls.device_id_to_physical_device_id(device_id)
            handle = pymxml.nvmlDeviceGetHandleByIndex(physical_device_id)
            major, minor = pymxml.nvmlDeviceGetCudaComputeCapability(handle)
            return DeviceCapability(major=major, minor=minor)
        except RuntimeError:
            return None

    @classmethod
    @with_mcml_context
    def has_device_capability(
        cls,
        capability: Union[tuple[int, int], int],
        device_id: int = 0,
    ) -> bool:
        try:
            return super().has_device_capability(capability, device_id)
        except RuntimeError:
            return False

    @classmethod
    @with_mcml_context
    def get_device_name(cls, device_id: int = 0) -> str:
        return "Device 4000"
        physical_device_id = device_id_to_physical_device_id(device_id)
        return cls._get_physical_device_name(physical_device_id)

    @classmethod
    @with_mcml_context
    def get_device_uuid(cls, device_id: int = 0) -> str:
        physical_device_id = cls.device_id_to_physical_device_id(device_id)
        handle = pymxml.nvmlDeviceGetHandleByIndex(physical_device_id)
        return pymxml.nvmlDeviceGetUUID(handle)

    @classmethod
    @with_mcml_context
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        physical_device_id = cls.device_id_to_physical_device_id(device_id)
        handle = pymxml.nvmlDeviceGetHandleByIndex(physical_device_id)
        return int(pymxml.nvmlDeviceGetMemoryInfo(handle).total)

    @classmethod
    @with_mcml_context
    def is_fully_connected(cls, physical_device_ids: list[int]) -> bool:
        """
        query if the set of gpus are fully connected by nvlink (1 hop)
        """
        handles = [
            pymxml.nvmlDeviceGetHandleByIndex(i) for i in physical_device_ids
        ]
        for i, handle in enumerate(handles):
            for j, peer_handle in enumerate(handles):
                if i < j:
                    try:
                        p2p_status = pymxml.nvmlDeviceGetP2PStatus(
                            handle,
                            peer_handle,
                            pymxml.NVML_P2P_CAPS_INDEX_NVLINK,
                        )
                        if p2p_status != pymxml.NVML_P2P_STATUS_OK:
                            return False
                    except pymxml.NVMLError:
                        logger.exception(
                            "NVLink detection failed. This is normal if"
                            " your machine has no NVLink equipped.")
                        return False
        return True

    @classmethod
    def _get_physical_device_name(cls, device_id: int = 0) -> str:
        handle = pymxml.nvmlDeviceGetHandleByIndex(device_id)
        return pymxml.nvmlDeviceGetName(handle)

    @classmethod
    @with_mcml_context
    def log_warnings(cls):
        device_ids: int = pymxml.nvmlDeviceGetCount()
        if device_ids > 1:
            device_names = [
                cls._get_physical_device_name(i) for i in range(device_ids)
            ]
            if (len(set(device_names)) > 1
                    and os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID"):
                logger.warning(
                    "Detected different devices in the system: %s. Please"
                    " make sure to set `CUDA_DEVICE_ORDER=PCI_BUS_ID` to "
                    "avoid unexpected behavior.",
                    ", ".join(device_names),
                )


class NonMxmlMetaxPlatform(MacaPlatformBase):

    @classmethod
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability:
        major, minor = torch.cuda.get_device_capability(device_id)
        return DeviceCapability(major=major, minor=minor)

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return "Device 4000"

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        device_props = torch.cuda.get_device_properties(device_id)
        return device_props.total_memory

    @classmethod
    def is_fully_connected(cls, physical_device_ids: List[int]) -> bool:
        logger.exception(
            "NVLink detection not possible, as context support was"
            " not found. Assuming no NVLink available.")
        return False


# Autodetect either NVML-enabled or non-NVML platform
# based on whether NVML is available.
mxml_available = False
try:
    try:
        pymxml.nvmlInit()
        mxml_available = True
    except Exception:
        # On Jetson, NVML is not supported.
        mxml_available = False
finally:
    if mxml_available:
        pymxml.nvmlShutdown()

MacaPlatform = MxmlPlatform if mxml_available else NonMxmlMetaxPlatform