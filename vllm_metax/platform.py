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
from vllm.logger import init_logger
from vllm_metax.utils import import_pymcml

from vllm.platforms.interface import DeviceCapability, Platform, PlatformEnum, _Backend, FlexibleArgumentParser

if TYPE_CHECKING:
    from vllm.config import ModelConfig, VllmConfig

logger = init_logger(__name__)

_P = ParamSpec("_P")
_R = TypeVar("_R")

pymcml = import_pymcml()

# pytorch 2.5 uses cudnn sdpa by default, which will cause crash on some models
# see https://github.com/huggingface/diffusers/issues/9704 for details
# torch.backends.cuda.enable_cudnn_sdp(False)
torch.backends.cuda.enable_cudnn_sdp(False)

def with_mcml_context(fn: Callable[_P, _R]) -> Callable[_P, _R]:

    @wraps(fn)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        pymcml.nvmlInit()
        try:
            return fn(*args, **kwargs)
        finally:
            pymcml.nvmlShutdown()

    return wrapper


class MetaXPlatformBase(Platform):
    _enum = PlatformEnum.CUDA
    device_name: str = "Metax"
    device_type: str = "cuda"
    dispatch_key: str = "CUDA"
    ray_device_key: str = "GPU"
    device_control_env_var: str = "CUDA_VISIBLE_DEVICES"

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
    def is_cuda(cls) -> bool:
        return True    

    @classmethod
    def is_cuda_alike(cls) -> bool:
        return True

    @classmethod
    def is_async_output_supported(cls, enforce_eager: Optional[bool]) -> bool:
        if enforce_eager:
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
        parallel_config = vllm_config.parallel_config
        scheduler_config = vllm_config.scheduler_config
        compilation_config = vllm_config.compilation_config
        model_config = vllm_config.model_config

        if parallel_config.worker_cls == "auto":
            if scheduler_config.is_multi_step:
                if envs.VLLM_USE_V1:
                    raise NotImplementedError(
                        "Multi-step scheduling is not supported (and not "
                        "needed) on vLLM V1. Please launch without "
                        "--num-scheduler-steps.")
                else:
                    parallel_config.worker_cls = \
                        "vllm.worker.multi_step_worker.MultiStepWorker"
            elif vllm_config.speculative_config:
                if envs.VLLM_USE_V1:
                    parallel_config.worker_cls = \
                            "vllm.v1.worker.gpu_worker.Worker"
                else:
                    parallel_config.worker_cls = \
                        "vllm.spec_decode.spec_decode_worker.create_spec_worker"
                    parallel_config.sd_worker_cls = \
                        "vllm.worker.worker.Worker"
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
            # if `VLLM_ATTENTION_BACKEND` is not set and we are using MLA, then
            # we default to FlashMLA backend, so we need to force the blocksize
            # here
            use_flashmla = (envs.VLLM_ATTENTION_BACKEND is None \
                or envs.VLLM_ATTENTION_BACKEND == "FLASHMLA")
            from vllm_metax.attention.ops.flashmla import is_flashmla_supported
            if use_flashmla and is_flashmla_supported()[0] \
                and cache_config.block_size != 64:
                cache_config.block_size = 64
                logger.info(
                    "Forcing kv cache block size to 64 for FlashMLA backend.")
                
        if (envs.VLLM_ALL2ALL_BACKEND == "deepep_high_throughput"
                and parallel_config.data_parallel_size > 1
                and vllm_config.compilation_config.use_cudagraph):
            logger.info(
                "Data Parallel: Forcing enforce eager to be True since DP "
                "with DeepEP high-throughput kernels are not CUDA Graph "
                "compatible. The DeepEP low-latency kernels are CUDA Graph "
                "compatible. Set the all_to_all backend to deepep_low_latency "
                "to use those kernels instead.")
            vllm_config.compilation_config.use_cudagraph = False
            vllm_config.model_config.enforce_eager = True
            # TODO (varun): Turning this ON gives incorrect results for the
            # Deepseek-V2-lite model.
            vllm_config.compilation_config.use_inductor = False

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
                             kv_cache_dtype, block_size, use_v1,
                             use_mla) -> str:
        if use_mla:
            if selected_backend == _Backend.CUTLASS_MLA_VLLM_V1:
                if use_v1:
                    logger.info_once("Using Cutlass MLA backend on V1 engine.")
                    return ("vllm.v1.attention.backends.mla."
                            "cutlass_mla.CutlassMLABackend")
                else:
                    logger.warning(
                        "Cutlass MLA backend is only supported on V1 engine")
            # TODO(lucas): refactor to  be more concise
            #  we should probably consider factoring out V1 here
            if selected_backend == _Backend.TRITON_MLA or block_size != 64:
                if use_v1:
                    logger.info_once("Using Metax Triton MLA backend on V1 engine.")
                    return ("vllm_metax.v1.attention.backends.mla.triton_mla.MetaxTritonMLABackend")
                else:
                    logger.info("Using Metax Triton MLA backend.")
                    return "vllm_metax.attention.backends.triton_mla.MetaxTritonMLABackend"
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
                        return ("vllm_metax.v1.attention.backends.mla."
                                "flashmla.MetaxFlashMLABackend")
                    else:
                        logger.info("Using FlashMLA backend.")
                        return ("vllm_metax.attention.backends."
                                "flashmla.MetaxFlashMLABackend")
        if use_v1:
            if selected_backend == _Backend.FLASHINFER:
                logger.info_once("Using Metax FlashInfer backend on V1 engine.")
                return "vllm_metax.v1.attention.backends.flashinfer.MetaxFlashInferBackend"
            if selected_backend == _Backend.FLEX_ATTENTION:
                logger.info("Using FlexAttenion backend on V1 engine.")
                return "vllm.v1.attention.backends.flex_attention.FlexAttentionBackend"  # noqa: E501
            if selected_backend == _Backend.TRITON_ATTN_VLLM_V1:
                logger.info_once("Using Triton backend on V1 engine.")
                return ("vllm.v1.attention.backends."
                        "triton_attn.TritonAttentionBackend")
            if cls.is_device_capability(100):
                # Prefer FlashInfer for V1 on Blackwell GPUs if installed
                try:
                    import flashinfer  # noqa: F401
                    logger.info_once(
                        "Using FlashInfer backend on V1 engine by default for "
                        "Blackwell (SM 10.0) GPUs.")
                    return ("vllm_metax.v1.attention.backends."
                            "flashinfer.MetaxFlashInferBackend")
                except ImportError:
                    logger.info_once(
                        "FlashInfer failed to import for V1 engine on "
                        "Blackwell (SM 10.0) GPUs; it is recommended to "
                        "install FlashInfer for better performance.")
                    pass
            if cls.has_device_capability(80):
                logger.info_once("Using Metax Flash Attention backend on V1 engine.")
                return ("vllm_metax.v1.attention.backends.flash_attn.MetaxFlashAttentionBackend")
        if selected_backend == _Backend.FLASHINFER:
            logger.info("Using FlashInfer backend.")
            return "vllm_metax.attention.backends.flashinfer.MetaxFlashInferImpl"
        elif selected_backend == _Backend.XFORMERS:
            logger.info("Using XFormers backend.")
            return "vllm.attention.backends.xformers.XFormersBackend"
        elif selected_backend == _Backend.DUAL_CHUNK_FLASH_ATTN:
            logger.info("Using DualChunkFlashAttention backend.")
            return ("vllm.attention.backends.dual_chunk_flash_attn."
                    "DualChunkFlashAttentionBackend")
        elif selected_backend == _Backend.FLASH_ATTN:
            pass
        elif selected_backend:
            raise ValueError(
                f"Invalid attention backend for {cls.device_name}, "
                f"with use_v1: {use_v1} use_mla: {use_mla}")

        target_backend = _Backend.FLASH_ATTN
        if not cls.has_device_capability(80):
            # Volta and Turing NVIDIA GPUs.
            logger.info(
                "Cannot use FlashAttention-2 backend for Volta and Turing "
                "GPUs.")
            target_backend = _Backend.XFORMERS
        elif dtype not in (torch.float16, torch.bfloat16):
            logger.info(
                "Cannot use FlashAttention-2 backend for dtype other than "
                "torch.float16 or torch.bfloat16.")
            target_backend = _Backend.XFORMERS
        elif block_size % 16 != 0:
            logger.info(
                "Cannot use FlashAttention-2 backend for block size not "
                "divisible by 16.")
            target_backend = _Backend.XFORMERS

        # FlashAttn is valid for the model, checking if the package is
        # installed.
        if target_backend == _Backend.FLASH_ATTN:
            try:
                import flash_attn  # noqa: F401
                from vllm_metax.attention.backends.flash_attn import (  # noqa: F401
                    FlashAttentionBackend, flash_attn_supports_fp8)
                    
                supported_sizes = \
                    FlashAttentionBackend.get_supported_head_sizes()
                if head_size not in supported_sizes:
                    logger.info(
                        "Cannot use FlashAttention-2 backend for head size %d.",
                        head_size)
                    target_backend = _Backend.XFORMERS
                fp8_kv_cache = (kv_cache_dtype is not None
                                and kv_cache_dtype.startswith("fp8"))
                if (fp8_kv_cache and not flash_attn_supports_fp8()):
                    logger.info(
                        "Cannot use FlashAttention backend for FP8 KV cache.")
                    logger.warning(
                        "Please use FlashInfer backend with FP8 KV Cache for "
                        "better performance by setting environment variable "
                        "VLLM_ATTENTION_BACKEND=FLASHINFER")
                    target_backend = _Backend.XFORMERS
            except ImportError:
                logger.info(
                    "Cannot use FlashAttention-2 backend because the "
                    "vllm.vllm_flash_attn package is not found. "
                    "Make sure that vllm_flash_attn was built and installed "
                    "(on by default).")
                target_backend = _Backend.XFORMERS

        if target_backend == _Backend.XFORMERS:
            logger.info("Using XFormers backend.")
            return "vllm.attention.backends.xformers.XFormersBackend"

        logger.info("Using Flash Attention backend.")

        logger.info_once("Using Metax Flash Attention backend on V0 engine.")
        return "vllm_metax.attention.backends.flash_attn.MetaxFlashAttentionBackend"

    @classmethod
    def get_punica_wrapper(cls) -> str:
        return "vllm.lora.punica_wrapper.punica_gpu.PunicaWrapperGPU"

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        return "vllm.distributed.device_communicators.cuda_communicator.CudaCommunicator"  # noqa

    @classmethod
    def supports_fp8(cls) -> bool:
        return cls.has_device_capability(89)

    @classmethod
    def supports_v1(cls, model_config: "ModelConfig") -> bool:
        return True

    @classmethod
    def use_custom_allreduce(cls) -> bool:
        return False
    
    @classmethod
    def get_piecewise_backend_cls(cls) -> str:
        return "vllm.compilation.cuda_piecewise_backend.CUDAPiecewiseBackend"  # noqa

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
    def pre_register_and_update(cls,
                                parser: Optional[FlexibleArgumentParser] = None
                                ) -> None:
        logger.info("[hook] platform:pre_register_and_update...")
        import vllm_metax.patch


# NVML utils
# Note that NVML is not affected by `CUDA_VISIBLE_DEVICES`,
# all the related functions work on real physical device ids.
# the major benefit of using NVML is that it will not initialize CUDA
class McmlMetaxPlatform(MetaXPlatformBase):

    @classmethod
    @with_mcml_context
    def get_device_capability(cls,
                              device_id: int = 0
                              ) -> Optional[DeviceCapability]:
        try:
            physical_device_id = cls.device_id_to_physical_device_id(device_id)
            handle = pymcml.nvmlDeviceGetHandleByIndex(physical_device_id)
            major, minor = pymcml.nvmlDeviceGetCudaComputeCapability(handle)
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
        handle = pymcml.nvmlDeviceGetHandleByIndex(physical_device_id)
        return pymcml.nvmlDeviceGetUUID(handle)

    @classmethod
    @with_mcml_context
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        physical_device_id = cls.device_id_to_physical_device_id(device_id)
        handle = pymcml.nvmlDeviceGetHandleByIndex(physical_device_id)
        return int(pymcml.nvmlDeviceGetMemoryInfo(handle).total)

    @classmethod
    @with_mcml_context
    def is_fully_connected(cls, physical_device_ids: list[int]) -> bool:
        """
        query if the set of gpus are fully connected by nvlink (1 hop)
        """
        handles = [
            pymcml.nvmlDeviceGetHandleByIndex(i) for i in physical_device_ids
        ]
        for i, handle in enumerate(handles):
            for j, peer_handle in enumerate(handles):
                if i < j:
                    try:
                        p2p_status = pymcml.nvmlDeviceGetP2PStatus(
                            handle,
                            peer_handle,
                            pymcml.NVML_P2P_CAPS_INDEX_NVLINK,
                        )
                        if p2p_status != pymcml.NVML_P2P_STATUS_OK:
                            return False
                    except pymcml.NVMLError:
                        logger.exception(
                            "NVLink detection failed. This is normal if"
                            " your machine has no NVLink equipped.")
                        return False
        return True

    @classmethod
    def _get_physical_device_name(cls, device_id: int = 0) -> str:
        handle = pymcml.nvmlDeviceGetHandleByIndex(device_id)
        return pymcml.nvmlDeviceGetName(handle)

    @classmethod
    @with_mcml_context
    def log_warnings(cls):
        device_ids: int = pymcml.nvmlDeviceGetCount()
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


class NonMcmlMetaxPlatform(MetaXPlatformBase):

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
nvml_available = False
try:
    try:
        pymcml.nvmlInit()
        nvml_available = True
    except Exception:
        # On Jetson, NVML is not supported.
        nvml_available = False
finally:
    if nvml_available:
        pymcml.nvmlShutdown()

MetaXPlatform = McmlMetaxPlatform if nvml_available else NonMcmlMetaxPlatform