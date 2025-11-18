# SPDX-License-Identifier: Apache-2.0
"""Code inside this file can safely assume cuda platform, e.g. importing
pynvml. However, it should not initialize cuda context.
"""

import contextlib
import os
from collections.abc import Callable
from functools import cache, wraps
from typing import TYPE_CHECKING, TypeVar

import torch
from typing_extensions import ParamSpec

import vllm.envs as envs
from vllm.logger import logger
from vllm_metax.utils import import_pymxml
from vllm.utils.torch_utils import cuda_device_count_stateless

from vllm.platforms.interface import DeviceCapability, Platform, PlatformEnum
from vllm.utils.argparse_utils import FlexibleArgumentParser

if TYPE_CHECKING:
    from vllm.attention.backends.registry import _Backend
    from vllm.config import VllmConfig

_P = ParamSpec("_P")
_R = TypeVar("_R")

pymxml = import_pymxml()

# pytorch 2.5 uses cudnn sdpa by default, which will cause crash on some models
# see https://github.com/huggingface/diffusers/issues/9704 for details
# torch.backends.cuda.enable_cudnn_sdp(False)
torch.backends.cuda.enable_cudnn_sdp(False)


def with_mxml_context(fn: Callable[_P, _R]) -> Callable[_P, _R]:
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
    device_name: str = "maca"
    device_type: str = "cuda"
    dispatch_key: str = "CUDA"
    ray_device_key: str = "GPU"
    dist_backend: str = "nccl"
    device_control_env_var: str = "CUDA_VISIBLE_DEVICES"

    supported_quantization: list[str] = [
        "awq",
        "gptq",
        "compressed-tensors",
        "compressed_tensors",
        "moe_wna16",
        "gguf",
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
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability | None:
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
    def is_sleep_mode_available(cls) -> bool:
        return True

    @classmethod
    def is_fully_connected(cls, device_ids: list[int]) -> bool:
        raise NotImplementedError

    @classmethod
    def log_warnings(cls):
        pass

    @classmethod
    def import_kernels(cls) -> None:
        """Import any platform-specific C kernels."""
        try:
            import vllm_metax._C  # noqa: F401
        except ImportError as e:
            logger.warning("Failed to import from vllm_metax._C: %r", e)
        with contextlib.suppress(ImportError):
            import vllm_metax._moe_C  # noqa: F401

    @classmethod
    def check_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        # Config Override
        parallel_config = vllm_config.parallel_config
        compilation_config = vllm_config.compilation_config
        model_config = vllm_config.model_config

        if parallel_config.worker_cls == "auto":
            parallel_config.worker_cls = "vllm.v1.worker.gpu_worker.Worker"

        cache_config = vllm_config.cache_config
        if cache_config and cache_config.block_size is None:
            cache_config.block_size = 16

        # TODO(lucas): handle this more gracefully
        # Note: model_config may be None during testing
        if (
            model_config is not None
            and model_config.use_mla
            and cache_config.block_size is not None
        ):
            use_sparse = hasattr(vllm_config.model_config.hf_config, "index_topk")
            # If `VLLM_ATTENTION_BACKEND` is not set and we are using MLA,
            # then we default to FlashMLA backend for non-blackwell GPUs,
            # else we default to CutlassMLA. For each case, we force the
            # required block_size.
            use_flashmla = False
            use_cutlass_mla = False
            use_flashinfer_mla = False

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
                use_flashmla = envs.VLLM_ATTENTION_BACKEND == "FLASHMLA"
                use_cutlass_mla = envs.VLLM_ATTENTION_BACKEND == "CUTLASS_MLA"
                use_flashinfer_mla = envs.VLLM_ATTENTION_BACKEND == "FLASHINFER_MLA"

            from vllm_metax.attention.ops.flashmla import is_flashmla_dense_supported

            if (
                use_flashmla
                and is_flashmla_dense_supported()[0]
                and cache_config.block_size % 64 != 0
            ):
                cache_config.block_size = 64
                logger.info("Forcing kv cache block size to 64 for FlashMLA backend.")

            if use_cutlass_mla and cache_config.block_size != 128:
                cache_config.block_size = 128
                logger.info(
                    "Forcing kv cache block size to 128 for CUTLASS_MLA backend."
                )

            if (
                use_flashinfer_mla
                and cache_config.block_size != 32
                and cache_config.block_size % 64 != 0
            ):
                cache_config.block_size = 64
                logger.info(
                    "Forcing kv cache block size to 64 for FlashInferMLA backend."
                )

            # TODO(Chen): remove this hacky code
            if use_sparse and cache_config.block_size != 64:
                cache_config.block_size = 64
                logger.info(
                    "Forcing kv cache block size to 64 for FlashMLASparse backend."
                )
        # lazy import to avoid circular import
        from vllm.config import CUDAGraphMode

        compilation_config = vllm_config.compilation_config
        if (
            envs.VLLM_ALL2ALL_BACKEND == "deepep_high_throughput"
            and parallel_config.data_parallel_size > 1
            and compilation_config.cudagraph_mode != CUDAGraphMode.NONE
        ):
            # TODO: Piecewise Cuda graph might be enabled
            # if torch compile cache key issue fixed
            # See https://github.com/vllm-project/vllm/pull/25093
            logger.info(
                "WideEP: Disabling CUDA Graphs since DeepEP high-throughput "
                "kernels are optimized for prefill and are incompatible with "
                "CUDA Graphs. "
                "In order to use CUDA Graphs for decode-optimized workloads, "
                "set VLLM_ALL2ALL_BACKEND to another option, such as "
                "deepep_low_latency, pplx, or allgather_reducescatter."
            )
            compilation_config.cudagraph_mode = CUDAGraphMode.NONE

        # Disable cascade attention for Maca platform currently
        if vllm_config.model_config is not None:
            model_config.disable_cascade_attn = True

    @classmethod
    def get_current_memory_usage(
        cls, device: torch.types.Device | None = None
    ) -> float:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        return torch.cuda.max_memory_allocated(device)

    @classmethod
    def get_vit_attn_backend(cls, head_size: int, dtype: torch.dtype) -> "_Backend":
        from vllm.attention.backends.registry import _Backend

        # TODO(Hank) Need to check which is better between
        # TORCH_SDPA or FLASH_ATTN on Maca platform
        FLASH_ATTN_V1 = (
            "vllm_metax.v1.attention.backends.flash_attn.MacaFlashAttentionBackend"  # noqa: E501
        )
        from vllm.attention.selector import is_attn_backend_supported

        if is_default_fa_supported := is_attn_backend_supported(
            FLASH_ATTN_V1, head_size, dtype, allow_import_error=False
        ):
            return _Backend.FLASH_ATTN
        else:
            use_sdpa_attention_reason = {}
            if not is_default_fa_supported.head_size:
                use_sdpa_attention_reason["head_size"] = head_size
            if not is_default_fa_supported.dtype:
                use_sdpa_attention_reason["dtype"] = dtype
            logger.warning(
                "Fallback to Backend TORCH_SDPA as vit_attn_backend since %s is "
                "not supported on FLASH_ATTN.",
                ", ".join(f"{k}={v}" for k, v in use_sdpa_attention_reason.items()),
            )
            return _Backend.TORCH_SDPA

    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend,
        head_size,
        dtype,
        kv_cache_dtype,
        block_size,
        use_v1,
        use_mla,
        has_sink,
        use_sparse,
    ) -> str:
        from vllm.attention.backends.registry import _Backend

        if use_mla:
            if not use_v1:
                raise RuntimeError(
                    "MLA attention backends require the V1 engine. "
                    "Set VLLM_USE_V1=1 to enable them."
                )

            from vllm_metax.attention.ops.flashmla import is_flashmla_dense_supported
            from vllm_metax.attention.utils.fa_utils import flash_attn_supports_mla

            if use_sparse:
                logger.info_once("Using Sparse MLA backend on V1 engine.")
                return (
                    "vllm_metax.v1.attention.backends.mla.flashmla_sparse."
                    "MacaFlashMLASparseBackend"
                )

            use_cutlassmla = selected_backend == _Backend.CUTLASS_MLA or (
                selected_backend is None and block_size % 128 == 0
            )
            use_flashinfermla = selected_backend == _Backend.FLASHINFER_MLA or (
                selected_backend is None and (block_size == 32 or block_size % 64 == 0)
            )
            use_flashmla = selected_backend == _Backend.FLASHMLA or (
                selected_backend is None and is_flashmla_dense_supported()[0]
            )
            use_flashattn_mla = selected_backend == _Backend.FLASH_ATTN_MLA or (
                selected_backend is None and flash_attn_supports_mla()
            )
            use_triton = selected_backend == _Backend.TRITON_MLA or (
                selected_backend is None
            )

            if use_flashmla:
                if block_size % 64 != 0:
                    logger.warning(
                        "FlashMLA backend is not supported for block size %d"
                        " (currently only supports block size 64).",
                        block_size,
                    )
                else:
                    logger.info_once("Using FlashMLA backend on V1 engine.")
                    return "vllm_metax.v1.attention.backends.mla.flashmla.MacaFlashMLABackend"  # noqa: E501
            if use_triton:
                logger.info_once("Using Triton MLA backend on V1 engine.")
                return "vllm_metax.v1.attention.backends.mla.triton_mla.MacaTritonMLABackend"  # noqa: E501
            # default mla
            logger.warning(
                "Selected MLA backend is not valid, falling back to Triton MLA."
            )
            return (
                "vllm_metax.v1.attention.backends.mla.triton_mla.MacaTritonMLABackend"  # noqa: E501
            )
        if use_v1:
            assert not use_mla
            FLASHINFER_V1 = (
                "vllm_metax.v1.attention.backends.flashinfer.MacaFlashInferBackend"  # noqa: E501
            )
            FLEX_ATTENTION_V1 = "vllm_metax.v1.attention.backends.flex_attention.MacaFlexAttentionBackend"  # noqa: E501
            TRITON_ATTN = "vllm_metax.v1.attention.backends.triton_attn.MacaTritonAttentionBackend"  # noqa: E501
            FLASH_ATTN_V1 = (
                "vllm_metax.v1.attention.backends.flash_attn.MacaFlashAttentionBackend"  # noqa: E501
            )
            TREE_ATTN_V1 = (
                "vllm_metax.v1.attention.backends.tree_attn.MacaTreeAttentionBackend"  # noqa: E501
            )

            if selected_backend == _Backend.FLASHINFER:
                logger.info_once("Using FlashInfer backend on V1 engine.")
                from vllm.v1.attention.backends.utils import set_kv_cache_layout

                set_kv_cache_layout("HND")
                return FLASHINFER_V1
            elif selected_backend == _Backend.FLEX_ATTENTION:
                logger.info_once("Using FlexAttention backend on V1 engine.")
                return FLEX_ATTENTION_V1
            elif selected_backend == _Backend.TRITON_ATTN:
                logger.info_once("Using Triton backend on V1 engine.")
                return TRITON_ATTN
            elif selected_backend == _Backend.FLASH_ATTN:
                logger.info_once("Using Flash Attention backend on V1 engine.")
                return FLASH_ATTN_V1
            elif selected_backend == _Backend.TREE_ATTN:
                logger.info_once("Using Tree Attention backend on V1 engine.")
                return TREE_ATTN_V1

            from vllm.attention.selector import is_attn_backend_supported

            # Default backends for V1 engine
            # FlashAttention is the default for MetaX GPUs
            if is_default_backend_supported := is_attn_backend_supported(
                FLASH_ATTN_V1, head_size, dtype, allow_import_error=False
            ):
                logger.info_once("Using Flash Attention backend on V1 engine.")
                return FLASH_ATTN_V1
            if is_default_backend_supported := is_attn_backend_supported(
                FLASHINFER_V1, head_size, dtype
            ):
                from vllm.v1.attention.backends.utils import set_kv_cache_layout

                logger.info_once(
                    "Using FlashInfer backend with HND KV cache layout on "
                    "V1 engine by default for MetaX GPUs."
                )
                set_kv_cache_layout("HND")

                return FLASHINFER_V1
            if has_sink:
                logger.info_once("Using Triton backend on V1 engine.")
                return TRITON_ATTN

            use_flex_attention_reason = {}
            if not is_default_backend_supported.head_size:
                use_flex_attention_reason["head_size"] = head_size
            if not is_default_backend_supported.dtype:
                use_flex_attention_reason["dtype"] = dtype

            logger.info_once(
                "Using FlexAttention backend for %s on V1 engine.",
                ", ".join(f"{k}={v}" for k, v in use_flex_attention_reason.items()),
            )
            return FLEX_ATTENTION_V1

        raise RuntimeError(
            "V0 attention backends have been removed. Set VLLM_USE_V1=1 "
            "to select a supported backend."
        )

    @classmethod
    def get_punica_wrapper(cls) -> str:
        return "vllm.lora.punica_wrapper.punica_gpu.PunicaWrapperGPU"

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        return (
            "vllm.distributed.device_communicators.cuda_communicator.CudaCommunicator"  # noqa
        )

    @classmethod
    def supports_fp8(cls) -> bool:
        return False

    @classmethod
    def use_custom_allreduce(cls) -> bool:
        return False

    @classmethod
    def opaque_attention_op(cls) -> bool:
        return True

    @classmethod
    def get_static_graph_wrapper_cls(cls) -> str:
        return "vllm.compilation.cuda_graph.CUDAGraphWrapper"

    @classmethod
    def device_count(cls) -> int:
        return cuda_device_count_stateless()

    @classmethod
    def check_if_supports_dtype(cls, torch_dtype: torch.dtype):
        if torch_dtype == torch.float8_e4m3fn or torch_dtype == torch.float8_e5m2:  # noqa
            raise ValueError("FP8 is not supported on GPUs ")

    @classmethod
    def insert_blocks_to_device(
        cls,
        src_cache: torch.Tensor,
        dst_cache: torch.Tensor,
        src_block_indices: torch.Tensor,
        dst_block_indices: torch.Tensor,
    ) -> None:
        """Copy blocks from src_cache to dst_cache on GPU."""
        _src_cache = src_cache[:, src_block_indices]
        dst_cache[:, dst_block_indices] = _src_cache.to(dst_cache.device)

    @classmethod
    def swap_out_blocks_to_host(
        cls,
        src_cache: torch.Tensor,
        dst_cache: torch.Tensor,
        src_block_indices: torch.Tensor,
        dst_block_indices: torch.Tensor,
    ) -> None:
        """Copy blocks from GPU to host (CPU)."""
        _src_cache = src_cache[:, src_block_indices]
        dst_cache[:, dst_block_indices] = _src_cache.cpu()

    @classmethod
    def support_hybrid_kv_cache(cls) -> bool:
        return True

    @classmethod
    def support_static_graph_mode(cls) -> bool:
        return True

    @classmethod
    def pre_register_and_update(
        cls, parser: FlexibleArgumentParser | None = None
    ) -> None:
        # TODO(m01016): update cudagraph max capture size  here
        logger.info("Pre-registering and updating Maca platform.")


# NVML utils
# Note that NVML is not affected by `CUDA_VISIBLE_DEVICES`,
# all the related functions work on real physical device ids.
# the major benefit of using NVML is that it will not initialize CUDA
class MxmlPlatform(MacaPlatformBase):
    @classmethod
    @cache
    @with_mxml_context
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability | None:
        try:
            physical_device_id = cls.device_id_to_physical_device_id(device_id)
            handle = pymxml.nvmlDeviceGetHandleByIndex(physical_device_id)
            major, minor = pymxml.nvmlDeviceGetCudaComputeCapability(handle)
            return DeviceCapability(major=major, minor=minor)
        except RuntimeError:
            return None

    @classmethod
    @with_mxml_context
    def has_device_capability(
        cls,
        capability: tuple[int, int] | int,
        device_id: int = 0,
    ) -> bool:
        try:
            return super().has_device_capability(capability, device_id)
        except RuntimeError:
            return False

    @classmethod
    @with_mxml_context
    def get_device_name(cls, device_id: int = 0) -> str:
        return "Device 4000"

    @classmethod
    @with_mxml_context
    def get_device_uuid(cls, device_id: int = 0) -> str:
        physical_device_id = cls.device_id_to_physical_device_id(device_id)
        handle = pymxml.nvmlDeviceGetHandleByIndex(physical_device_id)
        return pymxml.nvmlDeviceGetUUID(handle)

    @classmethod
    @with_mxml_context
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        physical_device_id = cls.device_id_to_physical_device_id(device_id)
        handle = pymxml.nvmlDeviceGetHandleByIndex(physical_device_id)
        return int(pymxml.nvmlDeviceGetMemoryInfo(handle).total)

    @classmethod
    @with_mxml_context
    def is_fully_connected(cls, physical_device_ids: list[int]) -> bool:
        """
        query if the set of gpus are fully connected by nvlink (1 hop)
        """
        handles = [pymxml.nvmlDeviceGetHandleByIndex(i) for i in physical_device_ids]
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
                            " your machine has no NVLink equipped."
                        )
                        return False
        return True

    @classmethod
    def _get_physical_device_name(cls, device_id: int = 0) -> str:
        return "Device 4000"
        # handle = pymxml.nvmlDeviceGetHandleByIndex(device_id)
        # return pymxml.nvmlDeviceGetName(handle)

    @classmethod
    @with_mxml_context
    def log_warnings(cls):
        device_ids: int = pymxml.nvmlDeviceGetCount()
        if device_ids > 1:
            device_names = [cls._get_physical_device_name(i) for i in range(device_ids)]
            if (
                len(set(device_names)) > 1
                and os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID"
            ):
                logger.warning(
                    "Detected different devices in the system: %s. Please"
                    " make sure to set `CUDA_DEVICE_ORDER=PCI_BUS_ID` to "
                    "avoid unexpected behavior.",
                    ", ".join(device_names),
                )


class NonMxmlMetaxPlatform(MacaPlatformBase):
    @classmethod
    @cache
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
    def is_fully_connected(cls, physical_device_ids: list[int]) -> bool:
        logger.exception(
            "MetaXLink detection not possible, as context support was"
            " not found. Assuming no MetaXLink available."
        )
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
MacaPlatform.log_warnings()


# Note: Put all env Override here for Maca platform
envs.VLLM_USE_FLASHINFER_SAMPLER = False
envs.VLLM_USE_STANDALONE_COMPILE = False
