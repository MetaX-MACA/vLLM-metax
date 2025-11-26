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
    from vllm.attention.backends.registry import AttentionBackendEnum
    from vllm.config import VllmConfig
    from vllm.config.cache import CacheDType
else:
    AttentionBackendEnum = None
    VllmConfig = None
    CacheDType = None

_P = ParamSpec("_P")
_R = TypeVar("_R")

pymxml = import_pymxml()

# pytorch 2.5 uses cudnn sdpa by default, which will cause crash on some models
# see https://github.com/huggingface/diffusers/issues/9704 for details
# torch.backends.cuda.enable_cudnn_sdp(False)
torch.backends.cuda.enable_cudnn_sdp(False)


@cache
def _get_backend_priorities(
    use_mla: bool,
    device_capability: DeviceCapability,
) -> list[AttentionBackendEnum]:
    """Get backend priorities with lazy import to avoid circular dependency."""
    from vllm.attention.backends.registry import AttentionBackendEnum

    if use_mla:
        return [
            AttentionBackendEnum.FLASHMLA,
            AttentionBackendEnum.TRITON_MLA,
            AttentionBackendEnum.CUTLASS_MLA,
            # AttentionBackendEnum.FLASHINFER_MLA,
            # AttentionBackendEnum.FLASH_ATTN_MLA,
            AttentionBackendEnum.FLASHMLA_SPARSE,
        ]
    else:
        return [
            AttentionBackendEnum.FLASH_ATTN,
            AttentionBackendEnum.FLASHINFER,
            AttentionBackendEnum.TRITON_ATTN,
            AttentionBackendEnum.TREE_ATTN,
            AttentionBackendEnum.FLEX_ATTENTION,
        ]


def register_attention_backends() -> None:
    # Pre-register all attention backends
    logger.info("Pre-registering attention backends.")
    from vllm.attention.backends.registry import (
        AttentionBackendEnum,
        register_backend,
    )

    register_backend(
        AttentionBackendEnum.FLASHMLA,
        "vllm_metax.v1.attention.backends.mla.flashmla.MacaFlashMLABackend",
    )
    register_backend(
        AttentionBackendEnum.FLASHMLA_SPARSE,
        "vllm_metax.v1.attention.backends.mla.flashmla_sparse.MacaFlashMLASparseBackend",
    )
    register_backend(
        AttentionBackendEnum.TRITON_MLA,
        "vllm_metax.v1.attention.backends.mla.triton_mla.MacaTritonMLABackend",
    )
    register_backend(
        AttentionBackendEnum.FLASH_ATTN,
        "vllm_metax.v1.attention.backends.flash_attn.MacaFlashAttentionBackend",
    )
    register_backend(
        AttentionBackendEnum.FLASHINFER,
        "vllm_metax.v1.attention.backends.flashinfer.MacaFlashInferBackend",
    )
    register_backend(
        AttentionBackendEnum.TRITON_ATTN,
        "vllm_metax.v1.attention.backends.triton_attn.MacaTritonAttentionBackend",
    )
    register_backend(
        AttentionBackendEnum.TREE_ATTN,
        "vllm_metax.v1.attention.backends.tree_attn.MacaTreeAttentionBackend",
    )
    register_backend(
        AttentionBackendEnum.FLEX_ATTENTION,
        "vllm_metax.v1.attention.backends.flex_attention.MacaFlexAttentionBackend",
    )


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
    def get_vit_attn_backend(
        cls, head_size: int, dtype: torch.dtype
    ) -> "AttentionBackendEnum":
        register_attention_backends()
        from vllm.attention.backends.registry import AttentionBackendEnum

        # TODO(Hank) Need to check which is better between
        # TORCH_SDPA or FLASH_ATTN on Maca platform

        backend_class = AttentionBackendEnum.FLASH_ATTN.get_class()
        if backend_class.supports_head_size(head_size) and backend_class.supports_dtype(
            dtype
        ):
            return AttentionBackendEnum.FLASH_ATTN
        else:
            logger.error(
                "Fallback to Backend TORCH_SDPA as vit_attn_backend since head_size or dtype is "
                "not supported on FLASH_ATTN."
            )
            return AttentionBackendEnum.TORCH_SDPA

    @classmethod
    def get_valid_backends(
        cls,
        head_size,
        dtype,
        kv_cache_dtype,
        block_size,
        use_mla,
        has_sink,
        use_sparse,
        device_capability,
        attn_type,
    ) -> tuple[
        list[tuple["AttentionBackendEnum", int]],
        dict["AttentionBackendEnum", list[str]],
    ]:
        valid_backends_priorities = []
        invalid_reasons = {}

        backend_priorities = _get_backend_priorities(use_mla, device_capability)
        for priority, backend in enumerate(backend_priorities):
            try:
                backend_class = backend.get_class()
                invalid_reasons_i = backend_class.validate_configuration(
                    head_size,
                    dtype,
                    kv_cache_dtype,
                    block_size,
                    use_mla,
                    has_sink,
                    use_sparse,
                    device_capability,
                    attn_type,
                )
            except ImportError:
                invalid_reasons_i = ["ImportError"]
            if invalid_reasons_i:
                invalid_reasons[backend] = invalid_reasons_i
            else:
                valid_backends_priorities.append((backend, priority))

        return valid_backends_priorities, invalid_reasons

    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend: "AttentionBackendEnum",
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: "CacheDType | None",
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        attn_type: str | None = None,
    ) -> str:
        register_attention_backends()

        from vllm.attention import AttentionType

        if attn_type is None:
            attn_type = AttentionType.DECODER

        device_capability = cls.get_device_capability()
        assert device_capability is not None

        # First try checking just the selected backend, if there is one.
        if selected_backend is not None:
            try:
                backend_class = selected_backend.get_class()
                invalid_reasons = backend_class.validate_configuration(
                    head_size,
                    dtype,
                    kv_cache_dtype,
                    None,
                    use_mla,
                    has_sink,
                    use_sparse,
                    device_capability,
                    attn_type,
                )
            except ImportError:
                invalid_reasons = ["ImportError"]
            if invalid_reasons:
                raise ValueError(
                    f"Selected backend {selected_backend} is not valid for "
                    f"this configuration. Reason: {invalid_reasons}"
                )
            else:
                logger.info("Using %s backend.", selected_backend)
                return selected_backend.get_path()

        # No selected backend or the selected backend is invalid,
        # so we try finding a valid backend.
        valid_backends_priorities, invalid_reasons = cls.get_valid_backends(
            head_size,
            dtype,
            kv_cache_dtype,
            None,
            use_mla,
            has_sink,
            use_sparse,
            device_capability,
            attn_type,
        )
        reasons_str = (
            "{"
            + ", ".join(
                f"{backend.name}: [{', '.join(reasons)}]"
                for backend, reasons in invalid_reasons.items()
            )
            + "}"
        )
        config_str = (
            f"head_size: {head_size}, dtype: {dtype}, "
            f"kv_cache_dtype: {kv_cache_dtype}, block_size: {block_size}, "
            f"use_mla: {use_mla}, has_sink: {has_sink}, use_sparse: {use_sparse}"
        )
        if invalid_reasons:
            logger.info_once(
                f"Some attention backends are not valid for {cls.device_name} with "
                f"{config_str}. Reasons: {reasons_str}."
            )
        if len(valid_backends_priorities) == 0:
            raise ValueError(
                f"No valid attention backend found for {cls.device_name} "
                f"with {config_str}. Reasons: {reasons_str}."
            )

        # We have found some valid backends. Select the one with the
        # highest priority.
        logger.info(
            "Valid backends: %s", [b[0].name for b in valid_backends_priorities]
        )
        sorted_indices = sorted(
            range(len(valid_backends_priorities)),
            key=lambda i: valid_backends_priorities[i][1],
        )
        selected_index = sorted_indices[0]
        selected_backend = valid_backends_priorities[selected_index][0]
        logger.info(
            "Using %s backend.",
            selected_backend.name,
        )

        return selected_backend.get_path()

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
        """Pre-register and update Maca platform."""
        register_attention_backends()
        # TODO(m01016): update cudagraph max capture size  here


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
# envs.VLLM_USE_STANDALONE_COMPILE = False
envs.VLLM_USE_TRTLLM_ATTENTION = False
envs.VLLM_USE_CUDNN_PREFILL = False
envs.VLLM_USE_TRTLLM_RAGGED_DEEPSEEK_PREFILL = False
