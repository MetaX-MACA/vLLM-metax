# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import os
import sys

import pytest
import torch

import vllm_metax
import vllm_metax.platform as metax_platform
from vllm_metax.platform import MacaPlatform, MacaPlatformBase, MxsmlMacaPlatform


@pytest.mark.requires_maca
def test_current_platform_is_plugin_platform(platform):
    from vllm.platforms import current_platform

    assert isinstance(current_platform, MacaPlatformBase)
    assert type(current_platform) is MxsmlMacaPlatform
    assert isinstance(current_platform, MacaPlatform)


def test_maca_platform_identity_and_dispatch_contract(platform):
    assert MacaPlatformBase._enum.name == "OOT"
    assert MacaPlatform.device_name == "maca"
    assert MacaPlatform.device_type == "cuda"
    assert MacaPlatform.dispatch_key == "CUDA"
    assert MacaPlatform.ray_device_key == "GPU"
    assert MacaPlatform.dist_backend == "nccl"
    assert MacaPlatform.device_control_env_var == "CUDA_VISIBLE_DEVICES"
    assert platform().is_cuda_alike() is True
    assert platform().is_maca() is True


@pytest.mark.requires_maca
def test_live_device_contract(platform):
    capability = platform.get_device_capability()
    assert capability is not None
    assert capability.major >= 0 and capability.minor >= 0
    assert platform.device_count() == torch.cuda.device_count()
    assert platform.get_device_name(0)
    assert platform.get_device_total_memory(0) > 0


@pytest.mark.requires_maca
def test_source_kernel_import_is_plugin_owned(platform):
    if os.environ.get("USE_PRECOMPILED_KERNEL", "0") != "0":
        pytest.skip("source-kernel provider not selected")

    platform.import_kernels()
    import vllm_metax._C_stable_libtorch as core_extension
    import vllm_metax._moe_C_stable_libtorch as moe_extension

    assert "vllm_metax" in core_extension.__file__
    assert "vllm_metax" in moe_extension.__file__
    assert "mcoplib" not in core_extension.__file__
    assert "mcoplib" not in moe_extension.__file__
    assert not [name for name in sys.modules if name == "mcoplib" or name.startswith("mcoplib.")]


def test_attention_backend_priority_contract():
    assert [backend.name for backend in metax_platform._get_backend_priorities(True, None)] == [
        "FLASHMLA",
        "TRITON_MLA",
        "FLASHMLA_SPARSE",
    ]
    assert [backend.name for backend in metax_platform._get_backend_priorities(False, None)] == [
        "FLASH_ATTN",
        "TRITON_ATTN",
        "FLASHINFER",
        "FLEX_ATTENTION",
        "TURBOQUANT",
    ]


def test_vision_attention_backend_policy(platform):
    assert [backend.name for backend in platform.get_supported_vit_attn_backends()] == [
        "FLASH_ATTN",
        "TORCH_SDPA",
    ]
    assert platform.get_vit_attn_backend(64, torch.float16).name == "FLASH_ATTN"
    assert platform.get_vit_attn_backend(72, torch.float16).name == "TORCH_SDPA"


def test_platform_native_feature_contract(platform):
    instance = platform()
    assert instance.is_sleep_mode_available()
    assert instance.is_cumem_allocator_available()
    assert instance.opaque_attention_op()
    assert instance.support_hybrid_kv_cache()
    assert instance.support_static_graph_mode()
    assert instance.use_custom_op_collectives()
    assert instance.use_custom_allreduce() is False
    assert instance.get_punica_wrapper().endswith("PunicaWrapperGPU")
    assert instance.get_device_communicator_cls().startswith("vllm_metax.")


def test_quantization_capability_gating(platform):
    supports_fp8 = platform.supports_fp8()
    if not supports_fp8:
        with pytest.raises(ValueError, match="FP8 support"):
            platform.verify_quantization("fp8")
    else:
        platform.verify_quantization("fp8")


def test_environment_validation_accepts_plugin_variables(monkeypatch):
    monkeypatch.setenv("VLLM_METAX_ENABLE_MRV2_HWQ_BALANCING", "1")
    monkeypatch.delenv("VLLM_USE_V1", raising=False)
    MacaPlatform.validate_environ(hard_fail=True)
