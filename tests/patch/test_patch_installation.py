# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import importlib
import sys
from pathlib import Path

import pytest

import vllm_metax.patch.bugfix  # noqa: F401
import vllm_metax.patch.enhancement  # noqa: F401
import vllm_metax.patch.models  # noqa: F401


PATCH_ROOT = Path(__file__).parents[2] / "vllm_metax" / "patch"
PATCH_MODULES = sorted(
    "vllm_metax."
    + ".".join(path.relative_to(PATCH_ROOT.parent).with_suffix("").parts)
    for path in PATCH_ROOT.rglob("*.py")
    if path.name != "__init__.py" and "template" not in path.parts
)


@pytest.mark.parametrize("module_name", PATCH_MODULES)
def test_every_plugin_patch_module_is_activated(module_name):
    assert module_name in sys.modules


def test_device_allocator_submodule_is_redirected_to_plugin():
    import vllm.device_allocator.cumem as upstream_module
    import vllm_metax.device_allocator.cumem as plugin_module

    assert upstream_module is plugin_module


def test_distributed_pynccl_wrapper_is_replaced_by_plugin():
    module = importlib.import_module(
        "vllm.distributed.device_communicators.pynccl_wrapper"
    )
    assert module.NCCLLibrary.__module__.startswith("vllm_metax.")


def test_nccl_library_discovery_uses_mccl_resolver():
    import vllm.utils.nccl as target
    from vllm_metax.patch.enhancement.distributed import utils_patch

    assert target.find_nccl_library is utils_patch.find_mccl_library


def test_scheduler_schedule_is_plugin_replacement():
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm_metax.patch.enhancement.performance import pp_batch_balance

    assert Scheduler.schedule is pp_batch_balance._schedule_with_pp_balance


def test_speculative_config_override_is_plugin_replacement():
    from vllm.config.speculative import SpeculativeConfig
    from vllm_metax.patch.bugfix import draft_config_overrides

    assert (
        SpeculativeConfig.hf_config_override.__module__
        == draft_config_overrides.__name__
    )


def test_group_coordinator_patch_uses_maca_collective_policy():
    from vllm.distributed.parallel_state import GroupCoordinator
    from vllm_metax.patch.bugfix import parallel_state

    assert (
        GroupCoordinator._should_use_all_gather.__module__
        == parallel_state.__name__
    )


def test_lora_kernel_bounds_wrapper_is_installed():
    from vllm.lora.ops.triton_ops import fused_moe_lora_op as upstream
    from vllm_metax.patch.bugfix import fused_moe_lora_expert_bounds as plugin

    assert upstream._run_fused_moe_lora_small_batch is plugin._run_fused_moe_lora_small_batch
    assert upstream._fused_moe_lora_small_batch_kernel is plugin._fused_moe_lora_small_batch_kernel
