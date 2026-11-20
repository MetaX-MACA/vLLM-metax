# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import os

import pytest
import torch

# The plugin test suite validates native source kernels by default. Explicitly
# setting the variable to 1 remains possible for a separately approved mcoplib run.
os.environ.setdefault("USE_PRECOMPILED_KERNEL", "0")
os.environ.setdefault("VLLM_USE_V1", "1")

import vllm_metax

vllm_metax.register_out_of_tree()
vllm_metax.register_attn_backends()
vllm_metax.register_model()


@pytest.fixture(scope="session")
def platform():
    from vllm_metax.platform import MacaPlatform

    return MacaPlatform


@pytest.fixture(scope="session")
def device() -> torch.device:
    from vllm_metax.platform import MacaPlatform

    return torch.device(MacaPlatform.device_type)


def pytest_collection_modifyitems(config, items):
    marker = pytest.mark.requires_maca
    for item in items:
        if "requires_maca" in item.keywords:
            item.add_marker(marker)


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "requires_maca: test requires a live MACA device"
    )


@pytest.fixture(autouse=True)
def _skip_without_maca(request):
    if request.node.get_closest_marker("requires_maca"):
        from vllm.platforms import current_platform

        if not current_platform.is_maca() or not torch.cuda.is_available():
            pytest.skip("requires a live MACA device")
