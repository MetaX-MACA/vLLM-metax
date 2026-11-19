# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import pytest

from vllm_metax.utils import import_pymxsml


def test_import_pymxsml_resolves_plugin_vendored_module():
    module = import_pymxsml()
    assert module.__name__ == "vllm_metax.third_party.pymxsml"
    assert module.__file__.endswith("vllm_metax/third_party/pymxsml.py")


@pytest.mark.requires_maca
def test_pymxsml_reports_maca_devices_and_isa():
    module = import_pymxsml()
    module.nvmlInit()
    try:
        count = module.nvmlDeviceGetCount()
        assert count > 0
        assert module.mxSmlGetDeviceIsaVersion(0) > 0
    finally:
        module.nvmlShutdown()
