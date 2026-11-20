# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#
# Note: Preserve the connector factory's lazy loading. Importing Mooncake's
# native transfer engine during general plugin registration can corrupt native
# teardown even for runs that do not use a KV connector.
# Affected versions: vLLM 0.30.0 (ced6857afa0e).
# Remove at: upstream provides ready notifications without batch metadata.
"""Install early-ready only when the built-in Mooncake connector is selected."""

import importlib

from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory

from vllm_metax.patch.utils import patch

_FACTORY = "vllm.distributed.kv_transfer.kv_connector.factory"
_MOONCAKE = "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector"
_get_class = KVConnectorFactory.get_connector_class.__func__
_get_class_by_name = KVConnectorFactory.get_connector_class_by_name.__func__


def _install(connector_cls):
    if connector_cls.__module__ == _MOONCAKE:
        importlib.import_module(
            "vllm_metax.patch.enhancement.performance.mooncake_early_ready"
        )
    return connector_cls


@patch(_FACTORY, "KVConnectorFactory.get_connector_class")  # type: ignore[misc]
@classmethod
def get_connector_class(cls, kv_transfer_config):
    return _install(_get_class(cls, kv_transfer_config))


@patch(  # type: ignore[misc]
    _FACTORY, "KVConnectorFactory.get_connector_class_by_name"
)
@classmethod
def get_connector_class_by_name(cls, connector_name):
    return _install(_get_class_by_name(cls, connector_name))
