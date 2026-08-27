# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Wiring for the msprobe (MindStudio Probe) precision-debugging toolkit.

Mirrors the `--additional-config` UX vllm-ascend already established
(`dump_config` / `dump_config_path`) so a dump captured on Maca can be
compared against one captured on Ascend/CUDA with the same msprobe config.
See vllm-ascend's developer_guide/performance_and_debug/msprobe_guide.md
for the upstream reference.
"""

import json
import tempfile
from typing import Any

from vllm.logger import logger


def _requested_dump_config(additional_config: Any) -> dict | str | None:
    """Return the raw `dump_config`/`dump_config_path` value, or None."""
    if not isinstance(additional_config, dict):
        return None
    if "dump_config" in additional_config:
        return additional_config["dump_config"]
    if "dump_config_path" in additional_config:
        return additional_config["dump_config_path"]
    return None


def is_msprobe_dump_requested(additional_config: Any) -> bool:
    """Whether the user asked for msprobe precision dump via --additional-config."""
    return _requested_dump_config(additional_config) is not None


def resolve_worker_cls(additional_config: Any, enforce_eager: bool | None) -> str:
    """Pick the `Worker` class for `check_and_update_config` to install.

    Returns the stock GPU worker unless a msprobe dump was requested, in
    which case it returns `MacaWorker`. Raises if a dump was requested
    without `--enforce-eager`, since CUDA graph capture bypasses the
    Python-level hooks msprobe relies on.
    """
    if not is_msprobe_dump_requested(additional_config):
        return "vllm.v1.worker.gpu_worker.Worker"
    if enforce_eager is False:
        raise ValueError(
            "msprobe precision dump (--additional-config with "
            "dump_config/dump_config_path) requires --enforce-eager: "
            "CUDA graph capture bypasses the Python-level hooks msprobe "
            "relies on."
        )
    return "vllm_metax.v1.worker.gpu_worker.MacaWorker"


def build_precision_debugger(additional_config: Any) -> Any | None:
    """Instantiate a msprobe `PrecisionDebugger` from `config_path`.

    `PrecisionDebugger.__init__` does not take a model; the model is passed
    to `debugger.start(model=...)` on every `execute_model` call instead, see
    `vllm_metax.v1.worker.gpu_worker.MacaWorker`.

    Returns None if no dump was requested. Raises if a dump was requested but
    the `mindstudio-probe` package is not installed, since silently skipping
    the dump would be more confusing than failing at startup.
    """
    dump_config = _requested_dump_config(additional_config)
    if dump_config is None:
        return None

    try:
        from msprobe.pytorch import PrecisionDebugger
    except ImportError as e:
        raise RuntimeError(
            "msprobe precision dump was requested via --additional-config "
            "(dump_config/dump_config_path) but the `mindstudio-probe` "
            "package is not installed. Install it with "
            "`pip install mindstudio-probe`."
        ) from e

    if isinstance(dump_config, dict):
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            prefix="msprobe_dump_config_",
            delete=False,
        ) as tmp:
            json.dump(dump_config, tmp)
            config_path = tmp.name
    else:
        config_path = dump_config

    logger.info("msprobe precision dump enabled, config_path=%s", config_path)
    return PrecisionDebugger(config_path=config_path)
