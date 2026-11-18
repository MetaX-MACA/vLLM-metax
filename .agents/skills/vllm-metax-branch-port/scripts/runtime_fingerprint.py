#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Compute a deterministic fingerprint for the approved branch-port environment."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import sys
from collections.abc import Sequence
from typing import Any


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment_details() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "torch": package_version("torch"),
        "vllm": package_version("vllm"),
        "vllm_metax": package_version("vllm-metax"),
        "use_precompiled_kernel": os.environ.get("USE_PRECOMPILED_KERNEL"),
        "vllm_target_device": os.environ.get("VLLM_TARGET_DEVICE"),
        "vllm_plugins": os.environ.get("VLLM_PLUGINS"),
    }


def compute_fingerprint() -> dict[str, Any]:
    details = environment_details()
    digest = hashlib.sha256(json.dumps(details, sort_keys=True).encode()).hexdigest()
    return {"fingerprint": digest, "details": details}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    result = compute_fingerprint()
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(result["fingerprint"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
