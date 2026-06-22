# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _run_version(command: list[str]) -> str | None:
    try:
        output = subprocess.check_output(
            command, stderr=subprocess.STDOUT, text=True, timeout=10
        )
        lines = output.splitlines()
        return lines[0] if lines else None
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def collect_preflight() -> dict[str, object]:
    maca_path = os.environ.get("MACA_PATH") or os.environ.get("MACA_HOME")
    cucc_path = os.environ.get("CUCC_PATH")
    if not cucc_path and maca_path:
        cucc_path = str(Path(maca_path) / "tools" / "cu-bridge")

    version_file = Path(maca_path) / "Version.txt" if maca_path else None
    cuda_path = os.environ.get("CUDA_PATH")
    if not cuda_path and cucc_path:
        cuda_path_candidate = Path(cucc_path) / "CUDA_DIR"
        cuda_path = str(cuda_path_candidate if cuda_path_candidate.exists() else Path(cucc_path))

    commands = {
        "cmake_maca": shutil.which("cmake_maca"),
        "cmake": shutil.which("cmake"),
        "mxcc": shutil.which("mxcc"),
        "cucc": _first_existing(
            [
                Path(cucc_path) / "bin" / "cucc",
                Path(cucc_path) / "tools" / "cucc",
            ]
        )
        if cucc_path
        else None,
        "nvcc": _first_existing([Path(cuda_path) / "bin" / "nvcc"]) if cuda_path else None,
    }
    normalized_commands = {
        name: str(value) if value else None for name, value in commands.items()
    }
    missing = [
        name
        for name in ("cmake_maca", "mxcc", "cucc", "nvcc")
        if normalized_commands.get(name) is None
    ]

    maca_version = None
    if version_file and version_file.is_file():
        try:
            maca_version = version_file.read_text(encoding="utf-8").strip()
        except OSError:
            maca_version = None

    info: dict[str, object] = {
        "python": sys.version.replace("\n", " "),
        "maca_path": maca_path,
        "cucc_path": cucc_path,
        "cuda_path": cuda_path,
        "maca_version": maca_version,
        "commands": normalized_commands,
        "command_versions": {
            name: _run_version([path, "--version"])
            for name, path in normalized_commands.items()
            if path
        },
        "missing": missing,
        "ok": not missing and bool(maca_path),
    }
    try:
        import torch

        info["torch"] = {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
        }
    except Exception as err:
        info["torch"] = {"error": str(err)}
    return info


def main() -> int:
    info = collect_preflight()
    print(json.dumps(info, indent=2, sort_keys=True))
    return 0 if info["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
