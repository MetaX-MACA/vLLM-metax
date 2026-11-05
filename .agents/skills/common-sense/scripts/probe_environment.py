# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Compatibility wrapper for the independent agent environment/source probe."""

import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-source", type=Path, required=True)
    parser.add_argument("--metax-source", type=Path, required=True)
    parser.add_argument("--runtime-cwd", type=Path, default=Path.cwd())
    args = parser.parse_args()
    metax_source = args.metax_source.resolve()
    environment_tool = Path(__file__).resolve().parent / "metax_upgrade/environment.py"
    if not environment_tool.is_file():
        parser.error(
            f"Independent upgrade environment tool not found: {environment_tool}"
        )
    # NOTE(metax): Source inventory must not import unfinished MetaX domain code.
    command = [
        sys.executable,
        "-B",
        str(environment_tool),
        "--source-probe",
        "--vllm-source",
        str(args.vllm_source.resolve()),
        "--metax-source",
        str(metax_source),
        "--runtime-cwd",
        str(args.runtime_cwd.resolve()),
    ]
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
