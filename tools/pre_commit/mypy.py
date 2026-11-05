# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run mypy on changed files.

This script is designed to be used as a pre-commit hook. It runs mypy
on changed files under vllm_metax/. Tests are excluded from type checking.

Usage:
    python tools/pre_commit/mypy.py <python_version> <changed_files...>

Args:
    python_version: Python version to use (e.g., "3.11") or "local" to use
        the local Python version.
    changed_files: List of changed files to check.

"""

import subprocess
import sys


def mypy(
    targets: list[str],
    python_version: str | None,
    follow_imports: str | None,
    file_group: str,
) -> int:
    """Run mypy on the given targets.

    Args:
        targets: List of files or directories to check.
        python_version: Python version to use (e.g., "3.11") or None to use
            the default mypy version.
        follow_imports: Value for the --follow-imports option or None to use
            the default mypy behavior.
        file_group: The file group name for logging purposes.

    Returns:
        The return code from mypy.

    """
    args = ["mypy"]
    if python_version is not None:
        args += ["--python-version", python_version]
    if follow_imports is not None:
        args += ["--follow-imports", follow_imports]
    print(f"$ {' '.join(args)} {file_group}")
    return subprocess.run(args + targets, check=False).returncode


def main():
    python_version = sys.argv[1]
    changed_files = [path for path in sys.argv[2:] if path.startswith("vllm_metax/")]

    if python_version == "local":
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}"

    if not changed_files:
        return 0
    return mypy(changed_files, python_version, None, "vllm_metax")


if __name__ == "__main__":
    sys.exit(main())
