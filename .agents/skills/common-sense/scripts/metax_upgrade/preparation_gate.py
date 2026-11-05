# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Connect shared run state to the environment-check skill's preparation records."""

from contextlib import contextmanager
import os
from pathlib import Path
import sys

sys.path.insert(
    0, str(Path(__file__).resolve().parents[3] / "environment-check/scripts")
)
from metax_environment import preparation  # noqa: E402


def uses_plan(run_dir, state):
    return (
        state["inputs"]["mode"] == "full"
        or state["inputs"].get("environment_protocol") == "preparation-plan-v1"
        or (Path(run_dir) / "PREPARATION.json").exists()
    )


def confirmation(run_dir, state, fingerprint):
    if uses_plan(run_dir, state):
        return preparation.require_ready(run_dir, state["inputs"], fingerprint)
    return None


def kernel_variables(run_dir):
    if not run_dir or not (Path(run_dir) / "PREPARATION.json").exists():
        return {}
    _, plan = preparation.load_plan(run_dir, approved=False)
    return preparation.kernel_variables(plan)


@contextmanager
def kernel_environment(run_dir):
    values = kernel_variables(run_dir)
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
