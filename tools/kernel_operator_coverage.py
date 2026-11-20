# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Report dispatcher operators observed in passing MetaX kernel tests."""

import argparse
import ast
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
NATIVE_NAMESPACES = {"_C", "_C_cache_ops", "_C_cuda_utils", "_moe_C"}
DISPATCH_KEYS = (
    "CPU",
    "CUDA",
    "CompositeExplicitAutograd",
    "CompositeImplicitAutograd",
)
NATIVE_MODULES = ("vllm_metax._C_stable_libtorch", "vllm_metax._moe_C_stable_libtorch")


def python_registrations() -> dict[str, str]:
    registrations = {}
    for path in sorted((ROOT / "vllm_metax").rglob("*.py")):
        for call in ast.walk(ast.parse(path.read_bytes(), filename=str(path))):
            if not (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == "direct_register_custom_op"
            ):
                continue
            name = next(
                (kw.value for kw in call.keywords if kw.arg == "op_name"),
                call.args[0] if call.args else None,
            )
            if isinstance(name, ast.Constant) and isinstance(name.value, str):
                registrations[f"vllm::{name.value}"] = str(path.relative_to(ROOT))
    return registrations


class OperatorProfile:
    def __init__(self, operators: set[str]) -> None:
        self.operators = operators
        self.current = set[str]()
        self.passing_tests: dict[str, set[str]] = defaultdict(set)
        self.expected_failures: dict[str, set[str]] = defaultdict(set)

    @pytest.hookimpl(wrapper=True)
    def pytest_runtest_call(self, item: pytest.Item):
        self.current.clear()
        profile = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
        )
        try:
            with profile:
                return (yield)
        finally:
            self.current.update(
                event.name for event in profile.events() if event.name in self.operators
            )

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        if report.when != "call":
            return
        if report.passed:
            observations = self.passing_tests
            test = report.nodeid.split("[", 1)[0]
        elif report.skipped and hasattr(report, "wasxfail"):
            observations = self.expected_failures
            test = report.nodeid
        else:
            return
        for name in self.current:
            observations[name].add(test)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if os.environ.setdefault("USE_PRECOMPILED_KERNEL", "0") != "0":
        parser.error("the coverage report requires USE_PRECOMPILED_KERNEL=0")

    baseline = set(torch._C._dispatch_get_all_op_names())
    from vllm_metax.platform import MacaPlatform

    MacaPlatform.import_kernels()
    native = {
        name
        for name in set(torch._C._dispatch_get_all_op_names()) - baseline
        if name.partition("::")[0] in NATIVE_NAMESPACES
    }
    python = python_registrations()
    profile = OperatorProfile(native | python.keys())
    pytest_args = args.pytest_args
    if pytest_args[:1] == ["--"]:
        pytest_args = pytest_args[1:]
    exit_code = pytest.main(pytest_args or ["-q", "tests/kernels"], plugins=[profile])
    if any(name == "mcoplib" or name.startswith("mcoplib.") for name in sys.modules):
        parser.exit(1, "mcoplib loaded during source-mode operator coverage\n")

    loaded = set(torch._C._dispatch_get_all_op_names())
    operators = {}
    for name in sorted(profile.operators):
        keys = [
            key
            for key in DISPATCH_KEYS
            if name in loaded
            and torch._C._dispatch_has_kernel_for_dispatch_key(name, key)
        ]
        tests = sorted(profile.passing_tests.get(name, ()))
        expected_failures = sorted(profile.expected_failures.get(name, ()))
        if tests:
            status = "observed"
        elif expected_failures:
            status = "expected_failure"
        elif name not in loaded:
            status = "not_loaded"
        elif keys:
            status = "unobserved"
        else:
            status = "schema_only"
        operators[name] = {
            "kind": "native" if name in native else "python",
            "file": python.get(name),
            "dispatch_keys": keys,
            "status": status,
            "tests": tests,
            "expected_failures": expected_failures,
        }
    report = {
        "provider": "source",
        "pytest_exit_code": int(exit_code),
        "native_extensions": {
            name: sys.modules[name].__file__ for name in NATIVE_MODULES
        },
        "operators": operators,
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    return int(exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
