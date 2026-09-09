# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Validated task inputs. YAML compatibility and defaults end at this boundary."""

from dataclasses import dataclass
import json
import math
from pathlib import Path
import shlex
from typing import Any
from tools.batched_test.paths import (
    TOOL_ROOT,
    normalize_config,
    option_value,
    resolve_path,
)


def positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a positive integer")
    try:
        result = int(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{field} must be a positive integer") from exc
    if result <= 0 or str(value).strip() != str(result):
        raise ValueError(f"{field} must be a positive integer")
    return result


def positive_seconds(value: Any, field: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{field} must be finite and positive")
    return result


def cli_args(raw: Any) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        return tuple(shlex.split(raw))
    if isinstance(raw, dict):
        result = []
        for key, value in raw.items():
            result.append(str(key))
            if value is not None:
                result.append(str(value))
        return tuple(result)
    if isinstance(raw, list):
        return tuple(str(item) for item in raw)
    raise ValueError("extra_args must be a mapping, list or command-line string")


@dataclass(frozen=True)
class ServeSpec:
    # Tensor-parallel ranks per model replica.
    tp: int = 1
    # Pipeline-parallel stages per replica.
    pp: int = 1
    # Data-parallel replicas; total GPUs = tp * pp * dp.
    dp: int = 1
    # Executor backend for local runs; multi-node deployment forces mp.
    backend: str = "mp"
    # Fraction of device memory vLLM may reserve, in (0, 1].
    memory_utilization: float = 0.8
    # Maximum context length in tokens.
    max_model_len: int = 4096
    # Optional vLLM runner override, e.g. pooling for embeddings.
    runner: str | None = None
    # Normalized argv tokens appended to the serving command.
    extra_args: tuple[str, ...] = ()

    @property
    def required_gpus(self) -> int:
        return self.tp * self.pp * self.dp

    @classmethod
    def parse(cls, raw: dict | None):
        raw = raw or {}
        memory = float(raw.get("gpu_memory_utilization", 0.8))
        if not 0 < memory <= 1:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
        return cls(
            **{key: positive_int(raw.get(key, 1), key) for key in ("tp", "pp", "dp")},
            backend=str(raw.get("distributed_executor_backend", "mp")),
            memory_utilization=memory,
            max_model_len=positive_int(raw.get("max_model_len", 4096), "max_model_len"),
            runner="pooling" if raw.get("task") == "embed" else None,
            extra_args=cli_args(raw.get("extra_args")),
        )


@dataclass(frozen=True)
class ModelSpec:
    # Human-readable model name used in report and artifact identifiers.
    name: str
    # Resolved local model path, Hub repository ID, or remote URI.
    model_path: str
    # Validated serving and parallelism configuration.
    serve: ServeSpec
    # Immutable model environment overrides; allocation controls device visibility.
    env: tuple[tuple[str, str], ...] = ()
    # Inference service readiness limit in seconds after launch.
    startup_timeout: float = 1200
    # Supported suite names to execute in declaration order.
    infer_types: tuple[str, ...] = ()

    @classmethod
    def parse(cls, raw: dict, *, base_dir: Path = TOOL_ROOT):
        raw = normalize_config(raw, base_dir)
        if not raw.get("name") or not raw.get("model_path"):
            raise ValueError("Each model needs name and model_path")
        env = raw.get("extra_env") or {}
        if isinstance(env, list):
            merged = {}
            for item in env:
                if not isinstance(item, dict):
                    raise ValueError("extra_env list entries must be mappings")
                merged.update(item)
            env = merged
        if not isinstance(env, dict):
            raise ValueError("extra_env must be a mapping or list of mappings")
        modes = raw.get("infer_type") or []
        if isinstance(modes, str):
            modes = [modes]
        if not isinstance(modes, list) or not all(
            isinstance(mode, str) for mode in modes
        ):
            raise ValueError("infer_type must be a string or list of strings")
        return cls(
            name=str(raw["name"]),
            model_path=str(raw["model_path"]),
            serve=ServeSpec.parse(raw.get("serve_config")),
            env=tuple(sorted((str(k), str(v)) for k, v in env.items())),
            startup_timeout=positive_seconds(raw.get("timeout", 1200), "timeout"),
            infer_types=tuple(dict.fromkeys(modes)),
        )

    def tag(self, kind: str) -> str:
        s = self.serve
        if kind == "inference":
            return f"{self.name}[tp{s.tp}pp{s.pp}dp{s.dp}]"
        return f"{self.name}_tp{s.tp}_pp{s.pp}_dp{s.dp}"


@dataclass(frozen=True)
class RunPolicy:
    # Seconds allowed to wait for GPUs/nodes after leaving the executor queue.
    allocation_timeout: float = 28800
    # Seconds allowed for task execution after resource allocation.
    execution_timeout: float = 3600
    # Maximum HTTP request duration in seconds, capped by the execution deadline.
    request_timeout: float = 600
    # Seconds per local process wait / descendant wait group during cleanup.
    cleanup_timeout: float = 10
    # Seconds between unsuccessful resource allocation attempts.
    allocation_poll: float = 1

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            positive_seconds(getattr(self, name), name)


@dataclass(frozen=True)
class CaseSpec:
    # Text question, local image path/URL, or embedding query, depending on suite.
    input: str
    # Case-insensitive alternatives; any match passes a text/image case.
    keywords: tuple[str, ...] = ()
    # Maximum generated completion tokens; unused for embedding cases.
    max_tokens: int = 256
    # Texts expected to be more similar to the embedding query.
    positive: tuple[str, ...] = ()
    # Texts expected to be less similar to the embedding query.
    negative: tuple[str, ...] = ()


@dataclass(frozen=True)
class SuiteSpec:
    # Suite selector: text-only, single-image, or embedding.
    kind: str
    # Validated cases in execution order; must be nonempty.
    cases: tuple[CaseSpec, ...]


@dataclass(frozen=True)
class InferenceSpec:
    # Ordered suites to run against the same model service.
    suites: tuple[SuiteSpec, ...]
    # Minimum fraction of passing cases, in (0, 1]; CLI currently uses 1.0.
    pass_threshold: float = 1.0

    def __post_init__(self):
        if not self.suites or not all(suite.cases for suite in self.suites):
            raise ValueError("Inference needs at least one nonempty suite")
        if not 0 < self.pass_threshold <= 1:
            raise ValueError("pass_threshold must be in (0, 1]")


@dataclass(frozen=True)
class BenchmarkSpec:
    # Freeze normalized parameters rather than rereading a mutable file at run time.
    # Frozen parameter rows: each (key, JSON value) preserves the original value type.
    parameters: tuple[tuple[tuple[str, str], ...], ...]
    # vLLM benchmark dataset selector, e.g. random.
    dataset: str = "random"
    # Whether generation should continue past EOS during benchmarking.
    ignore_eos: bool = False
    # Number of sweep repetitions per parameter combination.
    runs: int = 3
    # Seconds sweep may wait for its rank0 server to become ready.
    startup_timeout: float = 3600
    # Seconds the benchmark client may wait for service readiness.
    ready_timeout: float = 6000

    @classmethod
    def parse(cls, raw: dict | None, *, base_dir: Path = TOOL_ROOT):
        raw = raw or {}
        path = raw.get("bench_param")
        if not path:
            raise ValueError("benchmark.bench_param is required for performance tests")
        source = Path(resolve_path(path, base_dir))
        data = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(data, list) or not data:
            raise ValueError("Benchmark parameters must be a nonempty list")
        for row in data:
            if not isinstance(row, dict):
                raise ValueError("Benchmark parameters must be mappings")
            for key, value in row.items():
                if isinstance(value, str):
                    row[key] = option_value(
                        "--" + key.replace("_", "-"), value, source.parent
                    )
        return cls(
            parameters=tuple(
                tuple((str(k), json.dumps(v)) for k, v in row.items()) for row in data
            ),
            dataset=str(raw.get("dataset_name", "random")),
            ignore_eos=bool(raw.get("ignore_eos", False)),
            runs=positive_int(raw.get("sweep_num_runs", 3), "sweep_num_runs"),
            startup_timeout=positive_seconds(
                raw.get("startup_timeout", 3600), "startup_timeout"
            ),
            ready_timeout=positive_seconds(
                raw.get("ready_timeout", 6000), "ready_timeout"
            ),
        )

    def parameter_dicts(self) -> list[dict]:
        return [{k: json.loads(v) for k, v in row} for row in self.parameters]
