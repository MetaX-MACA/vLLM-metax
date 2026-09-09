# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Pure command/rank planning plus service startup and readiness checks."""

from collections import deque
from dataclasses import dataclass
import os
from pathlib import Path
import shlex
import time
import urllib.request

from tools.batched_test.runtime import RunContext
from tools.batched_test.specs import BenchmarkSpec, ModelSpec

CRITICAL_WORDS = ("EngineCore encountered an issue", "ioctl create queue block timeout")


@dataclass(frozen=True)
class RankSpec:
    # Backend node index used to route this rank to SSH or local launch.
    node_id: int
    # Complete argv for this rank, including rank index and distributed address.
    command: tuple[str, ...]
    # Normalized node/model environment including allocated GPU visibility.
    env: tuple[tuple[str, str], ...]
    # Absolute log path on the machine hosting this rank.
    log: str


@dataclass(frozen=True)
class DeploymentPlan:
    # HTTP service port exposed by local rank0.
    port: int
    # Ordered ranks; index zero is local, all remaining ranks are headless remotes.
    ranks: tuple[RankSpec, ...]


def serve_command(
    model: ModelSpec,
    port: int,
    *,
    rank: int = 0,
    nnodes: int | None = None,
    master_addr: str | None = None,
    master_port: int | None = None,
) -> list[str]:
    s = model.serve
    cmd = [
        "vllm",
        "serve",
        model.model_path,
        "--host",
        "localhost",
        "--port",
        str(port),
        "-tp",
        str(s.tp),
        "-pp",
        str(s.pp),
        "-dp",
        str(s.dp),
        "--trust-remote-code",
        "--gpu-memory-utilization",
        str(s.memory_utilization),
        "--max-model-len",
        str(s.max_model_len),
        "--distributed-executor-backend",
        "mp" if nnodes is not None else s.backend,
    ]
    if s.runner:
        cmd += ["--runner", s.runner]
    if nnodes is not None:
        cmd += [
            "--nnodes",
            str(nnodes),
            "--node-rank",
            str(rank),
            "--master-addr",
            str(master_addr),
            "--master-port",
            str(master_port),
        ]
        if rank:
            cmd.append("--headless")
    return cmd + list(s.extra_args)


def plan_deployment(context: RunContext) -> DeploymentPlan:
    port = context.port()
    lease = context.lease
    master_port = context.port(distributed=True) if lease.distributed else None
    ranks = []
    for rank, node in enumerate(lease.nodes):
        env = {
            **dict(node.env),
            **dict(context.model.env),
            "CUDA_VISIBLE_DEVICES": ",".join(map(str, node.gpu_ids)),
        }
        cmd = serve_command(
            context.model,
            port,
            rank=rank,
            nnodes=len(lease.nodes) if lease.distributed else None,
            master_addr=lease.nodes[0].hostname,
            master_port=master_port,
        )
        log = (
            str(context.work_dir / "serve.log")
            if rank == 0
            else f"/tmp/batched_test_{context.work_dir.name}_rank{rank}.log"
        )
        ranks.append(RankSpec(node.node_id, tuple(cmd), tuple(env.items()), log))
    return DeploymentPlan(port, tuple(ranks))


def launch_remotes(context: RunContext, plan: DeploymentPlan):
    for rank in plan.ranks[1:]:
        context.check()
        context.session.start_remote(
            lambda rank=rank: context.backend.start_remote(
                rank.node_id,
                list(rank.command),
                dict(rank.env),
                rank.log,
            )
        )


def log_command(log: str, command: list[str], env: dict):
    with open(log, "a", encoding="utf-8") as stream:
        stream.write(shlex.join(command) + "\n" + str(env) + "\n" + "-" * 80 + "\n")


def start_service(context: RunContext, plan: DeploymentPlan):
    rank0 = plan.ranks[0]
    log_command(rank0.log, list(rank0.command), dict(rank0.env))
    process = context.session.start(
        list(rank0.command),
        {**os.environ, **dict(rank0.env)},
        rank0.log,
    )
    launch_remotes(context, plan)
    return process


def wait_ready(context: RunContext, plan: DeploymentPlan, process):
    started = time.monotonic()
    cursor = 0
    log = Path(plan.ranks[0].log)
    try:
        while time.monotonic() - started < context.model.startup_timeout:
            context.check()
            code = process.poll()
            if code is not None:
                raise RuntimeError(f"vLLM serve exited with code {code}")
            if log.exists():
                with log.open(encoding="utf-8", errors="replace") as stream:
                    stream.seek(cursor)
                    content = stream.read()
                    cursor = stream.tell()
                for word in ("Traceback", *CRITICAL_WORDS):
                    if word in content:
                        raise RuntimeError(
                            f"Serve log contains {word}: {content[-500:]}"
                        )
            try:
                with urllib.request.urlopen(
                    f"http://localhost:{plan.port}/health",
                    timeout=min(5, context.request_timeout()),
                ) as response:
                    if response.status == 200:
                        return
            except (OSError, TimeoutError):
                pass
            context.cancel.wait(0.2)
        raise TimeoutError(
            f"Service did not start within {context.model.startup_timeout:g}s"
        )
    except Exception:
        if log.exists():
            with log.open(encoding="utf-8", errors="replace") as stream:
                print("".join(deque(stream, maxlen=50)))
        raise


def bench_command(model: ModelSpec, spec: BenchmarkSpec, port: int) -> list[str]:
    cmd = [
        "vllm",
        "bench",
        "serve",
        "--model",
        model.model_path,
        "--host",
        "localhost",
        "--port",
        str(port),
        "--dataset-name",
        spec.dataset,
        "--trust-remote-code",
        "--ready-check-timeout-sec",
        f"{spec.ready_timeout:g}",
    ]
    return cmd + (["--ignore-eos"] if spec.ignore_eos else [])


def sweep_command(
    serve: list[str], bench: list[str], spec: BenchmarkSpec, params: Path, output: Path
) -> list[str]:
    return [
        "vllm",
        "bench",
        "sweep",
        "serve",
        "--server-ready-timeout",
        f"{spec.startup_timeout:g}",
        "--serve-cmd",
        shlex.join(serve),
        "--bench-cmd",
        shlex.join(bench),
        "--bench-params",
        str(params),
        "--output-dir",
        str(output),
        "--num-runs",
        str(spec.runs),
        "--show-stdout",
    ]


def client_commands(base: list[str], spec: BenchmarkSpec) -> list[str]:
    commands = []
    for params in spec.parameter_dicts():
        command = list(base)
        for key, value in params.items():
            command += ["--" + key.replace("_", "-"), str(value)]
        commands.append(shlex.join(command))
    return commands
