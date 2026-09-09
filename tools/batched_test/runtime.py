# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""One owner for task allocation, cancellation, process cleanup and release."""

from dataclasses import dataclass, field
from pathlib import Path
import os
import signal
import subprocess
import threading
import time
from typing import Callable, Protocol

import psutil

from tools.batched_test import utils
from tools.batched_test.resources import ResourceBackend, ResourceLease
from tools.batched_test.results import Phase, Status, TaskOutcome, TaskResult
from tools.batched_test.specs import ModelSpec, RunPolicy


class Cancelled(RuntimeError):
    pass


class ProcessSession:
    def __init__(self, cancel: threading.Event, cleanup_timeout: float):
        # Per-task cancellation event shared with the runner and HTTP clients.
        self.cancel = cancel
        # Seconds allowed for each local process / descendant wait during stop().
        self.cleanup_timeout = cleanup_timeout
        # Serializes process registration, observation and cleanup across task/monitor threads.
        self._lock = threading.RLock()
        # Owned local Popen leaders, each launched in a new session.
        self._local: list[subprocess.Popen] = []
        # Stop callbacks for started remote ranks; failures remain registered for retry.
        self._remote: list[Callable[[], None]] = []
        # Observed descendant identities retained if their parent exits or they create another session.
        self._children: dict[int, psutil.Process] = {}

    def check(self):
        if self.cancel.is_set():
            raise Cancelled("Model run cancelled")

    def start(self, cmd: list[str], env: dict, log_file: str):
        with self._lock:
            self.check()
            process = utils.run_cmd(cmd=cmd, env=env, log_file=log_file)
            self._local.append(process)
            return process

    def start_remote(self, start: Callable[[], Callable[[], None]]):
        with self._lock:
            self.check()
            self._remote.append(start())

    def observe(self):
        # Retain descendant identities while sweep is alive: its server creates
        # another session and can outlive the sweep process on a crash.
        with self._lock:
            for process in self._local:
                try:
                    for child in psutil.Process(process.pid).children(recursive=True):
                        self._children[child.pid] = child
                except psutil.NoSuchProcess:
                    pass

    def wait(self, process):
        while True:
            self.check()
            self.observe()
            try:
                return process.wait(timeout=0.2)
            except subprocess.TimeoutExpired:
                continue

    def stop(self):
        with self._lock:
            errors = []
            try:
                self.observe()
            except Exception as exc:
                errors.append(exc)
            children = list(self._children.values())
            for child in reversed(children):
                try:
                    if not child.is_running():
                        continue
                    if os.getpgid(child.pid) == child.pid:
                        os.killpg(child.pid, signal.SIGKILL)
                    else:
                        child.kill()
                except (ProcessLookupError, psutil.NoSuchProcess):
                    pass
                except Exception as exc:
                    errors.append(exc)
            for process in self._local:
                try:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait(timeout=self.cleanup_timeout)
                except Exception as exc:
                    errors.append(exc)
            for stop in self._remote[:]:
                try:
                    stop()
                    self._remote.remove(stop)
                except Exception as exc:
                    errors.append(exc)
            _, alive = psutil.wait_procs(children, timeout=self.cleanup_timeout)
            for process in alive:
                try:
                    if (
                        process.is_running()
                        and process.status() != psutil.STATUS_ZOMBIE
                    ):
                        errors.append(RuntimeError(f"Child {process.pid} did not exit"))
                except psutil.NoSuchProcess:
                    pass
            if errors:
                raise RuntimeError("; ".join(str(exc) for exc in errors))
            self._local.clear()
            self._children.clear()


@dataclass
class RunContext:
    # Validated model settings for this execution.
    model: ModelSpec
    # Absolute per-task artifact directory, including a configuration fingerprint.
    work_dir: Path
    # Allocation, execution, request and cleanup timeout limits.
    policy: RunPolicy
    # Per-task event set by batch cancellation or the deadline monitor.
    cancel: threading.Event
    # Owner of all local processes and remote stop callbacks for this execution.
    session: ProcessSession
    # Resource reservation held until process cleanup succeeds.
    lease: ResourceLease
    # Injected local/cluster adapter used to start remote ranks.
    backend: ResourceBackend
    # Batch-shared port reservation manager; not owned exclusively by this context.
    ports: utils.PortManager
    # Monotonic timestamp recorded immediately after resource allocation.
    started_at: float
    # Current execution phase updated by the task for failure reporting.
    phase: Phase = Phase.STARTING
    # Ports reserved by this execution; released by TaskRunner after cleanup.
    _ports: list[int] = field(default_factory=list)

    def check(self):
        self.session.check()
        if time.monotonic() - self.started_at >= self.policy.execution_timeout:
            raise TimeoutError("Exceeded per-model execution timeout")

    def port(self, distributed=False):
        self.check()
        port = self.ports.get_next_available_port(
            start_port=29500 if distributed else 8000,
            max_port=29650 if distributed else 9000,
        )
        self._ports.append(port)
        return port

    def request_timeout(self):
        self.check()
        return min(
            self.policy.request_timeout,
            max(
                0.01,
                self.policy.execution_timeout - (time.monotonic() - self.started_at),
            ),
        )


class Task(Protocol):
    # Model settings needed to calculate the resource request.
    model: ModelSpec
    # Task category used by scheduling and reporting.
    kind: str
    # Stable filesystem-safe task directory name including the fingerprint.
    artifact_id: str

    def execute(self, context: RunContext) -> TaskOutcome: ...


class TaskRunner:
    def __init__(self, backend: ResourceBackend, ports: utils.PortManager):
        # Batch-shared resource allocator and remote-launch adapter.
        self.backend = backend
        # Batch-shared port manager injected into every run context.
        self.ports = ports

    def run(
        self, task: Task, work_dir: Path, policy: RunPolicy, cancel: threading.Event
    ) -> TaskResult:
        lease = None
        context = None
        session = ProcessSession(cancel, policy.cleanup_timeout)
        monitor_done = threading.Event()
        timed_out = threading.Event()
        monitor = None
        monitor_errors = []
        result = TaskResult(Status.ERROR, Phase.ALLOCATING, 0)
        start = time.monotonic()

        def watch():
            while not monitor_done.wait(0.1):
                if time.monotonic() - context.started_at >= policy.execution_timeout:
                    timed_out.set()
                    cancel.set()
                if cancel.is_set():
                    try:
                        session.stop()
                    except Exception:
                        # Cleanup below retries and records failures before release.
                        pass
                    return
                try:
                    session.observe()
                except Exception as exc:
                    monitor_errors.append(str(exc))
                    cancel.set()

        try:
            while lease is None:
                session.check()
                if time.monotonic() - start >= policy.allocation_timeout:
                    raise TimeoutError("GPU allocation timed out")
                lease = self.backend.allocate(task.model.serve.required_gpus)
                if lease is None:
                    cancel.wait(policy.allocation_poll)
            context = RunContext(
                task.model,
                work_dir / task.artifact_id,
                policy,
                cancel,
                session,
                lease,
                self.backend,
                self.ports,
                time.monotonic(),
            )
            session.check()
            context.work_dir.mkdir(parents=True, exist_ok=True)
            monitor = threading.Thread(target=watch, daemon=True)
            monitor.start()
            result.outcome = task.execute(context)
            context.check()
            result.status = result.outcome.status
            result.phase = Phase.COMPLETE
        except Cancelled as exc:
            result.status = Status.CANCELLED
            result.error = str(exc)
        except TimeoutError as exc:
            result.status = Status.TIMEOUT
            result.error = str(exc)
        except Exception as exc:
            result.status = Status.ERROR
            result.error = f"{type(exc).__name__}: {exc}"
        finally:
            if result.status not in (Status.PASSED, Status.FAILED):
                result.phase = context.phase if context else Phase.ALLOCATING
            monitor_done.set()
            if monitor is not None:
                monitor.join()
            if context is not None:
                result.outcome.artifacts.setdefault(
                    "artifact_dir", str(context.work_dir)
                )
                result.outcome.artifacts.setdefault("log_dir", str(context.work_dir))
            if monitor_errors:
                result.status = Status.ERROR
                result.error = "Process monitoring failed: " + "; ".join(monitor_errors)
            if timed_out.is_set():
                result.status = Status.TIMEOUT
                result.error = (
                    f"Exceeded per-model timeout ({policy.execution_timeout:g}s)"
                )
            try:
                session.stop()
                if lease is not None:
                    lease.release()
                if context is not None:
                    for port in context._ports:
                        self.ports.release_port(port)
            except Exception as exc:
                result.cleanup_error = f"Cleanup failed; retaining resources: {exc}"
                if result.status != Status.TIMEOUT:
                    result.status = Status.ERROR
                    result.phase = Phase.CLEANUP
            result.elapsed = time.monotonic() - start
        return result
