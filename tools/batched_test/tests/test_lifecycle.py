# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""CPU-only contract tests for the batched test runtime and task implementations."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import csv
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time
from types import SimpleNamespace

import psutil
import pytest

from tools.batched_test import deployment, suites
from tools.batched_test.api_client import ApiConnection, ChatCompletionClient, EmbeddingClient
from tools.batched_test.launch import Scheduler, SchedulerArgs
from tools.batched_test.model_worker import InferenceTask, BenchmarkTask
from tools.batched_test.reporting import CsvReport, ResumeIndex, inference_row
from tools.batched_test.resources import LocalBackend, ClusterBackend, ResourceLease, NodeAllocation
from tools.batched_test.results import CaseResult, Phase, Status, TaskOutcome, TaskResult
from tools.batched_test.runtime import Cancelled, ProcessSession, TaskRunner
from tools.batched_test.specs import ModelSpec, ServeSpec, InferenceSpec, SuiteSpec, CaseSpec, BenchmarkSpec, RunPolicy
from tools.batched_test.utils import PortManager


class Resources:
    def __init__(self):
        self.released = []
        self.occupied = False
        self.lock = threading.Lock()

    def allocate(self, count):
        with self.lock:
            if self.occupied:
                return []
            self.occupied = True
            return [0]

    def release(self, ids):
        with self.lock:
            self.released.append(list(ids))
            self.occupied = False


@pytest.fixture
def runtime(monkeypatch):
    # Reservation concurrency is tested separately from host socket availability.
    monkeypatch.setattr(PortManager, 'is_port_available', lambda self, port: True)
    manager = Resources()
    ports = PortManager()
    return TaskRunner(LocalBackend(manager), ports), manager, ports


def model(**kwargs):
    return ModelSpec('mock', 'mock-path', ServeSpec(), **kwargs)


def task():
    return InferenceTask(model(infer_types=('text-only',)), InferenceSpec((
        SuiteSpec('text-only', (CaseSpec('question', ('answer',)),)),
    )))


class Job:
    model = model()
    kind = 'inference'
    artifact_id = 'mock'

    def __init__(self, action):
        self.action = action

    def execute(self, context):
        return self.action(context)


def policy(**kwargs):
    return RunPolicy(allocation_poll=0.01, **kwargs)


def test_task_construction_has_no_io_or_allocations(monkeypatch):
    monkeypatch.setattr(PortManager, 'get_next_available_port', lambda *a, **kw: pytest.fail('eager port'))
    task()


def test_timeout_isolated_and_cleanup_precedes_result(runtime, tmp_path):
    runner, manager, ports = runtime
    def hung(context):
        context.port()
        assert context.cancel.wait(3)
        context.check()
    stop = threading.Event()
    result = runner.run(Job(hung), tmp_path, policy(execution_timeout=0.1), stop)
    assert result.status == Status.TIMEOUT
    assert manager.released == [[0]]
    assert not ports.occupied_ports
    healthy = threading.Event()
    result = runner.run(Job(lambda ctx: TaskOutcome()), tmp_path, policy(), healthy)
    assert result.status == Status.PASSED
    assert not healthy.is_set()


def test_allocation_wait_excluded_from_execution_timeout(runtime, tmp_path):
    runner, manager, _ = runtime
    started = time.monotonic()
    allocate = manager.allocate
    manager.allocate = lambda count: [] if time.monotonic() - started < 0.25 else allocate(count)
    result = runner.run(Job(lambda ctx: TaskOutcome()), tmp_path,
                        policy(execution_timeout=0.1), threading.Event())
    assert result.status == Status.PASSED
    assert result.elapsed >= 0.25


def test_allocation_wait_cancels_without_launch(runtime, tmp_path):
    runner, manager, ports = runtime
    allocating = threading.Event()
    def allocate(count):
        allocating.set()
        return []
    manager.allocate = allocate
    cancel = threading.Event()
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(runner.run, Job(lambda ctx: pytest.fail('executed')),
                             tmp_path, policy(), cancel)
        assert allocating.wait(3)
        cancel.set()
        result = future.result(timeout=3)
    assert result.status == Status.CANCELLED
    assert result.phase == Phase.ALLOCATING
    assert not ports.occupied_ports


def scheduler(runner):
    obj = Scheduler.__new__(Scheduler)
    obj._runner = runner
    obj.max_workers = 1
    return obj


def test_scheduler_queue_time_excluded(runtime, tmp_path):
    runner, _, _ = runtime
    def work(context):
        time.sleep(0.04)
        return TaskOutcome()
    results = list(scheduler(runner)._run_tasks([Job(work) for _ in range(8)], tmp_path,
                                               policy(execution_timeout=0.2)))
    assert all(result.status == Status.PASSED for _, result in results)


def test_scheduler_closing_stops_running_and_queued(runtime, tmp_path):
    runner, manager, ports = runtime
    def hung(context):
        assert context.cancel.wait(3)
        context.check()
    jobs = [Job(lambda ctx: TaskOutcome()), Job(hung), Job(hung)]
    runs = scheduler(runner)._run_tasks(jobs, tmp_path, policy())
    next(runs)
    runs.close()
    assert not manager.occupied
    assert not ports.occupied_ports


@pytest.mark.parametrize('passed, expected', [(False, Status.FAILED), (True, Status.PASSED)])
def test_inference_task_accuracy(runtime, tmp_path, monkeypatch, passed, expected):
    runner, _, _ = runtime
    monkeypatch.setattr(deployment, 'start_service', lambda *args: None)
    monkeypatch.setattr(deployment, 'wait_ready', lambda *args: None)
    monkeypatch.setattr('tools.batched_test.model_worker.run_suites', lambda *args: (
        CaseResult('text-only', 0, passed, 'response'),))
    result = runner.run(task(), tmp_path, policy(), threading.Event())
    assert result.status == expected
    row = inference_row(task(), result)
    assert row['Stage'] == ('NORMAL_END' if passed else 'ACCURACY_FAILED')


@pytest.mark.parametrize('ratio, skip', [('0%', False), ('99%', False), ('100%', True), ('invalid', False)])
def test_resume_legacy_accuracy(tmp_path, ratio, skip):
    path = tmp_path / 'results.csv'
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['Model', 'Stage', 'Correct Ratio', 'Model Path'])
        writer.writeheader()
        writer.writerow({'Model': task().model.tag('inference'), 'Stage': 'NORMAL_END',
                         'Correct Ratio': ratio, 'Model Path': task().model.model_path})
    assert ResumeIndex(str(path)).should_skip(task()) is skip


def test_resume_detects_configuration_and_case_changes(tmp_path):
    original = task()
    result = TaskResult(Status.PASSED, Phase.COMPLETE, 1,
                        TaskOutcome(cases=(CaseResult('text-only', 0, True, 'answer'),)))
    path = tmp_path / 'results.csv'
    with CsvReport(path, [original], 'inference') as report:
        report.write(original, result)
    resume = ResumeIndex(str(path))
    assert resume.should_skip(original)
    assert not resume.should_skip(replace(original, model=replace(original.model, env=(('FLAG', '1'),))))
    assert not resume.should_skip(replace(original, spec=InferenceSpec((
        SuiteSpec('text-only', (CaseSpec('different question', ('answer',)),)),))))


def test_commands_do_not_accumulate():
    spec = BenchmarkSpec(((('max_concurrency', '1'),), (('max_concurrency', '2'),)))
    base = ['vllm', 'bench', 'serve']
    commands = deployment.client_commands(base, spec)
    assert base == ['vllm', 'bench', 'serve']
    assert all(cmd.count('--max-concurrency') == 1 for cmd in commands)
    assert commands[1].endswith('--max-concurrency 2')


def test_port_concurrency_and_independent_managers(monkeypatch):
    def available(self, port):
        time.sleep(0.001)
        return True
    monkeypatch.setattr(PortManager, 'is_port_available', available)
    manager = PortManager()
    with ThreadPoolExecutor(16) as pool:
        ports = list(pool.map(lambda _: manager.get_next_available_port(), range(64)))
    assert len(set(ports)) == 64
    assert not PortManager().occupied_ports
    for port in ports:
        manager.release_port(port)
    assert not manager.occupied_ports


def test_cancel_before_process_launch(tmp_path):
    cancel = threading.Event()
    session = ProcessSession(cancel, 1)
    cancel.set()
    with pytest.raises(Cancelled):
        session.start([sys.executable, '-c', 'pass'], os.environ.copy(), str(tmp_path / 'log'))


def test_cleanup_kills_separate_session_child_before_release(runtime, tmp_path):
    runner, manager, _ = runtime
    pid_file = tmp_path / 'child.pid'
    child_pid = None
    parent = None
    def work(context):
        nonlocal child_pid, parent
        code = ('import subprocess,sys,time; from pathlib import Path; '
                'p=subprocess.Popen([sys.executable,"-c","import time;time.sleep(60)"], start_new_session=True); '
                f'Path({str(pid_file)!r}).write_text(str(p.pid)); time.sleep(60)')
        parent = context.session.start([sys.executable, '-c', code], os.environ.copy(),
                                       str(context.work_dir / 'log'))
        deadline = time.monotonic() + 5
        while not pid_file.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        child_pid = int(pid_file.read_text())
        return TaskOutcome()
    release = manager.release
    def checked_release(ids):
        assert parent.poll() is not None
        assert not psutil.pid_exists(child_pid) or psutil.Process(child_pid).status() == psutil.STATUS_ZOMBIE
        release(ids)
    manager.release = checked_release
    try:
        result = runner.run(Job(work), tmp_path, policy(cleanup_timeout=0.1), threading.Event())
        assert result.status == Status.PASSED
        assert manager.released == [[0]]
    finally:
        for pid in (parent.pid if parent else None, child_pid):
            if pid:
                try:
                    os.killpg(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        if parent:
            parent.wait()


def test_partial_startup_and_remote_cleanup_failure_retain_resources(runtime, tmp_path):
    runner, manager, ports = runtime
    stopped = []
    def work(context):
        context.port()
        def failed_stop():
            stopped.append('first')
            raise RuntimeError('SSH unavailable')
        context.session.start_remote(lambda: failed_stop)
        context.session.start_remote(lambda: lambda: stopped.append('second'))
        raise RuntimeError('rank startup failed')
    result = runner.run(Job(work), tmp_path, policy(), threading.Event())
    assert result.status == Status.ERROR
    assert 'SSH unavailable' in result.cleanup_error
    assert stopped == ['first', 'second']
    assert manager.occupied and not manager.released
    assert ports.occupied_ports


def test_benchmark_timeout_cleans_sweep(runtime, tmp_path, monkeypatch):
    runner, manager, ports = runtime
    spec = BenchmarkSpec(((('max_concurrency', '1'),),))
    monkeypatch.setattr(deployment, 'sweep_command', lambda *args: [
        sys.executable, '-c', 'import time;time.sleep(60)'])
    result = runner.run(BenchmarkTask(model(), spec), tmp_path,
                        policy(execution_timeout=0.1), threading.Event())
    assert result.status == Status.TIMEOUT
    assert manager.released == [[0]] and not ports.occupied_ports


def test_cancelled_client_does_not_send_next_request():
    connection = object.__new__(ApiConnection)
    connection.stop_event = threading.Event()
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='ok'))])
    connection.request_client = lambda: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    client = ChatCompletionClient(connection=connection)
    responses = client.run_text_only(['one', 'two'], model='mock')
    assert next(responses) == 'ok'
    connection.stop_event.set()
    with pytest.raises(RuntimeError, match='cancelled'):
        next(responses)
    assert len(calls) == 1
    assert not issubclass(EmbeddingClient, ChatCompletionClient)


def test_multi_suite_planning_and_empty_case_validation(tmp_path):
    text = tmp_path / 'text.yaml'
    image = tmp_path / 'image.yaml'
    text.write_text('- question: q\n  keywords: [answer]\n')
    image.write_text('- picture_url: local.png\n  keywords: [cat]\n')
    spec = suites.plan_inference(model(infer_types=('text-only', 'single-image')),
                                 text_case=str(text), image_case=str(image))
    assert [suite.kind for suite in spec.suites] == ['text-only', 'single-image']
    text.write_text('[]')
    with pytest.raises(ValueError, match='nonempty'):
        suites.plan_inference(model(infer_types=('text-only',)), text_case=str(text), image_case=None)


@pytest.mark.parametrize('value', [0, -1, 1.5, True, 'abc'])
def test_invalid_parallelism_rejected(value):
    with pytest.raises(ValueError, match='positive integer'):
        ModelSpec.parse({'name': 'mock', 'model_path': 'mock', 'serve_config': {'tp': value}})


def test_legacy_string_extra_args_normalized():
    spec = ServeSpec.parse({'extra_args': '--enable-expert-parallel --name "two words"'})
    assert spec.extra_args == ('--enable-expert-parallel', '--name', 'two words')


def test_cluster_allocation_has_explicit_node_gpu_mapping():
    manager = SimpleNamespace(
        allocate=lambda count: [0, 1], gpu_per_node=8, release=lambda ids: None,
        get_node_hostname=lambda i: f'node{i}', get_base_env=lambda i: {'NIC': f'eth{i}'},
    )
    lease = ClusterBackend(manager).allocate(16)
    assert lease.distributed
    assert [node.hostname for node in lease.nodes] == ['node0', 'node1']
    assert all(node.gpu_ids == tuple(range(8)) for node in lease.nodes)


def test_dry_run_does_not_initialize_gpu_or_ssh(tmp_path):
    config = tmp_path / 'models.yaml'
    config.write_text('- name: mock\n  model_path: mock\n')
    obj = Scheduler(SchedulerArgs(work_dir=str(tmp_path), model_config=str(config),
                                  text_case='missing', image_case='missing', dry_run=True))
    obj.record_environment = lambda: pytest.fail('environment collection during dry run')
    obj.run_all()
    assert obj._runner is None


def test_mixed_suites_execute_all_and_close_connection(runtime, tmp_path, monkeypatch):
    runner, _, _ = runtime
    calls = []
    class Connection:
        def __init__(self, **kwargs):
            pass
        def close(self):
            calls.append('closed')
    class Chat:
        def __init__(self, **kwargs):
            pass
        def run_text_only(self, *args, **kwargs):
            calls.append('text')
            yield 'answer'
        def run_single_image(self, *args, **kwargs):
            calls.append('image')
            yield 'cat'
    class Embedding:
        def __init__(self, **kwargs):
            pass
        def embed(self, texts):
            calls.append('embedding')
            return [[1, 0], [1, 0], [0, 1]]
    monkeypatch.setattr(suites, 'ApiConnection', Connection)
    monkeypatch.setattr(suites, 'ChatCompletionClient', Chat)
    monkeypatch.setattr(suites, 'EmbeddingClient', Embedding)
    spec = InferenceSpec((
        SuiteSpec('text-only', (CaseSpec('q', ('answer',)),)),
        SuiteSpec('single-image', (CaseSpec('image', ('cat',)),)),
        SuiteSpec('embedding', (CaseSpec('query', positive=('positive',), negative=('negative',)),)),
    ))
    def work(context):
        cases = suites.run_suites(context, spec, 8000)
        assert len(cases) == 3 and all(case.passed for case in cases)
        return TaskOutcome(cases=cases)
    result = runner.run(Job(work), tmp_path, policy(), threading.Event())
    assert result.status == Status.PASSED
    assert calls == ['text', 'image', 'embedding', 'closed']


def test_deployment_plan_distributed_commands_and_gpu_env(runtime, tmp_path):
    runner, _, _ = runtime
    runner.backend = SimpleNamespace(allocate=lambda count: ResourceLease((
        NodeAllocation(0, 'host-a', tuple(range(8)), (('NIC', 'eth0'),)),
        NodeAllocation(1, 'host-b', tuple(range(8)), (('NIC', 'eth1'),)),
    ), lambda: None, distributed=True))
    def work(context):
        context.model = replace(context.model, serve=ServeSpec(tp=16), env=(('CUDA_VISIBLE_DEVICES', '99'),))
        plan = deployment.plan_deployment(context)
        first, second = plan.ranks
        assert '--headless' not in first.command and '--headless' in second.command
        assert first.command[first.command.index('--master-addr') + 1] == 'host-a'
        assert second.command[second.command.index('--node-rank') + 1] == '1'
        assert dict(second.env)['CUDA_VISIBLE_DEVICES'] == '0,1,2,3,4,5,6,7'
        assert dict(second.env)['NIC'] == 'eth1'
        return TaskOutcome()
    result = runner.run(Job(work), tmp_path, policy(), threading.Event())
    assert result.status == Status.PASSED


def test_skipped_task_is_reported_without_runtime(tmp_path):
    original = task()
    path = tmp_path / 'results.csv'
    with CsvReport(path, [original], 'inference') as report:
        row = report.write(original, TaskResult(Status.SKIPPED, Phase.COMPLETE, 0))
    assert row['Correct Ratio'] == '100%'
    assert row['Stage'] == 'NORMAL_END'
    assert ResumeIndex(str(path)).should_skip(original)


def test_singleton_removal_cluster_manager_instances(monkeypatch):
    from types import ModuleType
    monkeypatch.setitem(sys.modules, 'paramiko', ModuleType('paramiko'))
    from tools.batched_test.mp_manager import MPClusterManager
    one = MPClusterManager([{'ssh': {'hostname': 'one'}}])
    two = MPClusterManager([{'ssh': {'hostname': 'two'}}])
    one.allocate(1)
    assert one.get_node_hostname(0) == 'one'
    assert two.get_node_hostname(0) == 'two'
    assert not two.occupied_nodes
    # Retaining rank0 blocks another cluster task instead of starting it locally
    # while claiming a different machine as rank0.
    assert one.allocate(1) == []


def test_same_legacy_tag_configs_get_distinct_resume_identity(tmp_path):
    original = task()
    other = replace(original, model=replace(original.model, serve=ServeSpec(
        extra_args=('--enable-expert-parallel',))))
    tasks = Scheduler._label_tasks([original, other, original])
    assert len(tasks) == 2
    assert len({t.report_tag for t in tasks}) == 2
    assert all('#' in t.report_tag for t in tasks)
    assert len({t.artifact_id for t in tasks}) == 2
    result = TaskResult(Status.PASSED, Phase.COMPLETE, 1,
                        TaskOutcome(cases=(CaseResult('text-only', 0, True, 'answer'),)))
    path = tmp_path / 'results.csv'
    with CsvReport(path, tasks, 'inference') as report:
        report.write(tasks[0], result)
    resume = ResumeIndex(str(path))
    assert resume.should_skip(tasks[0])
    assert not resume.should_skip(tasks[1])


def test_inference_scheduler_skips_without_allocating(tmp_path):
    config = tmp_path / 'models.yaml'
    cases = tmp_path / 'text.yaml'
    config.write_text('- name: mock\n  model_path: mock-path\n  infer_type: [text-only]\n')
    cases.write_text('- question: question\n  keywords: [answer]\n')
    resume_path = tmp_path / 'previous.csv'
    original = task()
    with CsvReport(resume_path, [original], 'inference') as report:
        report.write(original, TaskResult(Status.SKIPPED, Phase.COMPLETE, 0))
    obj = Scheduler(SchedulerArgs(
        work_dir=str(tmp_path / 'output'), model_config=str(config),
        text_case=str(cases), image_case='not-needed', resume_csv=str(resume_path),
    ))
    obj._initialize_runtime = lambda: pytest.fail('skipped task initialized runtime')
    obj.run_inference()
    path = Path(obj.work_dir) / 'inference' / 'inference_results.csv'
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]['Stage'] == 'NORMAL_END'
    assert rows[0]['Correct Ratio'] == '100%'


def test_empty_performance_report_keeps_header(tmp_path):
    path = tmp_path / 'empty.csv'
    with CsvReport(path, [], 'performance'):
        pass
    with path.open() as stream:
        reader = csv.DictReader(stream)
        assert 'task_name' in reader.fieldnames
        assert list(reader) == []


def test_gpu_manager_allocation_with_package_driver_import(monkeypatch):
    import importlib
    from types import ModuleType
    import tools.batched_test as package

    driver = ModuleType('tools.batched_test.pymxml')
    driver.nvmlInit = lambda: None
    driver.nvmlShutdown = lambda: None
    driver.nvmlDeviceGetCount = lambda: 2
    driver.nvmlDeviceGetHandleByIndex = lambda i: i
    driver.nvmlDeviceGetMemoryInfo = lambda i: SimpleNamespace(
        used=(100 if i == 0 else 1000) * 1024**2,
        free=1000 * 1024**2, total=2000 * 1024**2,
    )
    monkeypatch.setitem(sys.modules, 'tools.batched_test.pymxml', driver)
    monkeypatch.setattr(package, 'pymxml', driver, raising=False)
    monkeypatch.delitem(sys.modules, 'tools.batched_test.gpu_manager', raising=False)
    module = importlib.import_module('tools.batched_test.gpu_manager')
    try:
        manager = module.GPUManager()
        assert manager.allocate(1) == [0]
        assert manager.allocate(1) == []
        manager.release([0])
        assert manager.allocate(1) == [0]
        assert module.GPUManager().occupied_gpus == set()
    finally:
        sys.modules.pop('tools.batched_test.gpu_manager', None)
        monkeypatch.delattr(package, 'gpu_manager', raising=False)
