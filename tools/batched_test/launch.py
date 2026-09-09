# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# This script is used for model auto testing

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
from typing import ClassVar
import os
import yaml
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

from pprint import pprint
import sys
import threading

# Support both `python launch.py` and `python -m tools.batched_test.launch`.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.batched_test import utils
from tools.batched_test.model_worker import InferenceTask, BenchmarkTask
from tools.batched_test.specs import ModelSpec, ServeSpec, BenchmarkSpec, RunPolicy
from tools.batched_test.suites import plan_inference
from tools.batched_test.runtime import TaskRunner
from tools.batched_test.resources import LocalBackend, ClusterBackend
from tools.batched_test.results import TaskResult, TaskOutcome, Status, Phase
from tools.batched_test.reporting import CsvReport, ResumeIndex
from tools.batched_test.paths import normalize_config, resolve_path

from tqdm import tqdm


def cal_gpu_count(model_cfg: dict):
    return ServeSpec.parse(model_cfg.get("serve_config")).required_gpus


@dataclass(kw_only=True)
class SchedulerArgs:
    # Output root, normalized relative to the invocation directory.
    work_dir: str
    # Absolute model YAML filename after CLI path resolution.
    model_config: str

    # Default text suite YAML; only read when a selected task needs it.
    text_case: str
    # Default image suite YAML; only read when a selected task needs it.
    image_case: str
    # Optional long-text YAML used to add one deterministic case per text suite.
    long_text_case: str | None = None
    # Optional embedding suite YAML required by embedding tasks.
    embedding_case: str | None = None
    # Optional previous inference CSV; its adjacent manifest strengthens resume matching.
    resume_csv: str | None = None

    # Optional cluster YAML; None selects local GPU execution.
    cluster_config: str | None = None
    # Whether to run correctness tests for selected models.
    infer: bool = False
    # Whether to run performance sweeps after any correctness tests.
    perf: bool = False
    # Requested task concurrency; None uses GPU count locally or one for clusters.
    concurrency: int | None = None
    # Execution deadline in seconds after allocation; cleanup can extend past it.
    model_timeout: int = 3600

    # Comma-separated required GPU counts used to filter models.
    gpus: str | None = None
    # Comma-separated model tags matched with OR semantics.
    tag: str | None = None
    # Optional absolute output filename for the normalized selected YAML subset.
    dump_selected: str | None = None
    # Print selection without initializing GPU/SSH managers or running tests.
    dry_run: bool = False

    # Description displayed by the command-line argument parser.
    parser_help: ClassVar[str] = "Model Auto Testing Scheduler"

    def __post_init__(self):
        cwd = Path.cwd()
        for name in (
            "model_config",
            "cluster_config",
            "text_case",
            "image_case",
            "long_text_case",
            "embedding_case",
            "resume_csv",
        ):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, resolve_path(value, cwd))
        for name in ("work_dir", "dump_selected"):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, resolve_path(value, cwd, legacy=False))

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "SchedulerArgs":
        return cls(
            work_dir=args.work_dir,
            model_config=args.model_config,
            cluster_config=args.cluster_config,
            text_case=args.text_case,
            image_case=args.image_case,
            long_text_case=args.long_text_case,
            embedding_case=args.embedding_case,
            resume_csv=args.resume_csv,
            infer=args.infer,
            perf=args.perf,
            concurrency=args.concurrency,
            model_timeout=args.model_timeout,
            gpus=args.gpus,
            tag=args.tag,
            dump_selected=args.dump_selected,
            dry_run=args.dry_run,
        )

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--work-dir",
            type=str,
            default="/workspace/model_test",
            help="Save result for all kind of tests. Default to: </workspace/model_test>",
        )

        parser.add_argument(
            "--model-config",
            metavar="CONFIG_YAML_FILE",
            type=str,
            default=os.path.join(os.path.dirname(__file__), "configs", "model.yaml"),
            help="Model config file path. Default to: <configs/model.yaml>",
        )

        parser.add_argument(
            "--cluster-config",
            metavar="CONFIG_YAML_FILE",
            type=str,
            help="Cluster config file path.",
        )

        parser.add_argument(
            "--infer",
            action="store_true",
            help="Specify this to run inference test.",
        )

        parser.add_argument(
            "--text-case",
            metavar="LM_CASE_FILE",
            type=str,
            default=os.path.join(
                os.path.dirname(__file__), "configs", "inference", "text_case.yaml"
            ),
            help="Cases used for inference test. Default to: <configs/inference/text_case.yaml>",
        )

        parser.add_argument(
            "--image-case",
            metavar="IMAGE_CASE_FILE",
            type=str,
            default=os.path.join(
                os.path.dirname(__file__), "configs", "inference", "image_case.yaml"
            ),
            help="Cases used for inference test. Default to: <configs/inference/image_case.yaml>",
        )

        parser.add_argument(
            "--long-text-case",
            metavar="LONG_TEXT_CASE_FILE",
            type=str,
            default=None,
            help="Optional long-context text cases (YAML). When specified, these cases are "
            "run in addition to the short text cases. Each case may specify 'max_tokens' "
            "(default 512). Example: configs/inference/long_text_case.yaml",
        )

        parser.add_argument(
            "--embedding-case",
            metavar="EMBEDDING_CASE_FILE",
            type=str,
            default=os.path.join(
                os.path.dirname(__file__), "configs", "inference", "embedding_case.yaml"
            ),
            help="Cases used for embedding models (infer_type: embedding). Each case has "
            "'query', 'positive' and 'negative' texts; correct iff mean cosine(query, "
            "positive) > mean cosine(query, negative). Default: "
            "<configs/inference/embedding_case.yaml>",
        )

        parser.add_argument(
            "--resume-csv",
            metavar="RESUME_CSV",
            type=str,
            help="Resume from the failed case in specified inference_result.csv",
        )

        # Model selection / filtering
        parser.add_argument(
            "--gpus",
            type=str,
            default=None,
            help=(
                "Only run models that require the given number(s) of GPUs (tp*pp*dp). Comma-separated, e.g. '1,2,4,8'. "
                "If not set, default to '1,2,4,8'."
            ),
        )

        parser.add_argument(
            "--tag",
            type=str,
            default=None,
            help=(
                "Only run models matching the given tag(s). Comma-separated, e.g. 'moe' or 'dense' or 'moe,vl'. "
                "If not set, run all models. Models without 'tags' in model.yaml are treated as tag 'dense'."
            ),
        )

        parser.add_argument(
            "--dump-selected",
            type=str,
            default=None,
            help="Dump the selected model subset to a yaml file and continue running.",
        )

        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Print the selected model list (name / gpu_count / moe) then exit.",
        )

        parser.add_argument(
            "--perf",
            action="store_true",
            help="Specify this to run performance benchmark.",
        )

        parser.add_argument(
            "--concurrency",
            type=int,
            default=None,
            help="Max number of models to run concurrently. Default: GPU count on local machine; 1 on cluster. Use 1 for serial.",
        )

        parser.add_argument(
            "--model-timeout",
            type=int,
            default=3600,
            help="Execution timeout after GPU allocation, excluding queue time. Stops the model and waits for cleanup. Default: 3600 seconds.",
        )


class Scheduler:
    def __init__(self, args: SchedulerArgs):
        # Normalized CLI options shared by planning and execution.
        self.args = args
        # Selected configuration dictionaries, sorted by required GPU count descending.
        self.model_list = self._load_yaml_config(args.model_config)
        self.model_list = sorted(
            self._filter_model_list(self.model_list), key=cal_gpu_count, reverse=True
        )
        # Timestamped output directory for this batch.
        self.work_dir = os.path.join(args.work_dir, utils.current_dt())
        # Model-only operations must not initialize GPU drivers or SSH clients.
        # Lazily initialized lifecycle owner; stays None for dry runs or fully skipped batches.
        self._runner = None
        # Effective executor concurrency after local/cluster capacity limits are applied.
        self.max_workers = None
        # Default execution policy; performance derives a shorter allocation timeout.
        self.policy = RunPolicy(execution_timeout=args.model_timeout)
        if args.concurrency is not None and args.concurrency < 1:
            raise ValueError("--concurrency must be >= 1")

    def _initialize_runtime(self):
        if self._runner is not None:
            return
        if self.args.cluster_config:
            from tools.batched_test.mp_manager import MPClusterManager

            manager = MPClusterManager(self._load_yaml_config(self.args.cluster_config))
            backend, capacity = ClusterBackend(manager), 1
        else:
            from tools.batched_test.gpu_manager import GPUManager

            manager = GPUManager()
            backend, capacity = LocalBackend(manager), manager.get_gpu_count()
        self.max_workers = min(self.args.concurrency or capacity, capacity)
        if self.max_workers < 1:
            raise ValueError("No GPUs available")
        self._runner = TaskRunner(backend, utils.PortManager())

    def _run_tasks(self, tasks, work_dir: Path, policy: RunPolicy):
        if not tasks:
            return
        self._initialize_runtime()
        executor = ThreadPoolExecutor(max_workers=self.max_workers)
        pending = {}
        try:
            for task in tasks:
                cancel = threading.Event()
                future = executor.submit(
                    self._runner.run, task, work_dir, policy, cancel
                )
                pending[future] = (task, cancel)
            while pending:
                done, _ = concurrent.futures.wait(
                    pending,
                    timeout=0.2,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for future in done:
                    task, _ = pending.pop(future)
                    yield task, future.result()
        finally:
            for future, (_, cancel) in pending.items():
                cancel.set()
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)

    def _load_yaml_config(self, config_yaml: str) -> list[dict]:
        with open(config_yaml, "r") as f:
            config = yaml.safe_load(f)

        if not isinstance(config, list):
            raise ValueError(f"Expected a YAML list: {config_yaml}")
        base = Path(config_yaml).parent
        return [normalize_config(row, base) for row in config]

    def _parse_gpus_filter(self) -> set[int] | None:
        """Parse --gpus like '1,2,4' into a set of ints.

        Default behavior:
          - If --gpus is not provided, run models requiring {1,2,4,8} GPUs by default.
          - If --gpus is provided, use the user-specified set.
        """
        if not self.args.gpus:
            return {1, 2, 4, 8}

        # A whitespace-only filter uses the same defaults as an omitted filter.
        if str(self.args.gpus).strip() == "":
            return {1, 2, 4, 8}

        out: set[int] = set()
        for part in str(self.args.gpus).split(","):
            part = part.strip()
            if not part:
                continue
            try:
                out.add(int(part))
            except ValueError as e:
                raise ValueError(
                    f"Invalid --gpus value '{part}'. Expected comma-separated integers."
                ) from e
        return out or {1, 2, 4, 8}

    def _get_tags(self, model_cfg: dict) -> set[str]:
        """Return normalized tags for a model.

        Design:
        - If model_cfg has no 'tags' (or it's empty), treat it as {'dense'}.
        - If model_cfg.tags is a string, treat it as a single tag.
        - Tags are lower-cased strings.
        """
        tags = model_cfg.get("tags", None)
        if not tags:
            return {"None"}
        if isinstance(tags, str):
            tags = [tags]
        out = {str(t).strip().lower() for t in tags if str(t).strip()}
        return out or {"None"}

    def _parse_tag_filter(self) -> set[str] | None:
        """Parse --tag like 'moe,dense' into a set of tags (OR semantics)."""
        if not self.args.tag:
            return None
        out: set[str] = set()
        for part in str(self.args.tag).split(","):
            part = part.strip().lower()
            if part:
                out.add(part)
        return out or None

    def _filter_model_list(self, models: list[dict]) -> list[dict]:
        """Filter models by --gpus and --tag.

        Notes:
        - GPU count is computed as tp*pp*dp.
        - Tag filter uses OR semantics (match any tag).
        - Models without 'tags' are treated as tag 'dense'.
        - We attach derived fields prefixed with '_' for logging/debugging.
        """
        gpus_filter = self._parse_gpus_filter()
        tag_filter = self._parse_tag_filter()

        selected: list[dict] = []
        for m in models:
            req = cal_gpu_count(m)
            if gpus_filter is not None and req not in gpus_filter:
                continue

            tags = self._get_tags(m)
            if tag_filter is not None and tags.isdisjoint(tag_filter):
                continue

            mm = dict(m)  # avoid mutating original
            mm["_required_gpus"] = req
            mm["_tags"] = sorted(tags)
            selected.append(mm)

        # Optional: dump selected subset for reproducibility
        if self.args.dump_selected:
            self._dump_selected_models(selected)

        return selected

    def _dump_selected_models(self, selected: list[dict]) -> None:
        dump_path = self.args.dump_selected
        os.makedirs(os.path.dirname(dump_path), exist_ok=True)

        # Strip derived keys before dumping
        dump_models: list[dict] = []
        for m in selected:
            mm = {k: v for k, v in m.items() if not str(k).startswith("_")}
            dump_models.append(mm)

        # Write one model per list-item, with a blank line between models for readability
        with open(dump_path, "w", encoding="utf-8") as f:
            for i, m in enumerate(dump_models):
                if i > 0:
                    f.write("\n")
                yaml.safe_dump(
                    [m],
                    f,
                    sort_keys=False,
                    allow_unicode=True,
                    default_flow_style=False,
                )

        print(f"[Scheduler] Dumped selected models to: {dump_path}")

    def _print_selected_models(self) -> None:
        rows = []
        for m in self.model_list:
            name = m.get("name", "<unknown>")
            g = m.get("_required_gpus", "?")
            tags = ",".join(m.get("_tags") or [])
            rows.append((str(name), g, tags))
        rows.sort(key=lambda x: (int(x[1]) if str(x[1]).isdigit() else 10**9, x[0]))

        gpus_filter = self._parse_gpus_filter()
        tag_filter = self._parse_tag_filter()
        if gpus_filter is not None:
            print(f"[Scheduler] GPU count filter: {sorted(gpus_filter)}")
        if tag_filter is not None:
            print(f"[Scheduler] Tag filter (OR): {sorted(tag_filter)}")

        print(f"[Scheduler] Selected {len(rows)} model(s):")
        name_width = max((len(name) for name, _, _ in rows), default=0)
        gpu_width = max((len(str(g)) for _, g, _ in rows), default=0)
        for name, g, tags in rows:
            print(
                f"  - {name:<{name_width}} | gpus={str(g):>{gpu_width}} | tags={tags}"
            )

    def record_environment(self):
        log_file = os.path.join(self.work_dir, "collect_env.txt")
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        from tools.batched_test import collect_env

        with open(log_file, "w") as f:
            env_info = collect_env.get_pretty_env_info()
            f.write(env_info)

    def run_inference(self):
        tasks = []
        for cfg in self.model_list:
            model = ModelSpec.parse(cfg, base_dir=Path(self.args.model_config).parent)
            spec = plan_inference(
                model,
                text_case=self.args.text_case,
                image_case=self.args.image_case,
                embedding_case=self.args.embedding_case,
                long_text_case=self.args.long_text_case,
            )
            tasks.append(InferenceTask(model, spec))
        tasks = self._label_tasks(tasks)
        resume = ResumeIndex(self.args.resume_csv)
        todo = [task for task in tasks if not resume.should_skip(task)]
        skipped = [task for task in tasks if resume.should_skip(task)]
        work_dir = Path(self.work_dir) / "inference"
        results = []
        runs = self._run_tasks(todo, work_dir, self.policy)
        try:
            with CsvReport(
                work_dir / "inference_results.csv", tasks, "inference"
            ) as report:
                with tqdm(
                    total=len(tasks),
                    desc="Inference",
                    unit="model",
                    mininterval=0.5,
                    maxinterval=2.0,
                ) as progress:
                    for task in skipped:
                        result = TaskResult(
                            Status.SKIPPED,
                            Phase.COMPLETE,
                            0,
                            TaskOutcome(reason="Resumed from last result"),
                        )
                        results.append(report.write(task, result))
                        progress.update(1)
                    for task, result in runs:
                        results.append(report.write(task, result))
                        progress.update(1)
        finally:
            runs.close()
        self._print_inference_summary(results)

    @staticmethod
    def _label_tasks(tasks):
        from collections import Counter

        # Exact duplicates need only one execution. Distinct configurations with
        # the same legacy tag need separate report rows and resume identities.
        tasks = list({task.fingerprint: task for task in tasks}.values())
        counts = Counter(task.model.tag(task.kind) for task in tasks)
        return [
            replace(
                task, report_name=f"{task.model.tag(task.kind)}#{task.fingerprint[:12]}"
            )
            if counts[task.model.tag(task.kind)] > 1
            else task
            for task in tasks
        ]

    @staticmethod
    def _print_inference_summary(results: list[dict]) -> None:
        """Print a compact summary of inference results."""
        if not results:
            return

        passed = [r for r in results if r.get("Stage") == "NORMAL_END"]
        failed = [r for r in results if r.get("Stage") not in ("NORMAL_END",)]

        print(f"\n{'=' * 60}")
        print(
            f"Inference Summary: {len(results)} total | "
            f"{len(passed)} PASS | {len(failed)} FAIL"
        )
        print(f"{'=' * 60}")

        # Group by status
        by_stage: dict[str, list[dict]] = {}
        for r in failed:
            stage = r.get("Stage", "?")
            by_stage.setdefault(stage, []).append(r)

        for stage, items in by_stage.items():
            print(f"\n  [{stage}] ({len(items)}):")
            for r in items:
                reason = (r.get("Reason", "") or "")[:150]
                print(f"    - {r['Model']}: {reason}")

        # Print passed models compactly
        if passed:
            names = [r["Model"] for r in passed]
            print(f"\n  [PASSED] ({len(passed)}): {', '.join(names)}")

    def run_performance(self):
        tasks = [
            BenchmarkTask(
                ModelSpec.parse(cfg, base_dir=Path(self.args.model_config).parent),
                BenchmarkSpec.parse(cfg.get("benchmark")),
            )
            for cfg in self.model_list
        ]
        tasks = self._label_tasks(tasks)
        work_dir = Path(self.work_dir) / "performance"
        policy = replace(self.policy, allocation_timeout=14400)
        runs = self._run_tasks(tasks, work_dir, policy)
        results = []
        try:
            with CsvReport(
                work_dir / "bench_tasks_result.csv", tasks, "performance"
            ) as report:
                for task, result in runs:
                    results.append(report.write(task, result))
        finally:
            runs.close()
        ok = sum(row["status"] == "success" for row in results)
        print(
            f"Performance Summary: {len(results)} total | {ok} OK | {len(results) - ok} failed"
        )
        pprint(results)

    def run_all(self):
        if self.args.dry_run:
            self._print_selected_models()
            return
        self.record_environment()
        try:
            if self.args.infer:
                self.run_inference()

            if self.args.perf:
                self.run_performance()

        except KeyboardInterrupt:
            print("Ctrl-C detected, terminating all tests...")
            # TaskRunner has already cleaned up all active tasks.

        except Exception as e:
            print(f"Script has unexpected exited: {e}")


# Intentional public compatibility alias for scripts importing the old CLI name.
SchedularArgs = SchedulerArgs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=SchedulerArgs.parser_help)
    SchedulerArgs.add_cli_args(parser)

    args = parser.parse_args()

    sche = Scheduler(SchedulerArgs.from_cli_args(args))
    sche.run_all()
