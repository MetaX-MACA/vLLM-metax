# SPDX-License-Identifier: Apache-2.0
# This script is used for model auto testing

import argparse
from dataclasses import dataclass
import threading
from typing import ClassVar
import os
import yaml
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed

from pprint import pprint
import net_utils

from tqdm import tqdm
from ray_manager import RayClusterManager
from gpu_manager import GPUManager


@dataclass(kw_only=True)
class SchedularArgs:
    work_dir: str
    model_config: str

    text_case: str
    image_case: str
    resume_csv: str | None = None

    cluster_config: str | None = None
    infer: bool = False
    perf: bool = False

    gpus: str = None  # comma-separated GPU counts to run (e.g., '1,2,4,8')
    tag: str | None = (
        None  # comma-separated tags to select (e.g., 'moe' or 'dense' or 'moe,vl'); default: all
    )
    dump_selected: str = None  # dump selected model configs to yaml
    dry_run: bool = False  # print selected models then exit

    parser_name: ClassVar[str] = "schedular"
    parser_help: ClassVar[str] = "Model Auto Testing Scheduler"

    def __post_init__(self):
        pass

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "SchedularArgs":
        if args.resume_csv is not None:
            assert os.path.exists(args.resume_csv)
            # resume_csv = os.path.join(os.path.abspath)
        return cls(
            work_dir=args.work_dir,
            model_config=args.model_config,
            cluster_config=args.cluster_config,
            text_case=args.text_case,
            image_case=args.image_case,
            resume_csv=args.resume_csv,
            infer=args.infer,
            perf=args.perf,
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
            help="Only run models that require the given number(s) of GPUs. Comma-separated, e.g. '1,2,4,8'. GPU count is computed as tp*pp*dp.",
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


stop_event = threading.Event()


class Scheduler:
    def __init__(self, args: SchedularArgs):
        self.args = args
        self.model_list = self._load_yaml_config(args.model_config)
        self.model_list = self._filter_model_list(self.model_list)
        self.work_dir = os.path.join(args.work_dir, net_utils.current_dt())
        if args.cluster_config:
            cluster_nodes_config = self._load_yaml_config(args.cluster_config)
            self.gpu_manager = RayClusterManager(cluster_nodes_config)
            # TODO(hank) not allow concurrency on cluster mode now
            max_workers = 1
        else:
            self.gpu_manager = GPUManager()
            max_workers = self.gpu_manager.get_gpu_count()

        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def _load_yaml_config(self, config_yaml: str) -> list[dict]:
        with open(config_yaml, "r") as f:
            config = yaml.safe_load(f)
        return config

    def _required_gpus(self, model_cfg: dict) -> int:
        sc = model_cfg.get("serve_config") or {}
        tp = int(sc.get("tp", 1) or 1)
        pp = int(sc.get("pp", 1) or 1)
        dp = int(sc.get("dp", 1) or 1)
        return tp * pp * dp

    def _parse_gpus_filter(self) -> set[int] | None:
        """Parse --gpus like '1,2,4' into a set of ints."""
        if not self.args.gpus:
            return None
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
        return out or None

    def _get_tags(self, model_cfg: dict) -> set[str]:
        """Return normalized tags for a model.

        Design:
        - If model_cfg has no 'tags' (or it's empty), treat it as {'dense'}.
        - If model_cfg.tags is a string, treat it as a single tag.
        - Tags are lower-cased strings.
        """
        tags = model_cfg.get("tags")
        if not tags:
            return {"dense"}
        if isinstance(tags, str):
            tags = [tags]
        out = {str(t).strip().lower() for t in tags if str(t).strip()}
        return out or {"dense"}

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
            req = self._required_gpus(m)
            if gpus_filter is not None and req not in gpus_filter:
                continue

            tags = self._get_tags(m)
            if tag_filter is not None and tags.isdisjoint(tag_filter):
                continue

            mm = dict(m)  # avoid mutating original
            mm["_required_gpus"] = req
            mm["_tags"] = sorted(tags)
            mm["_is_moe"] = "moe" in tags
            selected.append(mm)

        # Optional: dump selected subset for reproducibility
        if self.args.dump_selected:
            self._dump_selected_models(selected)

        return selected

    def _resolve_dump_selected_path(self, path: str) -> str:
        """Resolve dump path.

        If the user passed a bare filename (no directory component), place it under
        the same configs directory used by --model-config default (./configs).
        """
        if os.path.dirname(path) == "":
            base_dir = os.path.join(os.path.dirname(__file__), "configs")
            return os.path.join(base_dir, path)
        return path

    def _dump_selected_models(self, selected: list[dict]) -> None:
        dump_path = os.path.abspath(
            self._resolve_dump_selected_path(self.args.dump_selected)
        )
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
        for name, g, tags in rows:
            print(f"  - {name} | gpus={g} | tags={tags}")

    def record_environment(self):
        log_file = os.path.join(self.work_dir, "collect_env.txt")
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        import collect_env

        with open(log_file, "w") as f:
            env_info = collect_env.get_pretty_env_info()
            f.write(env_info)

    def run_inference(self):
        all_results = []
        futures = []

        assert os.path.exists(self.args.text_case), (
            f"Case file not found: {self.args.text_case}"
        )
        assert os.path.exists(self.args.image_case), (
            f"Case file not found: {self.args.image_case}"
        )

        infer_work_dir = os.path.join(self.work_dir, "inference")
        csv_file_path = net_utils.prepare_dir(
            os.path.join(infer_work_dir, "inference_results.csv")
        )

        from model_worker import InferWorker

        for cfg in self.model_list:
            worker = InferWorker(
                work_dir=infer_work_dir,
                model_cfg=cfg,
                text_case=self.args.text_case,
                image_case=self.args.image_case,
                last_resume=self.args.resume_csv,
                gpu_manager=self.gpu_manager,
            )
            future = self.executor.submit(worker.run, stop_event)
            futures.append(future)

        with open(csv_file_path, mode="w", newline="", encoding="utf-8") as f_csv:
            csv_writer = csv.DictWriter(
                f_csv,
                fieldnames=["Model", "Correct Ratio", "Stage", "Reason", "Model Path"],
                restval="",
            )
            csv_writer.writeheader()
            with tqdm(
                total=len(self.model_list),
                desc="Inference",
                unit="model",
                mininterval=0.5,
                maxinterval=2.0,
            ) as pbar:
                pbar.refresh()
                refresh_stop = threading.Event()

                def _refresher():
                    while not refresh_stop.wait(5.0):
                        pbar.refresh()

                refresher_thread = threading.Thread(target=_refresher, daemon=True)
                refresher_thread.start()

                try:
                    for f in as_completed(futures):
                        result = f.result()
                        pbar.update(1)
                        all_results.append(result)
                        csv_writer.writerow(result)
                        f_csv.flush()
                finally:
                    refresh_stop.set()
                    refresher_thread.join()

        pprint(all_results)

    def run_performance(self):
        all_results = []
        futures = []

        bench_work_dir = os.path.join(self.work_dir, "performance")
        from model_worker import BenchSweepWorker

        for cfg in self.model_list:
            worker = BenchSweepWorker(
                work_dir=bench_work_dir, model_cfg=cfg, gpu_manager=self.gpu_manager
            )
            future = self.executor.submit(worker.run, stop_event)
            futures.append(future)

        for f in as_completed(futures):
            result = f.result()
            all_results.append(result)

    def run_all(self):
        self.record_environment()
        if self.args.dry_run:
            self._print_selected_models()
            return
        try:
            if self.args.infer:
                self.run_inference()

            if self.args.perf:
                self.run_performance()

        except KeyboardInterrupt:
            print("Ctrl-C detected, terminating all tests...")
            stop_event.set()

        except Exception as e:
            print(f"Script has unexpected exited: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=SchedularArgs.parser_help)
    SchedularArgs.add_cli_args(parser)

    args = parser.parse_args()

    sche = Scheduler(SchedularArgs.from_cli_args(args))
    sche.run_all()
