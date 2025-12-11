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

import gpu_manager


@dataclass
class SchedularArgs:
    base_dir: str
    work_dir: str
    model_config: str

    inference_case: str

    infer: bool = False
    perf: bool = False

    parser_name: ClassVar[str] = "schedular"
    parser_help: ClassVar[str] = "Model Auto Testing Scheduler"

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "SchedularArgs":
        return cls(
            base_dir=args.base_dir,
            work_dir=args.work_dir,
            model_config=args.model_config,
            inference_case=args.inference_case,
            infer=args.infer,
            perf=args.perf,
        )

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--base-dir",
            type=str,
            default=os.path.dirname(__file__),
            help="Base directory of the script.",
        )

        parser.add_argument(
            "--work-dir",
            type=str,
            default="/workspace/model_test",
            help="Base working directory for all tests.",
        )

        parser.add_argument(
            "--model-config",
            type=str,
            default=os.path.join(os.path.dirname(__file__), "configs", "model.yaml"),
            help="Path to the model configuration file.",
        )

        parser.add_argument(
            "--infer",
            action="store_true",
            help="Need run inference functional tests.",
        )

        parser.add_argument(
            "--inference-case",
            type=str,
            default=os.path.join(
                os.path.dirname(__file__), "configs", "inference", "inference_case.yaml"
            ),
            help="Path to the inference case configuration file.",
        )

        parser.add_argument(
            "--perf",
            action="store_true",
            help="Need run performance tests.",
        )


stop_event = threading.Event()


class Scheduler:
    def __init__(self, args: SchedularArgs):
        self.args = args

        self.gpu_manager = gpu_manager.GPUManager()
        self.model_list = self._load_yaml_config(args.model_config)
        self.work_dir = os.path.join(args.work_dir, net_utils.current_dt())
        self.executor = ThreadPoolExecutor(max_workers=self.gpu_manager.get_gpu_count())

    def _load_yaml_config(self, config_yaml: str) -> list[dict]:
        with open(config_yaml, "r") as f:
            config = yaml.safe_load(f)
        return config

    def record_environment(self):
        log_file = os.path.join(self.work_dir, "collect_env.txt")
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)

        import collect_env

        with open(log_file, "w") as f:
            env_info = collect_env.get_pretty_env_info()
            f.write(env_info)

    def run_functional_test(self):
        all_results = []
        futures = []

        assert os.path.exists(self.args.inference_case), (
            f"Case file not found: {self.args.inference_case}"
        )

        infer_work_dir = os.path.join(self.work_dir, "inference")
        csv_file_path = net_utils.prepare_dir(
            os.path.join(infer_work_dir, "inference_results.csv")
        )

        from model_worker import InferWorker

        for cfg in self.model_list:
            worker = InferWorker(
                base_dir=self.args.base_dir,
                work_dir=infer_work_dir,
                model_cfg=cfg,
                case_file=self.args.inference_case,
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

            for f in as_completed(futures):
                result = f.result()
                all_results.append(result)

                csv_writer.writerow(result)
                f_csv.flush()

        pprint(all_results)

    def run_performance_test(self):
        all_results = []
        futures = []

        bench_work_dir = os.path.join(self.work_dir, "performance")
        from model_worker import BenchSweepWorker

        for cfg in self.model_list:
            worker = BenchSweepWorker(
                base_dir=self.args.base_dir,
                work_dir=bench_work_dir,
                model_cfg=cfg,
            )
            future = self.executor.submit(worker.run)
            futures.append(future)

        for f in as_completed(futures):
            result = f.result()
            all_results.append(result)

    def run_all(self):
        # self.record_environment()
        try:
            if self.args.infer:
                self.run_functional_test()

            elif self.args.perf:
                self.run_performance_test()

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
