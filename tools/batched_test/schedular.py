# SPDX-License-Identifier: Apache-2.0
# This script is used for model auto testing

import datetime
import os
import yaml
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed

from pprint import pprint
import net_utils

import gpu_manager


class Scheduler:
    def __init__(self, recipy: str):
        self.gpu_manager = gpu_manager.GPUManager()
        self.recipy = self._load_recipy(recipy)

        self.model_list = self._load_model_config()
        self.work_dir = self.recipy["work_dir"] + net_utils.current_dt()
        self.pool = ThreadPoolExecutor(max_workers=self.gpu_manager.get_gpu_count())

    def _load_recipy(self, recipy_path: str) -> dict:
        with open(recipy_path, "r") as f:
            recipy = yaml.safe_load(f)
        return recipy

    def _load_model_config(self) -> list[dict]:
        config_yaml = os.path.join(
            os.path.dirname(__file__), self.recipy["model_config"]
        )

        with open(config_yaml, "r") as f:
            config = yaml.safe_load(f)

        model_list = []
        for model_cfg in config:
            model_list.append(model_cfg)

        return model_list

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

        case_file = os.path.join(
            os.path.dirname(__file__), self.recipy["inference_case"]
        )

        assert os.path.exists(case_file), f"Case file not found: {case_file}"

        csv_file_path = net_utils.prepare_dir(
            os.path.join(self.work_dir, "inference_results.csv"), is_file=True
        )

        from model_worker import InferWorker

        for cfg in self.model_list:
            worker = InferWorker(
                scheduler=self,
                model_cfg=cfg,
                case_file=case_file,
                work_dir=self.work_dir,
            )
            future = self.pool.submit(worker.run)
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

        bench_param_folder = net_utils.prepare_dir(
            os.path.join(
                os.path.dirname(__file__), self.recipy["bench_param_config_folder"]
            ),
            is_file=False,
        )

        from model_worker import BenchmarkWorker

        for cfg in self.model_list:
            worker = BenchmarkWorker(
                bench_param_folder=bench_param_folder,
                scheduler=self,
                model_cfg=cfg,
                work_dir=self.work_dir,
            )
            future = self.pool.submit(worker.run)
            futures.append(future)

        for f in as_completed(futures):
            result = f.result()
            all_results.append(result)

    def run_all(self):
        self.record_environment()

        # -------------------------------------------
        # Do functional test for each model
        # self.run_functional_test()

        # -------------------------------------------
        # Do performance test for each model
        # (Not implemented yet)
        self.run_performance_test()

        # -------------------------------------------
        # Do eval test for each model
        # (Not implemented yet)


if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), "recipy.yaml")

    sche = Scheduler(config_path)
    sche.run_all()
