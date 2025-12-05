# SPDX-License-Identifier: Apache-2.0
# This script is used for model auto testing

import os
import yaml
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from pprint import pprint

import gpu_manager

# ------------- user-tweakable defaults -------------
NUM_GPUS = 8  # total GPUs on machine
MODEL_CONFIG_FILE = "model_jiajia.yaml"  # model configuration file
MODEL_FUNCTIONAL_TEST_RESULT_DIR = f"/workspace/model_functional_test_{datetime.now().strftime('%Y_%m_%d_%H_%M')}"  # where to store test results
# ---------------------------------------------------


class Scheduler:
    def __init__(
        self,
        num_gpus=NUM_GPUS,
        model_config_file=MODEL_CONFIG_FILE,
        work_dir=MODEL_FUNCTIONAL_TEST_RESULT_DIR,
    ):
        self.num_gpus = num_gpus
        self.model_config_file = model_config_file
        self.work_dir = work_dir

        self.gpu_manager = gpu_manager.GPUManager(gpu_count=num_gpus)
        self.pool = ThreadPoolExecutor(max_workers=self.num_gpus)
        self.model_list = self._load_model_config(file_path=model_config_file)

    def _load_model_config(self, file_path: str) -> list[dict]:
        with open(file_path, "r") as f:
            config = yaml.safe_load(f)

        model_list = []
        for model_cfg in config:
            model_list.append(model_cfg)

        return model_list

    def run_functional_test(self):
        results = []
        futures = []

        from model_worker import ModelWorker

        for cfg in self.model_list:
            print(f"Adding model {cfg['name']} to the job queue.")
            worker = ModelWorker(
                self,
                model_cfg=cfg,
                work_dir=os.path.join(self.work_dir, "model_functional_test"),
            )
            future = self.pool.submit(worker.run)
            futures.append(future)

        for f in as_completed(futures):
            results.append(f.result())

        pprint(results)
        return results

    def record_environment(self):
        log_file = os.path.join(self.work_dir, "environment_info.txt")
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)

        import collect_env

        with open(log_file, "w") as f:
            env_info = collect_env.get_pretty_env_info()
            f.write(env_info)

    def run_all(self):
        self.record_environment()

        # -------------------------------------------
        # Do functional test for each model
        self.run_functional_test()

        # -------------------------------------------
        # Do performance test for each model
        # (Not implemented yet)

        # -------------------------------------------
        # Do eval test for each model
        # (Not implemented yet)


def main():
    pass


if __name__ == "__main__":
    sche = Scheduler()
    sche.run_all()
