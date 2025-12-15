# SPDX-License-Identifier: Apache-2.0
import shlex
import time
import psutil
import os
import abc
from enum import Enum, auto

import signal
import threading

import gpu_manager
import net_utils
import contextlib
from api_client import ChatCompletionClient


class Worker(abc.ABC):
    def __init__(self, work_dir: str, model_cfg: dict):
        self.work_dir = work_dir
        self.model_cfg = model_cfg

        self.config_manager = ModelConfigManager(model_cfg)
        self.port_manager = net_utils.PortManager()
        self.gpu_manager = gpu_manager.GPUManager()
        self.port = self.port_manager.get_next_available_port()
        self.related_gpu_ids = []

    @abc.abstractmethod
    def run(self, stop_event: threading.Event):
        raise NotImplementedError("Worker must implement run method.")

    def _wait_and_allocate_gpus(self, timeout: int = 3600) -> list[int]:
        # Block until required GPUs are allocated
        assert self.related_gpu_ids == [], "GPUs have already been allocated."

        required_gpus = self.config_manager.calc_required_gpus()

        t0 = time.time()
        while time.time() - t0 < timeout:
            occupied_gpus = self.gpu_manager.allocate(required_gpus)
            print(
                f"[{self.model_cfg['name']}] Trying to allocate {required_gpus} GPUs..."
            )

            if len(occupied_gpus) > 0:
                if occupied_gpus[0] == -1:
                    raise ValueError(
                        "Requested more GPUs than available on the system."
                    )
                print(f"[{self.model_cfg['name']}] Allocated GPUs: {occupied_gpus}")
                self.related_gpu_ids = occupied_gpus
                return occupied_gpus
            time.sleep(10)

    def _cleanup(self):
        self.port_manager.release_port(self.port)
        self.gpu_manager.release(self.related_gpu_ids)


class ModelConfigManager:
    def __init__(self, model_cfg: dict):
        self.model_cfg = model_cfg
        self.serve_cfg = model_cfg.get("serve_config", {})

    def get_field(self, field_name: str, default=None):
        return self.model_cfg.get(field_name, default)

    def calc_required_gpus(self) -> int:
        serve_config = self.model_cfg.get("serve_config", {})
        tp = serve_config.get("tp", 1)
        pp = serve_config.get("pp", 1)
        dp = serve_config.get("dp", 1)
        return tp * pp * dp

    def prepare_serve_cmd(self, host: str | None, port: int) -> list[str]:
        # Prepare command
        serve_config = self.model_cfg.get("serve_config", {})
        cmd = [
            "vllm",
            "serve",
            self.model_cfg["model_path"],
            "--host",
            host if host is not None else "localhost",
            "--port",
            str(port),
            "-tp",
            str(serve_config.get("tp", 1)),
            "-pp",
            str(serve_config.get("pp", 1)),
            "-dp",
            str(serve_config.get("dp", 1)),
            "--trust-remote-code",
            "--gpu-memory-utilization",
            str(serve_config.get("gpu_memory_utilization", 0.9)),
            "--swap-space",
            str(serve_config.get("swap_space", 16)),
            "--max-model-len",
            str(serve_config.get("max_model_len", 4096)),
            "--distributed-executor-backend",
            serve_config.get("distributed_executor_backend", "ray"),
        ]

        extra_args = serve_config.get("extra_args")
        if extra_args:
            if isinstance(extra_args, dict):
                for key, value in extra_args.items():
                    cmd.append(str(key))
                    if value is not None:
                        cmd.append(str(value))
            elif isinstance(extra_args, list):
                for item in extra_args:
                    cmd.append(str(item))

        return cmd

    def prepare_bench_cmd(self, host: str | None, port: int) -> list[str]:
        bench_cfg = self.model_cfg.get("benchmark", {})
        bench_cmd = [
            "vllm",
            "bench",
            "serve",
            "--model",
            self.model_cfg["model_path"],
            "--host",
            host if host is not None else "localhost",
            "--port",
            str(port),
            "--dataset-name",
            bench_cfg.get("dataset_name", "random"),
            "--ignore-eos" if bench_cfg.get("ignore_eos") else "",
            "--trust-remote-code",
        ]
        return bench_cmd

    def prepare_sweep_cmd(
        self, host: str | None, port: int, output_dir: str
    ) -> list[str]:
        # Prepare sweep command
        bench_cfg = self.model_cfg.get("benchmark", {})

        serve_cmd = self.prepare_serve_cmd(host, port)
        bench_cmd = self.prepare_bench_cmd(host, port)
        param_file = bench_cfg.get("bench_param")

        assert serve_cmd is not None, "Serve command is not prepared."
        assert bench_cmd is not None, "Benchmark command is not prepared."
        assert os.path.exists(os.path.abspath(param_file)), (
            f"Benchmark parameters file {param_file} does not exist."
        )

        sweep_cmd = [
            "vllm",
            "bench",
            "sweep",
            "serve",
            "--serve-cmd",
            " ".join(serve_cmd),
            "--bench-cmd",
            " ".join(bench_cmd),
            "--bench-params",
            param_file,
            "--output-dir",
            output_dir,
            "--num-runs",
            str(bench_cfg.get("sweep_num_runs", "3")),
        ]

        return sweep_cmd

    def prepare_extra_env(self, occupied_gpus: list[int] | None) -> dict:
        # Prepare environment variables
        run_env = {}

        if occupied_gpus is not None:
            run_env["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = "1"
            run_env["CUDA_VISIBLE_DEVICES"] = ",".join(
                str(idx) for idx in occupied_gpus
            )

        extra_env = self.model_cfg.get("extra_env")
        if extra_env:
            if isinstance(extra_env, dict):
                # transfer all key-value pairs to string
                run_env.update({str(k): str(v) for k, v in extra_env.items()})
            elif isinstance(extra_env, list):
                for item in extra_env:
                    if isinstance(item, dict):
                        for k, v in item.items():
                            run_env[str(k)] = str(v)

        return run_env

    def get_infer_type(self) -> list[str]:
        return self.model_cfg.get("infer_type", [])


class InferWorker(Worker):
    class InferenceStatus(Enum):
        INIT = auto()
        STARTING_SERVER = auto()
        INFERENCING = auto()
        NORMAL_END = auto()

    def __init__(self, text_case: str, image_case: str, model_cfg: dict, work_dir: str):
        super().__init__(work_dir=work_dir, model_cfg=model_cfg)

        self.text_case = text_case
        self.image_case = image_case
        self.api_serve_process = None
        self.status = self.InferenceStatus.INIT
        self.serve_cfg = model_cfg.get("serve_config", {})
        self.model_tag = f"{model_cfg['name']}_tp{self.serve_cfg.get('tp', 1)}_pp{self.serve_cfg.get('pp', 1)}_dp{self.serve_cfg.get('dp', 1)}"

    def _get_text_only_cases(self) -> list[dict]:
        import yaml

        with open(self.text_case, "r", encoding="utf-8") as f:
            test_cases = yaml.safe_load(f)
        return test_cases

    def _get_image_cases(self) -> list[dict]:
        import yaml

        with open(self.image_case, "r", encoding="utf-8") as f:
            test_cases = yaml.safe_load(f)
        return test_cases

    def _do_text_only_inference(self, log_file: str) -> float:
        client = ChatCompletionClient(host="localhost", port=self.port)
        text_cases = self._get_text_only_cases()
        questions = [case["question"] for case in text_cases]

        # Get generator for responses
        content_gen = client.run_text_only(questions)

        corrected_responses = 0
        with open(log_file, "a") as f:
            # Zip test cases with yielded responses to match them
            for test_case, content in zip(text_cases, content_gen):
                keywords = test_case.get("keywords", [])

                # Check if any keyword is in the content (case-insensitive)
                if any(str(k).lower() in content.lower() for k in keywords):
                    corrected_responses += 1

                f.write(
                    f"[{self.model_cfg['name']}] Question: {test_case['question']}\n"
                )
                f.write(f"[{self.model_cfg['name']}] Response:\n")
                f.write(content + "\n")
                f.write("-" * 40 + "\n")
        return corrected_responses / len(text_cases)

    def _do_single_image_inference(self, log_file: str) -> float:
        client = ChatCompletionClient(host="localhost", port=self.port)
        image_cases = self._get_image_cases()
        image_urls = [case["picture_url"] for case in image_cases]

        # Get generator for responses
        content_gen = client.run_single_image(
            model=client.get_model(),
            max_completion_tokens=256,
            image_urls=image_urls,
        )

        corrected_responses = 0
        with open(log_file, "a") as f:
            # Zip test cases with yielded responses to match them
            for test_case, content in zip(image_cases, content_gen):
                keywords = test_case.get("keywords", [])

                # Check if any keyword is in the content (case-insensitive)
                if any(str(k).lower() in content.lower() for k in keywords):
                    corrected_responses += 1

                f.write(
                    f"[{self.model_cfg['name']}] Image URL: {test_case['picture_url']}\n"
                )
                f.write(f"[{self.model_cfg['name']}] Response:\n")
                f.write(content + "\n")
                f.write("-" * 40 + "\n")

        return corrected_responses / len(image_cases)

    def run(self, stop_event: threading.Event):
        self.stop_event = stop_event
        try:
            # Step 1. alloc GPU
            self._wait_and_allocate_gpus()
            # self.related_gpu_ids = [0,1]

            # Step 2. launch serve
            self._launch_vllm_serve()

            # Step 3. client testing
            return self._post_client_test()
        except Exception as e:
            print(f"[{self.model_cfg['name']}] Inference failed: {e}")
            return self._warp_failure(str(e))
        finally:
            self._cleanup()

    def _post_client_test(self):
        self._check_api_service_ready(timeout=600, blocking=True)

        correct_ratio = self._chat_completion()
        return {
            "Model": self.model_cfg["name"],
            "Correct Ratio": str(correct_ratio * 100) + "%",
            "Stage": self.InferenceStatus.NORMAL_END.name,
            "Reason": "",
            "Model Path": self.model_cfg["model_path"],
        }

    def _warp_failure(self, e: str):
        return {
            "Model": self.model_cfg["name"],
            "Correct Ratio": "0%",
            "Stage": self.status.name,
            "Reason": str(e),
            "Model Path": self.model_cfg["model_path"],
        }

    def _launch_vllm_serve(self):
        # Asynchronously launch vLLM serve process
        self.status = self.InferenceStatus.STARTING_SERVER

        assert len(self.related_gpu_ids) > 0, (
            "No GPUs allocated for launching vLLM serve."
        )

        # Prepare logfile
        log_file = net_utils.prepare_dir(
            os.path.join(self.work_dir, f"{self.model_tag}_serve.log")
        )

        # Prepare command
        cmd = self.config_manager.prepare_serve_cmd(host=None, port=self.port)

        # Set environment variable
        extra_env = self.config_manager.prepare_extra_env(self.related_gpu_ids)

        env_copy = os.environ.copy()
        env_copy.update(extra_env)

        # Log the command and environment
        with open(log_file, "a") as f:
            cmd_str = f"[{self.model_cfg['name']}] command: {' '.join(cmd)}"
            f.write(cmd_str + "\n" + "-" * 80 + "\n")
            f.write(extra_env.__str__() + "\n" + "-" * 80 + "\n")
            f.flush()
            print(cmd_str)

        # Launch the command
        self.api_serve_process = net_utils.run_cmd(
            cmd=cmd, log_file=log_file, env=env_copy
        )

    def _check_api_service_ready(self, blocking=True, timeout=600):
        # Block until the API service is up or timeout
        t0 = time.time()

        print(f"[{self.model_cfg['name']}] Waiting for service on port {self.port}...")
        while time.time() - t0 < timeout:
            # Check if process has exited
            if self.stop_event.is_set():
                raise KeyboardInterrupt("Stop event set, terminating service check.")

            return_code = self.api_serve_process.poll()
            if return_code is not None:
                raise RuntimeError(
                    f"[{self.model_cfg['name']}] vLLM serve process exited unexpectedly with code {return_code}."
                )

            # Check if port is open
            if not self.port_manager.is_port_available(self.port):
                print(f"[{self.model_cfg['name']}] Service is up on port {self.port}.")
                return True

            if not blocking:
                return False

        raise TimeoutError(
            f"[{self.model_cfg['name']}] Service did not start within {timeout} seconds, aborted."
        )

    def _chat_completion(self) -> float:
        infer_type = self.model_cfg.get("infer_type", [])
        assert len(infer_type) > 0, "infer_type must be specified in model_cfg."

        self.status = self.InferenceStatus.INFERENCING

        # Load test cases from YAML
        assert os.path.exists(self.text_case), (
            f"Case file {self.text_case} does not exist."
        )
        assert os.path.exists(self.image_case), (
            f"Case file {self.image_case} does not exist."
        )

        if "single-image" in infer_type:
            log_file = net_utils.prepare_dir(
                os.path.join(
                    self.work_dir, f"{self.model_tag}_single_image_inference.log"
                )
            )
            return self._do_single_image_inference(log_file)

        if "text-only" in infer_type:
            log_file = net_utils.prepare_dir(
                os.path.join(self.work_dir, f"{self.model_tag}_text_only_inference.log")
            )
            return self._do_text_only_inference(log_file)

    def _shutdown_process(self):
        serve_process = self.api_serve_process

        if serve_process is None:
            return

        try:
            parent = psutil.Process(serve_process.pid)
            children = parent.children(recursive=True)
            for child in children:
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            parent.kill()
            parent.wait()
        except psutil.NoSuchProcess:
            pass

        # Double check the gpu worker processes
        worker_pid = self.gpu_manager.get_gpu_process_pid(self.related_gpu_ids)

        # kill GPU worker zombie processes in case they are not cleaned up
        for pid in worker_pid:
            if psutil.pid_exists(pid):
                try:
                    p = psutil.Process(pid)
                    p.kill()
                except Exception as e:
                    print(
                        f"[{self.model_cfg['name']}] Error killing GPU worker process {pid}: {e}"
                    )

        print(f"[{self.model_cfg['name']}] Serve cleaned up successfully.")

    def _cleanup(self):
        """
        Additional cleanup after serve is stopped.

        :param self: Description
        :param args: Description
        :param kwargs: Description
        """
        super()._cleanup()
        self._shutdown_process()


class BenchSweepWorker(Worker):
    def __init__(self, work_dir: str, model_cfg: dict):
        super().__init__(work_dir=work_dir, model_cfg=model_cfg)
        self.sweep_process = None

        self.serve_cfg = model_cfg.get("serve_config", {})
        self.model_tag = f"{model_cfg['name']}_tp{self.serve_cfg.get('tp', 1)}_pp{self.serve_cfg.get('pp', 1)}_dp{self.serve_cfg.get('dp', 1)}"

    def run(self):
        try:
            self._wait_and_allocate_gpus()
            self._launch_bench_sweep()

        except Exception as e:
            return self.warp_failure(str(e))

        finally:
            self._cleanup()

    def _launch_bench_sweep(self):
        result_dir = os.path.join(self.work_dir, self.model_tag)

        sweep_cmd = self.config_manager.prepare_sweep_cmd(
            host=None, port=self.port, output_dir=result_dir
        )
        extra_env = self.config_manager.prepare_extra_env(self.related_gpu_ids)

        # Log the process output
        log_file = net_utils.prepare_dir(
            os.path.join(self.work_dir, f"{self.model_tag}_serve.log")
        )

        with open(log_file, "a") as f:
            f.write(self.model_cfg["name"])
            f.write("\n" + "-" * 80 + "\n")
            f.write(" ".join(sweep_cmd))
            f.write("\n" + "-" * 80 + "\n")
            f.write(extra_env.__str__())
            f.write("\n" + "-" * 80 + "\n")
            f.flush()

        self.sweep_process = net_utils.run_cmd(
            cmd=sweep_cmd, env={**os.environ, **extra_env}, log_file=log_file
        )

        self.sweep_process.wait()

    def warp_failure(self, e: str):
        # Implement failure handling for performance testing here
        print(f"[{self.model_cfg['name']}] Benchmark failed: {e}")

    def _cleanup(self):
        super()._cleanup()
