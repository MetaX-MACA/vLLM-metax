# SPDX-License-Identifier: Apache-2.0
import time
import psutil
import os
import abc
from enum import Enum, auto

from schedular import Scheduler
import net_utils


class InferenceStatus(Enum):
    INIT = auto()
    STARTING_SERVER = auto()

    # inference testing only
    INFERENCE_TESTING = auto()

    NORMAL_END = auto()


class Worker(abc.ABC):
    @abc.abstractmethod
    def run(self):
        raise NotImplementedError("Worker must implement run method.")


class ServeBuilder:
    related_gpu_ids = []
    work_dir = None

    def __init__(self, scheduler: Scheduler, work_dir: str, model_cfg: dict):
        self.scheduler = scheduler
        self.port_manager = net_utils.PortManager()
        self.gpu_manager = scheduler.gpu_manager

        self.work_dir = work_dir
        self.model_cfg = model_cfg

        self.port = self.port_manager.get_next_available_port()
        self.model_tag = f"{model_cfg['name']}_tp{model_cfg.get('tp', 1)}_pp{model_cfg.get('pp', 1)}_dp{model_cfg.get('dp', 1)}"

    def _calc_required_gpus(self, inference_cfg: dict) -> int:
        tp = inference_cfg.get("tp", 1)
        pp = inference_cfg.get("pp", 1)
        dp = inference_cfg.get("dp", 1)
        return tp * pp * dp

    def _wait_and_allocate_gpus(self, timeout: int = 3600) -> list[int]:
        # Block until required GPUs are allocated
        assert self.related_gpu_ids == [], "GPUs have already been allocated."

        serve_config = self.model_cfg.get("serve_config", {})
        required_gpus = self._calc_required_gpus(serve_config)

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

    def _prepare_serve_cmd(self) -> list[str]:
        # Prepare command
        serve_config = self.model_cfg.get("serve_config", {})
        cmd = [
            "vllm",
            "serve",
            self.model_cfg["model_path"],
            "--port",
            str(self.port),
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

    def _prepare_extra_env(self) -> dict:
        # Prepare environment variables
        run_env = {}
        run_env["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = "1"
        run_env["CUDA_VISIBLE_DEVICES"] = ",".join(
            str(idx) for idx in self.related_gpu_ids
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

    def _cleanup(self):
        self.port_manager.release_port(self.port)
        self.gpu_manager.release(self.related_gpu_ids)


class InferWorker(ServeBuilder, Worker):
    def __init__(self, case_file, **kwargs):
        super().__init__(**kwargs)
        self.case_file = case_file
        self.status = InferenceStatus.INIT
        self.api_serve_process = None
        # Update work_dir
        self.work_dir = net_utils.prepare_dir(
            os.path.join(self.work_dir, "functional"), is_file=False
        )

    def run(self):
        try:
            # Step 1. alloc GPU
            self._wait_and_allocate_gpus()

            # Step 2. launch serve
            self._launch_vllm_serve()

            # Step 3. client testing
            return self._post_client_test()

        except Exception as e:
            return self._warp_failure(str(e))
        finally:
            self._cleanup()

    def _post_client_test(self):
        correct_ratio = self._chat_completion()
        return {
            "Model": self.model_cfg["name"],
            "Correct Ratio": str(correct_ratio * 100) + "%",
            "Stage": InferenceStatus.NORMAL_END.name,
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
        self.status = InferenceStatus.STARTING_SERVER

        assert len(self.related_gpu_ids) > 0, (
            "No GPUs allocated for launching vLLM serve."
        )

        # Prepare logfile
        log_file = net_utils.prepare_dir(
            os.path.join(self.work_dir, f"{self.model_tag}_serve.log"), is_file=True
        )

        # Prepare command
        cmd = self._prepare_serve_cmd()

        # Set environment variable
        extra_env = self._prepare_extra_env()

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
        from api_client import ChatCompletionClient

        self._check_api_service_ready(timeout=600, blocking=True)

        self.status = InferenceStatus.INFERENCE_TESTING
        log_file = net_utils.prepare_dir(
            os.path.join(self.work_dir, f"{self.model_tag}_inference.log"), is_file=True
        )

        # Load test cases from YAML
        assert os.path.exists(self.case_file), (
            f"Case file {self.case_file} does not exist."
        )
        import yaml

        with open(self.case_file, "r", encoding="utf-8") as f:
            test_cases = yaml.safe_load(f)

        client = ChatCompletionClient(host="localhost", port=self.port)
        model = client.get_model()
        print(f"Using model: {model}")

        questions = [case["question"] for case in test_cases]

        # Get generator for responses
        responses_gen = client.create_chat_completion(
            questions=questions, model=model, stream=False
        )

        corrected_responses = 0
        with open(log_file, "a") as f:
            # Zip test cases with yielded responses to match them
            for test_case, response in zip(test_cases, responses_gen):
                content = response.choices[0].message.content
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

        print(
            f"[{self.model_cfg['name']}] Corrected responses: {corrected_responses} out of {len(test_cases)}"
        )
        return corrected_responses / len(test_cases)

    def _shutdown_process(self):
        if self.api_serve_process is None:
            print(f"[{self.model_cfg['name']}] No serve process to clean up.")
            return

        # Clean up the serve process and its children
        pid = self.api_serve_process.pid
        worker_pid = self.gpu_manager.get_gpu_process_pid(self.related_gpu_ids)

        if not psutil.pid_exists(pid) and len(worker_pid) == 0:
            print(
                f"[{self.model_cfg['name']}] PID {pid} does not exist, skipping cleanup."
            )
            return

        try:
            parent = psutil.Process(pid)
        except Exception as e:
            print(f"[{self.model_cfg['name']}] Error getting process info: {e}")
            return

        try:
            children = parent.children(recursive=True)
            for child in children:
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
        except psutil.NoSuchProcess:
            pass

        try:
            parent.kill()
            parent.wait()
        except psutil.NoSuchProcess:
            pass

        # kill GPU worker zombie processes in case they are not cleaned up
        for pid in worker_pid:
            if psutil.pid_exists(pid):
                try:
                    p = psutil.Process(pid)
                    p.kill()
                    p.wait()
                except Exception as e:
                    print(
                        f"[{self.model_cfg['name']}] Error killing GPU worker process {pid}: {e}"
                    )

        print(f"[{self.model_cfg['name']}] serve cleaned up.")

    def _cleanup(self):
        """
        Additional cleanup after serve is stopped.

        :param self: Description
        :param args: Description
        :param kwargs: Description
        """
        super()._cleanup()
        self._shutdown_process()


class BenchmarkWorker(ServeBuilder, Worker):
    def __init__(self, bench_param_folder: str, **kargs):
        super().__init__(**kargs)
        self.work_dir = net_utils.prepare_dir(
            os.path.join(self.work_dir, "performance"), is_file=False
        )
        self.bench_result_dir = self.work_dir
        self.bench_param_dir = bench_param_folder

    def run(self):
        try:
            self._wait_and_allocate_gpus()
            self._launch_bench_sweep()

        except Exception as e:
            return self.warp_failure(str(e))

        finally:
            self._cleanup()

    def _launch_bench_sweep(self):
        sweep_cmd = self._prepare_sweep_cmd()
        extra_env = self._prepare_extra_env()

        # Log the command and environment
        log_file = net_utils.prepare_dir(
            os.path.join(self.work_dir, f"{self.model_tag}_serve.log"), is_file=True
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

    def _prepare_bench_cmd(self):
        bench_cfg = self.model_cfg.get("benchmark", {})
        bench_cmd = [
            "vllm",
            "bench",
            "serve",
            "--model",
            self.model_cfg["model_path"],
            "--port",
            str(self.port),
            "--dataset-name",
            bench_cfg.get("dataset_name", "random"),
            "--ignore-eos" if bench_cfg.get("ignore_eos") else "",
        ]
        return bench_cmd

    def _prepare_bench_param_file(self) -> str:
        # Find benchmark parameters file
        bench_cfg = self.model_cfg.get("benchmark", {})
        bench_params_file = os.path.join(
            self.bench_param_dir, "bench_%d.json" % bench_cfg.get("priority", 1)
        )

        return bench_params_file

    def _prepare_sweep_cmd(self):
        # Prepare sweep command
        bench_cfg = self.model_cfg.get("benchmark", {})

        serve_cmd = self._prepare_serve_cmd()
        bench_cmd = self._prepare_bench_cmd()
        bench_params_file = self._prepare_bench_param_file()

        assert serve_cmd is not None, "Serve command is not prepared."
        assert bench_cmd is not None, "Benchmark command is not prepared."
        assert os.path.exists(bench_params_file), (
            f"Benchmark parameters file {bench_params_file} does not exist."
        )

        result_dir = net_utils.prepare_dir(
            os.path.join(self.work_dir, self.model_tag), is_file=False
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
            bench_params_file,
            "--output-dir",
            result_dir,
            "--num-runs",
            str(bench_cfg.get("sweep_num_runs", "3")),
        ]

        return sweep_cmd

    def warp_failure(self, e: str):
        # Implement failure handling for performance testing here
        print(f"[{self.model_cfg['name']}] Benchmark failed: {e}")

    def _cleanup(self):
        super()._cleanup()
