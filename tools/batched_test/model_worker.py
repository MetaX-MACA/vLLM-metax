# SPDX-License-Identifier: Apache-2.0
import time
import psutil
import os
import abc
from enum import Enum, auto

from schedular import Scheduler
import net_utils


class WorkerStatus(Enum):
    INIT = auto()
    ALLOCATING_GPU = auto()
    STARTING_SERVER = auto()
    INFERENCE_TESTING = auto()
    NORMAL_END = auto()


class Worker(abc.ABC):
    @abc.abstractmethod
    def run(self):
        raise NotImplementedError("Worker must implement run method.")


class ModelWorker(Worker):
    def __init__(self, scheduler: Scheduler, model_cfg, work_dir: str):
        self.scheduler = scheduler
        self.model_cfg = model_cfg
        self.work_dir = work_dir

        self.gpu_manager = scheduler.gpu_manager
        self.port_manager = net_utils.PortManager()
        self.related_gpu_ids = []
        self.api_serve_process = None
        self.port = self.port_manager.get_next_available_port()
        self.model_tag = f"{model_cfg['name']}_tp{model_cfg.get('tp', 1)}_pp{model_cfg.get('pp', 1)}_dp{model_cfg.get('dp', 1)}"
        self.status = WorkerStatus.INIT

    def _calc_required_gpus(self, model_config):
        tp = model_config.get("tp", 1)
        pp = model_config.get("pp", 1)
        dp = model_config.get("dp", 1)
        return tp * pp * dp

    def run(self):
        # The entry_points of worker thread

        # Get available port
        required_gpus = self._calc_required_gpus(self.model_cfg)

        try:
            # Step 1. alloc GPU
            self.related_gpu_ids = self._wait_and_allocate_gpus(required_gpus)

            # Step 2. launch serve
            self.api_serve_process = self._launch_vllm_serve(
                self.related_gpu_ids, self.model_cfg
            )

            # Step 3. test inference request
            self._test_inference()

            # Step 4. result recording
            return {
                "Model": self.model_cfg["name"],
                "Result": "Success",
                "Stage": WorkerStatus.NORMAL_END.name,
                "Reason": "",
                "Model Path": self.model_cfg["model_path"],
            }

        except Exception as e:
            print(f"[{self.model_cfg['name']}] Error occurred: {e}")
            return {
                "Model": self.model_cfg["name"],
                "Result": "Failed",
                "Stage": self.status.name,
                "Reason": str(e),
                "Model Path": self.model_cfg["model_path"],
            }

        finally:
            # cleanup
            self.port_manager.release_port(self.port)
            self.gpu_manager.release(self.related_gpu_ids)
            self._cleanup_serve()

    def _wait_and_allocate_gpus(self, required) -> list[int]:
        # Block until required GPUs are allocated
        self.status = WorkerStatus.ALLOCATING_GPU

        while True:
            occupied_gpus = self.gpu_manager.allocate(required)
            print(f"[{self.model_cfg['name']}] Trying to allocate {required} GPUs...")

            if len(occupied_gpus) > 0:
                if occupied_gpus[0] == -1:
                    raise ValueError(
                        "Requested more GPUs than available on the system."
                    )
                return occupied_gpus

            print(f"[{self.model_cfg['name']}] Waiting for GPU resources...")
            time.sleep(3)

    def _launch_vllm_serve(self, gpus_list: list[int], model_cfg):
        # Asynchronously launch vLLM serve process
        self.status = WorkerStatus.STARTING_SERVER

        assert len(gpus_list) > 0, "No GPUs allocated for launching vLLM serve."
        assert len(gpus_list) == self._calc_required_gpus(model_cfg), (
            "Allocated GPU count does not match required."
        )

        # Prepare logfile

        log_file = os.path.join(self.work_dir, f"{self.model_tag}_serve.log")
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)

        # Set environment variable
        extra_env = {}
        extra_env["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = "1"
        extra_env["CUDA_VISIBLE_DEVICES"] = ",".join(str(idx) for idx in gpus_list)

        cmd = [
            "vllm",
            "serve",
            self.model_cfg["model_path"],
            "--port",
            str(self.port),
            "-tp",
            str(self.model_cfg.get("tp", 1)),
            "-pp",
            str(self.model_cfg.get("pp", 1)),
            "-dp",
            str(self.model_cfg.get("dp", 1)),
            "--trust-remote-code",
            "--gpu-memory-utilization",
            str(self.model_cfg.get("gpu_memory_utilization", 0.9)),
            "--swap-space",
            str(self.model_cfg.get("swap_space", 16)),
            "--max-model-len",
            str(self.model_cfg.get("max_model_len", 4096)),
            "--distributed-executor-backend",
            self.model_cfg.get("distributed_executor_backend", "ray"),
        ]
        extra_args = self.model_cfg.get("extra_args")
        if extra_args:
            if isinstance(extra_args, dict):
                for key, value in extra_args.items():
                    cmd.append(str(key))
                    if value is not None:
                        cmd.append(str(value))
            elif isinstance(extra_args, list):
                for item in extra_args:
                    cmd.append(str(item))

        env_copy = os.environ.copy()
        env_copy.update(extra_env)

        cmd_str = f"[{self.model_cfg['name']}] command: {' '.join(cmd)}"

        with open(log_file, "a") as f:
            f.write(cmd_str + "\n" + "-" * 80 + "\n")
            f.write(extra_env.__str__() + "\n" + "-" * 80 + "\n")
            f.flush()
        print(cmd_str)

        return net_utils.run_cmd(cmd=cmd, log_file=log_file, env=env_copy)

    def _await_api_service_ready(self, blocking=True, timeout=600):
        # Block until the API service is up or timeout
        t0 = time.time()

        print(f"[{self.model_cfg['name']}] Waiting for service on port {self.port}...")
        while time.time() - t0 < timeout:
            # Check if process has exited
            return_code = self.api_serve_process.poll()
            if return_code is not None:
                raise RuntimeError(
                    "vLLM serve process exited unexpectedly with code %d." % return_code
                )

            # Check if port is open
            if not self.port_manager.is_port_available(self.port):
                print(f"[{self.model_cfg['name']}] Service is up on port {self.port}.")
                return True

            if not blocking:
                return False

        raise TimeoutError(f"Service did not start within {timeout} seconds.")

    def _test_inference(self):
        from api_client import ChatCompletionClient

        self._await_api_service_ready(timeout=600, blocking=True)

        self.status = WorkerStatus.INFERENCE_TESTING
        log_file = os.path.join(self.work_dir, f"{self.model_tag}_inference.log")
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)

        client = ChatCompletionClient(host="localhost", port=self.port)
        model = client.get_model()
        print(f"Using model: {model}")

        questions = [
            "Where's the capital of China?",
            "Who's the founder of Apple?",
            "What's the value of gravity in the earth?",
        ]
        for response in client.create_chat_completion(
            questions=questions, model=model, stream=False
        ):
            # Write response to log file
            with open(log_file, "a") as f:
                f.write(f"[{self.model_cfg['name']}] Response:\n")
                f.write(response.choices[0].message.content + "\n")
                f.write("-" * 40 + "\n")

    def _cleanup_serve(self):
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
