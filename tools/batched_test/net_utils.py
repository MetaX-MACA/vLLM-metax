# SPDX-License-Identifier: Apache-2.0
import socket
import subprocess
import os
import threading


class PortManager:
    _instance = None
    _lock = threading.Lock()  # class-level lock for singleton creation

    OCCUPIED_PORTS = set()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, host=""):
        self.global_mutex = threading.Lock()
        self.host = host

    def is_port_available(self, port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex((self.host, port)) != 0

    def get_next_available_port(self, start_port=8000, max_port=9000) -> int:
        with self.global_mutex:
            for port in range(start_port, max_port):
                if port in self.OCCUPIED_PORTS:
                    continue
                if self.is_port_available(port):
                    self.OCCUPIED_PORTS.add(port)
                    return port
            raise RuntimeError("No available port found")

    def release_port(self, port: int):
        with self.global_mutex:
            self.OCCUPIED_PORTS.discard(port)


def run_cmd(
    cmd: list[str], env, log_file: str | None = None, capture_output: bool = False
) -> subprocess.Popen:
    use_shell = isinstance(cmd, str)
    file_obj = None
    stdout_dest = None
    stderr_dest = None

    if log_file is not None:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        file_obj = open(log_file, "a")
        stdout_dest = file_obj
        stderr_dest = subprocess.STDOUT  # combine stderr with stdout
    elif capture_output:
        stdout_dest = subprocess.PIPE
        stderr_dest = subprocess.PIPE

    pid = subprocess.Popen(
        cmd, shell=use_shell, stdout=stdout_dest, stderr=stderr_dest, env=env
    )

    return pid  # return pid for background process


def prepare_dir(path: str) -> str:
    dir_path = os.path.dirname(os.path.abspath(path))
    os.makedirs(dir_path, exist_ok=True)
    return path


def current_dt() -> str:
    import datetime

    return datetime.datetime.now().strftime("%Y%m%d_%H%M")
