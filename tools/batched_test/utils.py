# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import socket
import subprocess
import os
import threading


class PortManager:
    def __init__(self, host=""):
        # Local host address probed for listening TCP ports.
        self.host = host
        # Protects port reservation and release across concurrent task threads.
        self.global_mutex = threading.Lock()
        # Ports reserved by this batch before services necessarily bind them.
        self.occupied_ports: set[int] = set()

    def is_port_available(self, port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex((self.host, port)) != 0

    def get_next_available_port(self, start_port=8000, max_port=9000) -> int:
        with self.global_mutex:
            for port in range(start_port, max_port):
                if port in self.occupied_ports:
                    continue
                if self.is_port_available(port):
                    self.occupied_ports.add(port)
                    return port
            raise RuntimeError("No available port found")

    def release_port(self, port: int):
        with self.global_mutex:
            self.occupied_ports.discard(port)


def run_cmd(cmd: list[str], env: dict, log_file: str) -> subprocess.Popen:
    """Start a task-owned session with combined stdout/stderr in its log."""
    os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
    with open(log_file, "a") as stream:
        return subprocess.Popen(
            cmd,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )


def current_dt() -> str:
    import datetime

    return datetime.datetime.now().strftime("%Y%m%d_%H%M")
