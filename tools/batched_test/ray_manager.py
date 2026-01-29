# SPDX-License-Identifier: Apache-2.0
import paramiko
import time
import os
from dataclasses import dataclass, asdict
from typing import Literal
import regex as re
import threading
from pprint import pprint

import shlex


@dataclass
class SSHConfig:
    hostname: str
    port: int
    user: str
    auth_type: Literal["password", "key"]
    password: str | None = None
    private_key: str | None = None

    def __post_init__(self):
        if self.auth_type not in ["password", "key"]:
            raise ValueError(
                f"auth_type must be 'password' or 'key', got '{self.auth_type}'"
            )

        if self.auth_type == "password" and not self.password:
            raise ValueError("password must be provided when auth_type is 'password'")

        if self.auth_type == "key" and not self.private_key:
            raise ValueError("private_key must be provided when auth_type is 'key'")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SSHConfig":
        return cls(**data)


@dataclass
class ClusterNode:
    ssh: SSHConfig
    nic: str

    def to_dict(self) -> dict:
        return {"ssh": self.ssh.to_dict(), "nic": self.nic}

    @classmethod
    def from_dict(cls, data: dict) -> "ClusterNode":
        nic = data.get("nic")
        if nic is None:
            ray_cfg = data.get("ray") or {}
            nic = ray_cfg.get("nic", "eth0")
        return cls(ssh=SSHConfig.from_dict(data["ssh"]), nic=nic)


def remote_command(ssh_info: SSHConfig, command: str) -> str:
    """
    Run remote command via ssh
    """
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        # 连接服务器
        if ssh_info.auth_type == "key":
            key_path = os.path.expanduser(ssh_info.private_key)
            private_key = paramiko.RSAKey.from_private_key_file(key_path)
            ssh.connect(
                hostname=ssh_info.hostname,
                port=ssh_info.port,
                username=ssh_info.user,
                pkey=private_key,
                timeout=10,
            )
        elif ssh_info.auth_type == "password":
            ssh.connect(
                hostname=ssh_info.hostname,
                port=ssh_info.port,
                username=ssh_info.user,
                password=ssh_info.password,
                timeout=10,
            )

        # 执行命令
        print(f"[远程:{ssh_info.hostname}:{ssh_info.port}] 执行命令: {command}")
        _, stdout, stderr = ssh.exec_command(command)

        # 获取输出
        output = stdout.read().decode("utf-8")
        error = stderr.read().decode("utf-8")

        # 获取退出状态码
        exit_status = stdout.channel.recv_exit_status()

        # 关闭连接
        ssh.close()

        # 检查退出状态码
        if exit_status != 0:
            raise RuntimeError(f"标准错误输出: {error}")

        # 如果有stderr输出，仅作为警告显示
        if error:
            print(f"警告 - 命令产生了标准错误输出: {error}")

        return output

    except Exception as e:
        print(f"SSH连接或命令执行异常: ")
        raise


def wrap_command_with_env(command: str | list[str], env_dict: dict) -> str:
    env_parts = [f"{k}={shlex.quote(str(v))}" for k, v in env_dict.items()]
    env_export = "export " + " ".join(env_parts)

    if isinstance(command, list):
        cmd_str = shlex.join(command)
    else:
        cmd_str = command.strip()

    full_cmd = f"bash -c '{env_export} && {cmd_str}'"
    return full_cmd


class RayClusterManager:
    _instance = None
    _lock = threading.Lock()  # class-level lock for singleton creation

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, cluster_config: dict):
        self.global_mutex = threading.Lock()
        self.occupied_nodes = set()

        self.all_nodes: list[ClusterNode] = self.init_node(cluster_config)
        pprint(self.all_nodes)
        self.gpu_per_node = 8
        self.all_gpu_nums = len(self.all_nodes) * self.gpu_per_node

    def init_node(self, node_config: list | dict) -> ClusterNode | list[ClusterNode]:
        if isinstance(node_config, list):
            return [self.init_node(item) for item in node_config]

        return ClusterNode.from_dict(node_config)

    def start_ray_master(
        self, master: ClusterNode, extra_serve_env: dict | None = None
    ) -> str:
        extra_env = {
            "GLOO_SOCKET_IFNAME": master.nic,
            "MCCL_SOCKET_IFNAME": master.nic,
            "MACA_PATH": "/opt/maca",
            "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
        }
        extra_env.update(extra_serve_env)

        ray_start_cmd = f"""
            ray stop --force && \
            ray start --head --num-gpus={self.gpu_per_node}
            """
        remote_cmd = wrap_command_with_env(ray_start_cmd, extra_env)
        output = remote_command(master.ssh, remote_cmd)

        assert output

        ray_address = None
        for line in output.split("\n"):
            if "ray start --address=" in line:
                ray_address = line.strip().split("=")[1]
                break

        if ray_address and "Ray runtime started" in output:
            print(f"Ray started with address: {ray_address}")
            return ray_address

        return None

    def start_ray_slaves(
        self,
        ray_address: str,
        slaves: list[ClusterNode],
        extra_serve_env: dict | None = None,
    ):
        for slave in slaves:
            extra_env = {
                "GLOO_SOCKET_IFNAME": slave.nic,
                "NCCL_SOCKET_IFNAME": slave.nic,
                "MACA_PATH": "/opt/maca",
                "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
            }
            extra_env.update(extra_serve_env)

            ray_start_cmd = f"""
                ray stop --force && \
                ray start --address={ray_address} --num-gpus={self.gpu_per_node}
                """
            remote_cmd = wrap_command_with_env(ray_start_cmd, extra_env)
            output = remote_command(slave.ssh, remote_cmd)

            if not (output and "Ray runtime started" in output):
                raise RuntimeError(f"ray start error on slaves: {output}")

    def check_ray_cluster(self, num_required: int) -> bool:
        # sleep to wait ray status init
        time.sleep(5)

        cmd = "ray status"
        output = remote_command(self.master.ssh, cmd)

        assert output, f"chekc_ray_cluster failed, no output from `{cmd}`"

        match = re.search(r"(\d+)\.\d+\s*GPU", output)
        if match:
            integer_part = int(match.group(1))

        return integer_part >= num_required

    def get_free_nodes(self):
        free_list = [
            i for i in range(len(self.all_nodes)) if i not in self.occupied_nodes
        ]
        return free_list

    def start_ray_serve(self, nodes_list: list[int], extra_serve_env=None):
        assert len(nodes_list) > 0, (
            "start ray serve with empty node_list is not allowed"
        )

        master = self.all_nodes[nodes_list[0]]
        slaves = [self.all_nodes[i] for i in nodes_list[1:]]

        ray_address = self.start_ray_master(master, extra_serve_env)
        self.start_ray_slaves(ray_address, slaves, extra_serve_env)

    def allocate(self, num_required: int) -> list[int]:
        if num_required > self.all_gpu_nums:
            raise ValueError("Requested more GPUs than available on the system.")

        needed_nodes = (num_required + self.gpu_per_node - 1) // self.gpu_per_node
        assert self.gpu_per_node * needed_nodes >= num_required

        with self.global_mutex:
            free_nodes = self.get_free_nodes()
            if len(free_nodes) < needed_nodes:
                return []
            allocated_nodes = free_nodes[:needed_nodes]
            self.occupied_nodes.update(allocated_nodes)
            return allocated_nodes

    def release(self, related_nodes: list[int]):
        ray_stop_raw = "ray stop --force"

        with self.global_mutex:
            for i in related_nodes:
                remote_command(self.all_nodes[i].ssh, ray_stop_raw)
                self.occupied_nodes.discard(i)
