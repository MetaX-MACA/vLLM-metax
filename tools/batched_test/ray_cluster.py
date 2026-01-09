# SPDX-License-Identifier: Apache-2.0
import paramiko
import time
import os
from dataclasses import dataclass, asdict
from typing import Literal
import regex as re


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
        return cls(ssh=SSHConfig.from_dict(data["ssh"]), nic=data.get("nic", "eth0"))


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


class RayClusterManager:
    def __init__(self, cluster_config: dict):
        self.master: ClusterNode = self.init_node(cluster_config["master"])
        self.slaves: list[ClusterNode] = []

        if slave_nodes := cluster_config.get("slaves", None):
            self.slaves = self.init_node(slave_nodes)

        self.gpu_pre_node = cluster_config["gpu_per_node"]
        self.all_gpu_nums = (1 + len(self.slaves)) * self.gpu_pre_node

    def init_node(self, node_config: dict) -> ClusterNode | list[ClusterNode]:
        if isinstance(node_config, list):
            return [self.init_node(item) for item in node_config]

        ssh_config = node_config.get("ssh")
        ray_config = node_config.get("ray")

        ssh = SSHConfig(
            hostname=ssh_config.get("hostname"),
            port=ssh_config.get("port"),
            user=ssh_config.get("user"),
            auth_type=ssh_config.get("auth_type"),
            password=ssh_config.get("ssh_password"),
            private_key=ssh_config.get("ssh_key"),
        )

        return ClusterNode(ssh=ssh, nic=ray_config.get("nic"))

    def start_ray_head(self) -> str:
        """
        启动Ray头节点
        :return: Ray集群地址
        """

        # 启动Ray头节点，设置GLOO_SOCKET_IFNAME环境变量
        ray_start_cmd = f"""
            ray stop --force && \
            export GLOO_SOCKET_IFNAME={self.master.nic} && \
            export NCCL_SOCKET_IFNAME={self.master.nic} && \
            export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1 && \
            export MACA_PATH=/opt/maca && \
            ray start --head --num-gpus={self.gpu_pre_node}
            """
        output = remote_command(self.master.ssh, ray_start_cmd)

        assert output

        # 提取Ray集群地址
        ray_address = None
        for line in output.split("\n"):
            if "ray start --address=" in line:
                ray_address = line.strip().split("=")[1]
                break

        if ray_address and "Ray runtime started" in output:
            print(f"Ray头节点已启动, 集群地址: {ray_address}")
            return ray_address

        print("无法获取Ray集群地址")
        return None

    def start_ray_workers(self, ray_address: str, num_of_slaves: int) -> list[int]:
        slave_indices = []
        for i in range(num_of_slaves):
            cluster_node = self.slaves[i]

            ray_start_cmd = f"""
                ray stop --force && \
                export GLOO_SOCKET_IFNAME={cluster_node.nic} && \
                export NCCL_SOCKET_IFNAME={cluster_node.nic} && \
                export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1 && \
                export MACA_PATH=/opt/maca && \
                ray start --address={ray_address} --num-gpus={self.gpu_per_node}
                """
            output = remote_command(cluster_node.ssh, ray_start_cmd)

            if not (output and "Ray runtime started" in output):
                raise RuntimeError(f"ray start error on slaves: {output}")

            slave_indices.append(i + 1)  # start from 1 since 0 is for master

        return slave_indices

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

    def allocate(self, num_required: int) -> list[int]:
        if num_required > self.all_gpu_nums:
            raise RuntimeError(
                f"Out of cards, needs {num_required},but got {{self.all_gpu_nums}}"
            )

        needed_slaves = (num_required + self.gpu_pre_node - 1) // self.gpu_pre_node - 1

        assert self.gpu_pre_node * (needed_slaves + 1) >= num_required
        assert len(self.slaves) >= needed_slaves

        ray_address = self.start_ray_head()
        slave_indices = self.start_ray_workers(ray_address, needed_slaves)

        assert self.check_ray_cluster(num_required), (
            "check_ray_cluster failed, no enough gpus on cluster"
        )

        # TODO(hank): add mutex here to allocate nodes
        return [0] + slave_indices

    def release(self, related_nodes: list[int]):
        ray_stop_raw = "ray stop --force"
        remote_command(self.master.ssh, ray_stop_raw)
        for _, cluster_node in enumerate(self.slaves):
            remote_command(cluster_node.ssh, ray_stop_raw)
