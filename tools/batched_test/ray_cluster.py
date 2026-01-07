# SPDX-License-Identifier: Apache-2.0
import subprocess
import paramiko
import time
import os
from dataclasses import dataclass, field, asdict

dataclass


class SSHConfig:
    ip: str
    port: int
    user: str
    auth_type: str
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
        """转换为字典"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SSHConfig":
        """从字典创建实例"""
        return cls(**data)


@dataclass
class ClusterNode:
    ssh: SSHConfig
    nic: str

    def to_dict(self) -> dict:
        """转换为字典"""
        return {"ssh": self.ssh.to_dict(), "nic": self.nic}

    @classmethod
    def from_dict(cls, data: dict) -> "ClusterNode":
        """从字典创建实例"""
        return cls(ssh=SSHConfig.from_dict(data["ssh"]), nic=data.get("nic", "eth0"))


def remote_command(ssh_info: SSHConfig, command: str) -> str:
    """
    Run remote command via ssh
    """
    print(f"[远程:{ssh_info.ip}:{ssh_info.port}] 执行命令: {command}")

    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        # 连接服务器
        if ssh_info.auth_type == "key":
            # 使用密钥认证
            key_path = os.path.expanduser(ssh_info.private_key)
            private_key = paramiko.RSAKey.from_private_key_file(key_path)
            ssh.connect(
                hostname=ssh_info.host,
                port=ssh_info.port,
                username=ssh_info.user,
                pkey=private_key,
                timeout=10,
            )
        elif ssh_info.auth_type == "password":
            # 使用密码认证
            ssh.connect(
                hostname=ssh_info.host,
                port=ssh_info.port,
                username=ssh_info.user,
                password=ssh_info.password,
                timeout=10,
            )

        # 执行命令
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


# ------------------------------
# Ray集群部署
# ------------------------------


class RayClusterManager:
    def __init__(self, cluster_config: dict):
        self.master: ClusterNode = self.init_node(cluster_config["master"])
        self.slaves: list[ClusterNode] = self.init_node(cluster_config["slaves"])
        self.gpu_pre_node = cluster_config["gpu_per_node"]

    def init_node(self, node_config: dict) -> ClusterNode | list[ClusterNode]:
        if isinstance(node_config, list):
            return [self.init_node(item) for item in node_config]

        ssh_config = node_config.get("ssh_config")
        ray_config = node_config.get("ray_config")

        ssh = SSHConfig(
            ip=ssh_config.get("ssh_hostname"),
            port=ssh_config.get("ssh_port"),
            user=ssh_config.get("user"),
            auth_type=ssh_config.get("auth_type"),
            password=ssh_config.get("ssh_password"),
            private_key=ssh_config.get("ssh_key"),
        )

        return ClusterNode(ssh=ssh, nic=ray_config.get("nic"))

    def start_ray_head(self, CLUSTER_CONFIG: dict) -> str:
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

        assert output is not None

        # 提取Ray集群地址
        for line in output.split("\n"):
            if "Ray runtime started" in line:
                # 解析ray address
                ray_address_line = [
                    l for l in output.split("\n") if "ray start --address=" in l
                ]
                if ray_address_line:
                    ray_address = ray_address_line[0].strip().split("=")[1]
                    print(f"Ray头节点已启动，集群地址: {ray_address}")
                    return ray_address

        print("无法获取Ray集群地址")
        return None

    def start_ray_workers(self, ray_address: str) -> bool:
        """
        启动Ray工作节点
        :param ray_address: Ray集群地址
        :return: 是否所有工作节点都成功启动
        """
        success = True
        for i, cluster_node in enumerate(self.slaves):
            ray_start_cmd = f"""
                export GLOO_SOCKET_IFNAME={cluster_node.nic} && \
                export NCCL_SOCKET_IFNAME={cluster_node.nic} && \
                export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1 && \
                export MACA_PATH=/opt/maca && \
                ray start --address={ray_address} --num-gpus={self.gpu_per_node}
                """
            output = remote_command(cluster_node.ssh, ray_start_cmd)

            if output and "Ray runtime started" in output:
                print(
                    f"Ray工作节点 {i + 1} ({cluster_node.ssh.host}:{cluster_node.ssh.port}) 已启动"
                )
            else:
                print(
                    f"Ray工作节点 {i + 1} ({cluster_node.ssh.host}:{cluster_node.ssh.port}) 启动失败"
                )
                success = False

        return success

    def check_ray_cluster(self) -> bool:
        """
        检查Ray集群状态
        :return: 集群是否正常
        """
        time.sleep(5)

        # 获取Ray节点列表
        cmd = "ray status"
        output = remote_command(self.master.ssh, cmd)

        if not output:
            return False

        print("Ray集群状态:")
        print(output)

        # 检查节点数量
        num_nodes = 1 + len(self.slaves)
        num_gpus = num_nodes * self.gpu_per_node

        if f"{float(num_gpus)} GPU" in output:
            print(f"Ray集群配置正确: {num_nodes}个节点，{num_gpus}张GPU")
            return True
        else:
            print("Ray集群配置不正确")
            return False

    def ray_init(self, node_cfg):
        print("=" * 60)
        print("VLLM多机部署自动化脚本")
        print("=" * 60)

        # 1. 启动Ray头节点
        print("\n1. 启动Ray头节点...")
        ray_address = self.start_ray_head(node_cfg)
        if not ray_address:
            raise RuntimeError("Ray头节点启动失败，退出脚本")

        # 2. 启动Ray工作节点
        print("\n2. 启动Ray工作节点...")
        if not self.start_ray_workers(ray_address, node_cfg):
            raise RuntimeError("Ray工作节点启动失败，退出脚本")

        # 3. 检查Ray集群状态
        print("\n3. 检查Ray集群状态...")
        if not self.check_ray_cluster(node_cfg):
            raise RuntimeError("Ray集群状态异常，退出脚本")
