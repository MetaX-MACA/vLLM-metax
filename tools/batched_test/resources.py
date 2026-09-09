# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Allocation adapters: only this layer interprets manager allocation IDs."""

from dataclasses import dataclass
from typing import Callable, Protocol


@dataclass(frozen=True)
class NodeAllocation:
    # Index in the backend node inventory; local backend uses zero.
    node_id: int
    # Reachable node address used for distributed initialization.
    hostname: str
    # Physical device indices on this node before CUDA_VISIBLE_DEVICES remapping.
    gpu_ids: tuple[int, ...]
    # Immutable node-specific communication and runtime environment.
    env: tuple[tuple[str, str], ...] = ()


@dataclass
class ResourceLease:
    # Allocated nodes and their GPUs, ordered with local rank0 first.
    nodes: tuple[NodeAllocation, ...]
    # Returns the reservation to its backend after processes have stopped.
    release_callback: Callable[[], None]
    # Whether to emit multi-node mp flags, including for a one-node cluster run.
    distributed: bool = False
    # Set only after successful release to make subsequent release calls harmless.
    released: bool = False

    def release(self):
        if not self.released:
            self.release_callback()
            self.released = True


class ResourceBackend(Protocol):
    def allocate(self, required_gpus: int) -> ResourceLease | None: ...
    def start_remote(
        self, node_id: int, cmd: list[str], env: dict, log: str
    ) -> Callable[[], None]: ...


class LocalBackend:
    def __init__(self, manager):
        # Batch-owned GPUManager providing atomic local device reservations.
        self.manager = manager

    def allocate(self, required_gpus: int) -> ResourceLease | None:
        ids = self.manager.allocate(required_gpus)
        if not ids:
            return None
        return ResourceLease(
            (NodeAllocation(0, "localhost", tuple(ids)),),
            lambda: self.manager.release(ids),
        )

    def start_remote(self, node_id, cmd, env, log):
        raise RuntimeError("Local backend cannot launch a remote rank")


class ClusterBackend:
    def __init__(self, manager):
        # Batch-owned MPClusterManager providing node reservations and SSH operations.
        self.manager = manager

    def allocate(self, required_gpus: int) -> ResourceLease | None:
        ids = self.manager.allocate(required_gpus)
        if not ids:
            return None
        try:
            if ids[0] != 0:
                raise RuntimeError(
                    "Cluster rank0 must be the local first configured node"
                )
            nodes = []
            remaining = required_gpus
            for node_id in ids:
                count = min(self.manager.gpu_per_node, remaining)
                nodes.append(
                    NodeAllocation(
                        node_id,
                        self.manager.get_node_hostname(node_id),
                        tuple(range(count)),
                        tuple(
                            (str(k), str(v))
                            for k, v in self.manager.get_base_env(node_id).items()
                        ),
                    )
                )
                remaining -= count
            return ResourceLease(
                tuple(nodes), lambda: self.manager.release(ids), distributed=True
            )
        except BaseException:
            self.manager.release(ids)
            raise

    def start_remote(self, node_id, cmd, env, log):
        pid = self.manager.start_headless_rank(node_id, cmd, env, log)
        return lambda: self.manager.stop_headless_rank(node_id, pid)
