# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#
# -----------------------------------------------------------------------------
# Note: Own static EPLB placement import, native weight transfers and load export.
#     Target EPLB has no static file lifecycle; reuse its policy and map commits.
#
# Affected versions: Reviewed vLLM v0.33.0 (4ba787ec9959).
#
# Remove at: Native EPLB supports static placement import and synchronized export.
# -----------------------------------------------------------------------------
"""Manage a fixed expert placement and its optional load-collection interval."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from vllm.distributed.eplb.eplb_communicator import (
    NixlEplbCommunicator,
    TorchDistGlooStagedEplbCommunicator,
)
from vllm.distributed.eplb.eplb_state import _commit_eplb_maps
from vllm.distributed.eplb.rebalance_execute import (
    move_from_buffer,
    move_to_buffer,
    rearrange_expert_weights_inplace,
)
from vllm.distributed.parallel_state import get_ep_group, get_eplb_group, get_node_count
from vllm.logger import init_logger
from vllm.utils.gpu_sync_debug import gpu_sync_allowed

from .placement import (
    map_from_payload,
    read_eplb_file,
    select_model_payload,
    validate_physical_to_logical_map,
    write_eplb_file,
)

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup
    from vllm.config import ParallelConfig
    from vllm.distributed.eplb.eplb_state import EplbModelState, EplbState

logger = init_logger(__name__)


class StaticEplbRecorder:
    """Own import, fixed-placement serving, and a one-shot placement export."""

    IMPORT_ENV = "VLLM_METAX_EPLB_IMPORT_PATH"
    EXPORT_ENV = "VLLM_METAX_EPLB_EXPORT_PATH"
    STEPS_ENV = "VLLM_METAX_EPLB_EXPORT_STEPS"

    def __init__(
        self,
        *,
        import_path: str | None = None,
        export_path: str | None = None,
        steps: int = 1000,
    ) -> None:
        if (
            import_path
            and export_path
            and os.path.realpath(import_path) == os.path.realpath(export_path)
        ):
            raise ValueError("EPLB import and export paths must differ")
        if export_path:
            if Path(export_path).suffix not in (".json", ".pt"):
                raise ValueError("EPLB export file must end in .json or .pt")
            if steps <= 0:
                raise ValueError(f"{self.STEPS_ENV} must be positive")
        self.import_path = import_path
        self.export_path = export_path
        self.steps = steps
        self.count = 0
        self.eplb_state: EplbState | None = None
        self.totals: dict[str, torch.Tensor] = {}
        # Read before model loading; dimensions are checked after all MoE models register.
        self.import_payload = read_eplb_file(import_path) if import_path else None

    @classmethod
    def is_enabled(cls) -> bool:
        return bool(os.getenv(cls.IMPORT_ENV) or os.getenv(cls.EXPORT_ENV))

    @classmethod
    def from_env(cls, parallel_config: ParallelConfig) -> StaticEplbRecorder | None:
        import_path = os.getenv(cls.IMPORT_ENV)
        export_path = os.getenv(cls.EXPORT_ENV)
        if not import_path and not export_path:
            return None
        if not parallel_config.enable_eplb:
            raise ValueError(
                "Static EPLB requires --enable-expert-parallel and --enable-eplb"
            )
        if parallel_config.enable_elastic_ep:
            raise ValueError("Static EPLB does not support elastic EP")
        steps = int(os.getenv(cls.STEPS_ENV, "1000")) if export_path else 1000
        return cls(import_path=import_path, export_path=export_path, steps=steps)

    @property
    def collecting(self) -> bool:
        return bool(self.export_path) and self.count < self.steps

    def _state(self) -> EplbState:
        if self.eplb_state is None:
            raise RuntimeError("Static EPLB recorder has not been attached")
        return self.eplb_state

    def attach(self, eplb_state: EplbState | None) -> None:
        """Apply any imported placement after both target and draft register."""
        if eplb_state is None or not eplb_state.model_states:
            raise ValueError("Static EPLB requires a model with native EPLB support")
        self.eplb_state = eplb_state
        if self.import_payload is not None:
            assert self.import_path is not None
            self.apply_maps(self.import_payload, path=self.import_path)
        self.clear_loads()
        if self.export_path:
            self.totals = {
                key: torch.zeros_like(state.expert_load_pass, dtype=torch.int64)
                for key, state in eplb_state.model_states.items()
            }
        eplb_state._metax_static_eplb_recorder = self
        self._set_recording(self.collecting)

    def _set_recording(self, enabled: bool) -> None:
        flag = self._state().should_record_tensor
        if flag is not None:
            flag.fill_(enabled)

    def clear_loads(self) -> None:
        """Discard profile/dummy load without changing collected serving totals."""
        for state in self._state().model_states.values():
            state.expert_load_pass.zero_()

    @staticmethod
    def _buffers_match_weights(weights: Any, buffers: Any) -> bool:
        return len(weights) == len(buffers) and all(
            weight.shape == buffer.shape
            and weight.dtype == buffer.dtype
            and weight.device == buffer.device
            for weight, buffer in zip(weights, buffers, strict=False)
        )

    def _rearrange_expert_weights(
        self, model_state: EplbModelState, mapping: torch.Tensor, group: ProcessGroup
    ) -> None:
        expert_weights = model_state.model.expert_weights
        buffers = model_state.expert_buffer
        current = model_state.physical_to_logical_map
        communicator = model_state.communicator
        if all(
            self._buffers_match_weights(weights, buffers) for weights in expert_weights
        ):
            rearrange_expert_weights_inplace(
                current, mapping, expert_weights, buffers, group, communicator
            )
            return

        if group.rank() == 0:
            logger.info(
                "Using layer-specific static EPLB staging buffers for %s",
                model_state.model_name,
            )
        # NIXL registers the original buffer addresses. Temporary layouts use Gloo.
        if isinstance(communicator, NixlEplbCommunicator):
            communicator = TorchDistGlooStagedEplbCommunicator(
                cpu_group=get_eplb_group().cpu_group
            )
            if group.rank() == 0:
                logger.info("Using Gloo for mixed-layout static EPLB import")
        old_indices = current.cpu()
        new_indices = mapping.cpu()
        for layer_idx, weights in enumerate(expert_weights):
            if torch.equal(old_indices[layer_idx], new_indices[layer_idx]):
                continue
            if not self._buffers_match_weights(weights, buffers):
                buffers = [torch.empty_like(weight) for weight in weights]
            metadata = move_to_buffer(
                num_local_experts=weights[0].shape[0],
                old_indices=old_indices[layer_idx].numpy(),
                new_indices=new_indices[layer_idx].numpy(),
                expert_weights=weights,
                expert_weights_buffers=buffers,
                stream=None,
                ep_rank=group.rank(),
                communicator=communicator,
                layer_idx=layer_idx,
            )
            move_from_buffer(
                expert_weights=weights,
                expert_weights_buffers=buffers,
                transfer_metadata=metadata,
                new_indices=new_indices[layer_idx].numpy(),
                ep_rank=group.rank(),
            )

    def apply_maps(self, payload: dict[str, Any], *, path: str = "<memory>") -> None:
        """Validate every model before moving weights, then commit native maps."""
        eplb_state = self._state()
        states = list(eplb_state.model_states.values())
        group = get_ep_group().device_group
        num_nodes = get_node_count()
        if group.size() % num_nodes != 0:
            num_nodes = 1
        mappings = []
        for index, state in enumerate(states):
            model_payload = select_model_payload(
                payload,
                model_name=state.model_name,
                model_index=index,
                model_count=len(states),
            )
            model = state.model
            mappings.append(
                map_from_payload(
                    model_payload,
                    num_layers=model.num_moe_layers,
                    num_physical_experts=model.num_physical_experts,
                    num_logical_experts=model.num_logical_experts,
                    ep_size=group.size(),
                    num_groups=model.num_expert_groups,
                    num_nodes=num_nodes,
                    old_map=state.physical_to_logical_map,
                    policy=eplb_state.policy,
                )
            )
        for state, mapping in zip(states, mappings, strict=False):
            if torch.equal(state.physical_to_logical_map.cpu(), mapping):
                continue
            self._rearrange_expert_weights(state, mapping, group)
            _commit_eplb_maps(state, mapping)
        torch.distributed.barrier(group=group)
        if group.rank() == 0:
            logger.info("Loaded static EPLB placement from %s", path)

    def step(
        self,
        is_dummy: bool = False,
        is_profile: bool = False,
        log_stats: bool = False,
    ) -> None:
        """Own static steps, including native profiling and dummy-load exclusion.

        Native periodic balancedness logging is not scheduled in static mode;
        ``log_stats`` is accepted as part of the shared step interface.
        """
        if is_dummy:
            self.clear_loads()
        if is_profile:
            # Reuse native buffer reservation without re-entering the patched step.
            self._state().rearrange(is_profile=True)
            self.clear_loads()
            return
        if not self.collecting:
            self.clear_loads()
            return
        for key, state in self._state().model_states.items():
            self.totals[key].add_(state.expert_load_pass)
            state.expert_load_pass.zero_()
        self.count += 1
        if self.count == self.steps:
            self.export_placement()
            self._set_recording(False)

    def export_placement(self) -> None:
        """Write a policy-generated placement; leave active weights/maps unchanged."""
        if not self.export_path:
            raise ValueError("Static EPLB export requires an export path")
        eplb_state = self._state()
        states = list(eplb_state.model_states.items())
        group = get_ep_group().device_group
        global_loads = [
            eplb_state._allreduce_list([self.totals[key].clone()])[0]
            for key, _ in states
        ]
        num_nodes = get_node_count()
        if group.size() % num_nodes != 0:
            num_nodes = 1
        records = []
        for (_, state), physical_load in zip(states, global_loads, strict=False):
            model = state.model
            logical_load = torch.zeros(
                (model.num_moe_layers, model.num_logical_experts),
                dtype=physical_load.dtype,
                device=physical_load.device,
            )
            logical_load.scatter_add_(
                1, state.physical_to_logical_map.long(), physical_load
            )
            # NOTE(metax): Export is a one-time host policy evaluation.
            with gpu_sync_allowed():
                placement = eplb_state.policy.rebalance_experts(
                    logical_load.cpu(),
                    model.num_physical_experts,
                    model.num_expert_groups,
                    num_nodes,
                    group.size(),
                    state.physical_to_logical_map.cpu(),
                )
            placement = validate_physical_to_logical_map(
                placement,
                num_layers=model.num_moe_layers,
                num_physical_experts=model.num_physical_experts,
                num_logical_experts=model.num_logical_experts,
            )
            records.append(
                {"model_name": state.model_name, "physical_to_logical_map": placement}
            )
        # A single-model file is also accepted by SGLang's init_expert_location.
        payload = (
            {"physical_to_logical_map": records[0]["physical_to_logical_map"]}
            if len(records) == 1
            else {"models": records}
        )
        if group.rank() == 0:
            write_eplb_file(self.export_path, payload)
            logger.info(
                "Exported static EPLB placement to %s after %d synchronized steps",
                self.export_path,
                self.count,
            )
