# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MetaX Engram tables for DeepSeek V4.1.

vLLM 0.32 implements the FP8 Engram path (E4M3 rows with UE8M0 block scales)
plus the host-offload / DP-shared storage helpers in
``vllm.models.deepseek_v41.common.engram``. MACA runs the INT8 (W4A8)
checkpoints, whose n-gram tables ship INT8 rows with one FP32 block scale per
32 columns, so this module subclasses the upstream classes and swaps only the
table layout and the lookup. Prefetch streams, DP gathering, huge pages and
the WKV path stay upstream.
"""

import os

import torch

from vllm.config import get_current_vllm_config
from vllm.distributed import get_engram_dp_group, get_engram_dp_size
from vllm.distributed.device_communicators.shm_broadcast import (
    SHM_PATH,
    check_shm_free_space,
)
from vllm.logger import init_logger
from vllm.models.deepseek_v41.common.engram import (
    DPSharedEngramStorage,
    EngramLayout,
    ParallelEngramEmbedding,
    _allocate_huge_page_storage,
    can_share_engram_tables,
)
from vllm.models.deepseek_v41.common.engram import Engram as BaseEngram
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.triton_utils import tl, triton

import vllm_metax.envs as mx_envs

from vllm_metax.model_executor.layers.online_rotation import (
    build_rotation,
    get_online_rotation_module_config,
)
from vllm_metax.model_executor.kernels.online_rotate import fused_engram_gate

logger = init_logger(__name__)


@triton.jit(
    do_not_specialize=[
        "vocab_start",
        "vocab_end",
        "num_rows",
        "ids_stride_t",
        "ids_stride_h",
        "GRID",
    ]
)
def _engram_lookup_kernel_int8(
    weight,
    scales,
    ids,
    out,
    vocab_start,
    vocab_end,
    num_rows,
    ids_stride_t,
    ids_stride_h,
    HEAD_START: tl.constexpr,
    LOCAL_HEADS: tl.constexpr,
    TOTAL_HEADS: tl.constexpr,
    DIM: tl.constexpr,
    QUANT_BLOCK: tl.constexpr,
    BLOCK_R: tl.constexpr,
    GRID,
):
    """Gather INT8 rows, apply their FP32 block scales, write BF16.

    Same shape/grid contract as the upstream FP8 lookup; only the
    dequantization differs (INT8 values times FP32 block scales instead of
    E4M3 times decoded UE8M0 exponents). Only this rank's heads are read;
    padded heads write zeros for the all-gather. ``weight``/``scales`` may
    address pinned host memory through UVA.
    """
    cols = tl.arange(0, DIM)
    scale_cols = cols // QUANT_BLOCK
    for base in tl.range(tl.program_id(0) * BLOCK_R, num_rows, GRID * BLOCK_R):
        rows = base + tl.arange(0, BLOCK_R)
        valid = rows < num_rows
        head = HEAD_START + rows % LOCAL_HEADS
        token = (rows // LOCAL_HEADS).to(tl.int64)
        index = tl.load(
            ids + token * ids_stride_t + head * ids_stride_h,
            mask=valid & (head < TOTAL_HEADS),
            other=-1,
        ).to(tl.int64)
        owned = valid & (head < TOTAL_HEADS)
        owned &= (index >= vocab_start) & (index < vocab_end)
        local = tl.where(owned, index - vocab_start, 0)
        values = tl.load(
            weight + local[:, None] * DIM + cols[None, :],
            mask=owned[:, None],
            other=0,
        )
        scale = tl.load(
            scales + local[:, None] * (DIM // QUANT_BLOCK) + scale_cols[None, :],
            mask=owned[:, None],
            other=0.0,
        )
        tl.store(
            out + rows[:, None] * DIM + cols[None, :],
            (values.to(tl.float32) * scale).to(tl.bfloat16),
            mask=valid[:, None],
        )


class MacaInt8DPSharedEngramStorage(DPSharedEngramStorage):
    """DP-shared host tables with INT8 rows and FP32 block scales.

    Reuses the upstream allocation, ``cudaHostRegister`` and finalizer code;
    only the byte layout of the single shared mapping differs from the FP8
    tables.
    """

    def __init__(self, num_rows: int, dim: int, block_size: int, group) -> None:
        self.group = group
        weight_bytes = num_rows * dim
        scale_bytes = num_rows * (dim // block_size) * 4
        storage = self._allocate(weight_bytes + scale_bytes)
        self.weight = storage[:weight_bytes].view(torch.int8).view(-1, dim)
        self.weight_scale_inv = (
            storage[weight_bytes:].view(torch.float32).view(-1, dim // block_size)
        )
        self._views: tuple[torch.Tensor, torch.Tensor] | None = None


class MacaInt8EngramEmbedding(ParallelEngramEmbedding):
    """Engram tables for the W4A8 checkpoints: INT8 rows + FP32 block scales.

    The W4A8 export ships ``engram.embed.weight`` (INT8) plus
    ``engram.embed.weight_scale`` (one FP32 scale per 32 columns). The check
    keeps the upstream ``weight_scale_inv`` parameter name; the checkpoint key
    is routed to it by ``weights_mapper_patch`` in the companion patch module.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.cpu_offload and hasattr(self.weight_scale_inv, "dummy_weight_value"):
            # Upstream fills the UE8M0 byte for scale 1.0 (127); these tables
            # store FP32 scales, whose identity is 1.0.
            self.weight_scale_inv.dummy_weight_value = 1.0

    def _allocate_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        scale_dim = self.dim // self.block_size
        if self.dp_shared_memory:
            group = get_engram_dp_group()
            assert group is not None
            storage = MacaInt8DPSharedEngramStorage(
                self.part_num_embeddings, self.dim, self.block_size, group
            )
            self._shared_memory = storage
            self._weight_loader = storage.load_weight
            return storage.weight, storage.weight_scale_inv
        if not self.cpu_offload:
            return (
                torch.empty(self.part_num_embeddings, self.dim, dtype=torch.int8),
                torch.empty(self.part_num_embeddings, scale_dim, dtype=torch.float32),
            )
        if self.use_thp:
            weight_bytes = self.part_num_embeddings * self.dim
            packed = _allocate_huge_page_storage(
                weight_bytes + self.part_num_embeddings * scale_dim * 4
            )
            if packed is not None:
                self._packed = packed
                return (
                    packed[:weight_bytes].view(torch.int8).view(-1, self.dim),
                    packed[weight_bytes:].view(torch.float32).view(-1, scale_dim),
                )
        # Model initialization may be inside a CUDA device context.
        return (
            torch.empty(
                self.part_num_embeddings,
                self.dim,
                dtype=torch.int8,
                device="cpu",
                pin_memory=True,
            ),
            torch.empty(
                self.part_num_embeddings,
                scale_dim,
                dtype=torch.float32,
                device="cpu",
                pin_memory=True,
            ),
        )

    def lookup(
        self, indices: torch.Tensor, out: torch.Tensor, background: bool = False
    ) -> None:
        """Look up local heads into BF16 rows.

        ``out`` may only be BF16: the packed FP8 path (``EngramBatch``, which
        hands ``uint8`` rows to a FlashInfer MXFP8 WKV) is FP8-only and is
        disabled on MACA because it needs fused MXFP8 query quantization.
        """
        if out.dtype != torch.bfloat16:
            raise NotImplementedError(
                "INT8 Engram tables can only be gathered as BF16 rows; the "
                "packed FP8 EngramBatch path requires an FP8 checkpoint."
            )
        rows = indices.shape[0] * self.part_n_hash_cols
        if not rows:
            return
        weight, scales = self._storage()
        # The table dwarfs TLB reach, so a persistent grid near the SM count
        # beats one program per row; halve it to leave SMs for the main stream.
        tiles = triton.cdiv(rows, 16)
        grid = min(tiles, self._num_sms // 2 if background else self._num_sms)
        _engram_lookup_kernel_int8[(grid,)](
            weight,
            scales,
            indices,
            out,
            self.vocab_start_idx,
            self.vocab_end_idx,
            rows,
            indices.stride(0),
            indices.stride(1),
            HEAD_START=self.head_start,
            LOCAL_HEADS=self.part_n_hash_cols,
            TOTAL_HEADS=self.n_hash_cols,
            DIM=self.dim,
            QUANT_BLOCK=self.block_size,
            BLOCK_R=16,
            GRID=grid,
        )


def maca_can_share_engram_tables(layout: EngramLayout, block_size: int = 32) -> bool:
    """Whether co-located DP replicas exist and /dev/shm can hold the tables.

    Same contract as the upstream check, but the INT8 tables store FP32 block
    scales (4 bytes per 32 columns) instead of UE8M0 bytes.
    """
    if mx_envs.VLLM_METAX_SUPPORTS_FP8:
        return can_share_engram_tables(layout, block_size)
    if get_engram_dp_size() == 1:
        logger.warning_once(
            "Engram DP replicas are not co-located on one node; "
            "storing the offloaded tables per rank instead of sharing them."
        )
        return False
    num_bytes = sum(layout.num_embeddings) * (
        layout.head_dim + (layout.head_dim // block_size) * 4
    )
    group = get_engram_dp_group()
    assert group is not None
    # Only the leader allocates, so every replica follows its check.
    error = None
    if group.rank_in_group == 0:
        if not os.path.isdir(SHM_PATH):
            error = f"{SHM_PATH} is not mounted"
        else:
            try:
                check_shm_free_space(num_bytes, allocation_name="Engram tables")
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
    error = group.broadcast_object(error)
    if error is not None:
        logger.warning_once(
            "Sharding the offloaded Engram tables across DP replicas: %s", error
        )
    return error is None


class MacaEngram(BaseEngram):
    """Upstream Engram with the checkpoint's n-gram table dtype."""

    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None,
        layout: EngramLayout,
        layer_hash_index: int,
        use_sequence_parallel: bool,
        prefix: str,
        prefetch_stream: torch.cuda.Stream | None = None,
    ) -> None:
        super().__init__(
            config,
            quant_config,
            layout,
            layer_hash_index,
            use_sequence_parallel,
            prefix,
            prefetch_stream,
        )

        layer_id = layout.layer_ids[layer_hash_index]
        rotation_name = f"model.layers.{layer_id}.engram.global_rotation"
        rotation_config = get_online_rotation_module_config(config, rotation_name)

        if rotation_config is not None:
            self.global_rotation = build_rotation(rotation_config)
        elif not hasattr(self, "global_rotation"):
            self.global_rotation = None

    def _create_embedding(
        self, layout: EngramLayout, layer_hash_index: int
    ) -> ParallelEngramEmbedding:
        engram_config = get_current_vllm_config().engram_config
        assert engram_config is not None
        use_fp8 = mx_envs.VLLM_METAX_SUPPORTS_FP8
        embedding_cls = ParallelEngramEmbedding if use_fp8 else MacaInt8EngramEmbedding
        logger.info_once(
            "Engram n-gram tables use %s (VLLM_METAX_SUPPORTS_FP8=%d)",
            embedding_cls.__name__,
            int(use_fp8),
        )
        return embedding_cls(
            layout.num_embeddings[layer_hash_index],
            layout.head_dim,
            tuple(size for order in layout.primes[layer_hash_index] for size in order),
            cpu_offload=engram_config.cpu_offload,
            dp_shared_memory=bool(engram_config.dp_shared_memory),
            use_thp=engram_config.use_thp,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        hash_ids: torch.Tensor,
        token_mask: torch.Tensor | None = None,
        gate_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if getattr(self, "global_rotation", None) is None and gate_input is None:
            return super().forward(hidden_states, hash_ids, token_mask)

        emb = self.embed_tokens(hash_ids)
        wkv_out = self.wkv(emb.flatten(-2))
        kv = wkv_out[0] if isinstance(wkv_out, tuple) else wkv_out

        if kv.dtype not in (torch.bfloat16, torch.float32):
            kv = kv.to(torch.bfloat16)
        kv = kv.contiguous()

        if gate_input is not None:
            gate_input = gate_input.contiguous()

        out = fused_engram_gate(
            hidden_states.contiguous(),
            kv,
            self.q_weight,
            self.k_weight,
            self.eps,
            self.clamp_value,
            gate_input=gate_input,
        )

        if token_mask is not None:
            out = torch.where(token_mask.view(-1, 1, 1), out, hidden_states)

        return out
