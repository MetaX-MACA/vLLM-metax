# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# -----------------------------------------------------------------------------
# Note: Small-batch MoE-LoRA checks negative expert IDs but lacks the upper bound.
#     Pass the weight expert count before forming expert-specific weight pointers.
#
# Affected versions: vLLM 0.33.0 (4ba787ec9959).
#
# Remove at: The upstream small-batch kernel checks both expert bounds.
# -----------------------------------------------------------------------------
"""Bound expert weight accesses while preserving the upstream persistent kernel."""

import torch

from vllm.lora.ops.triton_ops.fused_moe_lora_op import _get_ptr, _pick_small_batch_chunk
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm_metax.patch.utils import patch


@patch(
    "vllm.lora.ops.triton_ops.fused_moe_lora_op", "_fused_moe_lora_small_batch_kernel"
)
@triton.heuristics({"EVEN_K": lambda args: args["K"] % args["BLOCK_K"] == 0})
@triton.jit
def _fused_moe_lora_small_batch_kernel(
    # ---- pointers ----
    x_ptr,
    A_ptrs,
    B_ptrs,
    out_ptr,
    topk_weights_ptr,
    expert_ids_ptr,  # (num_tokens * top_k_num,)
    token_lora_mapping_ptr,  # (num_tokens,)
    adapter_enabled_ptr,
    # ---- dims ----
    N,
    K,
    top_k_num,
    max_loras,
    work_total,  # = pair_slices * n_chunks_per_pair_slice
    pair_slices,  # = num_tokens * top_k_num * NUM_SLICES
    # ---- strides ----
    stride_xm,
    stride_xk,
    stride_A_lora,
    stride_A_expert,
    stride_A_r,
    stride_A_k,
    stride_B_lora,
    stride_B_expert,
    stride_B_n,
    stride_B_r,
    stride_om,
    stride_on,
    # ---- scalar (runtime ints, NOT constexpr) ----
    # n_tiles_per_program / n_chunks_per_pair_slice are deliberately
    # runtime: each distinct value would otherwise trigger a fresh Triton
    # compile -> fresh kernel binary -> fresh CUDA graph instance per
    # batch size. Production traces showed that variant explosion adding
    # ~5.9k graph instantiations on top of legacy. Runtime args mean one
    # shared binary across all chunk sizes.
    slice_n_offset,
    n_tiles_per_program,
    n_chunks_per_pair_slice,
    # ---- constexpr ----
    token_mapping_factor: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    ADD_INPUTS: tl.constexpr,
    BLOCK_R: tl.constexpr,
    actual_rank: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_SLICES: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    """Persistent fused MoE-LoRA kernel for naive_block_assignment inputs.

    Each program owns one (pair × slice × n_chunk) work item. A "chunk"
    covers `n_tiles_per_program` consecutive output-N tiles, all of which
    share a single shrink — so the rank-vector is computed once per
    program and the A weights for that (lora, expert, slice) are loaded
    once instead of n_tiles_per_program times.

    The wrapper picks `n_tiles_per_program` to keep the grid close to
    2*SM_count: at very small batch (work_total ≤ SM_count) the chunk
    size collapses to 1 and behaviour matches a per-tile GEMV; as batch
    grows the chunk grows so we trade some N-axis parallelism for shrink
    reuse. When `work_total` exceeds the launched grid, the outer stride
    loop drains the leftover work units serially.
    """
    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)

    offs_r = tl.arange(0, BLOCK_R)
    rank_mask = offs_r < actual_rank
    # Clamp OOB rank lanes so they address row 0 of A/B; the mask zeros
    # the loaded values. Required when BLOCK_R > actual_rank (e.g. rank=4
    # padded to 16) -- without clamping, tl.load would address the next
    # expert's memory.
    safe_offs_r = tl.where(rank_mask, offs_r, 0)
    offs_k = tl.arange(0, BLOCK_K)

    # Persistent stride loop: when grid < work_total each program walks
    # multiple work items. When grid == work_total the loop runs exactly
    # once and the kernel degenerates to the per-tile GEMV.
    for work_id in range(pid, work_total, num_programs):
        n_chunk_idx = work_id % n_chunks_per_pair_slice
        pair_slice_idx = work_id // n_chunks_per_pair_slice
        # NUM_SLICES is constexpr (typ. 1 or 2) so divmod folds.
        pair_idx = pair_slice_idx // NUM_SLICES
        slice_id = pair_slice_idx % NUM_SLICES

        # Resolve lora_id / expert_id; skip the body for inactive lanes.
        # Using a single `valid` flag instead of early `return` keeps the
        # outer stride loop alive — `return` would exit the whole program
        # and skip later work items assigned to this SM.
        token_idx = pair_idx // top_k_num
        lora_id = tl.load(token_lora_mapping_ptr + token_idx)
        valid = (lora_id >= 0) & (lora_id < max_loras)
        enabled = tl.load(adapter_enabled_ptr + tl.where(valid, lora_id, 0))
        valid = valid & (enabled != 0)
        expert_id = tl.load(expert_ids_ptr + pair_idx)
        valid = valid & (expert_id >= 0) & (expert_id < NUM_EXPERTS)

        if valid:
            cur_A_ptr = tl.load(A_ptrs + slice_id).to(
                tl.pointer_type(out_ptr.dtype.element_ty)
            )
            cur_B_ptr = tl.load(B_ptrs + slice_id).to(
                tl.pointer_type(out_ptr.dtype.element_ty)
            )
            A_base = cur_A_ptr + lora_id * stride_A_lora + expert_id * stride_A_expert
            B_base = cur_B_ptr + lora_id * stride_B_lora + expert_id * stride_B_expert

            x_row = pair_idx // token_mapping_factor
            x_row_ptr = x_ptr + x_row * stride_xm

            # SHRINK GEMV (once per program; reused across n_tiles_per_program
            # expand tiles below). Sum-reduction over BLOCK_K with fp32
            # accumulator — same precision path as the one_shot kernel.
            rank_vec = tl.zeros((BLOCK_R,), dtype=tl.float32)
            if EVEN_K:
                for kb in range(0, K, BLOCK_K):
                    cur_k = kb + offs_k
                    x_tile = tl.load(x_row_ptr + cur_k * stride_xk).to(tl.float32)
                    a_tile = tl.load(
                        A_base
                        + safe_offs_r[:, None] * stride_A_r
                        + cur_k[None, :] * stride_A_k,
                        mask=rank_mask[:, None],
                        other=0.0,
                    ).to(tl.float32)
                    rank_vec += tl.sum(a_tile * x_tile[None, :], axis=1)
            else:
                for kb in range(0, K, BLOCK_K):
                    cur_k = kb + offs_k
                    k_mask = cur_k < K
                    x_tile = tl.load(
                        x_row_ptr + cur_k * stride_xk, mask=k_mask, other=0.0
                    ).to(tl.float32)
                    a_tile = tl.load(
                        A_base
                        + safe_offs_r[:, None] * stride_A_r
                        + cur_k[None, :] * stride_A_k,
                        mask=rank_mask[:, None] & k_mask[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    rank_vec += tl.sum(a_tile * x_tile[None, :], axis=1)

            # EXPAND: walk n_tiles_per_program consecutive output-N tiles
            # using the same rank_vec. The loop is a runtime range (not
            # tl.static_range) so a single compiled kernel handles every
            # chunk size — see the note on the kernel signature.
            n_tile_start = n_chunk_idx * n_tiles_per_program
            out_row_ptr = out_ptr + slice_id * slice_n_offset + pair_idx * stride_om

            if MUL_ROUTED_WEIGHT:
                moe_w = tl.load(topk_weights_ptr + pair_idx).to(tl.float32)

            for nt in range(n_tiles_per_program):
                n_lo = (n_tile_start + nt) * BLOCK_N
                if n_lo < N:
                    offs_n = n_lo + tl.arange(0, BLOCK_N)
                    n_mask = offs_n < N
                    b_tile = tl.load(
                        B_base
                        + offs_n[:, None] * stride_B_n
                        + safe_offs_r[None, :] * stride_B_r,
                        mask=n_mask[:, None] & rank_mask[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    out_tile = tl.sum(b_tile * rank_vec[None, :], axis=1)

                    if MUL_ROUTED_WEIGHT:
                        out_tile = out_tile * moe_w

                    out_ptrs = out_row_ptr + offs_n * stride_on
                    if ADD_INPUTS:
                        prev = tl.load(out_ptrs, mask=n_mask, other=0.0).to(tl.float32)
                        tl.store(
                            out_ptrs,
                            (prev + out_tile).to(out_ptr.dtype.element_ty),
                            mask=n_mask,
                        )
                    else:
                        tl.store(
                            out_ptrs,
                            out_tile.to(out_ptr.dtype.element_ty),
                            mask=n_mask,
                        )


@patch("vllm.lora.ops.triton_ops.fused_moe_lora_op", "_run_fused_moe_lora_small_batch")
def _run_fused_moe_lora_small_batch(
    output: torch.Tensor,
    qcurr_hidden_states: torch.Tensor,
    lora_a_stacked: list[torch.Tensor],
    lora_b_stacked: list[torch.Tensor],
    topk_weights: torch.Tensor,
    expert_ids_flat: torch.Tensor,  # (num_tokens * top_k_num,)
    token_lora_mapping: torch.Tensor,
    top_k_num: int,
    adapter_enabled: torch.Tensor,
    mul_routed_weight: bool,
    add_inputs: bool = True,
) -> None:
    """Small-batch GEMV-style wrapper. Naive-block-assignment inputs only.

    Shape contract matches `_run_fused_moe_lora_one_shot`: `output` is
    `(num_tokens, top_k_num, num_slices * N_per_slice)` with
    contiguous-style strides, `expert_ids_flat` is the flattened
    `topk_ids` of shape `(num_tokens * top_k_num,)`, and the
    rank-padded LoRA weights live in `lora_a_stacked` /
    `lora_b_stacked`.

    The kernel is persistent over (pair × slice × n_chunk) work items —
    each program does one shrink and reuses the rank vector across
    `n_tiles_per_program` expand tiles. The chunk size scales with the
    pair-slice count so very small batches keep per-tile parallelism
    while medium batches cut redundant shrinks.
    """
    num_slices = len(lora_a_stacked)
    device = qcurr_hidden_states.device

    A0 = lora_a_stacked[0]
    B0 = lora_b_stacked[0]
    max_loras_w = A0.shape[0]
    num_experts = A0.shape[1]
    rank = A0.shape[2]
    K = A0.shape[3]
    N_per_slice = B0.shape[2]

    # Rank padding: floor 16 (tensor-core min K), ceil to next pow2. The
    # ≤64 cap is set conservatively for the prototype: at rank 64 the
    # per-program register footprint is rank_vec(64 fp32) + b_tile(BLOCK_N
    # × 64 fp32) = e.g. 128*64*4 = 32 KiB, comfortably within the 64 KiB
    # register file even with num_warps=8. Doubling to 128 would push us
    # against the limit and require shared-memory staging.
    assert rank <= 64, f"fused_moe_lora_small_batch supports rank<=64; got rank={rank}"
    BLOCK_R = max(triton.next_power_of_2(rank), 16)

    num_tokens = topk_weights.shape[0]
    M_grid = num_tokens * top_k_num
    if M_grid == 0 or num_slices == 0:
        return

    token_mapping_factor = 1 if mul_routed_weight else top_k_num

    A_ptrs = _get_ptr(lora_a_stacked, device)
    B_ptrs = _get_ptr(lora_b_stacked, device)

    assert output.dim() == 3, f"output must be 3-D, got {output.shape}"
    assert output.stride(0) == output.shape[1] * output.stride(1), (
        "fused_moe_lora_small_batch requires output.stride(0) == "
        f"top_k*stride(1); got shape={output.shape} strides={output.stride()}"
    )
    out_view = output.view(-1, output.shape[-1])

    # Block sizes. BLOCK_N=128 matches the one_shot's expand tile and gives
    # 6-24 N tiles for typical N ∈ [768, 3072], enough to saturate the SM
    # array once M_grid * num_slices reaches ~SM_count. BLOCK_K=128 halves
    # the K-loop trip count vs 64 and pays for itself once K ≥ 1024 (the
    # only regime we care about — hidden sizes are always large here).
    BLOCK_N = 128
    BLOCK_K = 128
    nw = 4
    ns = 3

    N_tiles = triton.cdiv(N_per_slice, BLOCK_N)
    pair_slices = M_grid * num_slices

    sm_count = current_platform.num_compute_units(device.index)
    n_tiles_per_program = _pick_small_batch_chunk(pair_slices, N_tiles, sm_count)
    n_chunks = triton.cdiv(N_tiles, n_tiles_per_program)
    work_total = pair_slices * n_chunks

    # Grid sizing: keep parallelism uncapped when work_total is small (so
    # very small batches still spread across SMs); cap at 2*SM_count once
    # we have plenty of work, letting the in-kernel stride loop drain the
    # remainder.
    grid_size = min(work_total, max(1, 2 * sm_count))
    grid = (grid_size,)

    _fused_moe_lora_small_batch_kernel[grid](
        qcurr_hidden_states,
        A_ptrs,
        B_ptrs,
        out_view,
        topk_weights,
        expert_ids_flat,
        token_lora_mapping,
        adapter_enabled,
        N_per_slice,
        K,
        top_k_num,
        max_loras_w,
        work_total,
        pair_slices,
        qcurr_hidden_states.stride(0),
        qcurr_hidden_states.stride(1),
        A0.stride(0),
        A0.stride(1),
        A0.stride(2),
        A0.stride(3),
        B0.stride(0),
        B0.stride(1),
        B0.stride(2),
        B0.stride(3),
        out_view.stride(0),
        out_view.stride(1),
        N_per_slice,
        n_tiles_per_program,
        n_chunks,
        token_mapping_factor=token_mapping_factor,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        ADD_INPUTS=add_inputs,
        BLOCK_R=BLOCK_R,
        actual_rank=rank,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        NUM_SLICES=num_slices,
        NUM_EXPERTS=num_experts,
        num_warps=nw,
        num_stages=ns,
    )
