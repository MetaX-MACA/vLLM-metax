# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------
# Note: Disable following ops
#       - tl.extra.cuda.gdc_wait()
#       - tl.extra.cuda.gdc_launch_dependents()
# -----------------------------------------

from vllm.triton_utils import tl, triton


import torch

from vllm.distributed import (
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

from vllm.lora.ops.triton_ops.utils import supports_pdl

_LORA_PTR_DICT: dict[tuple[int, ...], torch.tensor] = {}


def _get_ptr(lora_weights: list[torch.Tensor], device: torch.device):
    """
    `_LORA_PTR_DICT` collects the required information during `profile_run`,
    After this, it remains constant and subsequent usage is through LUT.
    Refer to:
    https://github.com/triton-lang/triton/blob/release/3.1.x/python/tutorials/08-grouped-gemm.py
    """
    key = tuple(lora_weight.data_ptr() for lora_weight in lora_weights)

    if (ptr_tensor := _LORA_PTR_DICT.get(key)) is not None:
        return ptr_tensor

    tensor_ptrs = []
    for lora_weight in lora_weights:
        tensor_ptrs.append(lora_weight.data_ptr())
    ptr_tensor = torch.tensor(tensor_ptrs, device=device, dtype=torch.uint64)

    _LORA_PTR_DICT[key] = ptr_tensor
    return _LORA_PTR_DICT.get(key)


# ----------------------------
# Optimized kernel
# ----------------------------
@triton.jit(
    do_not_specialize=[
        "num_valid_tokens",
        "EM",
        "stride_tl",
        "stride_el",
        "slice_a_size",
        "slice_c_size",
    ]
)
def _fused_moe_lora_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N,
    K,
    EM,
    num_valid_tokens,
    num_experts,
    lora_ids,
    adapter_enabled,
    # Strides
    stride_am,
    stride_ak,
    stride_bl,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_tl,
    stride_el,
    slice_a_size,
    slice_c_size,
    # Meta-parameters
    num_slice_a: tl.constexpr,
    num_slice_c: tl.constexpr,
    top_k: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    EVEN_K: tl.constexpr,
    EVEN_N: tl.constexpr,
    ADD_INPUTS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    USE_GDC: tl.constexpr,
    launch_pdl: tl.constexpr,
    IS_PRIMARY: tl.constexpr,
    # --- new toggles ---
    USE_CACHE_HINTS: tl.constexpr,   # enable cache_modifier/eviction_policy
    ASSUME_ALIGNED: tl.constexpr,    # enable tl.multiple_of hints (only if runtime checks pass)
):
    pid = tl.program_id(axis=0)
    slice_id = tl.program_id(axis=1)
    lora_idx = tl.program_id(axis=2)

    lora_id = tl.load(lora_ids + lora_idx).to(tl.int32)
    if lora_id == -1:
        return
    moe_enabled = tl.load(adapter_enabled + lora_id).to(tl.int32)
    if moe_enabled == 0:
        return

    grid_k = tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)

    pid_sk = (pid % SPLIT_K).to(tl.int32)
    pid_m_n = (pid // SPLIT_K).to(tl.int32)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = (pid_m_n // num_pid_in_group).to(tl.int32)
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid_m_n % num_pid_in_group) % group_size_m)
    pid_n = (pid_m_n % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr + lora_id).to(tl.int32)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    # int64 base offsets
    lora_id_i64 = lora_id.to(tl.int64)
    stride_el_i64 = tl.full((), stride_el, tl.int64)
    stride_tl_i64 = tl.full((), stride_tl, tl.int64)

    base_expert = expert_ids_ptr + lora_id_i64 * stride_el_i64
    expert_id = tl.load(
        base_expert + pid_m,
        mask=pid_m.to(tl.int64) < stride_el_i64,
        other=-1,
    ).to(tl.int32)
    if expert_id == -1:
        return

    # ---------------------------
    # (1) Remove % num_slice_*  (compile-time)
    # ---------------------------
    if num_slice_a == 1:
        slice_a_idx = tl.full((), 0, tl.int32)
    else:
        slice_a_idx = slice_id
    if num_slice_c == 1:
        slice_c_idx = tl.full((), 0, tl.int32)
    else:
        slice_c_idx = slice_id

    cur_a_ptr = a_ptr + slice_a_idx * slice_a_size
    cur_b_ptr = tl.load(b_ptr + slice_id).to(tl.pointer_type(c_ptr.dtype.element_ty))
    cur_c_ptr = c_ptr + slice_c_idx * slice_c_size

    # ---------------------------
    # Offsets + contiguity hints
    # ---------------------------
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)).to(tl.int32)
    offs_bn = tl.max_contiguous(offs_bn, BLOCK_SIZE_N)

    if EVEN_N:
        mask_n = None
    else:
        mask_n = offs_bn < N

    offs_k = (pid_sk * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)).to(tl.int32)
    offs_k = tl.max_contiguous(offs_k, BLOCK_SIZE_K)
    offs_k_i64 = offs_k.to(tl.int64)

    offs_token_id = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)).to(tl.int32)

    base_sorted = sorted_token_ids_ptr + lora_id_i64 * stride_tl_i64
    token_id_mask = offs_token_id < EM
    offs_token = tl.load(base_sorted + offs_token_id, mask=token_id_mask, other=0).to(tl.int32)
    token_mask = token_id_mask & (offs_token < num_valid_tokens)

    if top_k == 1:
        offs_token_div = offs_token
    elif top_k == 2:
        offs_token_div = offs_token >> 1
    elif top_k == 4:
        offs_token_div = offs_token >> 2
    elif top_k == 8:
        offs_token_div = offs_token >> 3
    else:
        offs_token_div = (offs_token // top_k).to(tl.int32)

    # Build A ptrs
    stride_am_i64 = tl.full((), stride_am, tl.int64)
    stride_ak_i64 = tl.full((), stride_ak, tl.int64)
    a_ptrs = (
        cur_a_ptr
        + offs_token_div.to(tl.int64)[:, None] * stride_am_i64
        + offs_k_i64[None, :] * stride_ak_i64
    )

    # Build B ptrs
    stride_bl_i64 = tl.full((), stride_bl, tl.int64)
    stride_be_i64 = tl.full((), stride_be, tl.int64)
    stride_bk_i64 = tl.full((), stride_bk, tl.int64)
    stride_bn_i64 = tl.full((), stride_bn, tl.int64)

    b_base = cur_b_ptr + lora_id_i64 * stride_bl_i64 + expert_id.to(tl.int64) * stride_be_i64
    b_ptrs = b_base + offs_k_i64[:, None] * stride_bk_i64 + offs_bn.to(tl.int64)[None, :] * stride_bn_i64

    # (3) Vectorization/alignment hints (only if runtime-verified)
    if ASSUME_ALIGNED:
        # Common matmul trick: tell compiler these pointers are well-aligned to enable wider ld/st.
        # Only safe if caller guarantees alignment/contiguity.
        tl.multiple_of(a_ptrs, (16, 16))
        tl.multiple_of(b_ptrs, (16, 16))

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    advance_a = (BLOCK_SIZE_K * SPLIT_K) * stride_ak_i64
    advance_b = (BLOCK_SIZE_K * SPLIT_K) * stride_bk_i64

    # ---------------------------
    # Inner loop with cache hints
    # ---------------------------
    if EVEN_K:
        if EVEN_N:
            for _ in range(0, grid_k):
                if USE_CACHE_HINTS:
                    b = tl.load(b_ptrs, cache_modifier=".cg", eviction_policy="evict_last")
                    a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0,
                                cache_modifier=".cg", eviction_policy="evict_first")
                else:
                    b = tl.load(b_ptrs)
                    a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
                accumulator += tl.dot(a, b)
                a_ptrs += advance_a
                b_ptrs += advance_b
        else:
            for _ in range(0, grid_k):
                if USE_CACHE_HINTS:
                    b = tl.load(b_ptrs, mask=mask_n[None, :], other=0.0,
                                cache_modifier=".cg", eviction_policy="evict_last")
                    a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0,
                                cache_modifier=".cg", eviction_policy="evict_first")
                else:
                    b = tl.load(b_ptrs, mask=mask_n[None, :], other=0.0)
                    a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
                accumulator += tl.dot(a, b)
                a_ptrs += advance_a
                b_ptrs += advance_b
    else:
        if EVEN_N:
            for k in range(0, grid_k):
                k_remaining = K - k * (BLOCK_SIZE_K * SPLIT_K)
                k_mask = (offs_k[:, None] < k_remaining)
                if USE_CACHE_HINTS:
                    b = tl.load(b_ptrs, mask=k_mask, other=0.0,
                                cache_modifier=".cg", eviction_policy="evict_last")
                    a = tl.load(a_ptrs,
                                mask=token_mask[:, None] & (offs_k[None, :] < k_remaining),
                                other=0.0,
                                cache_modifier=".cg", eviction_policy="evict_first")
                else:
                    b = tl.load(b_ptrs, mask=k_mask, other=0.0)
                    a = tl.load(a_ptrs,
                                mask=token_mask[:, None] & (offs_k[None, :] < k_remaining),
                                other=0.0)
                accumulator += tl.dot(a, b)
                a_ptrs += advance_a
                b_ptrs += advance_b
        else:
            for k in range(0, grid_k):
                k_remaining = K - k * (BLOCK_SIZE_K * SPLIT_K)
                k_mask = (offs_k[:, None] < k_remaining) & mask_n[None, :]
                if USE_CACHE_HINTS:
                    b = tl.load(b_ptrs, mask=k_mask, other=0.0,
                                cache_modifier=".cg", eviction_policy="evict_last")
                    a = tl.load(a_ptrs,
                                mask=token_mask[:, None] & (offs_k[None, :] < k_remaining),
                                other=0.0,
                                cache_modifier=".cg", eviction_policy="evict_first")
                else:
                    b = tl.load(b_ptrs, mask=k_mask, other=0.0)
                    a = tl.load(a_ptrs,
                                mask=token_mask[:, None] & (offs_k[None, :] < k_remaining),
                                other=0.0)
                accumulator += tl.dot(a, b)
                a_ptrs += advance_a
                b_ptrs += advance_b

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0).to(tl.float32)
        accumulator = accumulator * moe_weight[:, None]

    accumulator = accumulator.to(c_ptr.dtype.element_ty)

    # Store C
    offs_cn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)).to(tl.int32)
    stride_cm_i64 = tl.full((), stride_cm, tl.int64)
    stride_cn_i64 = tl.full((), stride_cn, tl.int64)

    c_ptrs = (
        cur_c_ptr
        + stride_cm_i64 * offs_token.to(tl.int64)[:, None]
        + stride_cn_i64 * offs_cn.to(tl.int64)[None, :]
    )

    if ASSUME_ALIGNED:
        tl.multiple_of(c_ptrs, (16, 16))

    if EVEN_N:
        c_mask = token_mask[:, None]
    else:
        c_mask = token_mask[:, None] & (offs_cn[None, :] < N)

    if SPLIT_K == 1:
        if ADD_INPUTS:
            prev = tl.load(c_ptrs, mask=c_mask, other=0.0)
            tl.store(c_ptrs, prev + accumulator, mask=c_mask)
        else:
            tl.store(c_ptrs, accumulator, mask=c_mask)
    else:
        tl.atomic_add(c_ptrs, accumulator, mask=c_mask, sem="relaxed")

def _assume_aligned_for_matmul_like(a: torch.Tensor,
                                   b: torch.Tensor,
                                   c: torch.Tensor,
                                   k_multiple: int = 8,
                                   align_bytes: int = 16) -> bool:
    # 1) Innermost dimension is contiguous (stride == 1)
    if a.stride(-1) != 1 or b.stride(-1) != 1 or c.stride(-1) != 1:
        return False
    # 2) data_ptr alignment
    if (a.data_ptr() % align_bytes) != 0 or (b.data_ptr() % align_bytes) != 0 or (c.data_ptr() % align_bytes) != 0:
        return False
    # 3) K / N dimensions preferably multiples of 8 (fp16 common v4/v8 coalescing)
    #    Here "b innermost dimension" represents K (your b_ptrs traverse along offs_k in the innermost dimension)
    if (a.shape[-1] % k_multiple) != 0 or (b.shape[-1] % k_multiple) != 0:
        return False
    return True

@torch.inference_mode()
def _fused_moe_lora_shrink(
    a_intermediate_cache1: torch.Tensor,
    # (num_slices, num_tokens, top_k_num, max_lora_rank)
    qcurr_hidden_states: torch.Tensor,  # (num_tokens, K,)
    lora_a_stacked: list[
        torch.Tensor
    ],  # [(max_loras, num_experts, max_lora_rank, K,),...]
    topk_weights: torch.Tensor,  # (num_tokens, top_k_num)
    sorted_token_ids: torch.Tensor,  # (max_loras, _)
    expert_ids: torch.Tensor,  # (max_loras, _ ,)
    num_tokens_post_padded: torch.Tensor,  # (max_loras, )
    top_k_num: int,
    lora_ids: torch.Tensor,
    adapter_enabled: torch.Tensor,
    ## adding for kernel
    device: torch.device,
    N: int,
    M: int,
    EM: int,
    K: int,
    num_tokens: int,
    num_experts: int,
    num_slices: int,
    block_size_m: int,
    block_size_n: int,
    block_size_k: int,
    group_size_m: int,
    num_warps: int,
    num_stages: int,
    split_k: int,
    mul_routed_weight: bool = False,
    use_gdc: bool = False,
) -> None:
    w1_lora_a_stacked = lora_a_stacked[0]
    shrink_config = {
        "BLOCK_SIZE_M": block_size_m,
        "BLOCK_SIZE_N": block_size_n,
        "BLOCK_SIZE_K": block_size_k,
        "GROUP_SIZE_M": group_size_m,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "SPLIT_K": split_k,
        "USE_GDC": use_gdc,
        "launch_pdl": use_gdc,
    }

    b_ptr = _get_ptr(lora_a_stacked, device)
    # Runtime check: whether it's safe to enable ASSUME_ALIGNED
    # A: qcurr_hidden_states (M x K)
    # B: last dimension of w1_lora_a_stacked is K
    # C: under a_intermediate_cache1 view, last dimension is rank (stride_cn=1)
    assume_aligned = _assume_aligned_for_matmul_like(
        qcurr_hidden_states,
        w1_lora_a_stacked,
        a_intermediate_cache1,
    )

    grid = lambda META: (
        split_k
        * triton.cdiv(EM, META["BLOCK_SIZE_M"])
        * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        len(lora_a_stacked),
        lora_a_stacked[0].shape[0],
    )

    # EVEN flags
    even_k = (K % (block_size_k * split_k) == 0)
    even_n = (N % block_size_n == 0)

    _fused_moe_lora_kernel[grid](
        qcurr_hidden_states,
        b_ptr,
        a_intermediate_cache1,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        N,
        K,
        EM,
        num_tokens,
        num_experts,
        lora_ids,
        adapter_enabled,
        qcurr_hidden_states.stride(0),
        qcurr_hidden_states.stride(1),
        w1_lora_a_stacked.stride(0),
        w1_lora_a_stacked.stride(1),
        w1_lora_a_stacked.stride(3),
        w1_lora_a_stacked.stride(2),
        a_intermediate_cache1.stride(2),
        a_intermediate_cache1.stride(3),
        sorted_token_ids.stride(0),
        expert_ids.stride(0),
        slice_a_size=qcurr_hidden_states.numel(),
        slice_c_size=a_intermediate_cache1.numel() // num_slices,
        num_slice_a=1,
        num_slice_c=num_slices,
        top_k=1 if mul_routed_weight else top_k_num,
        MUL_ROUTED_WEIGHT=False,
        EVEN_K=even_k,
        EVEN_N=even_n,
        ADD_INPUTS=False,
        IS_PRIMARY=True,
        USE_CACHE_HINTS=True,
        ASSUME_ALIGNED=assume_aligned,
        **shrink_config,
    )

@torch.inference_mode()
def _fused_moe_lora_expand(
    output: torch.Tensor,  # (num_tokens, top_k_num, N*len(lora_a_stacked),)
    a_intermediate_cache1: torch.Tensor,  # (num_slices, M, top_k_num, max_lora_rank)
    lora_b_stacked: list[
        torch.Tensor
    ],  # [(max_loras, num_experts, max_lora_rank, K,),...]
    topk_weights: torch.Tensor,  # (num_tokens, top_k_num)
    sorted_token_ids: torch.Tensor,  # (max_loras, _)
    expert_ids: torch.Tensor,  # (max_loras, _ ,)
    num_tokens_post_padded: torch.Tensor,  # (max_loras, )
    top_k_num: int,
    lora_ids: torch.Tensor,
    adapter_enabled: torch.Tensor,
    ## adding for kernel
    device: torch.device,
    N: int,
    M: int,
    EM: int,
    K: int,
    num_tokens: int,
    num_experts: int,
    num_slices: int,
    max_lora_rank: int,
    w1_output_dim_size: int,
    block_size_m: int,
    block_size_n: int,
    block_size_k: int,
    group_size_m: int,
    num_warps: int,
    num_stages: int,
    split_k: int,
    mul_routed_weight: bool = False,
    offset: int = 0,
    use_gdc: bool = False,
) -> None:
    b_ptr = _get_ptr(lora_b_stacked, device)
    K = max_lora_rank
    N = w1_output_dim_size

    w1_lora_b_stacked = lora_b_stacked[0]

    a_intermediate_cache1 = a_intermediate_cache1.view(
        -1, a_intermediate_cache1.shape[3]
    )

    expand_config = {
        "BLOCK_SIZE_M": block_size_m,
        "BLOCK_SIZE_N": block_size_n,
        "BLOCK_SIZE_K": block_size_k,
        "GROUP_SIZE_M": group_size_m,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "SPLIT_K": 1,  # Set split_k = 1 for expand calls
        "USE_GDC": use_gdc,
        "launch_pdl": use_gdc,  # triton kernel metadata
    }

    grid = lambda META: (
        triton.cdiv(EM, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        len(lora_b_stacked),
        lora_b_stacked[0].shape[0],
    )

    even_k = (K % block_size_k == 0)
    even_n = (N % block_size_n == 0)

    # Fast path: directly accumulate into the corresponding slice interval of output.
    out_view = output[:, :, offset : offset + num_slices * N]
    assume_aligned = _assume_aligned_for_matmul_like(
        a_intermediate_cache1,
        w1_lora_b_stacked,
        out_view,
    )
    slice_c_size = N * out_view.stride(2)

    _fused_moe_lora_kernel[grid](
        a_intermediate_cache1,
        b_ptr,
        out_view,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        N,
        K,
        EM,
        num_tokens,
        num_experts,
        lora_ids,
        adapter_enabled,
        a_intermediate_cache1.stride(0),
        a_intermediate_cache1.stride(1),
        w1_lora_b_stacked.stride(0),
        w1_lora_b_stacked.stride(1),
        w1_lora_b_stacked.stride(3),
        w1_lora_b_stacked.stride(2),
        out_view.stride(1),
        out_view.stride(2),
        sorted_token_ids.stride(0),
        expert_ids.stride(0),
        slice_a_size=a_intermediate_cache1.numel() // num_slices,
        slice_c_size=slice_c_size,
        num_slice_a=num_slices,
        num_slice_c=num_slices,
        top_k=1,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        EVEN_K=even_k,                      # NEW
        EVEN_N=even_n,
        ADD_INPUTS=True,
        IS_PRIMARY=False,
        USE_CACHE_HINTS=True,
        ASSUME_ALIGNED=assume_aligned,
        **expand_config,
    )


@torch.inference_mode()
def _fused_moe_lora(
    output: torch.Tensor,  # (num_tokens, top_k_num, N*len(lora_a_stacked),)
    qcurr_hidden_states: torch.Tensor,  # (num_tokens, K,)
    lora_a_stacked: list[
        torch.Tensor
    ],  # [(max_loras, num_experts, max_lora_rank, K,),...]
    lora_b_stacked: list[
        torch.Tensor
    ],  # [(max_loras, num_experts, N, max_lora_rank,),...]
    topk_weights: torch.Tensor,  # (num_tokens, top_k_num)
    sorted_token_ids: torch.Tensor,  # (max_loras, _)
    expert_ids: torch.Tensor,  # (max_loras, _ ,)
    num_tokens_post_padded: torch.Tensor,  # (max_loras, )
    max_lora_rank: int,
    top_k_num: int,
    lora_ids: torch.Tensor,
    adapter_enabled: torch.Tensor,
    shrink_block_size_m: int,
    shrink_block_size_n: int,
    shrink_block_size_k: int,
    shrink_group_size_m: int,
    shrink_num_warps: int,
    shrink_num_stages: int,
    shrink_split_k: int,
    expand_block_size_m: int,
    expand_block_size_n: int,
    expand_block_size_k: int,
    expand_group_size_m: int,
    expand_num_warps: int,
    expand_num_stages: int,
    expand_split_k: int,
    mul_routed_weight: bool = False,
    fully_sharded: bool = False,
    offset: int = 0,
) -> None:
    assert len(lora_a_stacked) == len(lora_b_stacked) > 0
    assert (
        sorted_token_ids.dim()
        == expert_ids.dim()
        == topk_weights.dim()
        == qcurr_hidden_states.dim()
        == 2
    )
    assert (
        sorted_token_ids.shape[0]
        == expert_ids.shape[0]
        == num_tokens_post_padded.shape[0]
    )
    assert output.shape[0] == topk_weights.shape[0]
    assert top_k_num == topk_weights.shape[1]
    device = qcurr_hidden_states.device
    num_slices = len(lora_a_stacked)
    w1_lora_b_stacked = lora_b_stacked[0]
    num_experts = lora_a_stacked[0].shape[1]
    N = max_lora_rank
    M = topk_weights.shape[0]
    EM = sorted_token_ids.shape[1]
    K = qcurr_hidden_states.shape[1]
    num_tokens = M * top_k_num
    w1_output_dim_size = w1_lora_b_stacked.shape[2]

    # NOTE: keep your existing workspace / zeroing strategy outside this snippet.
    # SPLIT_K==1: shrink will perform a complete overwrite -> memset is not needed
    if shrink_split_k == 1:
        a_intermediate_cache1 = torch.empty(
            (num_slices, M, top_k_num, max_lora_rank),
            dtype=output.dtype,
            device=device,
        )
    else:
        # SPLIT_K>1: shrink will use atomic_add for accumulation -> must clear to zero
        a_intermediate_cache1 = torch.zeros(
            (num_slices, M, top_k_num, max_lora_rank),
            dtype=output.dtype,
            device=device,
        )

    use_gdc = supports_pdl(device) and not fully_sharded

    _fused_moe_lora_shrink(
        a_intermediate_cache1,
        qcurr_hidden_states,
        lora_a_stacked,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        top_k_num,
        lora_ids,
        adapter_enabled,
        ## adding for kernel
        device,
        N,
        M,
        EM,
        K,
        num_tokens,
        num_experts,
        num_slices,
        shrink_block_size_m,
        shrink_block_size_n,
        shrink_block_size_k,
        shrink_group_size_m,
        shrink_num_warps,
        shrink_num_stages,
        shrink_split_k,
        mul_routed_weight,
        use_gdc=use_gdc,
    )

    if fully_sharded:
        if max_lora_rank == w1_lora_b_stacked.shape[-1]:
            a_intermediate_cache1 = tensor_model_parallel_all_reduce(
                a_intermediate_cache1
            )
        else:
            a_intermediate_cache1 = tensor_model_parallel_all_gather(
                a_intermediate_cache1
            )
            max_lora_rank = a_intermediate_cache1.shape[-1]

    _fused_moe_lora_expand(
        output,
        a_intermediate_cache1,
        lora_b_stacked,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        top_k_num,
        lora_ids,
        adapter_enabled,
        ## adding for kernel
        device,
        N,
        M,
        EM,
        K,
        num_tokens,
        num_experts,
        num_slices,
        max_lora_rank,
        w1_output_dim_size,
        expand_block_size_m,
        expand_block_size_n,
        expand_block_size_k,
        expand_group_size_m,
        expand_num_warps,
        expand_num_stages,
        expand_split_k,
        mul_routed_weight,
        offset,
        use_gdc=use_gdc,
    )

def _fused_moe_lora_fake(
    output: torch.Tensor,
    qcurr_hidden_states: torch.Tensor,
    lora_a_stacked: list[torch.Tensor],
    lora_b_stacked: list[torch.Tensor],
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    max_lora_rank: int,
    top_k_num: int,
    lora_ids: torch.Tensor,
    adapter_enabled: torch.Tensor,
    shrink_block_size_m: int,
    shrink_block_size_n: int,
    shrink_block_size_k: int,
    shrink_group_size_m: int,
    shrink_num_warps: int,
    shrink_num_stages: int,
    shrink_split_k: int,
    expand_block_size_m: int,
    expand_block_size_n: int,
    expand_block_size_k: int,
    expand_group_size_m: int,
    expand_num_warps: int,
    expand_num_stages: int,
    expand_split_k: int,
    mul_routed_weight: bool = False,
) -> None:
    return


def _fused_moe_lora_shrink_fake(
    a_intermediate_cache1: torch.Tensor,
    qcurr_hidden_states: torch.Tensor,
    lora_a_stacked: list[torch.Tensor],
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    top_k_num: int,
    lora_ids: torch.Tensor,
    adapter_enabled: torch.Tensor,
    device: torch.device,
    N: int,
    M: int,
    EM: int,
    K: int,
    num_tokens: int,
    num_experts: int,
    num_slices: int,
    block_size_m: int,
    block_size_n: int,
    block_size_k: int,
    group_size_m: int,
    num_warps: int,
    num_stages: int,
    split_k: int,
    mul_routed_weight: bool = False,
    use_gdc: bool = False,
) -> None:
    return


def _fused_moe_lora_expand_fake(
    output: torch.Tensor,
    a_intermediate_cache1: torch.Tensor,
    lora_b_stacked: list[torch.Tensor],
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    top_k_num: int,
    lora_ids: torch.Tensor,
    adapter_enabled: torch.Tensor,
    device: torch.device,
    N: int,
    M: int,
    EM: int,
    K: int,
    num_tokens: int,
    num_experts: int,
    num_slices: int,
    max_lora_rank: int,
    w1_output_dim_size: int,
    block_size_m: int,
    block_size_n: int,
    block_size_k: int,
    group_size_m: int,
    num_warps: int,
    num_stages: int,
    split_k: int,
    mul_routed_weight: bool = False,
    use_gdc: bool = False,
) -> None:
    return


try:
    direct_register_custom_op(
        op_name="maca_fused_moe_lora",
        op_func=_fused_moe_lora,
        mutates_args=["output"],
        fake_impl=_fused_moe_lora_fake,
    )

    direct_register_custom_op(
        op_name="maca_fused_moe_lora_shrink",
        op_func=_fused_moe_lora_shrink,
        mutates_args=["a_intermediate_cache1"],
        fake_impl=_fused_moe_lora_shrink_fake,
    )

    direct_register_custom_op(
        op_name="maca_fused_moe_lora_expand",
        op_func=_fused_moe_lora_expand,
        mutates_args=["output"],
        fake_impl=_fused_moe_lora_expand_fake,
    )

    fused_moe_lora = torch.ops.vllm.fused_moe_lora
    fused_moe_lora_shrink = torch.ops.vllm.fused_moe_lora_shrink
    fused_moe_lora_expand = torch.ops.vllm.fused_moe_lora_expand

except AttributeError:
    fused_moe_lora = _fused_moe_lora
    fused_moe_lora_shrink = _fused_moe_lora_shrink
    fused_moe_lora_expand = _fused_moe_lora_expand


def add_lora_fused_moe(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        lora_a_stacked: tuple[torch.Tensor, ...],
        lora_b_stacked: tuple[torch.Tensor, ...],
        topk_weights: torch.Tensor,
        sorted_token_ids: torch.Tensor,
        expert_ids: torch.Tensor,
        num_tokens_post_padded: torch.Tensor,
        max_lora_rank: int,
        top_k_num: int,
        shrink_config,
        expand_config,
        adapter_enabled: torch.Tensor,
        mul_routed_weight=False,
        fully_sharded: bool = False,
        offset: int = 0,
    ):
        """
        Performs a fused forward computation for LoRA of Mixture-of-Experts (MoE) layer.
        """
        (_, _, _, _, lora_ids, _) = self.token_mapping_meta.meta_args(x.size(0))
        _fused_moe_lora(
            y,
            x,
            lora_a_stacked,
            lora_b_stacked,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            max_lora_rank,
            top_k_num,
            lora_ids,
            adapter_enabled,
            shrink_config.get("BLOCK_SIZE_M", 64),
            shrink_config.get("BLOCK_SIZE_N", 64),
            shrink_config.get("BLOCK_SIZE_K", 32),
            shrink_config.get("GROUP_SIZE_M", 8),
            shrink_config.get("NUM_WARPS", 4),
            shrink_config.get("NUM_STAGES", 3),
            shrink_config.get("SPLIT_K", 1),
            expand_config.get("BLOCK_SIZE_M", 64),
            expand_config.get("BLOCK_SIZE_N", 64),
            expand_config.get("BLOCK_SIZE_K", 32),
            expand_config.get("GROUP_SIZE_M", 8),
            expand_config.get("NUM_WARPS", 4),
            expand_config.get("NUM_STAGES", 3),
            expand_config.get("SPLIT_K", 1),
            mul_routed_weight,
            fully_sharded,
            offset,
        )

import vllm.lora.ops.triton_ops.fused_moe_lora_op
import vllm.lora.punica_wrapper.punica_gpu
vllm.lora.punica_wrapper.punica_gpu.PunicaWrapperGPU.add_lora_fused_moe=add_lora_fused_moe
# vllm.lora.ops.triton_ops.fused_moe_lora_op._fused_moe_lora_kernel = (
#     _fused_moe_lora_kernel
# )
