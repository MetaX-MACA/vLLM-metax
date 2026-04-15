# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# -----------------------------------------
# Note: Disable following ops
#       - tl.extra.cuda.gdc_wait()
#       - tl.extra.cuda.gdc_launch_dependents()
#       - tl.make_tensor_descriptor
# -----------------------------------------

from vllm.triton_utils import tl, triton
from vllm.lora.ops.triton_ops.fused_moe_lora_op import (
    _get_lora_id,
    _get_expert_id,
    _get_token_offs,
    _get_c_ptrs,
)


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
    a_desc,
    b_ptr,
    b_desc,
    c_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    token_lora_mapping_ptr,
    # Matrix dimensions
    N,
    K,
    EM,
    num_valid_tokens,
    num_experts,
    top_k_num,
    lora_ids,
    adapter_enabled,
    max_loras,  # <<< PR2: rename, used for masks when grid axis-2 != max_loras
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
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
    # top_k_num or 1 depending on input token
    # is expanded by top_k or not
    token_mapping_factor: tl.constexpr,
    # whether use naive block assignment
    naive_block_assignment: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    ADD_INPUTS: tl.constexpr,
    USE_B_L2_CACHE: tl.constexpr,  # new, enable .ca load for B
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    USE_GDC: tl.constexpr,
    launch_pdl: tl.constexpr,
    IS_PRIMARY: tl.constexpr,
    USE_TMA: tl.constexpr,
    # sort_c determines whether tokens are stored in C in the order determined
    # by sorted_token_ids to enable later TMA loads from this tensor.
    #
    # When USE_TMA is enabled, the parameter combinations are:
    #   a_desc  | b_desc  | sort_c | Use Case
    #   --------|---------|--------|-----------------------------
    #   yes     | yes     | False  | expand kernel (num_slices=1)
    #   no      | yes     | True   | shrink kernel (num_slices=1)
    #   yes     | no      | False  | expand kernel (num_slices>1)
    #   no      | no      | True   | shrink kernel (num_slices>1)
    sort_c: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    slice_id = tl.program_id(axis=1)
    grid_k = tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)

    # calculate pid_m,pid_n
    lora_idx = tl.program_id(axis=2)
    pid_sk = pid % SPLIT_K
    pid_m_n = pid // SPLIT_K
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid_m_n // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid_m_n % num_pid_in_group) % group_size_m)
    pid_n = (pid_m_n % num_pid_in_group) // group_size_m

    offs = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)

    # Get lora_id
    lora_id = _get_lora_id(
        lora_ids,
        token_lora_mapping_ptr,
        lora_idx,
        pid_m,
        top_k_num,
        naive_block_assignment,
    )
    if lora_id == -1:
        return
    moe_enabled = tl.load(adapter_enabled + lora_id)
    if moe_enabled == 0:
        return
    if lora_id >= max_loras:
        return

    # Non-naive only: check num_tokens_post_padded
    if not naive_block_assignment:
        num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr + lora_id)
        if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
            return

    # Get expert_id
    expert_id = _get_expert_id(
        expert_ids_ptr,
        lora_id,
        pid_m,
        stride_el,
        max_loras,
        naive_block_assignment,
    )
    if expert_id == -1:
        return

    # Get token offsets
    offs_token = _get_token_offs(
        sorted_token_ids_ptr,
        lora_id,
        pid_m,
        offs,
        stride_tl,
        max_loras,
        num_valid_tokens,
        naive_block_assignment,
        BLOCK_SIZE_M,
    )
    # get a_ptr,b_ptr,c_ptr
    cur_a_ptr = a_ptr + (slice_id % num_slice_a) * slice_a_size
    cur_b_ptr = tl.load(b_ptr + slice_id).to(tl.pointer_type(c_ptr.dtype.element_ty))
    cur_c_ptr = c_ptr + (slice_id % num_slice_c) * slice_c_size

    offs_k = pid_sk * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    token_mask = offs_token < num_valid_tokens

    if USE_TMA and a_desc is not None:
        # Expand path - with TMA enabled, load from A using TMA descriptor
        offs_am = (
            slice_id * max_loras * EM
            + lora_id * EM
            + pid_m * BLOCK_SIZE_M // token_mapping_factor
        )
        offs_ak = pid_sk * BLOCK_SIZE_K
    else:
        # Shrink path - load hidden states based on order defined in
        # 'sorted_token_ids_ptr' then store them in c_ptr in this same sorted order
        tl.static_assert(a_desc is None, "a_desc must be none")
        a_ptrs = cur_a_ptr + (
            offs_token[:, None] // token_mapping_factor * stride_am
            + offs_k[None, :] * stride_ak
        )

    if USE_TMA:
        offs_bn = pid_n * BLOCK_SIZE_N
        offs_bk = pid_sk * BLOCK_SIZE_K
        if b_desc is None:
            # Note(@gnovack) - Allocation of TMA descriptors on-device
            # can cause conflicts when running in parallel via PDL
            # /----------------------- Metax Modification ------------------\
            # if USE_GDC and not IS_PRIMARY:
            #     tl.extra.cuda.gdc_wait()
            # \------------------------------------------------------------/

            pass
            # /----------------------- Metax Modification ------------------\
            # b_desc = tl.make_tensor_descriptor(
            #    cur_b_ptr,
            #    shape=[max_loras, num_experts, N, K],
            #    strides=[stride_bl, stride_be, stride_bn, stride_bk],
            #    block_shape=[1, 1, BLOCK_SIZE_N, BLOCK_SIZE_K],
            # )
            # \------------------------------------------------------------/
    else:
        offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int32)
        b_ptrs = (
            cur_b_ptr
            + lora_id * stride_bl
            + expert_id * stride_be
            + offs_k[:, None] * stride_bk
            + offs_bn[None, :] * stride_bn
        )

    # /----------------------- Metax Modification ------------------\
    # if USE_GDC and IS_PRIMARY:
    #     # GDC launch dependents hints the runtime system to launch dependent kernels.
    #     tl.extra.cuda.gdc_launch_dependents()
    # \------------------------------------------------------------/
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # /----------------------- Metax Modification ------------------\
    # if USE_GDC and not IS_PRIMARY:
    #     tl.extra.cuda.gdc_wait()
    # \------------------------------------------------------------/

    for k in range(0, grid_k):
        cur_k_offset = k * (BLOCK_SIZE_K * SPLIT_K)
        k_remaining = K - cur_k_offset
        # pre-fetch lora weight
        if b_desc is not None:
            b = (
                b_desc.load([lora_id, expert_id, offs_bn, offs_bk + cur_k_offset])
                .reshape(BLOCK_SIZE_N, BLOCK_SIZE_K)
                .T
            )
        else:
            # add (offs_bn < N) mask; optional .ca for B
            b_mask = (offs_k[:, None] < k_remaining) & (offs_bn[None, :] < N)
            if USE_B_L2_CACHE:
                b = tl.load(b_ptrs, mask=b_mask, other=0.0, cache_modifier=".ca")
            else:
                b = tl.load(b_ptrs, mask=b_mask, other=0.0)
            b_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_bk

        if a_desc is not None:
            a = a_desc.load([offs_am, offs_ak + cur_k_offset])
        else:
            a = tl.load(
                a_ptrs,
                mask=token_mask[:, None] & (offs_k[None, :] < k_remaining),
                other=0.0,
            )
            a_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_ak

        accumulator += tl.dot(a, b)

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0.0)
        accumulator = accumulator * moe_weight[:, None]
    accumulator = accumulator.to(c_ptr.dtype.element_ty)
    # Write back the block of the output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = _get_c_ptrs(
        cur_c_ptr,
        lora_id,
        pid_m,
        offs,
        offs_token,
        offs_cn,
        stride_cm,
        stride_cn,
        EM,
        BLOCK_SIZE_M,
        sort_c,
    )
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)

    if SPLIT_K == 1:
        if ADD_INPUTS:
            prev = tl.load(c_ptrs, mask=c_mask, other=0.0)
            tl.store(c_ptrs, prev + accumulator, mask=c_mask)
        else:
            tl.store(c_ptrs, accumulator, mask=c_mask)
    else:
        tl.atomic_add(c_ptrs, accumulator, mask=c_mask, sem="relaxed")


import vllm.lora.ops.triton_ops.fused_moe_lora_op

vllm.lora.ops.triton_ops.fused_moe_lora_op._fused_moe_lora_kernel = (
    _fused_moe_lora_kernel
)
