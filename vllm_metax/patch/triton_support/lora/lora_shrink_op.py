# SPDX-License-Identifier: Apache-2.0
# ------------------------------------------------
# Note: Substite `do_shrink_kernel` from vllm_metax
# ------------------------------------------------


from vllm_metax.patch.triton_support.lora.kernel_utils import do_shrink_kernel
from vllm.triton_utils import tl, triton


@triton.jit
def _lora_shrink_kernel(
    input_ptr,
    lora_ptr,
    out_ptr,
    M,
    N,
    K,
    token_indices_sorted_by_lora_ids,
    num_tokens_per_lora,
    lora_token_start_loc,
    lora_ids,
    scaling,
    input_d0_stride,
    input_d1_stride,
    lora_d0_stride,
    lora_d1_stride,
    lora_d2_stride,
    output_d0_stride,
    output_d1_stride,
    output_d2_stride,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    SLICE_NUM: tl.constexpr,
    USE_GDC: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    cta_n_num = tl.cdiv(N, BLOCK_N)
    cta_m_num = tl.cdiv(M, BLOCK_M)

    pid_sk_m_n = tl.program_id(axis=0)
    pid_sk = pid_sk_m_n % SPLIT_K

    pid_m_n = pid_sk_m_n // SPLIT_K
    num_pid_in_group = GROUP_SIZE_M * cta_n_num
    group_id = pid_m_n // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(cta_m_num - first_pid_m, GROUP_SIZE_M)

    # Column-major ordering within groups for better cache reuse
    pid_m = first_pid_m + ((pid_m_n % num_pid_in_group) % group_size_m)
    pid_n = (pid_m_n % num_pid_in_group) // group_size_m

    slice_id = tl.program_id(axis=1)
    lora_idx = tl.program_id(axis=2)

    lora_id = tl.load(lora_ids + lora_idx)
    if lora_id == -1:
        # Early exit for the no-lora case.
        return

    lora_m_size = tl.load(num_tokens_per_lora + lora_idx)

    cta_m_offset = pid_m * BLOCK_M
    if cta_m_offset >= lora_m_size:
        # Early exit CTA.
        return

    # num rows this CTA should process.
    cta_m_len = min(BLOCK_M, lora_m_size - cta_m_offset)

    # Identify all rows that this CTA should process.
    lora_m_indices_start = tl.load(lora_token_start_loc + lora_idx)
    cta_lora_seq_indices = (
        token_indices_sorted_by_lora_ids + lora_m_indices_start + cta_m_offset
    )
    # Load all relevant row indices.
    offset_m = tl.arange(0, BLOCK_M) % cta_m_len
    ram = tl.load(cta_lora_seq_indices + offset_m)

    do_shrink_kernel(
        pid_n,
        pid_sk,
        slice_id,
        lora_id,
        input_ptr,
        lora_ptr,
        out_ptr,
        N,
        K,
        cta_m_len,
        ram,  # array identifying the rows of Input ptr to operate on
        # input strides
        input_d0_stride,
        input_d1_stride,
        # lora strides
        lora_d0_stride,
        lora_d1_stride,
        lora_d2_stride,
        # output strides
        output_d0_stride,
        output_d1_stride,
        output_d2_stride,
        scaling,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        EVEN_K,
        SPLIT_K,
        SLICE_NUM,
        USE_GDC,
    )


import vllm.lora.ops.triton_ops.lora_shrink_op

vllm.lora.ops.triton_ops.lora_shrink_op._lora_shrink_kernel = _lora_shrink_kernel
