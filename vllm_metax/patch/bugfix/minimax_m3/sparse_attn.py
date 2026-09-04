# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#
# -----------------------------------------------------------------------------
# Note: A 128-token K/V tile needs 139264 B of shared memory, exceeding the
#       MetaX C550 limit of 65536 B. Process it as eight SUB_K=16 tiles instead:
#
#       selected K/V block (128 tokens)
#       +----+----+----+----+----+----+----+----+
#       | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 |
#       +----+----+----+----+----+----+----+----+
#         tile: load K -> QK -> softmax -> load V -> PV
#               (m_i, lse_i, acc_o) ----------------> next tile
#
#       Only the inner K/V loop changes; common's cache layout, FP8 scales,
#       top-k handling, and decode split-K logic are preserved.
#
# Remove at: Upstream adopts sub-tiling or the MetaX shared-memory limit permits
#            the original 128-token tile.
# -----------------------------------------------------------------------------
"""MetaX SUB_K fix for MiniMax-M3 block-sparse GQA attention kernels."""

from vllm.triton_utils import tl, triton

from vllm_metax.patch.utils import patch


@patch("vllm.models.minimax_m3.common.ops.sparse_attn", "_gqa_sparse_fwd_kernel")
@triton.heuristics(
    {
        "BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"]),
        # /-------------------- MetaX Modification --------------------\
        # Floored at 16: BLOCK_SIZE_QH (= BLOCK_SIZE_Q x BLOCK_SIZE_H) is the
        # M-dimension tile of `tl.dot(q, k)`; MetaX's MMA encoder requires it
        # to be >= 16 too, same as the decode kernel's existing BLOCK_SIZE_H
        # floor.
        "BLOCK_SIZE_H": lambda args: max(
            16, triton.next_power_of_2(args["gqa_group_size"])
        ),
        # \-------------------- MetaX Modification --------------------/
        "BLOCK_SIZE_QH": lambda args: args["BLOCK_SIZE_Q"]
        # /-------------------- MetaX Modification --------------------\
        * max(16, triton.next_power_of_2(args["gqa_group_size"])),
        # \-------------------- MetaX Modification --------------------/
        # /-------------------- MetaX Modification --------------------\
        # Largest power-of-two <= 128 that keeps both GEMM tiles >= 16:
        #   BLOCK_SIZE_H (>= gqa) x SUB_K >= 16  and  SUB_K x BLOCK_SIZE_D (>= 16).
        "SUB_K": lambda args: 16,
        # \-------------------- MetaX Modification --------------------/
    }
)
@triton.jit(do_not_specialize_on_alignment=["seq_lens", "prefix_lens"])
def _gqa_sparse_fwd_kernel(
    q_ptr,  # [total_q, num_heads, head_dim]
    kv_cache_ptr,  # main cache: [num_blocks, num_kv_heads, 128, 2*head_dim]
    k_scale_ptr,
    v_scale_ptr,
    t_ptr,  # topk_idx: [num_kv_heads, total_q, topk]
    o_ptr,  # [total_q, num_heads, head_dim]
    block_table_ptr,  # [num_reqs, max_blocks]
    cu_seqlens_q,
    cu_seqblocks_q,
    seq_lens,
    prefix_lens,
    num_kv_heads,
    gqa_group_size,
    head_dim,
    max_topk,
    num_q_loop,
    sm_scale,
    stride_qn,
    stride_qh,
    stride_qd,
    stride_kv_blk,
    stride_kv_h,
    stride_kv_pos,
    stride_kv_d,
    stride_ks_h,
    stride_ks_t,
    stride_vs_h,
    stride_vs_t,
    stride_th,
    stride_tn,
    stride_tk,
    stride_on,
    stride_oh,
    stride_od,
    stride_bt_b,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  # == SPARSE_BLOCK_SIZE (128)
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_QH: tl.constexpr,
    SUB_K: tl.constexpr,
    USE_FP8: tl.constexpr,  # fp8 KV cache: dequantize K/V to q.dtype on load
    KV_SCALE_MODE: tl.constexpr,  # 0: none, 1: scalar, 2: [kv_head, token]
):
    sm_scale_log2e = sm_scale * 1.4426950409
    pid_q = tl.program_id(0)
    pid_kh = tl.program_id(1)
    pid_b = tl.program_id(2)
    pid_h = pid_kh * gqa_group_size
    q_start = tl.load(cu_seqlens_q + pid_b)
    q_len = tl.load(cu_seqlens_q + pid_b + 1) - q_start
    q_block_start = tl.load(cu_seqblocks_q + pid_b)
    q_block_len = tl.load(cu_seqblocks_q + pid_b + 1) - q_block_start
    seq_len = tl.load(seq_lens + pid_b)
    prefix_len = tl.load(prefix_lens + pid_b)
    if pid_q * num_q_loop >= q_block_len:
        return
    real_q_loop = min(num_q_loop, q_block_len - pid_q * num_q_loop)
    bt_row = block_table_ptr + pid_b * stride_bt_b
    off_sk = tl.arange(0, SUB_K)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    d_mask = off_d < head_dim
    NUM_SUB: tl.constexpr = BLOCK_SIZE_K // SUB_K
    for j in range(real_q_loop):
        pid_q_j = pid_q * num_q_loop + j
        t_ptr_j = t_ptr + (q_block_start + pid_q_j) * stride_tn + pid_kh * stride_th
        # Valid block count from seq position (no sentinel): block_size_q == 1.
        q_abs = prefix_len + pid_q_j * BLOCK_SIZE_Q
        valid_blocks = (q_abs + BLOCK_SIZE_K) // BLOCK_SIZE_K
        real_topk = tl.minimum(max_topk, valid_blocks)
        q_ptrs = tl.make_block_ptr(
            base=q_ptr + q_start * stride_qn + pid_h * stride_qh,
            shape=(q_len, gqa_group_size, head_dim),
            strides=(stride_qn, stride_qh, stride_qd),
            offsets=(pid_q_j * BLOCK_SIZE_Q, 0, 0),
            block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_D),
            order=(2, 1, 0),
        )
        q = tl.load(q_ptrs, boundary_check=(0, 1, 2), padding_option="zero")
        off_q = tl.arange(0, BLOCK_SIZE_Q) + pid_q_j * BLOCK_SIZE_Q + prefix_len
        m_i = tl.full((BLOCK_SIZE_QH,), float("-inf"), dtype=tl.float32)
        lse_i = tl.full((BLOCK_SIZE_QH,), float("-inf"), dtype=tl.float32)
        acc_o = tl.zeros((BLOCK_SIZE_QH, BLOCK_SIZE_D), dtype=tl.float32)
        q = tl.reshape(q, BLOCK_SIZE_QH, BLOCK_SIZE_D)

        # MetaX: process each 128-token KV block in SUB_K-token sub-tiles so
        # the QK/PV working set stays below the per-SM shared-memory limit.
        # Carry the online-softmax state across both loops, making this
        # numerically equivalent to processing the full 128-token block.
        for _ in range(real_topk):
            blk = tl.load(t_ptr_j).to(tl.int32)
            t_ptr_j = t_ptr_j + stride_tk
            c = blk * BLOCK_SIZE_K
            page = tl.load(bt_row + blk).to(tl.int64)
            kv_base = kv_cache_ptr + page * stride_kv_blk + pid_kh * stride_kv_h
            for s in range(NUM_SUB):
                n_off = s * SUB_K
                off_sub = n_off + off_sk
                pos = c + off_sub  # [SUB_K] kv positions of this sub-tile
                pos_mask = pos < seq_len  # beyond seq_len -> padding
                k = tl.load(
                    kv_base
                    + off_sub[None, :] * stride_kv_pos
                    + off_d[:, None] * stride_kv_d,
                    mask=d_mask[:, None] & pos_mask[None, :],
                    other=0.0,
                )
                if USE_FP8:
                    k = k.to(q.dtype)
                    if KV_SCALE_MODE == 1:
                        k = (k * tl.load(k_scale_ptr)).to(q.dtype)
                    elif KV_SCALE_MODE == 2:
                        k_scale = tl.load(
                            k_scale_ptr
                            + pid_kh * stride_ks_h
                            + (page * BLOCK_SIZE_K + off_sub) * stride_ks_t,
                            mask=pos_mask,
                            other=1.0,
                        )
                        k = (k * k_scale[None, :]).to(q.dtype)
                # Build the causal mask with the current SUB_K offsets before
                # flattening Q and GQA-head dimensions, matching common's
                # q_abs_pos - key_offset >= block_start condition.
                off_q_sub = off_q[:, None] - off_sub[None, :]
                qk = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_H, SUB_K), dtype=tl.float32)
                qk += tl.where(off_q_sub[:, None, :] >= c, 0.0, float("-inf"))
                qk = tl.reshape(qk, BLOCK_SIZE_QH, SUB_K)
                qk += tl.dot(q, k) * sm_scale_log2e
                qk += tl.where(pos_mask[None, :], 0.0, float("-inf"))
                m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
                p = tl.exp2(qk - m_ij[:, None])
                l_ij = tl.sum(p, axis=1)
                acc_o = acc_o * tl.exp2(m_i - m_ij)[:, None]
                v = tl.load(
                    kv_base
                    + off_sub[:, None] * stride_kv_pos
                    + (head_dim + off_d[None, :]) * stride_kv_d,
                    mask=pos_mask[:, None] & d_mask[None, :],
                    other=0.0,
                )
                if USE_FP8:
                    v = v.to(q.dtype)
                    if KV_SCALE_MODE == 1:
                        v = (v * tl.load(v_scale_ptr)).to(q.dtype)
                    elif KV_SCALE_MODE == 2:
                        v_scale = tl.load(
                            v_scale_ptr
                            + pid_kh * stride_vs_h
                            + (page * BLOCK_SIZE_K + off_sub) * stride_vs_t,
                            mask=pos_mask,
                            other=1.0,
                        )
                        v = (v * v_scale[:, None]).to(q.dtype)
                acc_o += tl.dot(p.to(v.dtype), v)
                m_i = m_ij
                lse_i = m_ij + tl.log2(tl.exp2(lse_i - m_ij) + l_ij)
        acc_o = acc_o * tl.exp2(m_i - lse_i)[:, None]
        acc_o = tl.reshape(acc_o, BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_D)
        o_ptrs = tl.make_block_ptr(
            base=o_ptr + q_start * stride_on + pid_h * stride_oh,
            shape=(q_len, gqa_group_size, head_dim),
            strides=(stride_on, stride_oh, stride_od),
            offsets=(pid_q_j * BLOCK_SIZE_Q, 0, 0),
            block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_D),
            order=(2, 1, 0),
        )
        tl.store(o_ptrs, acc_o.to(o_ptr.dtype.element_ty), boundary_check=(0, 1, 2))


@patch("vllm.models.minimax_m3.common.ops.sparse_attn", "_gqa_sparse_decode_kernel")
@triton.heuristics(
    {
        "BLOCK_SIZE_H": lambda args: max(
            16, triton.next_power_of_2(args["gqa_group_size"])
        ),
        "BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"]),
        # /-------------------- MetaX Modification --------------------\
        # Largest power-of-two <= 128 keeping both GEMM tiles >= 16:
        #   BLOCK_SIZE_H (>= max(16,gqa)) x SUB_K >= 16  and  SUB_K x BLOCK_SIZE_D (>= 16).
        "SUB_K": lambda args: 16,
        # \-------------------- MetaX Modification --------------------/
    }
)
@triton.jit(do_not_specialize=["decode_query_len"])
def _gqa_sparse_decode_kernel(
    q_ptr,  # [total_q, num_heads, head_dim]
    kv_cache_ptr,  # main cache: [num_blocks, num_kv_heads, 128, 2*head_dim]
    k_scale_ptr,
    v_scale_ptr,
    t_ptr,  # topk_idx: [num_kv_heads, total_q, topk]
    o_ptr,  # partial out: [NUM_TOPK_CHUNKS, total_q, num_heads, head_dim]
    lse_ptr,  # partial lse (log2): [NUM_TOPK_CHUNKS, total_q, num_heads]
    block_table_ptr,  # [num_reqs, max_blocks]
    seq_lens,  # [num_reqs]
    total_q,
    gqa_group_size,
    head_dim,
    max_topk,
    sm_scale,
    decode_query_len,
    stride_qn,
    stride_qh,
    stride_qd,
    stride_kv_blk,
    stride_kv_h,
    stride_kv_pos,
    stride_kv_d,
    stride_ks_h,
    stride_ks_t,
    stride_vs_h,
    stride_vs_t,
    stride_th,
    stride_tn,
    stride_tk,
    stride_o_c,
    stride_o_b,
    stride_o_h,
    stride_o_d,
    stride_l_c,
    stride_l_b,
    stride_l_h,
    stride_bt_b,
    BLOCK_SIZE_K: tl.constexpr,  # == SPARSE_BLOCK_SIZE (128)
    NUM_TOPK_CHUNKS: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    SUB_K: tl.constexpr,
    USE_FP8: tl.constexpr,  # fp8 KV cache: dequantize K/V to q.dtype on load
    KV_SCALE_MODE: tl.constexpr,  # 0: none, 1: scalar, 2: [kv_head, token]
    USE_PDL: tl.constexpr,
):
    sm_scale_log2e = sm_scale * 1.4426950409
    # split-K over the topk dimension: pid(0) folds (query-token, chunk).
    pid_bc, pid_kh = tl.program_id(0), tl.program_id(1)
    pid_b = pid_bc % total_q
    pid_c = pid_bc // total_q
    req_id = pid_b // decode_query_len
    q_offset = pid_b - req_id * decode_query_len
    pid_h = pid_kh * gqa_group_size
    chunk_size_topk = (max_topk + NUM_TOPK_CHUNKS - 1) // NUM_TOPK_CHUNKS
    chunk_start_topk = pid_c * chunk_size_topk
    chunk_end_compiletime = chunk_start_topk + chunk_size_topk

    if USE_PDL:
        tl.extra.cuda.gdc_wait()

    seq_len = tl.load(seq_lens + req_id)
    query_pos = seq_len - decode_query_len + q_offset
    # Full-CG padding uses zero-length request rows. Clamp to an empty
    # attention range instead of letting padded rows produce negative lengths.
    kv_len = tl.maximum(query_pos + 1, 0)

    # Valid block count from seq_len (no sentinel): min(topk, cdiv(kv_len, blk)).
    idx_base = t_ptr + pid_kh * stride_th + pid_b * stride_tn
    num_blocks = (kv_len + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    real_topk = tl.minimum(max_topk, num_blocks)
    chunk_end_topk = tl.minimum(chunk_end_compiletime, real_topk)

    off_sk = tl.arange(0, SUB_K)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    d_mask = off_d < head_dim
    bt_row = block_table_ptr + req_id * stride_bt_b
    NUM_SUB: tl.constexpr = BLOCK_SIZE_K // SUB_K

    m_i = tl.full((BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)
    lse_i = tl.full((BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)
    acc_o = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_D), dtype=tl.float32)
    q_ptrs = tl.make_block_ptr(
        base=q_ptr + pid_b * stride_qn + pid_h * stride_qh,
        shape=(gqa_group_size, head_dim),
        strides=(stride_qh, stride_qd),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_H, BLOCK_SIZE_D),
        order=(1, 0),
    )
    q = tl.load(q_ptrs, boundary_check=(0, 1), padding_option="zero")

    cur_idx_ptr = idx_base + chunk_start_topk * stride_tk
    for _ in tl.range(chunk_start_topk, chunk_end_topk):
        blk = tl.load(cur_idx_ptr).to(tl.int32)
        cur_idx_ptr = cur_idx_ptr + stride_tk
        c = blk * BLOCK_SIZE_K
        page = tl.load(bt_row + blk).to(tl.int64)
        kv_base = kv_cache_ptr + page * stride_kv_blk + pid_kh * stride_kv_h
        for s in range(NUM_SUB):
            n_off = s * SUB_K
            off_sub = n_off + off_sk
            pos = c + off_sub  # [SUB_K] kv positions of this sub-tile
            pos_mask = pos < kv_len  # beyond kv_len -> padding
            k = tl.load(
                kv_base
                + off_sub[None, :] * stride_kv_pos
                + off_d[:, None] * stride_kv_d,
                mask=d_mask[:, None] & pos_mask[None, :],
                other=0.0,
            )
            if USE_FP8:
                k = k.to(q.dtype)
                if KV_SCALE_MODE == 1:
                    k = (k * tl.load(k_scale_ptr)).to(q.dtype)
                elif KV_SCALE_MODE == 2:
                    k_scale = tl.load(
                        k_scale_ptr
                        + pid_kh * stride_ks_h
                        + (page * BLOCK_SIZE_K + off_sub) * stride_ks_t,
                        mask=pos_mask,
                        other=1.0,
                    )
                    k = (k * k_scale[None, :]).to(q.dtype)
            qk = tl.dot(q, k) * sm_scale_log2e  # [BLOCK_SIZE_H, SUB_K]
            # causal: keep iff query_pos >= pos (equiv. pos < kv_len); padding
            # positions (pos >= kv_len) are masked out too.
            qk += tl.where(query_pos >= pos[None, :], 0.0, float("-inf"))
            qk += tl.where(pos_mask[None, :], 0.0, float("-inf"))
            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.exp2(qk - m_ij[:, None])
            l_ij = tl.sum(p, axis=1)
            acc_o = acc_o * tl.exp2(m_i - m_ij)[:, None]
            v = tl.load(
                kv_base
                + off_sub[:, None] * stride_kv_pos
                + (head_dim + off_d[None, :]) * stride_kv_d,
                mask=pos_mask[:, None] & d_mask[None, :],
                other=0.0,
            )
            if USE_FP8:
                v = v.to(q.dtype)
                if KV_SCALE_MODE == 1:
                    v = (v * tl.load(v_scale_ptr)).to(q.dtype)
                elif KV_SCALE_MODE == 2:
                    v_scale = tl.load(
                        v_scale_ptr
                        + pid_kh * stride_vs_h
                        + (page * BLOCK_SIZE_K + off_sub) * stride_vs_t,
                        mask=pos_mask,
                        other=1.0,
                    )
                    v = (v * v_scale[:, None]).to(q.dtype)
            acc_o += tl.dot(p.to(v.dtype), v)
            m_i = m_ij
            lse_i = m_ij + tl.log2(tl.exp2(lse_i - m_ij) + l_ij)

    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()

    # Empty chunks for active rows must store zero output; otherwise the merge
    # can hit 0 * NaN. All-empty padded rows may still produce NaNs in merge.
    scale = tl.where(lse_i > float("-inf"), tl.exp2(m_i - lse_i), tl.zeros_like(lse_i))
    acc_o = acc_o * scale[:, None]
    o_ptrs = tl.make_block_ptr(
        base=o_ptr + pid_c * stride_o_c + pid_b * stride_o_b + pid_h * stride_o_h,
        shape=(gqa_group_size, head_dim),
        strides=(stride_o_h, stride_o_d),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_H, BLOCK_SIZE_D),
        order=(1, 0),
    )
    tl.store(o_ptrs, acc_o.to(o_ptr.dtype.element_ty), boundary_check=(0, 1))
    lse_ptrs = tl.make_block_ptr(
        base=lse_ptr + pid_c * stride_l_c + pid_b * stride_l_b + pid_h * stride_l_h,
        shape=(gqa_group_size,),
        strides=(stride_l_h,),
        offsets=(0,),
        block_shape=(BLOCK_SIZE_H,),
        order=(0,),
    )
    tl.store(lse_ptrs, lse_i.to(lse_ptr.dtype.element_ty), boundary_check=(0,))
