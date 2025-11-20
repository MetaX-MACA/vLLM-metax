# 2025 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import vllm
from vllm_metax_plugin.patch.hook_registry import register_patch
from vllm.logger import init_logger

logger = init_logger(__name__)

import triton
from vllm.attention.ops.triton_decode_attention import _fwd_grouped_kernel_stage1, _fwd_kernel_stage2, decode_attention_fwd_normal

def _decode_grouped_att_m_fwd(
    q,
    k_buffer,
    v_buffer,
    att_out,
    Req_to_tokens,
    B_Seqlen,
    num_kv_splits,
    num_stages,
    sm_scale,
    page_size,
    logit_cap,
):
    BLOCK = 16
    Lk = k_buffer.shape[-1]
    Lv = v_buffer.shape[-1]

    # [TODO] work around shmem limit on MI3xx
    # if is_hip_ and Lk >= 576:
    #     BLOCK = 16

    if Lk == 576:
        BLOCK_DMODEL = 512
        BLOCK_DPE = 64
    elif Lk == 288:
        BLOCK_DMODEL = 256
        BLOCK_DPE = 32
    else:
        BLOCK_DMODEL = triton.next_power_of_2(Lk)
        BLOCK_DPE = 0
    BLOCK_DV = triton.next_power_of_2(Lv)

    batch, head_num = q.shape[0], q.shape[1]
    kv_group_num = q.shape[1] // k_buffer.shape[-2]

    BLOCK_H = 16
    NUM_KV_SPLITS = num_kv_splits
    grid = (
        batch,
        triton.cdiv(head_num, min(BLOCK_H, kv_group_num)),
        NUM_KV_SPLITS,
    )

    if num_stages == 1:
        extra_kargs = {"scenario":"mla"}
    elif num_stages == 2:
        extra_kargs = {"scenario" : "mla", "pipeline" : "cpasync"}
    else:
        KeyError("num_stages should be 1 or 2")  
    """
    num_stages = 2
    extra_kargs = {}
    if is_hip_:
        # https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html#mi300x-triton-kernel-performance-optimization
        # https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
        extra_kargs = {
            "waves_per_eu": 1,
            "matrix_instr_nonkdim": 16,
            "kpack": 2
        }
        num_stages = 1
    """

    _fwd_grouped_kernel_stage1[grid](
        q,
        k_buffer,
        v_buffer,
        sm_scale,
        Req_to_tokens,
        B_Seqlen,
        att_out,
        Req_to_tokens.stride(0),
        q.stride(0),
        q.stride(1),
        k_buffer.stride(-3),  # Assume (..., PAGE_SIZE, NUM_HEADS, HEAD_DIM)
        k_buffer.stride(-2),  # Assume (..., PAGE_SIZE, NUM_HEADS, HEAD_DIM)
        v_buffer.stride(-3),  # Assume (..., PAGE_SIZE, NUM_HEADS, HEAD_DIM)
        v_buffer.stride(-2),  # Assume (..., PAGE_SIZE, NUM_HEADS, HEAD_DIM)
        att_out.stride(0),
        att_out.stride(1),
        att_out.stride(2),
        kv_group_num=kv_group_num,
        q_head_num=head_num,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DPE=BLOCK_DPE,
        BLOCK_DV=BLOCK_DV,
        BLOCK_N=BLOCK,
        BLOCK_H=BLOCK_H,
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        PAGE_SIZE=page_size,
        logit_cap=logit_cap,
        num_warps=4,
        num_stages=num_stages,
        Lk=Lk,
        Lv=Lv,
        **extra_kargs,
    )

def _decode_softmax_reducev_fwd(
    logits,
    q,
    o,
    v_buffer,
    b_seq_len,
    num_kv_splits,
):

    batch, head_num = q.shape[0], q.shape[1]
    Lv = v_buffer.shape[-1]
    BLOCK_DV = triton.next_power_of_2(Lv)

    NUM_KV_SPLITS = num_kv_splits

    extra_kargs = {}
    """
    if is_hip_:
        # https://rocm.docs.amd.com/en/docs-6.2.0/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
        # https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
        extra_kargs = {
            "waves_per_eu": 4,
            "matrix_instr_nonkdim": 16,
            "kpack": 2
        }
    """

    grid = (batch, head_num)
    _fwd_kernel_stage2[grid](
        logits,
        o,
        b_seq_len,
        logits.stride(0),
        logits.stride(1),
        logits.stride(2),
        o.stride(0),
        o.stride(1),
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        BLOCK_DV=BLOCK_DV,
        Lv=Lv,
        num_warps=4,
        num_stages=2,
        **extra_kargs,
    )
    
def decode_attention_fwd_grouped(
    q,
    k_buffer,
    v_buffer,
    o,
    req_to_token,
    b_seq_len,
    attn_logits,
    num_kv_splits,
    num_stages,
    sm_scale,
    page_size,
    logit_cap=0.0,
):
    _decode_grouped_att_m_fwd(
        q,
        k_buffer,
        v_buffer,
        attn_logits,
        req_to_token,
        b_seq_len,
        num_kv_splits,
        num_stages,
        sm_scale,
        page_size,
        logit_cap,
    )
    _decode_softmax_reducev_fwd(attn_logits, q, o, v_buffer, b_seq_len,
                                num_kv_splits)

def decode_attention_fwd(
    q,
    k_buffer,
    v_buffer,
    o,
    req_to_token,
    b_seq_len,
    attn_logits,
    num_kv_splits,
    num_stages,
    sm_scale,
    page_size=1,
    logit_cap=0.0,
):
    assert num_kv_splits == attn_logits.shape[2]
    kv_group_num = q.shape[1] // v_buffer.shape[-2]

    if kv_group_num == 1:
        # MHA
        decode_attention_fwd_normal(
            q,
            k_buffer,
            v_buffer,
            o,
            req_to_token,
            b_seq_len,
            attn_logits,
            num_kv_splits,
            sm_scale,
            page_size,
            logit_cap,
        )
    else:
        # GQA/MQA/MLA
        decode_attention_fwd_grouped(
            q,
            k_buffer,
            v_buffer,
            o,
            req_to_token,
            b_seq_len,
            attn_logits,
            num_kv_splits,
            num_stages,
            sm_scale,
            page_size,
            logit_cap,
        )
        
vllm.attention.ops.triton_decode_attention._decode_grouped_att_m_fwd = _decode_grouped_att_m_fwd
vllm.attention.ops.triton_decode_attention._decode_softmax_reducev_fwd = _decode_softmax_reducev_fwd
vllm.attention.ops.triton_decode_attention.decode_attention_fwd_grouped = decode_attention_fwd_grouped
vllm.attention.ops.triton_decode_attention.decode_attention_fwd = decode_attention_fwd

register_patch("vllm.attention.ops.triton_decode_attention", "_decode_grouped_att_m_fwd", _decode_grouped_att_m_fwd)
register_patch("vllm.attention.ops.triton_decode_attention", "_decode_softmax_reducev_fwd", _decode_softmax_reducev_fwd)
register_patch("vllm.attention.ops.triton_decode_attention", "decode_attention_fwd_grouped", decode_attention_fwd_grouped)
register_patch("vllm.attention.ops.triton_decode_attention", "decode_attention_fwd", decode_attention_fwd)
