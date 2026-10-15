# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#
# -----------------------------------------------------------------------------
# Note: On the chunked-context MLA prefill path, upstream copies the gathered
# kv_c_normed tensor to kv_b_proj.weight.dtype before the projection. For
# compressed-tensors W8A8-int8 checkpoints kv_b_proj.weight is stored as
# int8, so this turns the model-dtype activations into int8 first; the int8
# scaled-MM kernel then quantizes them again with dynamic_scaled_int8_quant,
# which has no kernel overload for int8 ('Char') input and raises
# NotImplementedError. The fp8/uint8 cases are already excluded upstream;
# int8 needs the same exclusion because the int8 kernel quantizes internally.
#
# Mirrors the v0.23 fix (e587de89 "[Bugfix][MLA] Fix int8_w8a8 mla crash while
# chunk_prefill") that exists in vllm_metax's own MLA copy but was lost on the
# v0.26 base-vLLM path used by MultiHeadLatentAttentionWrapper.
#
# Affected versions: v0.26.0
#
# Remove at: Once upstream keeps model-dtype input for int8 W8A8 kv_b_proj
#            in MLACommonBaseImpl._compute_prefill_context.
# -----------------------------------------------------------------------------

import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from vllm.v1.attention.ops.merge_attn_states import merge_attn_states
from vllm.v1.attention.ops.triton_merge_attn_states import mask_empty_context

from vllm_metax.patch.utils import patch


@patch(
    "vllm.model_executor.layers.attention.mla_attention",
    "MLACommonBaseImpl._compute_prefill_context",
)
def _compute_prefill_context(
    self,
    q,
    kv_c_and_k_pe_cache,
    attn_metadata,
    k_scale,
):
    assert attn_metadata.prefill is not None
    prefill_metadata = attn_metadata.prefill
    assert prefill_metadata.prefill_backend is not None
    assert prefill_metadata.chunked_context is not None

    use_fp8_prefill = prefill_metadata.q_data_type == current_platform.fp8_dtype()

    output = None
    merge_output = None
    iters = len(prefill_metadata.chunked_context.seq_tot)
    workspace = prefill_metadata.chunked_context.workspace

    if use_fp8_prefill:
        q = q.to(prefill_metadata.q_data_type)

    for i in range(iters):
        toks = prefill_metadata.chunked_context.seq_tot[i]
        if self.kv_cache_dtype == "fp8_ds_mla":
            ops.cp_gather_and_upconvert_fp8_kv_cache(
                src_cache=kv_c_and_k_pe_cache,
                dst=workspace[:toks],
                block_table=prefill_metadata.block_table,
                workspace_starts=prefill_metadata.chunked_context.cu_seq_lens[i],
                batch_size=attn_metadata.num_prefills,
                seq_starts=prefill_metadata.chunked_context.starts[i],
            )
        elif not use_fp8_prefill:
            ops.gather_and_maybe_dequant_cache(
                src_cache=kv_c_and_k_pe_cache,
                dst=workspace,
                block_table=prefill_metadata.block_table,
                cu_seq_lens=prefill_metadata.chunked_context.cu_seq_lens[i],
                token_to_seq=prefill_metadata.chunked_context.token_to_seq[i],
                num_tokens=prefill_metadata.chunked_context.chunk_total_token[i],
                kv_cache_dtype=self.kv_cache_dtype,
                scale=k_scale,
                seq_starts=prefill_metadata.chunked_context.starts[i],
            )
        else:
            # FP8 path: gather cache without dequantization
            ops.cp_gather_cache(
                src_cache=kv_c_and_k_pe_cache,
                dst=workspace,
                block_table=prefill_metadata.block_table,
                cu_seq_lens=prefill_metadata.chunked_context.cu_seq_lens[i],
                batch_size=attn_metadata.num_prefills,
                seq_starts=prefill_metadata.chunked_context.starts[i],
            )

        # Extract kv_c_normed from workspace
        kv_c_normed = workspace[:toks][..., : self.kv_lora_rank]
        # When FP8 weights are used without FP8 prefill, kv_b_proj expects
        # model dtype input and will quantize internally.
        # For quantized layers (AWQ/GPTQ) that lack a .weight attribute,
        # use params_dtype which is the expected input dtype.
        _kv_b_proj_w_dtype = (
            self.kv_b_proj.weight.dtype
            if hasattr(self.kv_b_proj, "weight")
            else self.kv_b_proj.params_dtype
        )
        # For NVFP4, weights are packed uint8 -- keep input in model dtype
        # since the NVFP4 linear layer quantizes internally. Same for int8
        # W8A8 weights: the scaled-MM kernel quantizes model-dtype input.
        # ┌------------------------  Metax Modification -------------------------┐
        if (
            (use_fp8_prefill or _kv_b_proj_w_dtype != current_platform.fp8_dtype())
            and _kv_b_proj_w_dtype != torch.uint8
            and _kv_b_proj_w_dtype != torch.int8
        ):
            kv_c_normed = kv_c_normed.to(_kv_b_proj_w_dtype)
        # └------------------------- Metax Modification -------------------------┘

        k_pe = workspace[:toks][..., self.kv_lora_rank :].unsqueeze(1)
        kv_nope = self.kv_b_proj(kv_c_normed)[0].view(
            -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
        )

        # To Do: Use epilogue of kv_b_proj to generate fp8 kv_nope.
        if use_fp8_prefill:
            kv_nope = kv_nope.to(prefill_metadata.q_data_type)
            k_pe = k_pe.to(prefill_metadata.q_data_type)
        k_nope, v = kv_nope.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        k = self._concat_k_nope_k_pe(k_nope, k_pe)

        attn_output, attn_softmax_lse = (
            prefill_metadata.prefill_backend.run_prefill_context_chunk(
                chunk_idx=i,
                q=q,
                k=k,
                v=v,
            )
        )
        if prefill_metadata.chunked_context.has_empty_context[i]:
            mask_empty_context(
                attn_softmax_lse,
                attn_output,
                prefill_metadata.query_start_loc,
                prefill_metadata.chunked_context.cu_seq_lens[i],
            )

        if output is None:
            output = attn_output
            output_lse = attn_softmax_lse
        else:
            if merge_output is None:
                merge_output = torch.empty_like(output)
                merge_output_lse = torch.empty_like(output_lse)
            merge_attn_states(
                output=merge_output,
                output_lse=merge_output_lse,
                prefix_output=output,
                prefix_lse=output_lse,
                suffix_output=attn_output,
                suffix_lse=attn_softmax_lse,
            )
            output, merge_output = merge_output, output
            output_lse, merge_output_lse = merge_output_lse, output_lse

    return output, output_lse
