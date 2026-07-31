# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Custom Sparse Attention Indexer layers."""

import torch

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.distributed import get_pcp_group
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.pcp import maybe_gather_indexer_k
from vllm.platforms import current_platform
from vllm_metax.utils.deep_gemm import (
    bf16_mqa_logits,
    bf16_paged_mqa_logits,
)
from vllm.utils.torch_utils import (
    LayerNameType,
    _resolve_layer_name,
    direct_register_custom_op,
)
from vllm_metax.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadata,
)
from vllm.v1.attention.ops.common import pack_seq_triton, unpack_seq_triton
from vllm.v1.worker.workspace import current_workspace_manager
from vllm.model_executor.layers.sparse_attn_indexer import (
    kv_cache_as_quant_view,
    _merge_dcp_topk_global,
)

from vllm_metax import _custom_ops as mx_ops

logger = init_logger(__name__)

RADIX_TOPK_WORKSPACE_SIZE = 1024 * 1024


def _gather_workspace_shapes_bf16(
    total_seq_lens: int,
    head_dim: int,
    bf16_dtype: torch.dtype,
) -> tuple[tuple[int, int], torch.dtype]:
    """Return (values_shape, values_dtype) for
    the K-gather workspace."""
    return ((total_seq_lens, head_dim), bf16_dtype)


@eager_break_during_capture
def sparse_attn_indexer_bf16(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_bf16: torch.Tensor,
    q_scale: torch.Tensor | None,
    k_bf16: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor,
    skip_k_cache_insert: bool,
    use_pcp: bool,
    use_fp4_cache: bool = False,
    dcp_rank: int = 0,
    dcp_world_size: int = 1,
    cp_kv_cache_interleave_size: int = 1,
    skip_topk_buffer_clear: bool = False,
) -> torch.Tensor:
    # careful! this will be None in dummy run
    attn_metadata = get_forward_context().attn_metadata
    k_cache_prefix = _resolve_layer_name(k_cache_prefix)

    # assert isinstance(attn_metadata, dict)
    if not isinstance(attn_metadata, dict):
        # Reserve workspace for indexer during profiling run
        values_spec = _gather_workspace_shapes_bf16(
            total_seq_lens, head_dim, torch.bfloat16
        )
        # ----------------------------------------------------
        # Metax Note: we use bf16 instead of fp8 here, so we need to
        # preare workspace only for k_bf16, and skip k_scale (bf16 does not need scale)
        current_workspace_manager().get_simultaneous(
            values_spec,
            ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
        )

        # Dummy allocation to simulate for peak logits tensor memory during inference.
        # FP8 elements so elements == bytes
        max_logits_elems = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
        _ = torch.empty(
            max_logits_elems, dtype=torch.uint8, device=hidden_states.device
        )

        return sparse_attn_indexer_bf16_fake(
            hidden_states,
            k_cache_prefix,
            kv_cache,
            q_bf16,
            q_scale,
            k_bf16,
            weights,
            quant_block_size,
            scale_fmt,
            topk_tokens,
            head_dim,
            max_model_len,
            total_seq_lens,
            topk_indices_buffer,
            skip_k_cache_insert,
            use_pcp,
            use_fp4_cache,
        )
    attn_metadata_narrowed = attn_metadata[k_cache_prefix]
    assert isinstance(attn_metadata_narrowed, DeepseekV32IndexerMetadata)
    slot_mapping = attn_metadata_narrowed.slot_mapping
    has_decode = attn_metadata_narrowed.num_decodes > 0
    has_prefill = attn_metadata_narrowed.num_prefills > 0
    num_decode_tokens = attn_metadata_narrowed.num_decode_tokens

    assert q_scale is None, "q_scale is not needed for bf16 indexer"
    # During speculative decoding, k may be padded to the CUDA graph batch
    # size while slot_mapping only covers actual tokens. Truncate k to avoid
    # out-of-bounds reads in the kernel.
    # Keep PCP padding so every rank contributes the same all-gather shape.
    num_tokens = slot_mapping.shape[0]
    if use_pcp:
        num_tokens //= get_pcp_group().world_size
    if k_bf16 is not None:
        k_bf16 = k_bf16[:num_tokens]

    if not skip_k_cache_insert:
        assert k_bf16 is not None
        k_bf16, slot_mapping_for_cache = maybe_gather_indexer_k(
            k_bf16,
            slot_mapping,
            num_decode_tokens,
            use_pcp,
        )
        # scale_fmt can be None, but the function expects str
        assert scale_fmt is not None
        assert not use_fp4_cache, "Unfused FP4 Insert is not supported yet"
        mx_ops.indexer_k_quant_and_cache(
            k_bf16,
            kv_cache,
            slot_mapping_for_cache,
            quant_block_size,
            scale_fmt,
        )

    # The buffer must be pre-filled with -1 (the "no token" sentinel) before the
    # top-k kernels scatter valid indices into it. On the fused deepseek_v32
    # nvidia path, _fused_norm_rope_kernel already cleared the same
    # [:num_tokens, :topk] region earlier in this forward, so skip the redundant
    # fill.
    if not skip_topk_buffer_clear:
        topk_indices_buffer[: hidden_states.shape[0]] = -1
    if has_prefill:
        prefill_metadata = attn_metadata_narrowed.prefill
        assert prefill_metadata is not None

        # Get the full shared workspace buffers once (will allocate on first use).

        # ----------------------------------------------------
        # Metax Note: we use bf16 instead of fp8 here, so we need to
        # preare workspace only for k_bf16, and skip k_scale_full (bf16 does not need scale)
        workspace_manager = current_workspace_manager()
        values_spec = _gather_workspace_shapes_bf16(
            total_seq_lens, head_dim, torch.bfloat16
        )
        k_bf16_full = workspace_manager.get_simultaneous(
            ((total_seq_lens, head_dim), torch.bfloat16),
        )[0]
        for chunk in prefill_metadata.chunks:
            cu_seqlen_ks = chunk.cu_seqlen_ks
            cu_seqlen_ke = chunk.cu_seqlen_ke
            assert chunk.local_cu_seq_lens is not None
            k_bf16 = k_bf16_full[: chunk.max_local_total_seq_lens]

            # -----------------------------------------------
            # Metax Note: we use bf16 instead of fp8 here, so k_scale is
            # not needed and set to None
            k_scale = None  # k_scale_full[: chunk.max_local_total_seq_lens]
            if not chunk.skip_kv_gather and chunk.local_total_seq_lens > 0:
                mx_ops.cp_gather_indexer_k_quant_cache(
                    kv_cache,
                    k_bf16,
                    k_scale,
                    chunk.block_table,
                    chunk.local_cu_seq_lens,
                )

            q_slice = q_bf16[chunk.token_start : chunk.token_end]
            q_scale_slice = None  # noqa: F841

            topk_indices = topk_indices_buffer[
                chunk.token_start : chunk.token_end, :topk_tokens
            ]

            if chunk.local_total_seq_lens == 0:
                logits = q_slice.new_empty((q_slice.shape[0], 0), dtype=torch.float32)
                topk_indices.fill_(-1)
            else:
                q_slice_cast = q_slice
                k_quant_cast = k_bf16
                k_scale_cast = None  # noqa: F841
                # -----------------------------------------------
                # Metax Note: since we use bf16 so the args for
                # kv tuple is changed:
                #   - for fp8_mqa_logits it is a tuple of (k_fp8, k_scale),
                #   - and bf16_mqa_logits it is just k_bf16 (no scale)
                logits = bf16_mqa_logits(
                    q_slice_cast,
                    k_quant_cast,
                    weights[chunk.token_start : chunk.token_end],
                    cu_seqlen_ks,
                    cu_seqlen_ke,
                )
                num_rows = logits.shape[0]
                ops.top_k_per_row_prefill(
                    logits,
                    cu_seqlen_ks,
                    cu_seqlen_ke,
                    topk_indices,
                    num_rows,
                    logits.stride(0),
                    logits.stride(1),
                    topk_tokens,
                )

            _merge_dcp_topk_global(
                logits,
                topk_indices,
                topk_tokens,
                dcp_rank,
                dcp_world_size,
                cp_kv_cache_interleave_size,
                row_starts=chunk.cu_seqlen_ks,
            )

    if has_decode:
        decode_metadata = attn_metadata_narrowed.decode
        assert decode_metadata is not None
        # kv_cache size requirement [num_block, block_size, n_head, head_dim],
        # we only have [num_block, block_size, head_dim],
        kv_cache = kv_cache_as_quant_view(kv_cache, head_dim, False)
        decode_lens = decode_metadata.decode_lens
        if num_decode_tokens == 0:
            padded_q_bf16_decode_tokens = q_bf16[:1].reshape(1, 1, *q_bf16.shape[1:])
        elif decode_metadata.requires_padding:
            # pad in edge case where we have short chunked prefill length <
            # decode_threshold since we unstrictly split
            # prefill and decode by decode_threshold
            # (currently set to 1 + speculative tokens).
            padded_q_bf16_decode_tokens = pack_seq_triton(
                q_bf16[:num_decode_tokens], decode_lens
            )
        else:
            padded_q_bf16_decode_tokens = q_bf16[:num_decode_tokens].reshape(
                decode_lens.shape[0], -1, *q_bf16.shape[1:]
            )
        # TODO: move and optimize below logic with triton kernels
        batch_size = padded_q_bf16_decode_tokens.shape[0]
        next_n = padded_q_bf16_decode_tokens.shape[1]
        assert batch_size == decode_metadata.seq_lens.shape[0]
        num_padded_tokens = batch_size * next_n
        seq_lens = decode_metadata.seq_lens[:batch_size]
        logits = bf16_paged_mqa_logits(
            padded_q_bf16_decode_tokens,
            kv_cache,
            weights[:num_padded_tokens],
            seq_lens,
            decode_metadata.block_table,
            decode_metadata.schedule_metadata,
            max_model_len=max_model_len,
            clean_logits=False,
        )
        num_rows = logits.shape[0]
        topk_indices = topk_indices_buffer[:num_padded_tokens, :topk_tokens]

        use_cooperative_topk = (
            current_platform.is_cuda_alike()
            and topk_tokens in (512, 1024, 2048)
            and num_rows <= 32
            and logits.stride(0) % 4 == 0  # TMA 16-byte alignment
        )
        use_persistent_topk = current_platform.is_cuda_alike() and topk_tokens in (
            512,
            1024,
            2048,
        )
        if use_cooperative_topk:
            workspace_manager = current_workspace_manager()
            (topk_workspace,) = workspace_manager.get_simultaneous(
                ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
            )
            torch.ops._C.cooperative_topk(
                logits,
                seq_lens,
                topk_indices,
                topk_workspace,
                topk_tokens,
                attn_metadata_narrowed.max_seq_len,
            )
        elif use_persistent_topk:
            workspace_manager = current_workspace_manager()
            (topk_workspace,) = workspace_manager.get_simultaneous(
                ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
            )
            torch.ops._C.persistent_topk(
                logits,
                seq_lens,
                topk_indices,
                topk_workspace,
                topk_tokens,
                logits.shape[1],
            )
        else:
            ops.top_k_per_row_decode(
                logits,
                next_n,
                seq_lens,
                topk_indices,
                num_rows,
                logits.stride(0),
                logits.stride(1),
                topk_tokens,
            )

        if decode_metadata.global_seq_lens is not None:
            _merge_dcp_topk_global(
                logits,
                topk_indices,
                topk_tokens,
                dcp_rank,
                dcp_world_size,
                cp_kv_cache_interleave_size,
            )

        if decode_metadata.requires_padding:
            # if padded, we need to unpack
            # the topk indices removing padded tokens
            topk_indices = unpack_seq_triton(
                topk_indices.reshape(batch_size, -1, topk_indices.shape[-1]),
                decode_lens,
            )
            topk_indices_buffer[: topk_indices.shape[0], : topk_indices.shape[-1]] = (
                topk_indices
            )

    return topk_indices_buffer


def sparse_attn_indexer_bf16_fake(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_quant: torch.Tensor,
    q_scale: torch.Tensor | None,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor | None,
    skip_k_cache_insert: bool,
    use_pcp: bool,
    use_fp4_cache: bool = False,
    dcp_rank: int = 0,
    dcp_world_size: int = 1,
    cp_kv_cache_interleave_size: int = 1,
    skip_topk_buffer_clear: bool = False,
) -> torch.Tensor:
    return topk_indices_buffer


direct_register_custom_op(
    op_name="mx_sparse_attn_indexer_bf16",
    op_func=sparse_attn_indexer_bf16,
    mutates_args=["topk_indices_buffer"],
    fake_impl=sparse_attn_indexer_bf16_fake,
    dispatch_key=current_platform.dispatch_key,
)
