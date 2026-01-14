# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

# ------------------------------------------------------------------------
# Note: This file is a patch to DP4+MTP3 flashmla exceded Max_Spilts issue
# ------------------------------------------------------------------------

import torch
import numpy as np
from dataclasses import dataclass

import vllm.v1.attention.backends.utils
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.worker.ubatch_utils import UBatchSlice
from vllm.v1.attention.backends.utils import slice_query_start_locs
@dataclass
class MacaCommonAttentionMetadata(CommonAttentionMetadata):
    # /------------------------  Metax Modification -------------------------\
    # TODO(lucas): remove once we have FULL-CG spec-decode support
    def unpadded(
        self, num_actual_tokens: int, num_actual_reqs: int
    ) -> "MacaCommonAttentionMetadata":
        maybe_slice_reqs = lambda x: x[:num_actual_reqs] if x is not None else None
        return MacaCommonAttentionMetadata(
            query_start_loc=self.query_start_loc[: num_actual_reqs + 1],
            query_start_loc_cpu=self.query_start_loc_cpu[: num_actual_reqs + 1],
            seq_lens=self.seq_lens[:num_actual_reqs],
            seq_lens_cpu=self.seq_lens_cpu[:num_actual_reqs],
            num_computed_tokens_cpu=self.num_computed_tokens_cpu[:num_actual_reqs],
            num_reqs=num_actual_reqs,
            num_actual_tokens=num_actual_tokens,
            max_query_len=self.max_query_len,
            max_seq_len=self.max_seq_len,
            block_table_tensor=self.block_table_tensor[:num_actual_reqs],
            slot_mapping=self.slot_mapping[:num_actual_tokens],
            causal=self.causal,
            logits_indices_padded=self.logits_indices_padded,
            num_logits_indices=self.num_logits_indices,
            dcp_local_seq_lens=maybe_slice_reqs(self.dcp_local_seq_lens),
            dcp_local_seq_lens_cpu=maybe_slice_reqs(self.dcp_local_seq_lens_cpu),
        )
    # /------------------------  Metax Modification -------------------------\

def Maca_make_metadata_with_slice(
    ubatch_slice: UBatchSlice, attn_metadata: CommonAttentionMetadata
) -> CommonAttentionMetadata:
    """
    This function creates a new CommonAttentionMetadata that corresponds to
    the requests included in ubatch_slice
    """

    assert not ubatch_slice.is_empty(), f"Ubatch slice {ubatch_slice} is empty"

    request_slice = ubatch_slice.request_slice
    token_slice = ubatch_slice.token_slice

    start_locs = attn_metadata.query_start_loc_cpu
    first_req = request_slice.start
    first_tok = token_slice.start
    last_req = request_slice.stop - 1
    last_tok = token_slice.stop - 1

    assert start_locs[first_req] <= first_tok < start_locs[first_req + 1], (
        "Token slice start outside of first request"
    )
    # /------------------------  Metax Modification -------------------------\
    # NOTE: last token can be outside of the last request if we have CG padding.
    # \------------------------  Metax Modification -------------------------/

    # If the "middle" request has tokens in both ubatches, we have to split it.
    # If ubatch_slice is the first ubatch then we will be splitting the last
    # request. If it's the second microbatch, then we will be splitting the
    # first request
    splits_first_request = first_tok > start_locs[first_req]
    splits_last_request = last_tok < start_locs[last_req + 1] - 1

    query_start_loc_cpu = slice_query_start_locs(start_locs, request_slice)
    query_start_loc = slice_query_start_locs(
        attn_metadata.query_start_loc, request_slice
    )

    assert len(query_start_loc) >= 2, (
        f"query_start_loc must have at least 2 elements, got {len(query_start_loc)}"
    )

    if splits_first_request:
        tokens_skipped = first_tok - start_locs[first_req]
        query_start_loc[1:] -= tokens_skipped
        query_start_loc_cpu[1:] -= tokens_skipped
    seq_lens = attn_metadata.seq_lens[request_slice]
    seq_lens_cpu = attn_metadata.seq_lens_cpu[request_slice]

    if splits_last_request:
        tokens_skipped = query_start_loc_cpu[-1] - token_slice.stop
        query_start_loc[-1] -= tokens_skipped
        query_start_loc_cpu[-1] -= tokens_skipped

        # Make sure we don't modify the seq_lens tensors
        #  (not cudagraph compatible)
        seq_lens = seq_lens.clone()
        seq_lens_cpu = seq_lens_cpu.clone()
        seq_lens[-1] -= tokens_skipped
        seq_lens_cpu[-1] -= tokens_skipped

    max_seq_len = int(seq_lens_cpu.max())
    num_computed_tokens_cpu = attn_metadata.num_computed_tokens_cpu[request_slice]

    num_requests = request_slice.stop - request_slice.start
    num_actual_tokens = token_slice.stop - token_slice.start
    max_query_len = int(
        torch.max(torch.abs(query_start_loc_cpu[1:] - query_start_loc_cpu[:-1])).item()
    )

    # This is to account for the case where we are in a dummy
    # run and query_start_loc_cpu is full of 0s
    if max_query_len == 0:
        max_query_len = attn_metadata.max_query_len

    block_table_tensor = attn_metadata.block_table_tensor[request_slice]
    slot_mapping = attn_metadata.slot_mapping[token_slice]

    return CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens=seq_lens,
        seq_lens_cpu=seq_lens_cpu,
        num_computed_tokens_cpu=num_computed_tokens_cpu,
        num_reqs=num_requests,
        num_actual_tokens=num_actual_tokens,
        max_query_len=max_query_len,
        max_seq_len=max_seq_len,
        block_table_tensor=block_table_tensor,
        slot_mapping=slot_mapping,
    )

def Maca_split_decodes_and_prefills(
    common_attn_metadata: CommonAttentionMetadata,
    decode_threshold: int = 1,
    require_uniform: bool = False,
) -> tuple[int, int, int, int]:
    """
    Assuming a reordered batch, finds the boundary between prefill and decode
    requests.

    Args:
        common_attn_metadata: CommonAttentionMetadata object containing the
            batch metadata.
        decode_threshold: The maximum query length to be considered a decode.
        require_uniform: If True, requires that all decode requests have the
            same query length. When set, some queries may be considered prefills
            even if they are <= decode_threshold, in order to ensure uniformity.

    Returns:
        num_decodes: The number of decode requests.
        num_prefills: The number of prefill requests.
        num_decode_tokens: The number of tokens in the decode requests.
        num_prefill_tokens: The number of tokens in the prefill requests.
    """
    max_query_len = common_attn_metadata.max_query_len
    num_reqs = common_attn_metadata.num_reqs
    num_tokens = common_attn_metadata.num_actual_tokens
    query_start_loc = common_attn_metadata.query_start_loc_cpu

    if max_query_len <= decode_threshold and (
        not require_uniform or decode_threshold <= 1
    ):
        return num_reqs, 0, num_tokens, 0

    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    if query_lens[0].item() > decode_threshold:
        # first request is not decode, so no decode requests
        return 0, num_reqs, 0, num_tokens

    if require_uniform:
        is_prefill = query_lens != query_lens[0]
    else:
        # /------------------------  Metax Modification -------------------------\
        # 0-query len indicates a padded request; leave this at the back
        # of the batch with the prefills
        is_prefill = (query_lens > decode_threshold) | (query_lens == 0)
        # \------------------------  Metax Modification -------------------------/

    if not torch.any(is_prefill):
        return num_reqs, 0, num_tokens, 0

    first_prefill = is_prefill.int().argmax(dim=-1).item()
    assert torch.all(query_lens[:first_prefill] <= decode_threshold)
    num_decodes = first_prefill
    num_prefills = num_reqs - num_decodes
    num_decode_tokens = query_start_loc[first_prefill].item()
    num_prefill_tokens = num_tokens - num_decode_tokens
    return (num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens)

vllm.v1.attention.backends.utils.CommonAttentionMetadata = MacaCommonAttentionMetadata
CommonAttentionMetadata.unpadded = MacaCommonAttentionMetadata.unpadded
vllm.v1.attention.backends.utils.make_metadata_with_slice = Maca_make_metadata_with_slice
vllm.v1.attention.backends.utils.split_decodes_and_prefills = Maca_split_decodes_and_prefills