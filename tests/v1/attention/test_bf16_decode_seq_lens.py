# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import torch
import pytest

from vllm_metax.v1.attention.backends.mla import flashmla_sparse


def _build(query_starts, decode_tokens, seq_lens, req_id_per_token=None, *, num_decodes=None, prefill_tokens=0, num_prefills=0):
    if num_decodes is None:
        num_decodes = len(query_starts) - 1
    common = SimpleNamespace(
        num_actual_tokens=decode_tokens + prefill_tokens,
        query_start_loc_cpu=torch.tensor(query_starts, dtype=torch.int32),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
    )
    metadata = SimpleNamespace(
        num_decodes=num_decodes,
        num_prefills=num_prefills,
        num_decode_tokens=decode_tokens,
        req_id_per_token=req_id_per_token,
    )
    builder = SimpleNamespace(
        device=torch.device("cpu"),
        dummy_block_table=torch.zeros(decode_tokens, 1, dtype=torch.int32),
        max_model_len_tensor=torch.full(
            (decode_tokens,), 4096, dtype=torch.int32
        ),
    )
    with patch.object(flashmla_sparse, "get_mla_metadata", return_value=(None, None)):
        return (
            flashmla_sparse.FlashMLASparseMetadataBuilder
            ._build_bf16_separate_prefill_decode(builder, common, metadata)
        )


def test_uniform_decode_uses_contiguous_seq_lens_without_gather():
    with patch.object(
        torch.Tensor,
        "index_select",
        side_effect=AssertionError("uniform decode must not gather seq_lens"),
    ):
        output = _build([0, 3, 6], 6, [100, 101])

    assert output.decode.decode_query_len == 3
    assert output.decode.seq_lens.tolist() == [100, 101]


def test_padded_decode_preserves_generic_request_mapping():
    output = _build(
        [0, 1, 3, 3],
        3,
        [100, 101, 102],
        req_id_per_token=torch.tensor([0, 1, 1], dtype=torch.int32),
    )

    assert output.decode.decode_query_len == 1
    assert output.decode.seq_lens.tolist() == [100, 101, 101]


@pytest.mark.parametrize(
    "starts,tokens,lengths,mapping,expected,query_len",
    [
        ([0, 1, 2], 2, [100, 101], None, [100, 101], 1),
        ([0, 1, 3], 3, [100, 101], [0, 1, 1], [100, 101, 101], 1),
        ([0, 3, 3, 6], 6, [100, 101, 102], None, [100, 102], 3),
        ([0, 1, 3], 4, [100, 101], [0, 1, 1, 0], [100, 101, 101, 100], 1),
    ],
)
def test_decode_lengths_remain_aligned(starts, tokens, lengths, mapping, expected, query_len):
    request_ids = None if mapping is None else torch.tensor(mapping, dtype=torch.int32)
    output = _build(starts, tokens, lengths, request_ids)
    assert output.decode.seq_lens.tolist() == expected
    assert output.decode.decode_query_len == query_len
    assert output.num_decodes == len(expected)


def test_uniform_mixed_batch_uses_only_decode_prefix():
    output = _build([0, 2, 4, 10], 4, [100, 101, 102], num_decodes=2, prefill_tokens=6, num_prefills=1)
    assert output.decode.seq_lens.tolist() == [100, 101]
    assert output.decode.decode_query_len == 2
    assert output.num_prefills == 1
    assert output.num_prefill_tokens == 6


@pytest.mark.parametrize("prefill_tokens,num_prefills", [(0, 0), (5, 1)])
def test_no_decode_does_not_request_kernel_metadata(prefill_tokens, num_prefills):
    with patch.object(flashmla_sparse, "get_mla_metadata", side_effect=AssertionError("no decode kernel metadata")):
        # The helper normally replaces this external scheduler metadata provider.
        common = SimpleNamespace(num_actual_tokens=prefill_tokens)
        metadata = SimpleNamespace(num_decodes=0, num_prefills=num_prefills, num_decode_tokens=0)
        output = flashmla_sparse.FlashMLASparseMetadataBuilder._build_bf16_separate_prefill_decode(None, common, metadata)
    assert output.decode is None
    assert output.num_decode_tokens == 0
    assert output.num_prefill_tokens == prefill_tokens
