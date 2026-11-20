# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import pytest
import torch

from vllm_metax.platform import MacaPlatform

pytestmark = pytest.mark.requires_maca


@pytest.mark.parametrize(
    "num_seqs", [1, 2, 3, 4, 8, 13, 17, 32, 37, 256, 1023, 1024, 1025]
)
@pytest.mark.parametrize("vocab_size", [17, 256, 1019, 151936, 202048])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@torch.inference_mode()
def test_apply_repetition_penalties(device, num_seqs, vocab_size, dtype):
    MacaPlatform.import_kernels()
    rng = torch.Generator(device=device).manual_seed(0)
    logits = torch.randn(
        num_seqs, vocab_size, device=device, dtype=dtype, generator=rng
    )
    prompt_mask = torch.zeros_like(logits, dtype=torch.bool)
    output_mask = torch.zeros_like(prompt_mask)
    for mask in (prompt_mask, output_mask):
        indices = torch.randint(
            vocab_size,
            (num_seqs, max(1, vocab_size // 200)),
            device=device,
            generator=rng,
        )
        mask.scatter_(1, indices, True)
    penalties = torch.full((num_seqs,), 1.05, device=device, dtype=dtype)
    expected = logits.clone()
    for start in range(0, num_seqs, 64):
        rows = slice(start, start + 64)
        repeated = prompt_mask[rows] | output_mask[rows]
        values = logits[rows].float()
        scale = penalties[rows, None].float()
        scaled = torch.where(values > 0, values / scale, values * scale)
        expected[rows] = torch.where(repeated, scaled, values).to(dtype)

    torch.ops._C.apply_repetition_penalties_(
        logits, prompt_mask, output_mask, penalties
    )

    for start in range(0, num_seqs, 64):
        rows = slice(start, start + 64)
        torch.testing.assert_close(logits[rows], expected[rows], atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("num_seqs", [0, 3])
def test_apply_repetition_penalties_opcheck(device, num_seqs):
    MacaPlatform.import_kernels()
    logits = torch.ones(num_seqs, 17, device=device)
    prompt_mask = torch.ones_like(logits, dtype=torch.bool)
    output_mask = torch.zeros_like(prompt_mask)
    penalties = torch.full((num_seqs,), 1.05, device=device)
    torch.library.opcheck(
        torch.ops._C.apply_repetition_penalties_,
        (logits, prompt_mask, output_mask, penalties),
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize(("rows", "vocab"), [(1, 17), (17, 1019), (4, 151936)])
def test_repetition_penalties_change_each_seen_logit_once(device, dtype, rows, vocab):
    MacaPlatform.import_kernels()
    torch.manual_seed(0)
    logits = torch.randn(rows, vocab, device=device, dtype=dtype)
    columns = torch.arange(vocab, device=device)
    prompt = (columns % 13 == 0).expand(rows, -1).contiguous()
    output = (columns % 17 == 0).expand(rows, -1).contiguous()
    penalties = torch.linspace(1.05, 1.75, rows, device=device, dtype=dtype)
    original = logits.clone()
    repeated = prompt | output
    scale = penalties[:, None].float()
    penalized = torch.where(
        original > 0, original.float() / scale, original.float() * scale
    ).to(dtype)
    expected = torch.where(repeated, penalized, original)

    torch.ops._C.apply_repetition_penalties_(logits, prompt, output, penalties)

    torch.testing.assert_close(logits, expected, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(logits[~repeated], original[~repeated], atol=0, rtol=0)


@torch.inference_mode()
def test_apply_repetition_penalties_zero_seqs(device):
    MacaPlatform.import_kernels()
    logits = torch.empty(0, 17, device=device)
    mask = torch.empty_like(logits, dtype=torch.bool)
    penalties = torch.empty(0, device=device)

    torch.ops._C.apply_repetition_penalties_(logits, mask, mask, penalties)

    assert logits.shape == (0, 17)
