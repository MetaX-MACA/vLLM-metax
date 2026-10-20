# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerical correctness tests for the MetaX mctlass W8A8 scaled_mm op.

These run the real mctlassEx kernel on MetaX hardware and compare its output
against a PyTorch reference. The suite is skipped automatically when the
``mctlassEx`` library (and therefore a MetaX backend) is unavailable, so it is
safe to collect on non-MetaX CI.
"""

import pytest
import torch

# The mctlass python-api ops only import on a MetaX backend with mctlassEx
# present. Skip the whole module cleanly everywhere else.
_python_api_ops = pytest.importorskip(
    "vllm_metax.model_executor.layers.quantization._python_api_ops",
    reason="mctlassEx / MetaX backend not available",
)

mctlassEx_w8a8_scaled_mm_azp = _python_api_ops.mctlassEx_w8a8_scaled_mm_azp

pytestmark = pytest.mark.skipif(
    getattr(_python_api_ops, "mctlass_op", None) is None,
    reason="mctlass_op failed to initialize (mctlassEx not importable)",
)

# Output dtype of the kernel is bfloat16 (see kernel docstring).
OUT_DTYPE = torch.bfloat16

# (M, K, N): token count, hidden in, hidden out. The mctlass W8A8 kernel
# requires K and N to be multiples of 16 (see test_alignment_constraint below);
# M is unconstrained, so it carries the irregular / boundary values here.
SHAPES = [
    (1, 128, 128),
    (16, 256, 512),
    (64, 512, 256),
    (128, 1024, 1024),
    (37, 256, 128),  # irregular M
    (1234, 1568, 768),  # large, irregular M; K, N still multiples of 16
]
SEEDS = [0, 1]


def _quantize_per_token_int8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-token (per-row) int8 quantization.

    Returns the int8 tensor and per-token scales of shape (M, 1) float32.
    """
    amax = x.abs().to(torch.float32).amax(dim=1, keepdim=True).clamp(min=1e-6)
    scales = amax / 127.0
    q = (x.to(torch.float32) / scales).round().clamp(-128, 127).to(torch.int8)
    return q, scales


def _quantize_per_channel_int8(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-channel int8 quantization of a (N, K) weight.

    Quantizes along K so each output channel (row) has its own scale.
    Returns the int8 weight (N, K) and scales of shape (1, N) float32 — the
    layout the vllm-level caller passes (the op wrapper transposes internally).
    """
    amax = w.abs().to(torch.float32).amax(dim=1, keepdim=True).clamp(min=1e-6)
    scales = amax / 127.0  # (N, 1)
    q = (w.to(torch.float32) / scales).round().clamp(-128, 127).to(torch.int8)
    return q, scales.reshape(1, -1)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_mctlass_w8a8_scaled_mm(shape: tuple[int, int, int], seed: int) -> None:
    M, K, N = shape
    torch.manual_seed(seed)
    device = "cuda"

    # Start from realistic fp activations/weights, then quantize, so the test
    # reflects how the op is actually used in a W8A8 quantized model.
    a_f = torch.randn(M, K, dtype=torch.float32, device=device)
    w_f = torch.randn(N, K, dtype=torch.float32, device=device) * 0.1

    a_q, a_scales = _quantize_per_token_int8(a_f)
    b_q, b_scales = _quantize_per_channel_int8(w_f)

    out = torch.empty((M, N), dtype=OUT_DTYPE, device=device)
    mctlassEx_w8a8_scaled_mm_azp(out, a_q, b_q, a_scales, b_scales)

    # Reference: dequantize and matmul in fp32. b is (N, K) so use b^T.
    #   out[m, n] = a_scales[m] * b_scales[n] * sum_k a_q[m, k] * b_q[n, k]
    ref = a_q.to(torch.float32) @ b_q.to(torch.float32).T
    ref = ref * a_scales * b_scales  # (M, 1) * (1, N) broadcast

    out_f = out.to(torch.float32)
    ref_bf16 = ref.to(OUT_DTYPE).to(torch.float32)

    # The kernel accumulates in higher precision then rounds to bf16. Compare
    # against the bf16-rounded reference with a tolerance scaled to the output
    # magnitude (bf16 carries ~8 mantissa bits => ~2^-8 relative).
    scale = ref.abs().amax().clamp(min=1e-6).item()
    torch.testing.assert_close(
        out_f, ref_bf16, rtol=2.0 / 128, atol=scale * (2.0 / 128)
    )


@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_mctlass_w8a8_scaled_mm_scales_applied(seed: int) -> None:
    """Sanity check that both scales actually scale the output linearly."""
    M, K, N = 32, 128, 64
    torch.manual_seed(seed)
    device = "cuda"

    a_q = torch.randint(-64, 64, (M, K), dtype=torch.int8, device=device)
    b_q = torch.randint(-64, 64, (N, K), dtype=torch.int8, device=device)
    a_scales = torch.full((M, 1), 0.01, dtype=torch.float32, device=device)
    b_scales = torch.full((1, N), 0.02, dtype=torch.float32, device=device)

    out1 = torch.empty((M, N), dtype=OUT_DTYPE, device=device)
    mctlassEx_w8a8_scaled_mm_azp(out1, a_q, b_q, a_scales, b_scales)

    # Doubling a_scales should double the output.
    out2 = torch.empty((M, N), dtype=OUT_DTYPE, device=device)
    mctlassEx_w8a8_scaled_mm_azp(out2, a_q, b_q, a_scales * 2, b_scales)

    torch.testing.assert_close(
        out2.to(torch.float32), (out1.to(torch.float32) * 2), rtol=2.0 / 128, atol=1e-2
    )


@torch.inference_mode()
def test_mctlass_w8a8_scaled_mm_k_alignment_constraint() -> None:
    """Document that K must be a multiple of 16.

    The kernel has no tile for a misaligned K and raises while selecting one
    (``Can not find valid kid by rule``) before touching memory, so this is the
    safe constraint to assert. N misalignment is *not* exercised here: it does
    not raise but triggers an out-of-bounds device write that takes down the
    MACA runtime for the rest of the process, so callers must guarantee N % 16
    upstream.
    """
    M, K, N = 64, 257, 128  # K not a multiple of 16
    device = "cuda"
    a_q = torch.randint(-8, 8, (M, K), dtype=torch.int8, device=device)
    b_q = torch.randint(-8, 8, (N, K), dtype=torch.int8, device=device)
    out = torch.empty((M, N), dtype=OUT_DTYPE, device=device)
    a_scales = torch.full((M, 1), 0.01, dtype=torch.float32, device=device)
    b_scales = torch.full((1, N), 0.02, dtype=torch.float32, device=device)

    # No tile exists for misaligned K; the kernel raises IndexError from the
    # rule selector before any device memory is touched.
    with pytest.raises(IndexError):
        mctlassEx_w8a8_scaled_mm_azp(out, a_q, b_q, a_scales, b_scales)
