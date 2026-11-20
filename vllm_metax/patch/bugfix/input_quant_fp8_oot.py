# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#
# -----------------------------------------------------------------------------
# Purpose: route dynamic per-group FP8 input quantization to the fused MACA
#     kernel instead of the torch-native fallback.
#
#     vLLM dispatches out-of-tree platforms to ``CustomOp.forward_oot``, whose
#     base implementation is ``forward_native``. QuantFP8 therefore ran the
#     torch-native per-group quantizer on MACA for every block-scaled FP8 linear
#     (e.g. DeepSeek-V4 ``wo_b`` via BlockScaledMMLinearKernel.apply_weights).
#     That native path is (a) ~10x slower than the fused kernel - CUDA graph
#     capture of DeepSeek-V4-Flash-FP8 went from ~1.3s to ~18s per capture size,
#     which looks like a hang - and (b) issues a rank-3 mixed-dtype division
#     (bf16 / fp32 -> fp32) that traps during capture with
#     ``Xnack Error/ATU Fault(0x8)``: reported kernel
#     ``at::native::elementwise_kernel_2_2_cast<..., DivFunctor<float>>`` with
#     ``OffsetCalculator<3>`` (measured on DeepSeek-V4-Flash-FP8, tp8, MACA
#     3.9.0.31, command indices 246640 / 246432 / 248503 on three different
#     launch configurations).
#     ``fp8_utils.per_token_group_quant_fp8`` already calls the fused
#     ``torch.ops._C.per_token_group_fp8_quant`` on cuda-alike platforms - the
#     same op the MetaX MLA attention path uses - so OOT platforms can share it.
#     Block-scaled linear layers always hand over a contiguous activation
#     (``BlockScaledMMLinearKernel.apply_weights`` does ``x.view(-1, H)``), so
#     non-contiguous inputs keep the upstream native path.
#     The fused kernel matches the native result exactly except at rounding ties,
#     where it differs by at most one fp8 step (measured: 0.06% of elements).
#
# Affected versions: vLLM 0.29.1.dev0 (98dff2a81d), verified 2026-10-16.
#
# Remove at: Upstream provides a platform hook for QuantFP8 group quantization, or
#     MetaX registers a QuantFP8 OOT implementation, making this override
#     unnecessary.
# -----------------------------------------------------------------------------

import torch

from vllm_metax.patch import patch


@patch(
    "vllm.model_executor.layers.quantization.input_quant_fp8", "QuantFP8.forward_oot"
)
def QuantFP8_forward_oot(
    self,
    x: torch.Tensor,
    scale: torch.Tensor | None = None,
    scale_ub: torch.Tensor | None = None,
    use_triton: bool = False,
):
    # /-------------------- MetaX Modification --------------------\
    # 上游: CustomOp.forward_oot 直接回落到 forward_native,因此 MACA 上每个
    # block-scaled FP8 线性层都用 torch 原生 group-quant(多个临时张量 + rank-3
    # bf16/fp32 除法),既慢又会在抓图时触发 ATU fault。
    # 这里对满足融合算子前提的动态 group quant 改走
    # fp8_utils.per_token_group_quant_fp8(cuda-alike 平台内部即
    # torch.ops._C.per_token_group_fp8_quant),其余情况保持上游 native 行为。
    if (
        self.is_group_quant
        and not self.static
        and x.device.type == "cuda"
        and x.is_contiguous()
        and not self.tma_aligned_scales
        and x.shape[-1] % self.group_size == 0
    ):
        from vllm.model_executor.layers.quantization.utils import fp8_utils

        return fp8_utils.per_token_group_quant_fp8(
            x,
            group_size=self.group_size,
            column_major_scales=self.column_major_scales,
            tma_aligned_scales=False,
            use_ue8m0=self.use_ue8m0,
        )
    # \-------------------- MetaX Modification --------------------/

    return self.forward_native(x, scale, scale_ub, use_triton)
