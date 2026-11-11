# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#
# -----------------------------------------------------------------------------
# Note: Patch fused-MoE quantization helpers for MetaX INT8 execution paths:
#     MetaX's per-token INT8 quantizer, the W4A8 descriptor used by MetaX MoE
#     experts, and the string dtype that descriptor carries.
#
# Affected versions: vLLM 0.33.0
#     (4ba787ec9959e566d3b587a796dff07673324da5).
#
# Remove at: Upstream supports MetaX per-token INT8 quantization and the W4A8 descriptor
#     used by MetaX MoE experts, including its string-valued dtype dispatch.
# -----------------------------------------------------------------------------

import torch
from vllm.model_executor.layers.quantization.utils.int8_utils import (
    per_token_group_quant_int8,
)
from vllm.utils.math_utils import cdiv

from vllm import _custom_ops as ops
import vllm.model_executor.layers.fused_moe.utils as moe_utils
from vllm_metax.patch.utils import patch


@patch("vllm.model_executor.layers.fused_moe.utils", "_int8_quantize")
def _int8_quantize(
    A: torch.Tensor,
    A_scale: torch.Tensor | None,
    per_act_token: bool,
    block_shape: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Perform int8 quantization on the inputs.  If a block_shape
    is provided, the output will be blocked.
    """

    # If weights are per-channel (per_channel_quant=True), then
    # activations apply per-token quantization. Otherwise, assume
    # activation tensor-wise fp8/int8 quantization, dynamic or static
    if block_shape is None:
        if per_act_token:
            # ┌------------------------  Metax Modification -------------------------┐
            # A, A_scale = per_token_quant_int8(A)
            A, A_scale, _ = ops.scaled_int8_quant(A)
            # └------------------------- Metax Modification -------------------------┘
        elif A_scale is not None:
            # Static per-tensor: use the optimized CUDA kernel
            A, A_scale, _ = ops.scaled_int8_quant(A, scale=A_scale)
        elif A_scale is None:
            # Dynamic per-tensor: compute scale then quantize via kernel
            A_scale = torch.clamp(A.abs().max() / 127.0, min=1e-10)
            A, A_scale, _ = ops.scaled_int8_quant(A, scale=A_scale)
    else:
        assert not per_act_token
        assert len(block_shape) == 2
        _, block_k = block_shape[0], block_shape[1]
        A, A_scale = per_token_group_quant_int8(A, block_k)
        assert cdiv(A.size(-1), block_k) == A_scale.size(-1)

    return A, A_scale


@patch(  # type: ignore[prop-decorator, misc]
    "vllm.model_executor.layers.fused_moe.config",
    "FusedMoEQuantConfig.use_int4_w4a8",
    allow_missing=True,
)
@property
def use_int4_w4a8(self):
    return self._a1.dtype == "int8" and self._w1.dtype == "int4"


# Captured before the decorator below replaces the module attribute, so the
# dtypes MetaX does not intercept keep upstream's dispatch.
_moe_kernel_quantize_input_original = moe_utils.moe_kernel_quantize_input


@patch(
    "vllm.model_executor.layers.fused_moe.utils",
    "moe_kernel_quantize_input",
)
def moe_kernel_quantize_input(
    A: torch.Tensor,
    A_scale: torch.Tensor | None,
    quant_dtype: None | torch.dtype | str,
    per_act_token_quant: bool,
    block_shape: list[int] | None = None,
    is_scale_swizzled: bool = True,
    is_scale_transposed: bool = False,
    quantization_emulation: bool = False,
    mx_alignment: int = 0,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Quantize fused-MoE activations, accepting MetaX's string int8 dtype.

    MetaX's W4A8 MoE descriptor names the activation dtype with the *string*
    ``"int8"`` (see ``int4_w4a8_moe_quant_config`` and the ``use_int4_w4a8``
    property above), while upstream only compares against ``torch.int8``. The
    string would fall through to upstream's pass-through branch and reach the
    MoE kernel with unquantized (bf16) activations, so route it to the same
    (MetaX-patched) int8 quantizer as ``torch.int8``.

    Every other dtype keeps upstream's behavior.
    """
    # ┌------------------------  Metax Modification -------------------------┐
    if quant_dtype == "int8":
        if quantization_emulation:
            raise NotImplementedError(
                "moe_kernel_quantize_input does not support quant_dtype='int8'"
                " MOE quantization emulation. Please open an issue."
            )
        return _int8_quantize(A, A_scale, per_act_token_quant, block_shape)
    # └------------------------- Metax Modification -------------------------┘
    return _moe_kernel_quantize_input_original(
        A,
        A_scale,
        quant_dtype,
        per_act_token_quant,
        block_shape=block_shape,
        is_scale_swizzled=is_scale_swizzled,
        is_scale_transposed=is_scale_transposed,
        quantization_emulation=quantization_emulation,
        mx_alignment=mx_alignment,
    )
