# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd.
# All Rights Reserved.
# -----------------------------------------------
# Note: Dequantize Qwen3.5 MTP shared-expert gates that were quantized in the
#       checkpoint despite being excluded by its quantization configuration.
#
# Affected versions: v0.25.0 - v0.26.0 (ported from v0.25.0;
#                      validate weight loading on v0.26.0)
#
# Remove at: Remove after affected checkpoints are fixed or upstream supports
#            this checkpoint format.
# -----------------------------------------------
from collections.abc import Iterable
import torch
from vllm.logger import init_logger
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_metax.patch import patch

logger = init_logger(__name__)


def _dequantize_shared_expert_gate(
    weights: Iterable[tuple[str, torch.Tensor]],
) -> Iterable[tuple[str, torch.Tensor]]:
    """Dequantize MTP shared-expert gates stored as int8 despite being ignored."""
    weight_suffix = ".mlp.shared_expert_gate.weight"
    scale_suffix = f"{weight_suffix}_scale"
    pending_weights: dict[str, torch.Tensor] = {}
    pending_scales: dict[str, torch.Tensor] = {}
    for name, tensor in weights:
        if name.endswith(weight_suffix) and tensor.dtype == torch.int8:
            scale_name = f"{name}_scale"
            scale = pending_scales.pop(scale_name, None)
            if scale is None:
                pending_weights[name] = tensor
                continue
            quantized = tensor
        elif name.endswith(scale_suffix):
            weight_name = name.removesuffix("_scale")
            quantized = pending_weights.pop(weight_name, None)
            if quantized is None:
                pending_scales[name] = tensor
                continue
            name = weight_name
            scale = tensor
        else:
            yield name, tensor
            continue
        # /-------------------- Metax Modification ---------------------\
        tensor = (quantized.float() * scale.float()).to(scale.dtype)
        # \-------------------------------------------------------------/
        logger.warning("Dequantized ignored MTP weight %s during loading.", name)
        yield name, tensor
    unresolved = pending_weights.keys() | pending_scales.keys()
    if unresolved:
        raise ValueError(
            f"Incomplete quantized MTP shared-expert gate tensors: {sorted(unresolved)}"
        )


original_load_weights = AutoWeightsLoader.load_weights


@patch("vllm.model_executor.models.utils", "AutoWeightsLoader.load_weights")
def load_weights(
    self,
    weights: Iterable[tuple[str, torch.Tensor]],
    *,
    mapper=None,
) -> set[str]:
    module_type = type(self.module)
    if (
        module_type.__module__ == "vllm.model_executor.models.qwen3_5_mtp"
        and module_type.__name__ == "Qwen3_5MultiTokenPredictor"
    ):
        weights = _dequantize_shared_expert_gate(weights)
    return original_load_weights(self, weights, mapper=mapper)
