# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Union

import torch
from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.model_executor.layers.linear import (LinearBase, LinearMethodBase,
                                               UnquantizedLinearMethod)
from vllm.model_executor.layers.quantization.awq import AWQConfig
from vllm.model_executor.layers.quantization.awq import \
    AWQLinearMethod as vllm_AWQLinearMethod
from vllm.model_executor.layers.quantization.awq import (is_layer_skipped_awq,
                                                         logger)
from vllm.model_executor.layers.quantization.base_config import \
    QuantizeMethodBase
from vllm.utils import direct_register_custom_op

from vllm_metax import _custom_ops as ops
from vllm_metax.patch.model_executor.patch.hook_register import \
    register_quantization_config


@register_quantization_config("awq")
class MacaAWQConfig(AWQConfig):

    def get_supported_act_dtypes(self):
        return [torch.half, torch.bfloat16]

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[Union["LinearMethodBase", "QuantizeMethodBase"]]:
        if isinstance(layer, LinearBase):
            if is_layer_skipped_awq(prefix, self.modules_to_not_convert):
                return UnquantizedLinearMethod()
            return AWQLinearMethod(self)
        elif isinstance(layer, FusedMoE):
            # Lazy import to avoid circular import.
            from vllm_metax.quant_config.moe_wna16 import MacaMoeWNA16Config
            logger.warning_once(
                f"Layer '{prefix}' is not supported by AWQMoeMarlin. "
                "Falling back to Moe WNA16 kernels.")
            config = {
                "quant_method": "awq",
                "bits": self.weight_bits,
                "group_size": self.group_size,
                "zero_point": self.zero_point,
                "lm_head": False,
            }
            return MacaMoeWNA16Config.from_config(config).get_quant_method(
                layer, prefix)
        return None


class AWQLinearMethod(vllm_AWQLinearMethod):
    """Linear method for AWQ.

    Args:
        quant_config: The AWQ quantization config.
    """

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.qweight = torch.nn.Parameter(layer.qweight.data,
                                           requires_grad=False)
        layer.qzeros = torch.nn.Parameter(layer.qzeros.data,
                                          requires_grad=False)
        layer.scales = torch.nn.Parameter(layer.scales.data,
                                          requires_grad=False)
        # ┌------------------------  Metax Modification -------------------------┐
        # warmup
        if self.quant_config.group_size % 32:
            pass
        else:
            qweight = ops.awq_to_gptq_4bit(layer.qweight)
            layer.qweight = torch.nn.Parameter(qweight, requires_grad=False)
        # └------------------------- Metax Modification -------------------------┘

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:

        qweight = layer.qweight
        scales = layer.scales
        qzeros = layer.qzeros
        pack_factor = self.quant_config.pack_factor
        # ┌------------------------  Metax Modification -------------------------┐
        group_size = self.quant_config.group_size

        return torch.ops.vllm._apply_awq(x, qweight, scales, qzeros, bias,
                                         pack_factor, group_size)
        # └------------------------- Metax Modification -------------------------┘


def _apply_awq_fake(x: torch.Tensor, qweight: torch.Tensor,
                    scales: torch.Tensor, qzeros: torch.Tensor,
                    bias: torch.Tensor, pack_factor: int,
                    group_size: int) -> torch.Tensor:
    out_shape = ()
    if group_size % 32:
        out_shape = (x.shape[:-1] + (qweight.shape[-1] * pack_factor, ))
    else:
        out_shape = (x.shape[:-1] + (qweight.shape[0], ))
    return torch.empty(out_shape, dtype=x.dtype, device=x.device)


def _apply_awq(x: torch.Tensor, qweight: torch.Tensor, scales: torch.Tensor,
               qzeros: torch.Tensor, bias: torch.Tensor, pack_factor: int,
               group_size: int) -> torch.Tensor:

    out_shape = ()
    reshaped_x = x.reshape(-1, x.shape[-1])
    out = torch.empty(0)
    # num_tokens >= threshold
    FP16_MATMUL_HEURISTIC_CONDITION = x.shape[:-1].numel() >= 256  # noqa: F841
    # if (FP16_MATMUL_HEURISTIC_CONDITION and reshaped_x.dtype == torch.half) or self.quant_config.group_size != 128:
    if group_size % 32:
        out_shape = (x.shape[:-1] + (qweight.shape[-1] * pack_factor, ))
        out = ops.awq_dequantize(qweight, scales, qzeros, 0, 0, 0)
        out = torch.matmul(reshaped_x, out)
    else:
        num_out_channel = qweight.shape[0]
        out_shape = (x.shape[:-1] + (num_out_channel, ))
        temp_space = torch.empty(0, dtype=torch.float32, device=x.device)
        if reshaped_x.dtype == torch.bfloat16:
            temp_space = torch.zeros(reshaped_x.shape[0],
                                     num_out_channel,
                                     dtype=torch.float32,
                                     device=x.device)
        out = ops.awq_gemm(reshaped_x, qweight, qzeros, scales, pack_factor,
                           temp_space, reshaped_x.dtype == torch.bfloat16)
    if bias is not None:
        out.add_(bias)
    return out.reshape(out_shape)


direct_register_custom_op(
    op_name="_apply_awq",
    op_func=_apply_awq,
    mutates_args=[],
    fake_impl=_apply_awq_fake,
    tags=(torch.Tag.needs_fixed_stride_order, ),
)
