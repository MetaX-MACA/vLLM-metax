# SPDX-License-Identifier: Apache-2.0

import vllm

from vllm.logger import init_logger

logger = init_logger(__name__)

import torch
from typing import List, Optional
from vllm.utils import direct_register_custom_op
from vllm_metax import _custom_ops as ops

def get_supported_act_dtypes(self) -> List[torch.dtype]:
    return [torch.half, torch.bfloat16]

def _apply_awq_fake(x: torch.Tensor,
                    qweight: torch.Tensor,
                    scales: torch.Tensor,
                    qzeros: torch.Tensor,
                    bias: torch.Tensor,
                    pack_factor: int,
                    group_size: int) -> torch.Tensor:
    logger.info(f"[Plugin] Hooked _apply_awq_fake -> {_apply_awq_fake}")
    out_shape = ()
    if group_size % 32:
        out_shape = (x.shape[:-1] + (qweight.shape[-1] * pack_factor, ))
    else:
        out_shape = (x.shape[:-1] + (qweight.shape[0], ))
    return torch.empty(out_shape, dtype=x.dtype, device=x.device)

def _apply_awq(x: torch.Tensor,
               qweight: torch.Tensor,
               scales: torch.Tensor,
               qzeros: torch.Tensor,
               bias: torch.Tensor,
               pack_factor: int,
               group_size: int) -> torch.Tensor:

    out_shape = ()
    reshaped_x = x.reshape(-1, x.shape[-1])
    out = torch.empty(0)          
    # num_tokens >= threshold
    FP16_MATMUL_HEURISTIC_CONDITION = x.shape[:-1].numel() >= 256
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
            temp_space = torch.zeros(reshaped_x.shape[0], num_out_channel,
                                        dtype=torch.float32, device=x.device)
        out = ops.awq_gemm(reshaped_x, qweight, qzeros, scales,
                            pack_factor, temp_space,
                            True if reshaped_x.dtype == torch.bfloat16 else False)
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

def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
    logger.info(f"[Plugin] Hooked process_weights_after_loading -> {process_weights_after_loading}")

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
    
    return torch.ops.vllm._apply_awq(x, qweight, scales, qzeros, 
                                    bias, pack_factor, group_size)
# └------------------------- Metax Modification -------------------------┘
    
from vllm.model_executor.layers.quantization import awq

vllm.model_executor.layers.quantization.awq.AWQConfig.get_supported_act_dtypes = get_supported_act_dtypes
vllm.model_executor.layers.quantization.awq._apply_awq_fake = _apply_awq_fake
vllm.model_executor.layers.quantization.awq._apply_awq = _apply_awq
vllm.model_executor.layers.quantization.awq.AWQLinearMethod.process_weights_after_loading = process_weights_after_loading
vllm.model_executor.layers.quantization.awq.AWQLinearMethod.apply = apply




