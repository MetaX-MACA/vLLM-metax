# SPDX-License-Identifier: Apache-2.0

import vllm

from vllm.logger import init_logger

logger = init_logger(__name__)

import torch
from torch.nn.parameter import Parameter
from typing import Optional, List
from vllm.utils import direct_register_custom_op
from vllm_metax import _custom_ops as ops
from vllm.model_executor.layers.quantization.gptq import ExllamaState

def get_supported_act_dtypes(cls) -> List[torch.dtype]:
# ┌------------------------  Metax Modification -------------------------┐
    return [torch.half, torch.bfloat16]
# └------------------------- Metax Modification -------------------------┘



def _apply_gptq_fake(x: torch.Tensor,
                    qweight: torch.Tensor,
                    scales: torch.Tensor,
                    qzeros: torch.Tensor,
                    bias: torch.Tensor,
                    g_idx: torch.Tensor,
                    use_exllama: bool,
                    weight_bits: int,
                    group_size: int,
                    desc_act: bool) -> torch.Tensor:
    out_shape = x.shape[:-1] + (qweight.shape[-1], )
    return torch.empty(out_shape, dtype=x.dtype, device=x.device)

def _apply_gptq(x: torch.Tensor,
               qweight: torch.Tensor,
               scales: torch.Tensor,
               qzeros: torch.Tensor,
               bias: torch.Tensor,
               g_idx: torch.Tensor,
               use_exllama: bool,
               weight_bits: int,
               group_size: int,
               desc_act: bool) -> torch.Tensor:

    reshaped_x = x.reshape(-1, x.shape[-1])
    out_shape = x.shape[:-1] + (qweight.shape[-1], )

    perm_space = torch.empty(0)
    temp_space = torch.empty(0)
    if weight_bits == 4 or weight_bits == 8:
        if group_size == 128 or group_size == 64:
            if desc_act:
                perm_space = torch.empty(reshaped_x.shape[0], reshaped_x.shape[1],
                                            dtype=torch.float16, device=x.device)
            if reshaped_x.dtype == torch.bfloat16:
                temp_space = torch.zeros(reshaped_x.shape[0], qweight.shape[1],
                                            dtype=torch.float32, device=x.device)

    output = ops.gptq_gemm(reshaped_x, qweight, qzeros,
                            scales, g_idx,
                            use_exllama,
                            weight_bits,
                            group_size,
                            perm_space, temp_space,
                            True if reshaped_x.dtype == torch.bfloat16 else False)

    if bias is not None:
        output.add_(bias)
    return output.reshape(out_shape)

direct_register_custom_op(
    op_name="_apply_gptq",
    op_func=_apply_gptq,
    mutates_args=[],
    fake_impl=_apply_gptq_fake,
    tags=(torch.Tag.needs_fixed_stride_order, ),
)

def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
    # for torch.compile
    layer.qzeros = Parameter(layer.qzeros.data, requires_grad=False)
    layer.qweight = Parameter(layer.qweight.data, requires_grad=False)
    layer.g_idx = Parameter(layer.g_idx.data, requires_grad=False)
    layer.scales = Parameter(layer.scales.data, requires_grad=False)

    # exllama needs to shuffle the weight after the weight is loaded
    # here we do the shuffle on first forward pass
    
    # ┌------------------------  Metax Modification -------------------------┐
    if self.quant_config.group_size == 128 or self.quant_config.group_size == 64:
    # └------------------------- Metax Modification -------------------------┘
        if self.quant_config.desc_act:
            layer.g_idx.data = torch.argsort(layer.g_idx).to(torch.int)
        else:
            layer.g_idx.data = torch.empty((0, ),
                                            dtype=torch.int,
                                            device=layer.g_idx.device)
        layer.exllama_state = ExllamaState.READY
        ops.gptq_shuffle(layer.qweight, layer.g_idx,
                            self.quant_config.weight_bits)
        
    # ┌------------------------  Metax Modification -------------------------┐
        if layer.scales.dtype != torch.bfloat16:
            perm_space = torch.empty(0)
            temp_space = torch.empty(0)
            if self.quant_config.weight_bits == 4:
                # warmup
                reshaped_x = torch.randn(1, layer.qweight.shape[0]*8, dtype=layer.scales.dtype, device="cuda")
                _ = ops.gptq_gemm(reshaped_x, layer.qweight, layer.qzeros,
                                layer.scales, layer.g_idx,
                                layer.exllama_state == ExllamaState.READY,
                                self.quant_config.weight_bits,
                                self.quant_config.group_size,
                                perm_space, temp_space,
                                False)
            if self.quant_config.weight_bits == 8:
                # warmup
                reshaped_x = torch.randn(1, layer.qweight.shape[0]*4, dtype=layer.scales.dtype, device="cuda")
                _ = ops.gptq_gemm(reshaped_x, layer.qweight, layer.qzeros,
                                layer.scales, layer.g_idx,
                                layer.exllama_state == ExllamaState.READY,
                                self.quant_config.weight_bits,
                                self.quant_config.group_size,
                                perm_space, temp_space,
                                False)
    else:
        if layer.exllama_state == ExllamaState.UNINITIALIZED:
            if self.quant_config.desc_act:
                layer.g_idx.data = torch.argsort(layer.g_idx).to(torch.int)
            else:
                layer.g_idx.data = torch.empty((0, ),
                                                dtype=torch.int,
                                                device=layer.g_idx.device)
            layer.exllama_state = ExllamaState.READY
            ops.gptq_shuffle(layer.qweight, layer.g_idx,
                                self.quant_config.weight_bits)
    # └------------------------- Metax Modification -------------------------┘

def apply(self,
            layer: torch.nn.Module,
            x: torch.Tensor,
            bias: Optional[torch.Tensor] = None) -> torch.Tensor:
# ┌------------------------  Metax Modification -------------------------┐
    qweight = layer.qweight
    scales = layer.scales
    qzeros = layer.qzeros
    g_idx = layer.g_idx
    exllama_state = layer.exllama_state
    weight_bits = self.quant_config.weight_bits
    group_size = self.quant_config.group_size
    desc_act = self.quant_config.desc_act
    use_exllama =  exllama_state == ExllamaState.READY
    
    return torch.ops.vllm._apply_gptq(x, qweight, scales, qzeros, 
                                    bias, g_idx, use_exllama, weight_bits, 
                                    group_size, desc_act)
# └------------------------- Metax Modification -------------------------┘

import vllm.model_executor.layers.quantization.gptq

vllm.model_executor.layers.quantization.gptq.GPTQConfig.get_supported_act_dtypes  = classmethod(get_supported_act_dtypes)
vllm.model_executor.layers.quantization.gptq.GPTQLinearMethod.process_weights_after_loading  = process_weights_after_loading
vllm.model_executor.layers.quantization.gptq.GPTQLinearMethod.apply  = apply





