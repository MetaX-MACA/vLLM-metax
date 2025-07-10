import vllm
from vllm_metax_plugin.patch.hook_registry import register_patch
from vllm.logger import init_logger

logger = init_logger(__name__)

import torch
from typing import Union, Optional, Tuple
from vllm_metax_plugin import _custom_ops as ops
from vllm.model_executor.layers.quantization.kernels.scaled_mm.cutlass import CutlassScaledMMLinearKernel

def apply_weights_tuple(self,
                    layer: torch.nn.Module,
                    x: Union[torch.Tensor, Tuple[torch.dtype, torch.Tensor, torch.Tensor]],
                    bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    w_q, w_s, i_s, i_zp, azp_adj = self._get_weight_params(layer)
    symmetric = azp_adj is None
    if isinstance(x, tuple):
        assert symmetric # only supoort symmetric currently
        # C500-31755
        out_dtype = x[0]
        x_q = x[1]
        x_s = x[2]
        x_zp = None
    else:
        # ops.scaled_int8_quant supports both dynamic and static quant:
        # * dynamic, i_s is None and x_s computed from x.
        # * static, i_s is scalar and x_s is i_s.
        # symmetric = azp_adj is None
        out_dtype = x.dtype
        x_q, x_s, x_zp = ops.scaled_int8_quant(x,
                                            i_s,
                                            i_zp,
                                            symmetric=symmetric)

    if x_zp is not None:
        # Currently, static is always per-tensor and dynamic is per-token
        static = i_zp is not None
        azp = None if static else x_zp
        return ops.cutlass_scaled_mm_azp(x_q,
                                            w_q,
                                            scale_a=x_s,
                                            scale_b=w_s,
                                            out_dtype=out_dtype,
                                            azp_adj=azp_adj,
                                            azp=azp,
                                            bias=bias)
    return ops.cutlass_scaled_mm(x_q,
                                    w_q,
                                    scale_a=x_s,
                                    scale_b=w_s,
                                    out_dtype=out_dtype,
                                    bias=bias)

CutlassScaledMMLinearKernel.apply_weights_tuple = apply_weights_tuple
register_patch("vllm.model_executor.layers.quantization.kernels.scaled_mm.cutlass", "CutlassScaledMMLinearKernel.apply_weights_tuple", apply_weights_tuple)