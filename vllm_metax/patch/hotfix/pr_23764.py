# SPDX-License-Identifier: Apache-2.0

from typing import Any, Literal, Optional, Union

import torch
from torch.nn.parameter import Parameter
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
# yapf: disable
from vllm.model_executor.parameter import (BasevLLMParameter,
                                           BlockQuantScaleParameter,
                                           PerTensorScaleParameter)

from vllm.model_executor.layers.linear import ReplicatedLinear

from vllm.model_executor.parameter import (PackedColumnParameter, PackedvLLMParameter,
                                           get_tensor_model_parallel_rank)
#vllm/model_executor/layers/linear.py
class MergedReplicatedLinear(ReplicatedLinear):
    """Replicated linear layer.

    Args:
        input_size: input dimension of the linear layer.
        output_size: output dimension of the linear layer.
        bias: If true, add bias.
        skip_bias_add: If true, skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        quant_config: Quantization configure.
        prefix: The name of the layer in the state dict, including all parents
                        (e.g. model.layers.0.qkv_proj)
    """

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = True,
        skip_bias_add: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
    ):
        self.output_sizes = output_sizes
        super().__init__(input_size,
                         sum(output_sizes),
                         bias,
                         skip_bias_add,
                         params_dtype,
                         quant_config,
                         prefix=prefix,
                         return_bias=return_bias)

    def weight_loader(self,
                      param: Union[Parameter, BasevLLMParameter],
                      loaded_weight: torch.Tensor,
                      loaded_shard_id: Optional[int] = None):
        assert loaded_shard_id is not None
        assert loaded_shard_id < len(self.output_sizes)

        if isinstance(param, BlockQuantScaleParameter):
            from vllm.model_executor.layers.quantization.fp8 import (
                Fp8LinearMethod, Fp8MoEMethod)
            assert self.quant_method is not None
            assert isinstance(self.quant_method,
                              (Fp8LinearMethod, Fp8MoEMethod))
            weight_block_size = self.quant_method.quant_config.weight_block_size
            assert weight_block_size is not None
            block_n, _ = weight_block_size[0], weight_block_size[1]
            shard_offset = (
                (sum(self.output_sizes[:loaded_shard_id]) + block_n - 1) //
                block_n)
            shard_size = ((self.output_sizes[loaded_shard_id] + block_n - 1) //
                          block_n)
        elif isinstance(param, PerTensorScaleParameter):
            shard_offset = loaded_shard_id
            shard_size = 1
        else:
            shard_offset = sum(self.output_sizes[:loaded_shard_id])
            shard_size = self.output_sizes[loaded_shard_id]

        if isinstance(param, BasevLLMParameter):
            param.load_merged_column_weight(loaded_weight=loaded_weight,
                                            shard_id=loaded_shard_id,
                                            shard_offset=shard_offset,
                                            shard_size=shard_size,
                                            tp_rank=0)
        else:
            param.data[shard_offset:shard_offset + shard_size] = loaded_weight

# vllm/model_executor/parameter.py
def load_merged_column_weight(self, loaded_weight: torch.Tensor, **kwargs):

    shard_offset = kwargs.get("shard_offset")
    shard_size = kwargs.get("shard_size")
    if isinstance(
            self,
        (PackedColumnParameter,
          PackedvLLMParameter)) and self.packed_dim == self.output_dim:
        shard_size, shard_offset = self.adjust_shard_indexes_for_packing(
            shard_offset=shard_offset, shard_size=shard_size)

    param_data = self.data

    tp_rank = kwargs.get("tp_rank", get_tensor_model_parallel_rank())
    param_data = param_data.narrow(self.output_dim, shard_offset,
                                    shard_size)
    loaded_weight = loaded_weight.narrow(self.output_dim,
                                          tp_rank * shard_size, shard_size)
    assert param_data.shape == loaded_weight.shape
    param_data.copy_(loaded_weight)



import vllm.model_executor.layers.linear
vllm.model_executor.layers.linear.MergedReplicatedLinear = MergedReplicatedLinear

import vllm.model_executor.parameter
vllm.model_executor.parameter._ColumnvLLMParameter.load_merged_column_weight = load_merged_column_weight