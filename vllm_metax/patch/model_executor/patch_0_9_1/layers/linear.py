# SPDX-License-Identifier: Apache-2.0

import vllm
from vllm_metax.patch.hook_registry import register_patch
from vllm.logger import init_logger

logger = init_logger(__name__)

from vllm.distributed import (divide, get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              split_tensor_along_last_dim,
                              tensor_model_parallel_all_gather,
                              tensor_model_parallel_all_reduce)

import torch
import itertools
from typing import Optional
from vllm import envs
from torch.nn.parameter import Parameter, UninitializedParameter
from vllm.model_executor.layers.utils import dispatch_unquantized_gemm
from vllm.model_executor.layers.linear import (LinearMethodBase, 
    UnquantizedLinearMethod, adjust_marlin_shard, 
    adjust_bitblas_shard, adjust_bitsandbytes_4bit_shard)
from vllm.model_executor.utils import set_weight_attrs

def adjust_scalar_to_fused_array(param, loaded_weight, shard_id):
    """For fused modules (QKV and MLP) we have an array of length
    N that holds 1 scale for each "logical" matrix. So the param
    is an array of length N. The loaded_weight corresponds to 
    one of the shards on disk. Here, we slice the param based on 
    the shard_id for loading.
    """
    qkv_idxs = {"q": 0, "k": 1, "v": 2}

    if isinstance(shard_id, str):
        shard_id = qkv_idxs[shard_id]
    elif not isinstance(shard_id, int):
        raise ValueError(f"Unknown Shard Id {shard_id}")

    # AutoFP8 scales do not have a shape
    # compressed-tensors scales do have a shape
    if len(loaded_weight.shape) != 0:
        assert loaded_weight.shape[0] == 1
        loaded_weight = loaded_weight[0]

    # Support gemm_tn->gemm_nn
    if envs.MACA_VLLM_USE_TN_2_NN:
        return param[shard_id], loaded_weight.t()
    else:
        return param[shard_id], loaded_weight
    

def UnquantizedLinearMethod_create_weights(self, layer: torch.nn.Module,
                    input_size_per_partition: int,
                    output_partition_sizes: list[int], input_size: int,
                    output_size: int, params_dtype: torch.dtype,
                    **extra_weight_attrs):
    # Support gemm_tn->gemm_nn here
    if envs.MACA_VLLM_USE_TN_2_NN:
        weight = Parameter(torch.empty(input_size_per_partition,
                                    sum(output_partition_sizes),
                                    dtype=params_dtype),
                        requires_grad=False)
    else:
        weight = Parameter(torch.empty(sum(output_partition_sizes),
                                        input_size_per_partition,
                                        dtype=params_dtype),
                        requires_grad=False)
    set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
    layer.register_parameter("weight", weight)
    set_weight_attrs(weight, extra_weight_attrs)

def UnquantizedLinearMethod_apply(self,
            layer: torch.nn.Module,
            x: torch.Tensor,
            bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    # Support gemm_tn->gemm_nn here
    if envs.MACA_VLLM_USE_TN_2_NN and x.shape[-1] == layer.weight.shape[0]:
        return dispatch_unquantized_gemm()(x, layer.weight.t(), bias)
    else:
        return dispatch_unquantized_gemm()(x, layer.weight, bias)
        
def ReplicatedLinear_weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
    # If the weight on disk does not have a shape, give it one
    # (such scales for AutoFp8).
    # Special case for GGUF

    is_gguf_weight = getattr(param, "is_gguf_weight", False)
    is_gguf_weight_type = getattr(param, "is_gguf_weight_type", False)
    if is_gguf_weight_type:
        param.weight_type = loaded_weight.item()

    # Materialize GGUF UninitializedParameter
    if is_gguf_weight and isinstance(param, UninitializedParameter):
        param.materialize(loaded_weight.shape, dtype=loaded_weight.dtype)

    if len(loaded_weight.shape) == 0:
        loaded_weight = loaded_weight.reshape(1)

    # Support gemm_tn->gemm_nn here
    is_quantization = not isinstance(self.quant_method, UnquantizedLinearMethod)
    if envs.MACA_VLLM_USE_TN_2_NN and not is_quantization:
        loaded_weight = loaded_weight.t()

    assert param.size() == loaded_weight.size(), (
        f"Tried to load weights of size {loaded_weight.size()}"
        f"to a parameter of size {param.size()}")
    param.data.copy_(loaded_weight)
    
def ColumnParallelLinear_weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
    tp_rank = get_tensor_model_parallel_rank()
    output_dim = getattr(param, "output_dim", None)

    is_sharded_weight = getattr(param, "is_sharded_weight", False)
    use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit", False)
    # bitsandbytes loads the weights of the specific portion
    # no need to narrow
    is_sharded_weight = is_sharded_weight or use_bitsandbytes_4bit
    # Support gemm_tn->gemm_nn here
    is_quantization = not isinstance(self.quant_method, UnquantizedLinearMethod)

    # Special case for GGUF
    is_gguf_weight = getattr(param, "is_gguf_weight", False)
    is_gguf_weight_type = getattr(param, "is_gguf_weight_type", False)
    if is_gguf_weight_type:
        param.weight_type = loaded_weight.item()

    # Materialize GGUF UninitializedParameter
    if is_gguf_weight and isinstance(param, UninitializedParameter):
        final_shape = list(loaded_weight.shape)
        if output_dim is not None:
            tp_size = get_tensor_model_parallel_world_size()
            assert final_shape[output_dim] % tp_size == 0
            final_shape[output_dim] = final_shape[output_dim] // tp_size
        param.materialize(final_shape, dtype=loaded_weight.dtype)

    param_data = param.data
    if output_dim is not None and not is_sharded_weight:
        
        # Support gemm_tn->gemm_nn here
        if not envs.MACA_VLLM_USE_TN_2_NN or len(param_data.shape)==1 or is_quantization:
            shard_size = param_data.shape[output_dim] 
        else:
            shard_size = param_data.shape[int(not(output_dim))]
            
        start_idx = tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(output_dim, start_idx,
                                                shard_size)

    # Special case for loading scales off disk, which often do not
    # have a shape (such as in the case of AutoFP8).
    if len(loaded_weight.shape) == 0:
        loaded_weight = loaded_weight.reshape(1)

    # Support gemm_tn->gemm_nn here
    if envs.MACA_VLLM_USE_TN_2_NN and not is_quantization:
        loaded_weight = loaded_weight.t()

    assert param_data.shape == loaded_weight.shape
    param_data.copy_(loaded_weight)

def MergedColumnParallelLinear_weight_loader(self,
                    param: Parameter,
                    loaded_weight: torch.Tensor,
                    loaded_shard_id: Optional[int] = None):

    # Special case for GGUF
    # initialize GGUF param after we know the quantize type
    is_gguf_weight = getattr(param, "is_gguf_weight", False)
    is_gguf_weight_type = getattr(param, "is_gguf_weight_type", False)
    if is_gguf_weight_type:
        if loaded_shard_id is not None:
            param.data[loaded_shard_id].copy_(loaded_weight)
            param.shard_weight_type[loaded_shard_id] = loaded_weight.item()
        else:
            param.shard_weight_type = {
                i: loaded_weight.item()
                for i, _ in enumerate(self.output_sizes)
            }
        return

    if is_gguf_weight:
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()

        output_dim = getattr(param, "output_dim", None)
        shard_size = loaded_weight.size(output_dim) // tp_size
        start_idx = tp_rank * shard_size

        if loaded_shard_id is not None:
            loaded_weight = loaded_weight.narrow(output_dim, start_idx,
                                                    shard_size)
            param.shard_id.append(loaded_shard_id)
            param.shard_id_map[loaded_shard_id] = len(param.data_container)
            param.data_container.append(loaded_weight)
            if len(param.data_container) == 2:
                self.qweight = param.materialize_nested()
            return

    param_data = param.data
    output_dim = getattr(param, "output_dim", None)
    # Special case for AQLM codebooks.
    is_metadata = getattr(param, "is_metadata", False)
    # Special case for per-tensor scale to load scalar into fused array.
    needs_scalar_to_array = getattr(param, "needs_scalar_to_array", False)
    
    # Support gemm_tn->gemm_nn here
    is_quantization = not isinstance(self.quant_method, UnquantizedLinearMethod)
    
    if loaded_shard_id is None:
        # Loaded weight is already fused on disk (mlp).
        # (e.g., Phi-3's gate_up_proj).
        if output_dim is None:
            if needs_scalar_to_array:
                param_data, loaded_weight = adjust_scalar_to_fused_array(
                    param_data, loaded_weight, 0)

            assert param_data.shape == loaded_weight.shape
            param_data.copy_(loaded_weight)
            return
        current_shard_offset = 0
        use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit",
                                        False)
        shard_offsets: list[tuple[int, int, int]] = []
        for i, output_size in enumerate(self.output_sizes):
            shard_offsets.append((i, current_shard_offset, output_size))
            current_shard_offset += output_size
        packed_dim = getattr(param, "packed_dim", None)
        for shard_id, shard_offset, shard_size in shard_offsets:
            # Special case for Quantization.
            # If quantized, we need to adjust the offset and size to account
            # for the packing.
            if packed_dim == output_dim:
                shard_size = shard_size // param.pack_factor
                shard_offset = shard_offset // param.pack_factor
                # Special case for Marlin.
                shard_size, shard_offset = adjust_marlin_shard(
                    param, shard_size, shard_offset)

            shard_size, shard_offset = adjust_bitblas_shard(
                param, shard_size, shard_offset)

            if use_bitsandbytes_4bit:
                index = list(itertools.accumulate([0] + self.output_sizes))
                orig_offsets = {
                    str(i): (index[i], size)
                    for i, size in enumerate(self.output_sizes)
                }
                orig_offsets["total"] = (self.output_size, 0)
                shard_size, shard_offset = adjust_bitsandbytes_4bit_shard(
                    param, orig_offsets, str(shard_id))

            loaded_weight_shard = loaded_weight.narrow(
                output_dim, shard_offset, shard_size)
            self.weight_loader(param, loaded_weight_shard, shard_id)
        return

    assert loaded_shard_id < len(self.output_sizes)
    tp_rank = get_tensor_model_parallel_rank()
    tp_size = get_tensor_model_parallel_world_size()
    if output_dim is not None:
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // tp_size
        shard_size = self.output_sizes[loaded_shard_id] // tp_size
        # Special case for quantization.
        # If quantized, we need to adjust the offset and size to account
        # for the packing.
        packed_dim = getattr(param, "packed_dim", None)
        if packed_dim == output_dim:
            shard_size = shard_size // param.pack_factor
            shard_offset = shard_offset // param.pack_factor
            # Special case for Marlin.
            shard_size, shard_offset = adjust_marlin_shard(
                param, shard_size, shard_offset)
        shard_size, shard_offset = adjust_bitblas_shard(
            param, shard_size, shard_offset)

        use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit",
                                        False)
        is_sharded_weight = getattr(param, "is_sharded_weight", False)
        # bitsandbytes loads the weights of the specific portion
        # no need to narrow
        is_sharded_weight = is_sharded_weight or use_bitsandbytes_4bit

        if use_bitsandbytes_4bit:
            shard_size = loaded_weight.shape[output_dim]
            shard_offset = loaded_weight.shape[output_dim] * \
                loaded_shard_id

        # Support gemm_tn->gemm_nn here
        if not envs.MACA_VLLM_USE_TN_2_NN or is_quantization:
            param_data = param_data.narrow(output_dim, shard_offset,shard_size)
        else:
            param_data = param_data.narrow(int(not(output_dim)), shard_offset,shard_size)
            
        start_idx = tp_rank * shard_size
        if not is_sharded_weight:
            loaded_weight = loaded_weight.narrow(output_dim, start_idx,
                                                    shard_size)
    # Special case for AQLM codebooks.
    elif is_metadata:
        # metadata indicates fixed size concatenated along dim 0
        shard_size = loaded_weight.shape[0]
        shard_offset = loaded_shard_id * shard_size
        param_data = param_data.narrow(0, shard_offset, shard_size)

    # Special case for per-tensor scales in fused case.
    elif needs_scalar_to_array:
        param_data, loaded_weight = adjust_scalar_to_fused_array(
            param_data, loaded_weight, loaded_shard_id)

    else:
        ignore_warning = getattr(param, "ignore_warning", False)
        if not ignore_warning:
            logger.warning(
                "Loading a weight without `output_dim` attribute in "
                "MergedColumnParallelLinear, assume the weight is "
                "the same for all partitions.")
    # Support gemm_tn->gemm_nn here
    if envs.MACA_VLLM_USE_TN_2_NN and not is_quantization:
        loaded_weight = loaded_weight.t()
    assert param_data.shape == loaded_weight.shape
    param_data.copy_(loaded_weight)

def QKVParallelLinear_weight_loader(self,
                    param: Parameter,
                    loaded_weight: torch.Tensor,
                    loaded_shard_id: Optional[str] = None):

    # Special case for GGUF
    # initialize GGUF param after we know the quantize type
    is_gguf_weight = getattr(param, "is_gguf_weight", False)
    is_gguf_weight_type = getattr(param, "is_gguf_weight_type", False)
    if is_gguf_weight_type:
        idx_map = {"q": 0, "k": 1, "v": 2}
        if loaded_shard_id is not None:
            param.data[idx_map[loaded_shard_id]].copy_(loaded_weight)
            param.shard_weight_type[loaded_shard_id] = loaded_weight.item()
        else:
            param.shard_weight_type = {
                k: loaded_weight.item()
                for k in idx_map
            }
        return

    if is_gguf_weight:
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()

        output_dim = getattr(param, "output_dim", None)
        shard_size = loaded_weight.size(output_dim) // tp_size
        start_idx = tp_rank * shard_size

        if loaded_shard_id is not None:
            loaded_weight = loaded_weight.narrow(output_dim, start_idx,
                                                    shard_size)
            param.shard_id.append(loaded_shard_id)
            param.shard_id_map[loaded_shard_id] = len(param.data_container)
            param.data_container.append(loaded_weight)
            if len(param.data_container) == 3:
                self.qweight = param.materialize_nested()
            return

    param_data = param.data
    output_dim = getattr(param, "output_dim", None)
    # Special case for AQLM codebooks.
    is_metadata = getattr(param, "is_metadata", False)

    # Special case for per-tensor scales in fused case.
    needs_scalar_to_array = getattr(param, "needs_scalar_to_array", False)
    
    # Support gemm_tn->gemm_nn here
    is_quantization = not isinstance(self.quant_method, UnquantizedLinearMethod)
    
    if loaded_shard_id is None:
        # Loaded weight is already fused on disk (qkv).
        # (e.g., Phi-3's qkv_proj).
        if output_dim is None:
            if needs_scalar_to_array:
                param_data, loaded_weight = adjust_scalar_to_fused_array(
                    param_data, loaded_weight, 0)

            assert param_data.shape == loaded_weight.shape
            param_data.copy_(loaded_weight)
            return
        shard_offsets = [
            # (shard_id, shard_offset, shard_size)
            ("q", 0, self.total_num_heads * self.head_size),
            ("k", self.total_num_heads * self.head_size,
                self.total_num_kv_heads * self.head_size),
            ("v", (self.total_num_heads + self.total_num_kv_heads) *
                self.head_size, self.total_num_kv_heads * self.head_size),
        ]
        use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit",
                                        False)

        packed_dim = getattr(param, "packed_dim", None)
        for shard_id, shard_offset, shard_size in shard_offsets:
            # Special case for Quantized Weights.
            # If quantized, we need to adjust the offset and size to account
            # for the packing.
            if packed_dim == output_dim:
                shard_size = shard_size // param.pack_factor
                shard_offset = shard_offset // param.pack_factor

                # Special case for Marlin.
                shard_size, shard_offset = adjust_marlin_shard(
                    param, shard_size, shard_offset)

            if use_bitsandbytes_4bit:
                orig_qkv_offsets = {
                    "q": (0, self.total_num_heads * self.head_size),
                    "k": (self.total_num_heads * self.head_size,
                            self.total_num_kv_heads * self.head_size),
                    "v":
                    ((self.total_num_heads + self.total_num_kv_heads) *
                        self.head_size,
                        self.total_num_kv_heads * self.head_size),
                    "total":
                    ((self.total_num_heads + 2 * self.total_num_kv_heads) *
                        self.head_size, 0)
                }

                shard_size, shard_offset = adjust_bitsandbytes_4bit_shard(
                    param, orig_qkv_offsets, shard_id)

            loaded_weight_shard = loaded_weight.narrow(
                output_dim, shard_offset, shard_size)
            self.weight_loader(param, loaded_weight_shard, shard_id)
        return

    tp_rank = get_tensor_model_parallel_rank()
    assert loaded_shard_id in ["q", "k", "v"]

    # If output dim is defined, use the default loading process.
    if output_dim is not None:
        if loaded_shard_id == "q":
            shard_offset = 0
            shard_size = self.num_heads * self.head_size
        elif loaded_shard_id == "k":
            shard_offset = self.num_heads * self.head_size
            shard_size = self.num_kv_heads * self.head_size
        elif loaded_shard_id == "v":
            shard_offset = (self.num_heads +
                            self.num_kv_heads) * self.head_size
            shard_size = self.num_kv_heads * self.head_size
        # Special case for Quantized Weights.
        # If quantized, we need to adjust the offset and size to account
        # for the packing.
        packed_dim = getattr(param, "packed_dim", None)
        if packed_dim == output_dim:
            shard_size = shard_size // param.pack_factor
            shard_offset = shard_offset // param.pack_factor

            # Special case for Marlin.
            shard_size, shard_offset = adjust_marlin_shard(
                param, shard_size, shard_offset)

        use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit",
                                        False)
        is_sharded_weight = getattr(param, "is_sharded_weight", False)
        # bitsandbytes loads the weights of the specific portion
        # no need to narrow
        is_sharded_weight = is_sharded_weight or use_bitsandbytes_4bit

        if use_bitsandbytes_4bit:
            orig_qkv_offsets = {
                "q": (0, self.num_heads * self.head_size),
                "k": (self.num_heads * self.head_size,
                        self.num_kv_heads * self.head_size),
                "v":
                ((self.num_heads + self.num_kv_heads) * self.head_size,
                    self.num_kv_heads * self.head_size),
                "total":
                ((self.num_heads + 2 * self.num_kv_heads) * self.head_size,
                    0)
            }
            shard_size, shard_offset = adjust_bitsandbytes_4bit_shard(
                param, orig_qkv_offsets, loaded_shard_id)

        # Support gemm_tn->gemm_nn here
        if not envs.MACA_VLLM_USE_TN_2_NN or len(param_data.shape)==1 or is_quantization:
            param_data = param_data.narrow(output_dim, shard_offset,shard_size)
        else:
            param_data = param_data.narrow(int(not(output_dim)), shard_offset,shard_size)
        
        if loaded_shard_id == "q":
            shard_id = tp_rank
        else:
            shard_id = tp_rank // self.num_kv_head_replicas
        start_idx = shard_id * shard_size

        if not is_sharded_weight:
            loaded_weight = loaded_weight.narrow(output_dim, start_idx,
                                                    shard_size)

    # Special case for for AQLM codebooks.
    elif is_metadata:
        # metadata indicates fixed size concatenated along dim 0
        shard_size = loaded_weight.shape[0]
        shard_index = ["q", "k", "v"].index(loaded_shard_id)
        param_data = param_data.narrow(0, shard_index * shard_size,
                                        shard_size)
    # Special case for per-tensor scales in fused case.
    elif needs_scalar_to_array:
        param_data, loaded_weight = adjust_scalar_to_fused_array(
            param_data, loaded_weight, loaded_shard_id)
    else:
        ignore_warning = getattr(param, "ignore_warning", False)
        if not ignore_warning:
            logger.warning(
                "Loading a weight without `output_dim` attribute in "
                "QKVParallelLinear, assume the weight is the same "
                "for all partitions.")

    # Support gemm_tn->gemm_nn here
    if envs.MACA_VLLM_USE_TN_2_NN and not is_quantization:
        loaded_weight = loaded_weight.t()

    assert param_data.shape == loaded_weight.shape
    param_data.copy_(loaded_weight)

def RowParallelLinear_weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
    tp_rank = get_tensor_model_parallel_rank()
    tp_size = get_tensor_model_parallel_world_size()
    input_dim = getattr(param, "input_dim", None)
    use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit", False)
    is_sharded_weight = getattr(param, "is_sharded_weight", False)
    # bitsandbytes loads the weights of the specific portion
    # no need to narrow
    is_sharded_weight = is_sharded_weight or use_bitsandbytes_4bit

    # Special case for GGUF
    is_gguf_weight = getattr(param, "is_gguf_weight", False)
    is_gguf_weight_type = getattr(param, "is_gguf_weight_type", False)
    if is_gguf_weight_type:
        param.weight_type = loaded_weight.item()

    # Materialize GGUF UninitializedParameter
    if is_gguf_weight and isinstance(param, UninitializedParameter):
        weight_shape = list(loaded_weight.shape)
        if input_dim:
            weight_shape[input_dim] = weight_shape[input_dim] // tp_size
        param.materialize(tuple(weight_shape), dtype=loaded_weight.dtype)

    param_data = param.data
    # Support gemm_tn->gemm_nn here
    is_quantization = not isinstance(self.quant_method, UnquantizedLinearMethod)
    
    if input_dim is not None and not is_sharded_weight:
        # Support gemm_tn->gemm_nn here
        if not envs.MACA_VLLM_USE_TN_2_NN or is_quantization:
            shard_size = param_data.shape[input_dim]
        else:
            shard_size = param_data.shape[int(not(input_dim))]
        start_idx = tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(input_dim, start_idx,
                                                shard_size)

    # Special case for loading scales off disk, which often do not
    # have a shape (such as in the case of AutoFP8).
    if len(loaded_weight.shape) == 0:
        loaded_weight = loaded_weight.reshape(1)

    # Support gemm_tn->gemm_nn here
    if envs.MACA_VLLM_USE_TN_2_NN and not is_quantization:
        loaded_weight = loaded_weight.t()
    assert param_data.shape == loaded_weight.shape
    param_data.copy_(loaded_weight)
    
vllm.model_executor.layers.linear.adjust_scalar_to_fused_array = adjust_scalar_to_fused_array
vllm.model_executor.layers.linear.UnquantizedLinearMethod.create_weights = UnquantizedLinearMethod_create_weights
vllm.model_executor.layers.linear.UnquantizedLinearMethod.apply = UnquantizedLinearMethod_apply
vllm.model_executor.layers.linear.ReplicatedLinear.weight_loader = ReplicatedLinear_weight_loader
vllm.model_executor.layers.linear.ColumnParallelLinear.weight_loader = ColumnParallelLinear_weight_loader
vllm.model_executor.layers.linear.MergedColumnParallelLinear.weight_loader = MergedColumnParallelLinear_weight_loader
vllm.model_executor.layers.linear.QKVParallelLinear.weight_loader = QKVParallelLinear_weight_loader
vllm.model_executor.layers.linear.RowParallelLinear.weight_loader = RowParallelLinear_weight_loader

register_patch("vllm.model_executor.layers.linear", "adjust_scalar_to_fused_array", adjust_scalar_to_fused_array)
register_patch("vllm.model_executor.layers.linear", "UnquantizedLinearMethod.create_weights", UnquantizedLinearMethod_create_weights)
register_patch("vllm.model_executor.layers.linear", "UnquantizedLinearMethod.apply", UnquantizedLinearMethod_apply)
register_patch("vllm.model_executor.layers.linear", "ReplicatedLinear.weight_loader", ReplicatedLinear_weight_loader)
register_patch("vllm.model_executor.layers.linear", "ColumnParallelLinear.weight_loader", ColumnParallelLinear_weight_loader)
register_patch("vllm.model_executor.layers.linear", "MergedColumnParallelLinear.weight_loader", MergedColumnParallelLinear_weight_loader)
register_patch("vllm.model_executor.layers.linear", "QKVParallelLinear.weight_loader", QKVParallelLinear_weight_loader)
register_patch("vllm.model_executor.layers.linear", "RowParallelLinear.weight_loader", RowParallelLinear_weight_loader)


