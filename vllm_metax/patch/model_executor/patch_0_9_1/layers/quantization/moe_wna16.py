# SPDX-License-Identifier: Apache-2.0

import vllm
from vllm_metax.patch.hook_registry import register_patch
from vllm.logger import init_logger

logger = init_logger(__name__)

import torch
from typing import Optional
from vllm.distributed import get_tensor_model_parallel_rank, get_tp_group
from vllm.model_executor.layers.quantization.moe_wna16 import (MoeWNA16Config,
                                                               is_layer_skipped_quant,
                                                               check_marlin_supports_layer,
                                                               MoeWNA16Method)
from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.model_executor.layers.linear import LinearBase,UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase

class MetaxMoeWNA16Config(MoeWNA16Config):
    def __init__(self,  *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    # ┌------------------------  Metax Modification -------------------------┐
        self.use_marlin = False
    # └------------------------- Metax Modification -------------------------┘

    def get_quant_method(self, layer: torch.nn.Module,
                        prefix: str) -> Optional["QuantizeMethodBase"]:
        if is_layer_skipped_quant(prefix, self.modules_to_not_convert):
            return UnquantizedLinearMethod()
        elif isinstance(layer, LinearBase):
            # Avoid circular import
            from vllm.model_executor.layers.quantization.awq import AWQConfig
            from vllm.model_executor.layers.quantization.awq_marlin import (
                AWQMarlinConfig)
            from vllm.model_executor.layers.quantization.gptq import GPTQConfig
            from vllm.model_executor.layers.quantization.gptq_marlin import (
                GPTQMarlinConfig)
            if self.linear_quant_method == "gptq":
                if self.use_marlin:
                    return GPTQMarlinConfig.from_config(
                        self.full_config).get_quant_method(layer, prefix)
                else:
                    return GPTQConfig.from_config(
                        self.full_config).get_quant_method(layer, prefix)
            elif self.linear_quant_method == "awq":
                if self.use_marlin and check_marlin_supports_layer(
                        layer, self.group_size):
                    return AWQMarlinConfig.from_config(
                        self.full_config).get_quant_method(layer, prefix)
                else:
                    return AWQConfig.from_config(
                        self.full_config).get_quant_method(layer, prefix)
            else:
                raise ValueError("moe_wna16 only support gptq and awq.")
        elif isinstance(layer, FusedMoE):
            return MetaxMoeWNA16Method(self)
        return None
        
class MetaxMoeWNA16Method(MoeWNA16Method):
    @staticmethod
    def get_weight_loader(layer, weight_loader):

        def convert_awq_tensor(tensor, tensor_type):
            # convert awq qweight/qzeros to a standard format (assume int4)
            # qweight: (k, n // pack_factor_bit32) -> (n, k // pack_factor_bit8)
            # qzeros: (k // group_size, n // pack_factor_bit32) ->
            #         (n // pack_factor_bit8, k // group_size)
            # pack_factor_bit32 = 32 // weight_bits
            # pack_factor_bit8 = 8 // weight_bits

            # 0. suppose origin shape (a, b), dtype int32
            # 1. convert to uint8, shape (a, b) -> (a, 4 * b)
            size0 = tensor.size(0)
            tensor = tensor.view(torch.uint8)

            # 2. unpack to uint4 (only when weight_bits == 4)
            #    shape (a, 4 * b) -> (a, 4 * b, 2)
            shifter = torch.tensor([0, 4],
                                   dtype=torch.uint8,
                                   device=tensor.device)
            tensor = (tensor[:, :, None] >> shifter) & 0xF

            # 3. change order, see
            # https://github.com/casper-hansen/AutoAWQ/blob/v0.2.8/awq/utils/quant_utils.py
            # shape -> (a, 4 * b * pack_factor_bit8)
            reverse_awq_pack_order = [0, 4, 1, 5, 2, 6, 3, 7]
            tensor = tensor.view(-1, 8)[:, reverse_awq_pack_order]
            tensor = tensor.view(size0, -1)

            # 4. transpose, shape -> (4 * b * pack_factor_bit8, a)
            tensor = tensor.T.contiguous()

            # 5. repack (only when weight_bits == 4)
            # qweight shape -> (4 * b * pack_factor_bit8, a // pack_factor_bit8)
            # qzeros shape -> (4 * b, a)

            if tensor_type == "qweight":
                tensor = tensor[:, 1::2] * 16 + tensor[:, ::2]
            elif tensor_type == "qzeros":
                tensor = tensor[1::2, :] * 16 + tensor[::2, :]
            return tensor

        def convert_gptq_int4_qzeros(tensor):
            tensor = tensor.view(torch.uint8)
            shifter = torch.tensor([0, 4],
                                   dtype=torch.uint8,
                                   device=tensor.device)
            tensor = (tensor[:, :, None] >> shifter) & 0xF
            tensor = tensor + 1
            tensor = tensor[:, :, 0] + tensor[:, :, 1] * 16
            return tensor

        def moe_wna16_weight_loader(param: torch.nn.Parameter,
                                    loaded_weight: torch.Tensor,
                                    weight_name: str, shard_id: str,
                                    expert_id: int):
    # ┌------------------------  Metax Modification -------------------------┐
            if layer.ep_size > 1:
                global_expert_id = expert_id
                expert_id = layer._map_global_expert_id_to_local_expert_id(expert_id)
                if expert_id == -1:
                    return
    # └------------------------- Metax Modification -------------------------┘

            if "g_idx" in weight_name:
                return
            if not layer.quant_config.has_zp and "qzeros" in weight_name:
                return

            device = get_tp_group().device
            
    # ┌------------------------  Metax Modification -------------------------┐
            if layer.ep_size > 1:
                tp_rank = 0
            else:
                tp_rank = get_tensor_model_parallel_rank()
    # └------------------------- Metax Modification -------------------------┘

            loaded_weight = loaded_weight.to(device)
            shard_size = layer.intermediate_size_per_partition

            # convert gptq and awq weight to a standard format
            if layer.quant_config.linear_quant_method == "awq":
                assert layer.quant_config.weight_bits == 4
                if "weight" in weight_name:
                    loaded_weight = convert_awq_tensor(loaded_weight,
                                                       "qweight")
                elif "zeros" in weight_name:
                    loaded_weight = convert_awq_tensor(loaded_weight, "qzeros")
                else:
                    loaded_weight = loaded_weight.T
            elif layer.quant_config.linear_quant_method == "gptq":
                assert layer.quant_config.weight_bits in [4, 8]
                if "weight" in weight_name:
                    loaded_weight = loaded_weight.T.contiguous().view(
                        torch.uint8)
                elif "zeros" in weight_name:
                    # add 1 to gptq qzeros to align with awq
                    loaded_weight = loaded_weight.view(torch.uint8)
                    if layer.quant_config.weight_bits == 4:
                        loaded_weight = convert_gptq_int4_qzeros(
                            loaded_weight).T
                    else:
                        loaded_weight = loaded_weight.T + 1
                else:
                    loaded_weight = loaded_weight.T

            # repeat the qzeros/scales to fit new group size
            if layer.group_size_div_factor > 1 and \
                    "qzeros" in weight_name or "scales" in weight_name:
                loaded_weight = loaded_weight.repeat_interleave(
                    layer.group_size_div_factor, 1)

            if "w13_qzeros" in weight_name:
            # ┌------------------------  Metax Modification -------------------------┐
                if layer.ep_size > 1 :
                    tensor = loaded_weight.view(-1, param.data[expert_id].shape[0] // 2,
                                                loaded_weight.size(1))[tp_rank]
                else:
                    tensor = loaded_weight.view(layer.tp_size, -1,
                                                loaded_weight.size(1))[tp_rank]
            # └------------------------- Metax Modification -------------------------┘

                if shard_id == "w1":
                    param.data[expert_id, :shard_size // 2] = tensor
                else:
                    param.data[expert_id, shard_size // 2:] = tensor
            elif "w2_qzeros" in weight_name:
            # ┌------------------------  Metax Modification -------------------------┐

                if layer.ep_size > 1 :
                    param.data[expert_id] = loaded_weight.view(
                        loaded_weight.size(0), -1, param.data[expert_id].shape[1])[:, tp_rank]
                else:
                    param.data[expert_id] = loaded_weight.view(
                        loaded_weight.size(0), layer.tp_size, -1)[:, tp_rank]
            else:
                if layer.ep_size > 1:
                    expert_id = global_expert_id
                weight_loader(param, loaded_weight, weight_name, shard_id,
                              expert_id)
            # └------------------------- Metax Modification -------------------------┘

        return moe_wna16_weight_loader



vllm.model_executor.layers.quantization.moe_wna16.MoeWNA16Config = MetaxMoeWNA16Config