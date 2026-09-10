# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# -----------------------------------------------
# Note: fix MiniMax-M3-FP8 weight load
# Affected versions: v0.24.0+ (ported to v0.26.0; runtime validation pending)
# -----------------------------------------------
from collections.abc import Iterable
import torch
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.utils import (
    is_pp_missing_parameter,
)

from vllm_metax.patch import patch


@patch("vllm.models.minimax_m3.nvidia.model", "MiniMaxM3Model.model_load_weights")
def model_load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
    # q/k/v_proj -> fused qkv_proj; gate_proj/up_proj -> fused gate_up_proj
    # (dense MLP and shared expert). On sparse layers the indexer
    # index_q/index_k_proj fold into the same fused qkv_proj
    # (MinimaxM3QKVParallelLinearWithIndexer); these entries simply never match on
    # dense layers, whose checkpoints have no index_*_proj weights. Leading
    # dots keep `q_proj`/`k_proj` from matching `index_q_proj`/`index_k_proj`
    # (preceded by `_`, not `.`).
    stacked_params_mapping: list[tuple[str, str, int | str]] = [
        # (param_name, shard_name, shard_id)
        (".qkv_proj", ".q_proj", "q"),
        (".qkv_proj", ".k_proj", "k"),
        (".qkv_proj", ".v_proj", "v"),
        (".qkv_proj", ".index_q_proj", "index_q"),
        (".qkv_proj", ".index_k_proj", "index_k"),
        (".gate_up_proj", ".gate_proj", 0),
        (".gate_up_proj", ".up_proj", 1),
    ]
    # (param_name, weight_name, expert_id, shard_id)
    expert_params_mapping = self.get_expert_mapping()
    params_dict = dict(self.named_parameters())
    loaded_params: set[str] = set()
    for name, loaded_weight in weights:
        # The MTP module is not modeled yet.
        if "mtp." in name:
            continue
        # The checkpoint stores block scales as ``weight_scale_inv``; the
        # ModelOpt MXFP8 layers expose them as ``weight_scale``.
        # ------------------------  Metax Modification -------------------------
        # Metax(note): The checkpoint stores block scales as ``weight_scale_inv``,
        # and the FP8 layers also expose them as ``weight_scale_inv``.
        # There is no need for conversion here.
        # if "weight_scale_inv" in name:
        #     name = name.replace("weight_scale_inv", "weight_scale")
        # -------------------------------------------- -------------------------
        for param_name, weight_name, shard_id in stacked_params_mapping:
            if weight_name not in name:
                continue
            # Routed experts (w1/w2/w3) are handled below; don't let the
            # stacked mapping rewrite them.
            if ("block_sparse_moe.experts." in name) and name not in params_dict:
                continue
            name = name.replace(weight_name, param_name)
            if name.endswith(".bias") and name not in params_dict:
                continue
            if is_pp_missing_parameter(name, self):
                continue
            if name not in params_dict:
                continue
            param = params_dict[name]
            weight_loader = param.weight_loader
            weight_loader(param, loaded_weight, shard_id)
            break
        else:
            for (
                param_name,
                weight_name,
                expert_id,
                expert_shard_id,
            ) in expert_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if is_pp_missing_parameter(name, self):
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(
                    param,
                    loaded_weight,
                    name,
                    shard_id=expert_shard_id,
                    expert_id=expert_id,
                )
                break
            else:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                remapped = maybe_remap_kv_scale_name(name, params_dict)
                if remapped is None:
                    continue
                name = remapped
                if is_pp_missing_parameter(name, self):
                    continue
                # Modules not modeled yet (e.g. attention) are skipped until
                # they are ported.
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
        loaded_params.add(name)
    return loaded_params
