# SPDX-License-Identifier: Apache-2.0
from typing import TYPE_CHECKING, Any, Literal, cast, get_args
from vllm.config.model import ModelConfig
from vllm.config.parallel import ParallelConfig

LayerBlockType = Literal["attention", "linear_attention", "mamba"]

from vllm.transformers_utils.model_arch_config_convertor import (
    MODEL_ARCH_CONFIG_CONVERTORS,
    ModelArchConfigConvertorBase,
)


def maca_get_num_layers_by_block_type(
    self,
    parallel_config: ParallelConfig,
    block_type: LayerBlockType = "attention",
) -> int:
    # This function relies on 'layers_block_type' in hf_config,
    # for w/o this attribute, we will need to have workarounds like so
    attn_block_type = block_type == "attention"
    is_transformer = (
        not self.is_hybrid and not self.has_noops and not self.is_attention_free
    )
    start, end = self.get_layers_start_end_indices(parallel_config)

    if is_transformer:
        # Handle the basic case first
        return end - start if attn_block_type else 0
    elif self.is_attention_free:
        # Attention free
        # Note that this code assumes there
        # is only one type of attention-free block type.
        return 0 if attn_block_type else end - start
    elif self.has_noops:
        block_configs = self.hf_config.block_configs
        return sum(not bc.attention.no_op for bc in block_configs[start:end])
    else:
        # Hybrid model Jamba
        layers_block_type_value = getattr(
            self.hf_text_config, "layers_block_type", None
        )
        if layers_block_type_value is not None:
            if self.model_arch_config.text_model_type == "zamba2":
                if attn_block_type:
                    return sum(
                        t == "hybrid" for t in layers_block_type_value[start:end]
                    )
                else:
                    return self.get_num_layers(parallel_config)
            return sum(t == block_type for t in layers_block_type_value[start:end])

        # Hybrid model Minimax
        attn_type_list = getattr(self.hf_config, "attn_type_list", None)
        if attn_type_list:
            return sum(t == 1 for t in attn_type_list[start:end])

        # Hybrid model Qwen3Next Qwen3.5 Series
        layer_types_value = getattr(self.hf_text_config, "layer_types", None)
        if layer_types_value is not None:
            if block_type == "attention":
                return sum(t == "full_attention" for t in layer_types_value[start:end])
            elif block_type == "linear_attention":
                return sum(
                    t == "linear_attention" for t in layer_types_value[start:end]
                )
            else:
                return sum(t == block_type for t in layer_types_value[start:end])

        if (
            layers_block_type_value is None
            and attn_type_list is None
            and layer_types_value is None
        ):
            raise ValueError(
                "The model is an hybrid without a layers_block_type or an "
                "attn_type_list, or a layer_types in the hf_config, "
                f"cannot determine the num of {block_type} layers"
            )


class Qwen3_5MTPModelArchConfigConvertor(ModelArchConfigConvertorBase):
    def get_num_hidden_layers(self) -> int:
        return getattr(self.hf_text_config, "mtp_num_hidden_layers", 0)


ModelConfig.get_num_layers_by_block_type = maca_get_num_layers_by_block_type
MODEL_ARCH_CONFIG_CONVERTORS["qwen3_5"] = Qwen3_5MTPModelArchConfigConvertor
