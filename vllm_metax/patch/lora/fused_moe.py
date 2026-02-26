# SPDX-License-Identifier: Apache-2.0
from vllm import envs
from vllm.distributed.utils import divide
from vllm.model_executor.layers.fused_moe.config import (
    _get_config_dtype_str,
)
from vllm.model_executor.layers.fused_moe.fused_marlin_moe import (
    MarlinExperts,
)
from vllm_metax.model_executor.layers.fused_moe.fused_moe import (
    TritonExperts,
)
from vllm.model_executor.layers.fused_moe.fused_moe_modular_method import (
    FusedMoEModularMethod,
)
from vllm.model_executor.layers.fused_moe.gpt_oss_triton_kernels_moe import (
    UnfusedOAITritonExperts,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEModularKernel,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize import (
    MoEPrepareAndFinalizeNoEP,
)


def _inject_lora_into_fused_moe(self):
    moe_state_dict = {}
    top_k = self.base_layer.top_k

    self.base_layer.ensure_moe_quant_config_init()
    quant_config = self.base_layer.quant_method.moe_quant_config

    prepare_finalize = MoEPrepareAndFinalizeNoEP()
    m_fused_moe_fn = FusedMoEModularKernel(
        prepare_finalize,
        self.base_layer.quant_method.select_gemm_impl(
            prepare_finalize, self.base_layer
        ),
        self.base_layer.shared_experts,
    )
    if quant_config.use_mxfp4_w4a16:
        assert isinstance(
            m_fused_moe_fn.fused_experts, (MarlinExperts, UnfusedOAITritonExperts)
        )
    else:
        assert isinstance(m_fused_moe_fn.fused_experts, (MarlinExperts, TritonExperts))

    def fwd_decorator(layer, func):
        def wrapper(*args, **kwargs):
            moe_state_dict["hidden_states"] = kwargs["hidden_states"]
            moe_state_dict["topk_ids"] = kwargs["topk_ids"]
            moe_state_dict["topk_weights"] = kwargs["topk_weights"]
            moe_state_dict["expert_map"] = kwargs["expert_map"]
            moe_state_dict["apply_router_weight_on_input"] = kwargs[
                "apply_router_weight_on_input"
            ]
            result = func(*args, **kwargs)
            return result

        return wrapper

    def act_decorator(layer, func):
        def wrapper(*args, **kwargs):
            _, output, input = args

            hidden_states = moe_state_dict["hidden_states"]
            topk_weights = moe_state_dict["topk_weights"]
            curr_topk_ids = moe_state_dict["topk_ids"]

            expert_map = moe_state_dict["expert_map"]

            config_dtype = _get_config_dtype_str(
                dtype=hidden_states.dtype,
                use_fp8_w8a8=False,
                use_int8_w8a16=False,
                use_int4_w4a16=False,
            )
            CHUNK_SIZE = envs.VLLM_FUSED_MOE_CHUNK_SIZE
            num_tokens = hidden_states.size(0)
            M = min(num_tokens, CHUNK_SIZE)
            max_lora_rank = self.w13_lora_a_stacked[0].shape[-2]
            shrink_config, expand_config = self._get_lora_moe_configs(
                op_prefix="w13",
                num_loras=self.max_loras,
                rank=max_lora_rank,
                num_slices=self._w13_slices,
                M=M,
                layer=layer,
                top_k=top_k,
                config_dtype=config_dtype,
            )

            # get the block size of m from customized config or default config
            (
                sorted_token_ids_lora,
                expert_ids_lora,
                num_tokens_post_padded_lora,
            ) = self.punica_wrapper.moe_lora_align_block_size(
                curr_topk_ids,
                num_tokens,
                shrink_config["BLOCK_SIZE_M"],
                self.base_layer.local_num_experts,
                self.max_loras,
                self.adapter_enabled,
                expert_map,
            )

            moe_state_dict["sorted_token_ids_lora"] = sorted_token_ids_lora
            moe_state_dict["expert_ids_lora"] = expert_ids_lora
            moe_state_dict["num_tokens_post_padded_lora"] = num_tokens_post_padded_lora

            expert_ids_lora = expert_ids_lora.view(self.max_loras, -1)
            sorted_token_ids_lora = sorted_token_ids_lora.view(self.max_loras, -1)
            #

            self.punica_wrapper.add_lora_fused_moe(
                input.view(-1, top_k, input.shape[-1]),
                hidden_states,
                self.w13_lora_a_stacked,
                self.w13_lora_b_stacked,
                topk_weights,
                sorted_token_ids_lora,
                expert_ids_lora,
                num_tokens_post_padded_lora,
                max_lora_rank,
                top_k,
                shrink_config,  ## pass the shrink config
                expand_config,  ## pass the expand config
                self.adapter_enabled,
                fully_sharded=self.fully_sharded,
            )

            result = func(*args, **kwargs)

            moe_state_dict["intermediate_cache2"] = output
            return result

        return wrapper

    def moe_sum_decorator(layer, func):
        def wrapper(*args, **kwargs):
            hidden_states = moe_state_dict["hidden_states"]
            topk_weights = moe_state_dict["topk_weights"]

            config_dtype = _get_config_dtype_str(
                dtype=hidden_states.dtype,
                use_fp8_w8a8=False,
                use_int8_w8a16=False,
                use_int4_w4a16=False,
            )
            CHUNK_SIZE = envs.VLLM_FUSED_MOE_CHUNK_SIZE
            num_tokens = hidden_states.size(0)
            M = min(num_tokens, CHUNK_SIZE)
            max_lora_rank = self.w2_lora_a_stacked[0].shape[-2]
            shrink_config, expand_config = self._get_lora_moe_configs(
                op_prefix="w2",
                num_loras=self.max_loras,
                rank=max_lora_rank,
                num_slices=1,
                M=M,
                layer=layer,
                top_k=top_k,
                config_dtype=config_dtype,
            )

            sorted_token_ids_lora = moe_state_dict["sorted_token_ids_lora"]
            expert_ids_lora = moe_state_dict["expert_ids_lora"]
            num_tokens_post_padded_lora = moe_state_dict["num_tokens_post_padded_lora"]

            expert_ids_lora = expert_ids_lora.view(self.max_loras, -1)
            sorted_token_ids_lora = sorted_token_ids_lora.view(self.max_loras, -1)
            intermediate_cache2 = moe_state_dict["intermediate_cache2"]
            intermediate_cache3 = args[0]

            shard_size_w2 = divide(self.base_layer.hidden_size, self.tp_size)

            self.punica_wrapper.add_lora_fused_moe(
                intermediate_cache3,
                intermediate_cache2,
                self.w2_lora_a_stacked,
                self.w2_lora_b_stacked,
                topk_weights,
                sorted_token_ids_lora,
                expert_ids_lora,
                num_tokens_post_padded_lora,
                max_lora_rank,
                top_k,
                shrink_config,  ## pass the shrink config
                expand_config,  ## pass the expand config
                self.adapter_enabled,
                True,
                fully_sharded=self.fully_sharded,
                offset=shard_size_w2 * self.tp_rank if self.fully_sharded else 0,
            )

            result = func(*args, **kwargs)
            return result

        return wrapper

    fused_experts = m_fused_moe_fn.fused_experts

    m_fused_moe_fn.forward = fwd_decorator(self.base_layer, m_fused_moe_fn.forward)
    fused_experts.activation = act_decorator(self.base_layer, fused_experts.activation)
    fused_experts.moe_sum = moe_sum_decorator(self.base_layer, fused_experts.moe_sum)
    self.base_layer.quant_method = FusedMoEModularMethod(
        self.base_layer.quant_method, m_fused_moe_fn
    )


from vllm.lora.layers.fused_moe import FusedMoEWithLoRA

FusedMoEWithLoRA._inject_lora_into_fused_moe = _inject_lora_into_fused_moe
