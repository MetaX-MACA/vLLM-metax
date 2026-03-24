# SPDX-License-Identifier: Apache-2.0
from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import KVCacheSpec, MambaSpec
from vllm.model_executor.layers.mamba.abstract import MambaBase


def maca_get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
    if (
        vllm_config.speculative_config is not None
        and vllm_config.model_config.hf_config.model_type
        not in ["qwen3_next", "qwen3_5", "qwen3_5_moe"]
    ):
        raise NotImplementedError(
            "Mamba with speculative decoding is not supported yet."
        )
    mamba_block_size = vllm_config.cache_config.mamba_block_size
    page_size_padded = vllm_config.cache_config.mamba_page_size_padded
    return MambaSpec(
        shapes=self.get_state_shape(),
        dtypes=self.get_state_dtype(),
        block_size=mamba_block_size,
        page_size_padded=page_size_padded,
        mamba_type=self.mamba_type,
        mamba_cache_mode=vllm_config.cache_config.mamba_cache_mode,
        num_speculative_blocks=(
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        ),
    )


MambaBase.get_kv_cache_spec = maca_get_kv_cache_spec
