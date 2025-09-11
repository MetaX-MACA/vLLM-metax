# SPDX-License-Identifier: Apache-2.0
"""Inference-only Baichuan-MOE model."""
from transformers.configuration_utils import PretrainedConfig


class BaiChuanMoEConfig(PretrainedConfig):
    model_type = "baichuan-moe"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=64000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        hidden_act="silu",
        max_position_embeddings=4096,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        rope_base=1e6,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        moe_experts_fixed=0,
        moe_experts_selected=2,
        moe_experts_routed=8,
        num_experts_fixed_per_layer=None,  # "0,0,0,1,0,2"
        num_experts_selected_per_layer=None,  # "1,2,1,1,1,2"
        num_experts_routed_per_layer=None,  # "1,8,1,8,1,16"
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.rope_base = rope_base
        self.use_cache = use_cache
        self.moe_experts_fixed = moe_experts_fixed
        self.moe_experts_selected = moe_experts_selected
        self.moe_experts_routed = moe_experts_routed
        if num_experts_routed_per_layer:
            self.num_experts_routed_per_layer = [
                int(_.strip()) for _ in num_experts_routed_per_layer.split(",")
            ]
            assert len(
                self.num_experts_routed_per_layer) == self.num_hidden_layers
            assert all([_ >= 1 for _ in self.num_experts_routed_per_layer])
        else:
            self.num_experts_routed_per_layer = None

        if num_experts_selected_per_layer:
            self.num_experts_selected_per_layer = [
                int(_.strip())
                for _ in num_experts_selected_per_layer.split(",")
            ]
            assert len(
                self.num_experts_selected_per_layer) == self.num_hidden_layers
            assert all([
                x >= y for x, y in zip(self.num_experts_routed_per_layer,
                                       self.num_experts_selected_per_layer)
            ])
        else:
            self.num_experts_selected_per_layer = None

        if num_experts_fixed_per_layer:
            self.num_experts_fixed_per_layer = [
                int(_.strip()) for _ in num_experts_fixed_per_layer.split(",")
            ]
            assert len(
                self.num_experts_fixed_per_layer) == self.num_hidden_layers
        else:
            self.num_experts_fixed_per_layer = None

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


import copy
import math
from typing import Iterable, List, Optional, Tuple

import torch
from torch import nn
from vllm.attention import Attention, AttentionMetadata
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (get_pp_group, get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_reduce)
from vllm.model_executor.layers.activation import GeluAndMul, SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
#from vllm.model_executor.weight_utils import (default_weight_loader,
#                                              hf_model_weights_iterator)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.utils import (PPMissingLayer,
                                              is_pp_missing_parameter,
                                              make_layers, maybe_prefix)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.utils import set_weight_attrs
from vllm.sequence import IntermediateTensors

from vllm_metax.model_executor.layers.fused_moe.fused_moe import fused_moe


class MLP(nn.Module):

    def __init__(self,
                 hidden_size: int,
                 intermediate_size: int,
                 hidden_act: str,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj")
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act not in ["silu", "gelu"]:
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu and gelu are supported for now.")
        self.act_fn = SiluAndMul() if hidden_act == "silu" else GeluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        ret, _ = self.down_proj(x)

        return ret


class MixtralMLP(nn.Module):
    """
    This implementation is
    strictly equivalent to standard MoE with full capacity (no
    dropped tokens). It's faster since it formulates MoE operations
    in terms of block-sparse operations to accomodate imbalanced
    assignments of tokens to experts, whereas standard MoE either
    (1) drop tokens at the cost of reduced performance or (2) set
    capacity factor to number of experts and thus waste computation
    and memory on padding.
    """

    def __init__(self,
                 hidden_size,
                 intermediate_size,
                 hidden_act,
                 moe_experts_routed,
                 moe_experts_selected,
                 moe_experts_fixed,
                 quant_config: Optional[QuantizationConfig] = None,
                 params_dtype: Optional[torch.dtype] = None,
                 tp_size: Optional[int] = None,
                 prefix: str = ""):
        super().__init__()
        self.tp_size = tp_size or get_tensor_model_parallel_world_size()
        self.num_experts_routed = moe_experts_routed
        self.num_local_experts_routed = self.num_experts_routed // 1
        self.top_k = moe_experts_selected
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size // self.tp_size

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype
        self.router = ReplicatedLinear(
            self.hidden_size,
            self.num_experts_routed,
            bias=False,
            quant_config=quant_config,
            params_dtype=self.params_dtype,
        )

        self.ws = nn.Parameter(
            torch.empty(self.num_experts_routed,
                        2 * self.intermediate_size,
                        self.hidden_size,
                        device="cuda",
                        dtype=self.params_dtype))
        self.w2s = nn.Parameter(
            torch.empty(self.num_experts_routed,
                        self.hidden_size,
                        self.intermediate_size,
                        device="cuda",
                        dtype=self.params_dtype))

        set_weight_attrs(self.ws, {
            "weight_loader": self.weight_loader,
        })
        set_weight_attrs(self.w2s, {
            "weight_loader": self.weight_loader,
        })

        if moe_experts_fixed >= 1:
            self.local_experts_fixed = MLP(hidden_size,
                                           intermediate_size *
                                           moe_experts_fixed,
                                           hidden_act,
                                           quant_config=quant_config,
                                           prefix=f"{prefix}.mlp")
        else:
            self.local_experts_fixed = None

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor,
                      weight_name: str, expert_id: int):
        tp_rank = get_tensor_model_parallel_rank()
        param_data = param.data
        shard_size = self.intermediate_size
        shard = slice(tp_rank * shard_size, (tp_rank + 1) * shard_size)
        if weight_name.endswith("gate_proj.weight"):
            param_data[expert_id, 0:shard_size, :] = loaded_weight[shard, :]
        if weight_name.endswith("up_proj.weight"):
            param_data[expert_id,
                       shard_size:2 * shard_size, :] = loaded_weight[shard, :]
        if weight_name.endswith("down_proj.weight"):
            param_data[expert_id, :, :] = loaded_weight[:, shard]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        num_tokens, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_size)
        router_logits, _ = self.router(hidden_states)
        final_hidden_states = fused_moe(hidden_states,
                                        self.ws,
                                        self.w2s,
                                        router_logits,
                                        self.top_k,
                                        renormalize=True)

        if self.tp_size > 1:
            final_hidden_states = tensor_model_parallel_all_reduce(
                final_hidden_states)

        final_hidden_states = final_hidden_states.view(num_tokens, hidden_size)

        if self.local_experts_fixed:
            final_hidden_states += self.local_experts_fixed(
                hidden_states).reshape(num_tokens, hidden_size)
            final_hidden_states /= 2

        ret = final_hidden_states.reshape(num_tokens, hidden_size)
        return ret


class MixtralAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        rope_theta: float = 10000,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta

        self.W_pack = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=int(self.rope_theta),
            is_neox_style=True,
        )
        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              self.scaling,
                              num_kv_heads=self.num_kv_heads,
                              cache_config=cache_config,
                              quant_config=quant_config,
                              prefix=f"{prefix}.attn")

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        qkv, _ = self.W_pack(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        output, _ = self.o_proj(attn_output)
        return output


class DecoderLayer(nn.Module):

    def __init__(self,
                 config: BaiChuanMoEConfig,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = "") -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        # Requires transformers > 4.32.0
        rope_theta = getattr(config, "rope_base", 10000)
        self.self_attn = MixtralAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_attention_heads,
            rope_theta=rope_theta,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn")

        # Dense
        if config.moe_experts_routed == 1:
            self.mlp = MLP(hidden_size=config.hidden_size,
                           intermediate_size=config.intermediate_size,
                           hidden_act=config.hidden_act,
                           quant_config=quant_config,
                           prefix=f"{prefix}.mlp")
        # MoE
        else:
            self.mlp = MixtralMLP(config.hidden_size,
                                  config.intermediate_size,
                                  config.hidden_act,
                                  config.moe_experts_routed,
                                  config.moe_experts_selected,
                                  config.moe_experts_fixed,
                                  quant_config=quant_config,
                                  prefix=f"{prefix}.mlp")
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
    ) -> torch.Tensor:

        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)

        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


def layer_function(prefix, config, cache_config, quant_config):
    index = int(prefix.split(".")[-1])
    config_ = copy.deepcopy(config)

    config_.moe_experts_fixed = config.num_experts_fixed_per_layer[index]
    config_.moe_experts_selected = config.num_experts_selected_per_layer[index]
    config_.moe_experts_routed = config.num_experts_routed_per_layer[index]

    return DecoderLayer(config=config_,
                        cache_config=cache_config,
                        quant_config=quant_config,
                        prefix=prefix)


class Model(nn.Module):

    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.padding_idx = config.pad_token_id
        lora_vocab = (lora_config.lora_extra_vocab_size *
                      (lora_config.max_loras or 1)) if lora_config else 0
        self.vocab_size = config.vocab_size + lora_vocab
        self.org_vocab_size = config.vocab_size

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                self.vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
            )
        else:
            self.embed_tokens = PPMissingLayer()

        if config.num_experts_routed_per_layer:
            self.start_layer, self.end_layer, self.layers = make_layers(
                num_hidden_layers=config.num_hidden_layers,
                layer_fn=lambda prefix: layer_function(
                    prefix, config, cache_config, quant_config),
                prefix=f"{prefix}.layers",
            )
        else:
            self.start_layer, self.end_layer, self.layers = make_layers(
                num_hidden_layers=config.num_hidden_layers,
                layer_fn=lambda prefix: DecoderLayer(config=config,
                                                     cache_config=cache_config,
                                                     quant_config=quant_config,
                                                     prefix=prefix),
                prefix=f"{prefix}.layers",
            )

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors],
    ) -> torch.Tensor:
        if get_pp_group().is_first_rank:
            hidden_states = self.embed_tokens(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                kv_caches[i - self.start_layer],
                attn_metadata,
                residual,
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })

        hidden_states, _ = self.norm(hidden_states, residual)

        return hidden_states


class NormHead(nn.Module):

    def __init__(self, hidden_size, vocab_size, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((vocab_size, hidden_size)))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.norm_weight = nn.functional.normalize(self.weight)

    def forward(self, hidden_states):
        return nn.functional.linear(hidden_states, self.norm_weight)


class BaiChuanMoEForCausalLM(nn.Module, SupportsPP):
    # packed_modules_mapping = {
    #     "qkv_proj": [
    #         "q_proj",
    #         "k_proj",
    #         "v_proj",
    #     ],
    # }

    # # LoRA specific attributes
    # supported_lora_modules = [
    #     "qkv_proj",
    #     "o_proj",
    #     "embed_tokens",
    #     "lm_head",
    # ]
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }
    embedding_padding_modules = ["lm_head"]

    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.config = config
        self.quant_config = quant_config
        self.model = Model(vllm_config=vllm_config,
                           prefix=maybe_prefix(prefix, "model"))
        if get_pp_group().is_last_rank:
            self.unpadded_vocab_size = config.vocab_size
            if lora_config:
                self.unpadded_vocab_size += lora_config.lora_extra_vocab_size
            self.lm_head = ParallelLMHead(
                self.unpadded_vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                org_num_embeddings=config.vocab_size,
                padding_size=DEFAULT_VOCAB_PADDING_SIZE
                # We need bigger padding if using lora for kernel
                # compatibility
                if not lora_config else lora_config.lora_vocab_padding_size,
                prefix=maybe_prefix(prefix, "lm_head"))
            self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                    config.vocab_size)
            self.sampler = Sampler()
        else:
            self.lm_head = PPMissingLayer()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, kv_caches,
                                   attn_metadata, intermediate_tensors)
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def make_empty_intermediate_tensors(
            self, batch_size: int, dtype: torch.dtype,
            device: torch.device) -> IntermediateTensors:
        return IntermediateTensors({
            "hidden_states":
            torch.zeros((batch_size, self.config.hidden_size),
                        dtype=dtype,
                        device=device),
            "residual":
            torch.zeros((batch_size, self.config.hidden_size),
                        dtype=dtype,
                        device=device),
        })

    def sample(
        self,
        logits: Optional[torch.Tensor],
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("mlp.gate_up_proj", "mlp.gate_proj", 0),
            ("mlp.gate_up_proj", "mlp.up_proj", 1),
            ("mlp.local_experts_fixed.gate_up_proj",
             "mlp.local_experts_fixed.gate_proj", 0),
            ("mlp.local_experts_fixed.gate_up_proj",
             "mlp.local_experts_fixed.up_proj", 1),
        ]

        expert_params_mapping = [
            # (param_name, weight_name, expert_id)
            ("ws" if weight_name in ["gate_proj", "up_proj"] else "w2s",
             f"local_experts_routed.{expert_id}.{weight_name}.weight",
             expert_id) for expert_id in range(16)
            for weight_name in ["gate_proj", "down_proj", "up_proj"]
        ]

        params_dict = dict(self.named_parameters())

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for param_name, weight_name, expert_id in expert_params_mapping:
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)

                    if is_pp_missing_parameter(name, self):
                        continue

                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param,
                                  loaded_weight,
                                  weight_name,
                                  expert_id=expert_id)
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue

                    if is_pp_missing_parameter(name, self):
                        continue

                    param = params_dict.get(name)

                    if name == "lm_head.weight":
                        # do norm
                        norm_weight = nn.functional.normalize(loaded_weight)
                        weight_loader = getattr(param, "weight_loader",
                                                default_weight_loader)
                        weight_loader(param, norm_weight)
                    else:
                        weight_loader = getattr(param, "weight_loader",
                                                default_weight_loader)
                        weight_loader(param, loaded_weight)
