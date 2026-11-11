# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from vLLM 36ae8803291c, minimax_m3/common/dspark.py.
"""MiniMax DSpark using vLLM 0.32's portable DFlash attention/cache API."""

from collections.abc import Iterable

import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.model_executor.layers.activation import SiluAndMulWithClamp
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
    ParallelLMHead,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.models.qwen3_dflash import (
    DFlashQwen3Attention,
    DFlashQwen3DecoderLayer,
    _get_dflash_fc_input_size,
    _resolve_layer_attention,
)
from vllm.model_executor.models.qwen3_dspark import (
    DSparkConfidenceHead,
    DSparkMarkovHead,
    Qwen3DSparkForCausalLM,
    Qwen3DSparkModel,
)
from vllm.model_executor.models.utils import (
    WeightsMapper,
    get_draft_quant_config,
    map_draft_quant_config as map_draft_quant_config_layers,
    maybe_prefix,
)

CHECKPOINT_PREFIX = "language_model.model.dspark."


def rename_checkpoint_weights(weights: Iterable):
    for name, tensor in weights:
        name = name.removeprefix(CHECKPOINT_PREFIX).replace(".decoder_layer.", ".")
        if name.startswith("final_norm."):
            name = "norm." + name.removeprefix("final_norm.")
        yield name, tensor


def map_draft_quant_config(quant_config):
    if quant_config is None:
        return None
    quant_config.apply_vllm_mapper(
        WeightsMapper(
            orig_to_new_prefix={CHECKPOINT_PREFIX: "model."},
            orig_to_new_substr={".decoder_layer.": "."},
        )
    )
    # compressed-tensors intentionally leaves regex targets unchanged.
    # BF16 auxiliary heads move from checkpoint dspark.* to runtime model.*.
    if hasattr(quant_config, "ignore"):
        quant_config.ignore = [
            name.replace(r"dspark\.", r"model\.") if name.startswith("re:") else name
            for name in quant_config.ignore
        ]
    return quant_config


class MiniMaxM3DSparkMLP(nn.Module):
    def __init__(self, config, quant_config, prefix):
        super().__init__()
        if config.hidden_act != "swigluoai" or not config.dspark_use_dense_ffn:
            raise ValueError("MiniMax DSpark requires dense swigluoai FFNs")
        intermediate = config.dspark_ffn_hidden_size
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [intermediate] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = SiluAndMulWithClamp(
            swiglu_limit=config.swiglu_limit,
            alpha=config.swiglu_alpha,
            beta=config.swiglu_beta,
        )

    def forward(self, x):
        x, _ = self.gate_up_proj(x)
        x, _ = self.down_proj(self.act_fn(x))
        return x


class MiniMaxM3DSparkDecoderLayer(DFlashQwen3DecoderLayer):
    def __init__(
        self,
        vllm_config,
        *,
        config,
        layer_idx,
        cache_config=None,
        quant_config=None,
        prefix="",
    ):
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        sliding_window, causal = _resolve_layer_attention(config, layer_idx)
        self.self_attn = DFlashQwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            rope_parameters={
                "rope_theta": config.rope_theta,
                "partial_rotary_factor": config.partial_rotary_factor,
            },
            max_position=config.max_position_embeddings,
            head_dim=config.head_dim,
            rms_norm_eps=config.rms_norm_eps,
            sliding_window=sliding_window,
            causal=causal,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.self_attn.q_norm = GemmaRMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.self_attn.k_norm = GemmaRMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.mlp = MiniMaxM3DSparkMLP(config, quant_config, f"{prefix}.mlp")
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )


@support_torch_compile
class MiniMaxM3DSparkModel(Qwen3DSparkModel):
    decoder_layer_cls = MiniMaxM3DSparkDecoderLayer

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int = 0,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        speculative_config = vllm_config.speculative_config
        assert speculative_config is not None
        self.config = speculative_config.draft_model_config.hf_config
        self.vocab_size = self.config.vocab_size
        # NOTE(metax): Map standalone MiniMax exclusions before constructing layers.
        self.quant_config = map_draft_quant_config(get_draft_quant_config(vllm_config))
        map_draft_quant_config_layers(
            self.quant_config,
            self.hf_to_vllm_mapper,
            prefix,
            start_layer_id,
            self.config.num_hidden_layers,
        )

        drafter_config = getattr(self.config, "eagle_config", {})
        drafter_config.update(getattr(self.config, "dflash_config", {}))

        self.use_aux_hidden_state = drafter_config.get(
            "use_aux_hidden_state",
            getattr(self.config, "use_aux_hidden_state", True),
        )

        current_vllm_config = get_current_vllm_config()

        self.embed_tokens = VocabParallelEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )

        # Masked query slots are fed to the draft as `mask_token_id`. Most DFlash
        # checkpoints will have the mask embedding in the vocabulary embedding table
        # at that slot id. Some checkpoints (XiaomiMiMo/MiMo-V2.5-Pro-FP4-DFlash) ship
        # with a separate mask embedding tensor to use instead. When present, we load it
        # and substitute it for embed_tokens[mask_token_id] when computing embeddings.
        self.mask_token_id = drafter_config.get(
            "mask_token_id", getattr(self.config, "mask_token_id", None)
        )
        self.mask_embedding = nn.Parameter(
            torch.zeros(self.config.hidden_size, dtype=vllm_config.model_config.dtype),
            requires_grad=False,
        )
        self.has_separate_mask_embedding = False

        self.layers = nn.ModuleList(
            [
                self.decoder_layer_cls(
                    current_vllm_config,
                    config=self.config,
                    layer_idx=layer_idx,
                    cache_config=current_vllm_config.cache_config,
                    quant_config=self.quant_config,
                    prefix=maybe_prefix(prefix, f"layers.{layer_idx + start_layer_id}"),
                )
                for layer_idx in range(self.config.num_hidden_layers)
            ]
        )
        if self.use_aux_hidden_state:
            self.fc = ReplicatedLinear(
                input_size=_get_dflash_fc_input_size(
                    vllm_config,
                ),
                output_size=self.config.hidden_size,
                bias=False,
                params_dtype=vllm_config.model_config.dtype,
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, "fc"),
                return_bias=False,
            )
        # NOTE(metax): MiniMax stores Gemma offsets, including context precomputation.
        self.hidden_norm = GemmaRMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
        )
        self.norm = GemmaRMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
        )

        config = self.config
        draft_vocab_size = (
            getattr(config, "draft_vocab_size", None) or config.vocab_size
        )
        self.markov_head = DSparkMarkovHead(
            config.vocab_size,
            draft_vocab_size,
            config.markov_rank,
            prefix=maybe_prefix(prefix, "markov_head"),
            quant_config=self.quant_config,
            retain_weight_for_gather=(
                getattr(config, "dspark_draft_topk", None) is not None
            ),
        )
        self.confidence_head: DSparkConfidenceHead | None = None
        if getattr(config, "enable_confidence_head", False):
            with_markov = getattr(config, "confidence_head_with_markov", False)
            input_dim = config.hidden_size
            if with_markov:
                input_dim += config.markov_rank
            self.confidence_head = DSparkConfidenceHead(
                input_dim,
                prefix=maybe_prefix(prefix, "confidence_head"),
                bias=True,
                with_markov=with_markov,
            )

    def _build_context_kv_buffers(self, layers_attn, has_bias):
        # NOTE(metax): Keep Gemma normalization with the registered quant method;
        # upstream context precomputation uses ordinary RMSNorm weights.
        self._context_attentions = layers_attn
        self._k_norm_weights = torch.stack([a.k_norm.weight.data for a in layers_attn])

    def _project_context_kv(
        self, context_states, num_ctx, num_layers, num_kv_heads, head_dim
    ):
        normalized = self.hidden_norm(context_states)
        kv = []
        for attention in self._context_attentions:
            qkv, _ = attention.qkv_proj(normalized)
            kv.append(
                qkv[:, attention.q_size :].reshape(num_ctx, 2, num_kv_heads, head_dim)
            )
        all_kv = torch.stack(kv, dim=1).permute(2, 1, 0, 3, 4).contiguous()
        return all_kv[0], all_kv[1]

    def _normalize_context_k(self, all_k):
        # NOTE(metax): Gemma norm uses (weight + 1), unlike the Qwen grouped RMS op.
        return torch.stack(
            [
                layer.self_attn.k_norm(k)
                for layer, k in zip(self.layers, all_k, strict=True)
            ]
        )


class MiniMaxM3DSparkForCausalLM(Qwen3DSparkForCausalLM):
    model_cls = MiniMaxM3DSparkModel

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        speculative_config = vllm_config.speculative_config
        assert speculative_config is not None
        self.draft_model_config = speculative_config.draft_model_config
        self.config = self.draft_model_config.hf_config
        if getattr(self.config, "draft_vocab_size", None) is None:
            self.config.draft_vocab_size = getattr(self.config, "vocab_size", None)
        target_layer_num = vllm_config.model_config.get_total_num_hidden_layers()
        # NOTE(metax): The upstream DSpark constructor hardcodes the Qwen model.
        self.model = self.model_cls(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
            start_layer_id=target_layer_num,
        )

        logit_scale = getattr(self.config, "logit_scale", 1.0)
        self.lm_head = ParallelLMHead(
            self.config.draft_vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(
            self.config.draft_vocab_size, scale=logit_scale
        )
        self.target_vocab_size = vllm_config.model_config.get_vocab_size()
        # Only a pruned draft (subset of the target vocab) needs a d2t mapping to
        # scatter ids up. A draft >= the target vocab (e.g. padded to the target's
        # physical embedding size) maps identically, so no d2t is needed.
        if self.config.draft_vocab_size < self.target_vocab_size:
            self.draft_id_to_target_id = nn.Parameter(
                torch.zeros(self.config.draft_vocab_size, dtype=torch.long),
                requires_grad=False,
            )
        else:
            self.draft_id_to_target_id = None

    def load_weights(self, weights):
        # Only MXFP8 uses weight_scale. CT block-FP8 needs weight_scale_inv.
        def mapped():
            for name, tensor in rename_checkpoint_weights(weights):
                if self.draft_model_config.quantization in ("mxfp8", "modelopt_mxfp8"):
                    name = name.replace(".weight_scale_inv", ".weight_scale")
                yield name, tensor

        return super().load_weights(mapped())
