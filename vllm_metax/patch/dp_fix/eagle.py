# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------
# Note: fix triton kernel compilation errors
# ---------------------------------------------------------------------------

import torch

from vllm.config import (
    get_layers_from_vllm_config,
)
from vllm.v1.spec_decode.eagle import EagleProposer

import torch.nn as nn

# -------------------------------
# Metax Modification: import DeepseekV32IndexerCache
# -------------------------------
from vllm_metax.models.deepseek_v2 import DeepseekV32IndexerCache
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.models import supports_multimodal
from vllm.distributed.parallel_state import get_pp_group
from vllm.logger import init_logger

logger = init_logger(__name__)


"""
Diff with vllm.v1.spec_decode.eagle.SpecDecodeBaseProposer.load_model (vllm v0.15.0)
"""


# support dsv32, patch DeepseekV32IndexerCache
def load_model(self, target_model: nn.Module) -> None:
    draft_model_config = self.vllm_config.speculative_config.draft_model_config
    target_attn_layer_names = set(
        get_layers_from_vllm_config(self.vllm_config, AttentionLayerBase).keys()
    )
    # FIXME: support hybrid kv for draft model
    target_indexer_layer_names = set(
        get_layers_from_vllm_config(self.vllm_config, DeepseekV32IndexerCache).keys()
    )

    from vllm.compilation.backends import set_model_tag

    with set_model_tag("eagle_head"):
        self.model = get_model(
            vllm_config=self.vllm_config, model_config=draft_model_config
        )

    draft_attn_layer_names = (
        get_layers_from_vllm_config(self.vllm_config, AttentionLayerBase).keys()
        - target_attn_layer_names
    )
    indexer_layers = get_layers_from_vllm_config(
        self.vllm_config, DeepseekV32IndexerCache
    )
    draft_indexer_layer_names = indexer_layers.keys() - target_indexer_layer_names
    self.attn_layer_names = list(draft_attn_layer_names - draft_indexer_layer_names)
    self.indexer_layer_names = list(draft_indexer_layer_names)

    if self.indexer_layer_names:
        first_layer = self.indexer_layer_names[0]
        self.draft_indexer_metadata_builder = (
            indexer_layers[first_layer]
            .get_attn_backend()
            .get_builder_cls()(
                indexer_layers[first_layer].get_kv_cache_spec(self.vllm_config),
                self.indexer_layer_names,
                self.vllm_config,
                self.device,
            )
        )
    else:
        self.draft_indexer_metadata_builder = None

    if self.supports_mm_inputs:
        # Even if the target model is multimodal, we can also use
        # text-only draft models
        try:
            dummy_input_ids = torch.tensor([[1]], device=self.input_ids.device)
            self.model.embed_input_ids(dummy_input_ids, multimodal_embeddings=None)
        except (NotImplementedError, AttributeError, TypeError):
            logger.warning(
                "Draft model does not support multimodal inputs, "
                "falling back to text-only mode"
            )
            self.supports_mm_inputs = False

    if supports_multimodal(target_model):
        # handle multimodality
        if self.get_model_name(target_model) in [
            "Qwen2_5_VLForConditionalGeneration",
            "Qwen3VLForConditionalGeneration",
            "Qwen3VLMoeForConditionalGeneration",
        ]:
            self.model.config.image_token_index = target_model.config.image_token_id
        elif self.get_model_name(target_model) == "PixtralForConditionalGeneration":
            self.model.config.image_token_index = (
                target_model.config.vision_config.image_token_id
            )
        else:
            self.model.config.image_token_index = target_model.config.image_token_index
        target_language_model = target_model.get_language_model()
    else:
        target_language_model = target_model

    # share embed_tokens with the target model if needed
    if get_pp_group().world_size == 1:
        if hasattr(target_language_model.model, "embed_tokens"):
            target_embed_tokens = target_language_model.model.embed_tokens
        elif hasattr(target_language_model.model, "embedding"):
            target_embed_tokens = target_language_model.model.embedding
        else:
            raise AttributeError(
                "Target model does not have 'embed_tokens' or 'embedding' attribute"
            )

        share_embeddings = False
        if hasattr(self.model, "has_own_embed_tokens"):
            # EAGLE model
            if not self.model.has_own_embed_tokens:
                share_embeddings = True
                logger.info(
                    "Detected EAGLE model without its own embed_tokens in the"
                    " checkpoint. Sharing target model embedding weights with the"
                    " draft model."
                )
            elif (
                isinstance(target_embed_tokens.weight, torch.Tensor)
                and isinstance(self.model.model.embed_tokens.weight, torch.Tensor)
                # TODO: Offload to CPU for comparison to avoid extra GPU memory
                # usage in CI testing environments with limited GPU memory
                and torch.equal(
                    target_embed_tokens.weight.cpu(),
                    self.model.model.embed_tokens.weight.cpu(),
                )
            ):
                share_embeddings = True
                logger.info(
                    "Detected EAGLE model with embed_tokens identical to the target"
                    " model. Sharing target model embedding weights with the draft"
                    " model."
                )
            else:
                logger.info(
                    "Detected EAGLE model with distinct embed_tokens weights. "
                    "Keeping separate embedding weights from the target model."
                )
        else:
            # MTP model
            share_embeddings = True
            logger.info(
                "Detected MTP model. "
                "Sharing target model embedding weights with the draft model."
            )

        if share_embeddings:
            if hasattr(self.model.model, "embed_tokens"):
                del self.model.model.embed_tokens
            self.model.model.embed_tokens = target_embed_tokens
    else:
        logger.info(
            "The draft model's vocab embedding will be loaded separately"
            " from the target model."
        )

    # share lm_head with the target model if needed
    share_lm_head = False
    if hasattr(self.model, "has_own_lm_head"):
        # EAGLE model
        if not self.model.has_own_lm_head:
            share_lm_head = True
            logger.info(
                "Detected EAGLE model without its own lm_head in the checkpoint. "
                "Sharing target model lm_head weights with the draft model."
            )
        elif (
            hasattr(target_language_model, "lm_head")
            and isinstance(target_language_model.lm_head.weight, torch.Tensor)
            and isinstance(self.model.lm_head.weight, torch.Tensor)
            # TODO: Offload to CPU for comparison to avoid extra GPU memory
            # usage in CI testing environments with limited GPU memory
            and torch.equal(
                target_language_model.lm_head.weight.cpu(),
                self.model.lm_head.weight.cpu(),
            )
        ):
            share_lm_head = True
            logger.info(
                "Detected EAGLE model with lm_head identical to the target model. "
                "Sharing target model lm_head weights with the draft model."
            )
        else:
            logger.info(
                "Detected EAGLE model with distinct lm_head weights. "
                "Keeping separate lm_head weights from the target model."
            )
    else:
        # MTP model
        share_lm_head = True
        logger.info(
            "Detected MTP model. "
            "Sharing target model lm_head weights with the draft model."
        )

    if share_lm_head and hasattr(target_language_model, "lm_head"):
        if hasattr(self.model, "lm_head"):
            del self.model.lm_head
        self.model.lm_head = target_language_model.lm_head


EagleProposer.load_model = load_model
