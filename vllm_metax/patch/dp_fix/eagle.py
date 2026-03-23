# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

# ---------------------------------------------------------------------------
# Note: fix triton kernel compilation errors
# ---------------------------------------------------------------------------

import torch

from vllm.config import (
    get_layers_from_vllm_config,
)
from vllm.v1.spec_decode.eagle import EagleProposer

from vllm.config import (
    CUDAGraphMode,
)
from vllm.forward_context import set_forward_context
from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM
from vllm.triton_utils import triton
from vllm.v1.attention.backends.tree_attn import (
    TreeAttentionMetadata,
)
from vllm.v1.attention.backends.utils import (
    CommonAttentionMetadata,
)
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

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

PADDING_SLOT_ID = -1


class MacaEagleProposer(EagleProposer):
    # support dsv32, patch DeepseekV32IndexerCache
    def load_model(self, target_model: nn.Module) -> None:
        draft_model_config = self.vllm_config.speculative_config.draft_model_config
        target_attn_layer_names = set(
            get_layers_from_vllm_config(self.vllm_config, AttentionLayerBase).keys()
        )
        # FIXME: support hybrid kv for draft model
        target_indexer_layer_names = set(
            get_layers_from_vllm_config(
                self.vllm_config, DeepseekV32IndexerCache
            ).keys()
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
            ]:
                self.model.config.image_token_index = target_model.config.image_token_id
            elif self.get_model_name(target_model) == "PixtralForConditionalGeneration":
                self.model.config.image_token_index = (
                    target_model.config.vision_config.image_token_id
                )
            else:
                self.model.config.image_token_index = (
                    target_model.config.image_token_index
                )
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

            # /---------------------------- metax modified ----------------------------\
            # https://github.com/vllm-project/vllm/pull/34385
            # MTP models call compute_logits via shared_head.head (a
            # ParallelLMHead inside each MTP layer), not self.model.lm_head.
            # If the checkpoint omits a copy of the lm_head weights at the
            # MTP layer path, shared_head.head stays uninitialised and
            # produces NaN logits. Always share it explicitly.
            inner = getattr(self.model, "model", None)
            layers = getattr(inner, "layers", None) if inner else None
            if layers is not None:
                items = layers.values() if isinstance(layers, nn.ModuleDict) else layers
                for layer in items:
                    sh = getattr(layer, "shared_head", None)
                    if sh is not None and hasattr(sh, "head"):
                        del sh.head
                        sh.head = target_language_model.lm_head
                        logger.info(
                            "Shared target model lm_head with MTP shared_head.head."
                        )
            # \---------------------------- metax modified ----------------------------/

    @torch.inference_mode()
    def dummy_run(
        self,
        num_tokens: int,
        use_cudagraphs: bool = True,
        is_graph_capturing: bool = False,
    ) -> None:
        # Determine if CUDA graphs should be used for this run.
        cudagraphs_enabled = use_cudagraphs and self.use_cuda_graph

        # FIXME: when using tree-based specdec, adjust number of forward-passes
        # according to the depth of the tree.
        for fwd_idx in range(
            self.num_speculative_tokens if not is_graph_capturing else 1
        ):
            if fwd_idx <= 1:
                # /------------------------  Metax Modification -------------------------\
                if (
                    cudagraphs_enabled
                    and num_tokens <= self.compilation_config.max_cudagraph_capture_size
                ):
                    num_tokens_graph_padded = self.vllm_config.pad_for_cudagraph(
                        num_tokens
                    )
                else:
                    num_tokens_graph_padded = num_tokens

                num_tokens_dp_padded, num_tokens_across_dp = self._pad_batch_across_dp(
                    num_tokens_unpadded=num_tokens,
                    num_tokens_padded=num_tokens_graph_padded,
                )
                num_input_tokens = num_tokens_dp_padded
                if num_tokens_across_dp is not None:
                    num_tokens_across_dp[self.dp_rank] = num_input_tokens

                if (
                    cudagraphs_enabled
                    and num_input_tokens
                    <= self.compilation_config.max_cudagraph_capture_size
                ):
                    cudagraph_runtime_mode = CUDAGraphMode.PIECEWISE
                else:
                    cudagraph_runtime_mode = CUDAGraphMode.NONE
                # \------------------------- Metax Modification -------------------------/

            with set_forward_context(
                None,
                self.vllm_config,
                num_tokens=num_input_tokens,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
            ):
                if self.supports_mm_inputs:
                    input_ids = None
                    inputs_embeds = self.inputs_embeds[:num_input_tokens]
                else:
                    input_ids = self.input_ids[:num_input_tokens]
                    inputs_embeds = None

                self.model(
                    input_ids=input_ids,
                    positions=self._get_positions(num_input_tokens),
                    hidden_states=self.hidden_states[:num_input_tokens],
                    inputs_embeds=inputs_embeds,
                )

    def propose(
        self,
        # [num_tokens]
        target_token_ids: torch.Tensor,
        # [num_tokens] or [3, num_tokens] when M-RoPE is enabled
        target_positions: torch.Tensor,
        # [num_tokens, hidden_size]
        target_hidden_states: torch.Tensor,
        # [batch_size]
        next_token_ids: torch.Tensor,
        last_token_indices: torch.Tensor | None,
        common_attn_metadata: CommonAttentionMetadata,
        sampling_metadata: SamplingMetadata,
        mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        num_rejected_tokens_gpu: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_tokens = target_token_ids.shape[0]
        batch_size = next_token_ids.shape[0]

        if last_token_indices is None:
            last_token_indices = common_attn_metadata.query_start_loc[1:] - 1

        if self.method == "eagle3":
            assert isinstance(self.model, Eagle3LlamaForCausalLM)
            target_hidden_states = self.model.combine_hidden_states(
                target_hidden_states
            )
            assert target_hidden_states.shape[-1] == self.hidden_size
        # Shift the input ids by one token.
        # E.g., [a1, b1, b2, c1, c2, c3] -> [b1, b2, c1, c2, c3, c3]
        self.input_ids[: num_tokens - 1] = target_token_ids[1:]
        # Replace the last token with the next token.
        # E.g., [b1, b2, c1, c2, c3, c3] -> [a2, b2, b3, c2, c3, c4]
        self.input_ids[last_token_indices] = next_token_ids

        assert self.runner is not None

        if self.attn_metadata_builder is None:
            attn_metadata_builder = self._get_attention_metadata_builder()
        else:
            attn_metadata_builder = self.attn_metadata_builder

        attn_metadata = attn_metadata_builder.build_for_drafting(
            common_attn_metadata=common_attn_metadata, draft_index=0
        )
        # FIXME: support hybrid kv for draft model (remove separate indexer)
        if self.draft_indexer_metadata_builder:
            draft_indexer_metadata = (
                self.draft_indexer_metadata_builder.build_for_drafting(
                    common_attn_metadata=common_attn_metadata,
                    draft_index=0,
                )
            )
        else:
            draft_indexer_metadata = None
        # At this moment, we assume all eagle layers belong to the same KV
        # cache group, thus using the same attention metadata.
        per_layer_attn_metadata = {}
        for layer_name in self.attn_layer_names:
            per_layer_attn_metadata[layer_name] = attn_metadata

        for layer_name in self.indexer_layer_names:
            assert draft_indexer_metadata is not None
            per_layer_attn_metadata[layer_name] = draft_indexer_metadata

        # /------------------------  Metax Modification -------------------------\
        if (
            self.use_cuda_graph
            and num_tokens <= self.compilation_config.max_cudagraph_capture_size
        ):
            num_tokens_graph_padded = self.vllm_config.pad_for_cudagraph(num_tokens)
        else:
            num_tokens_graph_padded = num_tokens

        num_tokens_dp_padded, num_tokens_across_dp = self._pad_batch_across_dp(
            num_tokens_unpadded=num_tokens, num_tokens_padded=num_tokens_graph_padded
        )
        num_input_tokens = num_tokens_dp_padded
        if num_tokens_across_dp is not None:
            num_tokens_across_dp[self.dp_rank] = num_input_tokens

        if (
            self.use_cuda_graph
            and num_input_tokens <= self.compilation_config.max_cudagraph_capture_size
        ):
            cudagraph_runtime_mode = CUDAGraphMode.PIECEWISE
        else:
            cudagraph_runtime_mode = CUDAGraphMode.NONE
        # \------------------------- Metax Modification -------------------------/

        # copy inputs to buffer for cudagraph
        self._set_positions(num_tokens, target_positions)
        self.hidden_states[:num_tokens] = target_hidden_states

        if self.supports_mm_inputs:
            mm_embeds, is_mm_embed = mm_embed_inputs or (None, None)

            self.inputs_embeds[:num_tokens] = self.model.embed_input_ids(
                self.input_ids[:num_tokens],
                multimodal_embeddings=mm_embeds,
                is_multimodal=is_mm_embed,
            )

            input_ids = None
            inputs_embeds = self.inputs_embeds[:num_input_tokens]
        else:
            input_ids = self.input_ids[:num_input_tokens]
            inputs_embeds = None

        with set_forward_context(
            per_layer_attn_metadata,
            self.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
        ):
            ret_hidden_states = self.model(
                input_ids=input_ids,
                positions=self._get_positions(num_input_tokens),
                hidden_states=self.hidden_states[:num_input_tokens],
                inputs_embeds=inputs_embeds,
            )
            if self.method == "mtp":
                last_hidden_states = ret_hidden_states
                hidden_states = last_hidden_states
            else:
                last_hidden_states, hidden_states = ret_hidden_states
        sample_hidden_states = last_hidden_states[last_token_indices]
        logits = self.model.compute_logits(sample_hidden_states)

        # Early exit if there is only one draft token to be generated.
        if self.num_speculative_tokens == 1:
            draft_token_ids = logits.argmax(dim=-1)
            return draft_token_ids.view(-1, 1)

        if self.uses_mrope:
            positions = target_positions[:, last_token_indices]
        else:
            positions = target_positions[last_token_indices]
        if self.method in (
            "deepseek_mtp",
            "ernie_mtp",
            "longcat_flash_mtp",
            "pangu_ultra_moe_mtp",
        ):
            hidden_states = self.hidden_states[last_token_indices]
        else:
            hidden_states = hidden_states[last_token_indices]

        if isinstance(attn_metadata, TreeAttentionMetadata):
            # Draft using tree attention.
            draft_token_ids_list = self.propose_tree(
                batch_size=batch_size,
                logits=logits,
                positions=positions,
                hidden_states=hidden_states,
                common_attn_metadata=common_attn_metadata,
            )
            # [batch_size, num_tree_tokens]
            return torch.cat(draft_token_ids_list, dim=1)

        draft_token_ids = logits.argmax(dim=-1)

        if self.allowed_attn_types is not None and not isinstance(
            attn_metadata, self.allowed_attn_types
        ):
            raise ValueError(
                f"Unsupported attention metadata type for speculative "
                "decoding with num_speculative_tokens > 1: "
                f"{type(attn_metadata)}. Supported types are: "
                f"{self.allowed_attn_types}"
            )

        # Generate the remaining draft tokens.
        draft_token_ids_list = [draft_token_ids]
        # /------------------------  Metax Modification -------------------------\
        if (
            self.use_cuda_graph
            and batch_size <= self.compilation_config.max_cudagraph_capture_size
        ):
            batch_size_graph_padded = self.vllm_config.pad_for_cudagraph(batch_size)
        else:
            batch_size_graph_padded = batch_size

        batch_size_dp_padded, batch_size_across_dp = self._pad_batch_across_dp(
            num_tokens_unpadded=batch_size, num_tokens_padded=batch_size_graph_padded
        )
        input_batch_size = batch_size_dp_padded

        if batch_size_across_dp is not None:
            batch_size_across_dp[self.dp_rank] = input_batch_size

        if (
            self.use_cuda_graph
            and input_batch_size <= self.compilation_config.max_cudagraph_capture_size
        ):
            cudagraph_runtime_mode = CUDAGraphMode.PIECEWISE
        else:
            cudagraph_runtime_mode = CUDAGraphMode.NONE
        # \------------------------- Metax Modification -------------------------/

        common_attn_metadata.num_actual_tokens = batch_size
        common_attn_metadata.max_query_len = 1
        common_attn_metadata.query_start_loc = self.arange[: batch_size + 1]
        common_attn_metadata.query_start_loc_cpu = torch.from_numpy(
            self.token_arange_np[: batch_size + 1]
        ).clone()

        # In padded drafter batch, we need to adjust the sequence lengths
        # to remove the "padding" (i.e. rejected tokens).
        # Only apply this adjustment when we have rejected tokens
        # (i.e., not the first proposal).
        if self.num_speculative_tokens > 1 and num_rejected_tokens_gpu is not None:
            common_attn_metadata.seq_lens -= num_rejected_tokens_gpu
            # Invalidate the CPU-side shadows to avoid H<>D sync.
            common_attn_metadata._seq_lens_cpu = None
            common_attn_metadata._num_computed_tokens_cpu = None

        for token_index in range(self.num_speculative_tokens - 1):
            # Update the inputs.
            # cast to int32 is crucial when eagle model is compiled.
            # tensor.argmax() returns int64 by default.
            input_ids = draft_token_ids_list[-1].int()
            if self.uses_mrope:
                positions += 1
                # NOTE(woosuk): We should handle the case where the draft model
                # generates tokens beyond the max model length.
                # Since it is complex to remove such requests from the batch,
                # we keep them in the batch but adjust the position ids
                # and slot mappings to avoid the
                # out-of-range access during the model execution.
                # The draft tokens generated with this adjustment
                # should be ignored.
                exceeds_max_model_len = positions[0] >= self.max_model_len
                # Mask out the position ids that exceed the max model length.
                # Otherwise, we may get out-of-range error in RoPE.
                clamped_positions = torch.where(
                    exceeds_max_model_len.unsqueeze(0),
                    torch.zeros_like(positions),
                    positions,
                )
            else:
                positions += 1
                exceeds_max_model_len = positions >= self.max_model_len
                clamped_positions = torch.where(exceeds_max_model_len, 0, positions)
            # For data integrity when async scheduling, we shouldn't use in place
            # operations in case they are modified in next step's `prepare_input`
            # of main model.
            # Increment the sequence lengths.
            common_attn_metadata.seq_lens += 1
            # For the requests that exceed the max model length, we set the
            # sequence length to 1 to minimize their overheads in attention.
            common_attn_metadata.seq_lens.masked_fill_(exceeds_max_model_len, 1)

            # Also update the CPU-side shadow; NOTE: this is hacky and should be
            # removed in when common_attn_metadata.seq_lens_cpu is deprecated.
            if common_attn_metadata._seq_lens_cpu is not None:
                common_attn_metadata._seq_lens_cpu += 1
            if common_attn_metadata._num_computed_tokens_cpu is not None:
                common_attn_metadata._num_computed_tokens_cpu += 1

            # Compute the slot mapping.
            block_size = attn_metadata_builder.kv_cache_spec.block_size
            if self.uses_mrope:
                # all dimensions of positions are the same
                block_numbers = clamped_positions[0] // block_size
            else:
                block_numbers = clamped_positions // block_size
            block_ids = common_attn_metadata.block_table_tensor.gather(
                dim=1, index=block_numbers.view(-1, 1)
            )
            block_ids = block_ids.view(-1)
            if self.uses_mrope:
                common_attn_metadata.slot_mapping = (
                    block_ids * block_size + clamped_positions[0] % block_size
                )
            else:
                common_attn_metadata.slot_mapping = (
                    block_ids * block_size + clamped_positions % block_size
                )
            # Mask out the slot mappings that exceed the max model length.
            # Otherwise, the KV cache will be inadvertently updated with the
            # padding tokens.
            common_attn_metadata.slot_mapping.masked_fill_(
                exceeds_max_model_len, PADDING_SLOT_ID
            )

            # Rebuild attention metadata
            attn_metadata = attn_metadata_builder.build_for_drafting(  # type: ignore
                common_attn_metadata=common_attn_metadata, draft_index=token_index + 1
            )
            for layer_name in self.attn_layer_names:
                per_layer_attn_metadata[layer_name] = attn_metadata

            # copy inputs to buffer for cudagraph
            self.input_ids[:batch_size] = input_ids
            self._set_positions(batch_size, clamped_positions)
            self.hidden_states[:batch_size] = hidden_states
            if self.supports_mm_inputs:
                self.inputs_embeds[:batch_size] = self.model.embed_input_ids(input_ids)

                input_ids = None
                inputs_embeds = self.inputs_embeds[:input_batch_size]
            else:
                input_ids = self.input_ids[:input_batch_size]
                inputs_embeds = None

            # Run the model.
            with set_forward_context(
                per_layer_attn_metadata,
                self.vllm_config,
                num_tokens=input_batch_size,
                num_tokens_across_dp=batch_size_across_dp,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
            ):
                ret_hidden_states = self.model(
                    input_ids=input_ids,
                    positions=self._get_positions(input_batch_size),
                    hidden_states=self.hidden_states[:input_batch_size],
                    inputs_embeds=inputs_embeds,
                )
                if self.method == "mtp":
                    last_hidden_states = ret_hidden_states
                    hidden_states = ret_hidden_states
                else:
                    last_hidden_states, hidden_states = ret_hidden_states
            hidden_states = hidden_states[:batch_size]
            logits = self.model.compute_logits(last_hidden_states[:batch_size])
            draft_token_ids = logits.argmax(dim=-1)
            draft_token_ids_list.append(draft_token_ids)

        # [batch_size, num_speculative_tokens]
        draft_token_ids = torch.stack(draft_token_ids_list, dim=1)
        return draft_token_ids


EagleProposer.propose = MacaEagleProposer.propose
EagleProposer.dummy_run = MacaEagleProposer.dummy_run
EagleProposer.load_model = MacaEagleProposer.load_model
