# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

# -----------------------------------------------
# Note: Fix Triton kernel compilation errors in speculative Eagle decoding.
#
# Affected versions: v0.21.0
# Remove at: after triton3.6+metax is released.
# -----------------------------------------------

# -----------------------------------------------
# Note: Fix "ValueError: too many values to unpack (expected 2)" in
# SpecDecodeBaseProposer.propose() when the draft model forward
# returns more than 2 hidden states. This can happen when the
# model is compiled/wrapped (e.g. torch.compile) and the wrapper
# adds extra outputs to the forward return tuple.
#
# The fix makes the tuple unpacking defensive: when
# model_returns_tuple() is True and ret_hidden_states has >2
# elements, only the first two are taken instead of failing.
#
# Affected versions: v0.24.0, v0.25.0-dev
# Remove at: after root cause in upstream model compilation/wrapping
#            is identified and fixed.
# -----------------------------------------------

from vllm.triton_utils import tl, triton

import numpy as np
import torch
from vllm.v1.spec_decode.utils import (
    eagle_prepare_next_token_padded_kernel,
    next_power_of_2,
)
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

from vllm.v1.spec_decode.eagle import SpecDecodeBaseProposer
from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphWrapper
from vllm.forward_context import set_forward_context
from vllm.model_executor.models.deepseek_eagle3 import Eagle3DeepseekV2ForCausalLM
from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM
from vllm.model_executor.models.qwen3_dflash import DFlashQwen3ForCausalLM
from vllm.model_executor.models.qwen3_eagle3 import Eagle3Qwen3ForCausalLM

@triton.jit
def eagle_prepare_next_token_padded_kernel(
    sampled_token_ids_ptr,  # [num_reqs, num_sampled_tokens_per_req]
    discard_request_mask_ptr,  # [num_reqs]
    backup_next_token_ids_ptr,  # [num_reqs]
    next_token_ids_ptr,  # [num_reqs] (output)
    valid_sampled_tokens_count_ptr,  # [num_reqs] (output)
    vocab_size,  # tl.int32
    num_sampled_tokens_per_req,  # tl.int32 (num_spec_tokens + 1)
    num_reqs,  # tl.int32
    stride_sampled_token_ids,  # tl.int32 (stride for dim 0)
    BLOCK_SIZE_TOKENS: tl.constexpr,  # Power-of-2 >= num_sampled_tokens_per_req
):
    """
    Fused kernel for Eagle prepare_next_token_ids_padded. This kernel computes the
    number of valid (1 + accepted) tokens for each request, and the corresponding
    "next" token id to sample from during speculative decoding. This is the
    "last accepted token" from the sampled tokens, or the backup token if no
    tokens were accepted or if the request is marked as discarded.
    """
    req_idx = tl.program_id(axis=0)
    if req_idx >= num_reqs:
        return

    # Check if this request is discarded.
    is_discarded = tl.load(discard_request_mask_ptr + req_idx)

    if is_discarded:
        backup_token = tl.load(backup_next_token_ids_ptr + req_idx)
        # /------------------------  Metax Modification -------------------------\
        valid_count = tl.full((), 0, dtype=tl.int32)
        # \---------------------------------------------------------------------/
        tl.store(next_token_ids_ptr + req_idx, backup_token)
        tl.store(valid_sampled_tokens_count_ptr + req_idx, valid_count)
    else:
        # Count the number of valid tokens among the sampled tokens.
        token_offs = tl.arange(0, BLOCK_SIZE_TOKENS)
        token_mask = token_offs < num_sampled_tokens_per_req

        row_ptr = sampled_token_ids_ptr + req_idx * stride_sampled_token_ids
        token_ids = tl.load(row_ptr + token_offs, mask=token_mask, other=-1)

        # Rejected tokens are -1, valid tokens are in [0, vocab_size)
        is_valid_mask = (token_ids != -1) & (token_ids < vocab_size) & token_mask

        # /------------------------  Metax Modification -------------------------\
        valid_count = tl.sum(is_valid_mask.to(tl.int32))
        # \------------------------- Metax Modification -------------------------/
        if valid_count > 0:
            # Guaranteed to be well-defined since
            # valid_count > 0 implies is_valid_mask is not empty
            last_valid_index = tl.max(tl.where(is_valid_mask, token_offs, -1))

            # Select the token at that index, using a sum trick since
            # we don't want to load again to access token_ids[last_valid_index].
            last_valid_token = tl.sum(
                tl.where(token_offs == last_valid_index, token_ids, 0)
            )
            tl.store(next_token_ids_ptr + req_idx, last_valid_token)
        else:
            # No valid tokens found, use backup token
            backup_token = tl.load(backup_next_token_ids_ptr + req_idx)
            tl.store(next_token_ids_ptr + req_idx, backup_token)

        tl.store(valid_sampled_tokens_count_ptr + req_idx, valid_count)


class MacaEagleProposer(SpecDecodeBaseProposer):
    def prepare_next_token_ids_padded(
        self,
        sampled_token_ids: torch.Tensor,
        requests: dict[str, CachedRequestState],
        gpu_input_batch: InputBatch,
        discard_request_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        This function is used to prepare the inputs for speculative decoding.
        It calculates the next token ids and the number of valid sampled tokens
        for each request, considering the "discarded" requests whose next token
        is not sampled and comes from `request.get_token_id()` instead. This is denoted
        the "backup" token id. It also counts rejected tokens via `sampled_token_ids`.
        """
        # Precompute backup token IDs for discarded requests.
        num_reqs = gpu_input_batch.num_reqs
        for i in range(num_reqs):
            self.backup_next_token_ids.np[i] = requests[
                gpu_input_batch.req_ids[i]
            ].get_token_id(gpu_input_batch.num_tokens_no_spec[i] - 1)
        self.backup_next_token_ids.copy_to_gpu(num_reqs)
        backup_tokens_gpu = self.backup_next_token_ids.gpu

        batch_size, num_tokens = sampled_token_ids.shape
        device = sampled_token_ids.device

        assert discard_request_mask.dtype == torch.bool
        assert backup_tokens_gpu.dtype == torch.int32

        next_token_ids = torch.empty(batch_size, dtype=torch.int32, device=device)
        valid_sampled_tokens_count = next_token_ids.new_empty(batch_size)

        # Kernel grid: one program per request (row)
        grid = (batch_size,)

        # Find the next power of 2 for block sizes
        BLOCK_SIZE_TOKENS = next_power_of_2(num_tokens)
        eagle_prepare_next_token_padded_kernel[grid](
            sampled_token_ids,
            discard_request_mask,
            backup_tokens_gpu,
            next_token_ids,
            valid_sampled_tokens_count,
            gpu_input_batch.vocab_size,
            num_tokens,
            batch_size,
            sampled_token_ids.stride(0),
            BLOCK_SIZE_TOKENS=BLOCK_SIZE_TOKENS,
        )

        return next_token_ids, valid_sampled_tokens_count

    """Subclass with defensive ret_hidden_states unpacking."""
    def propose(
        self,
        num_speculative_tokens,
        # [num_tokens]
        target_token_ids: torch.Tensor,
        # [num_tokens] or [3, num_tokens] when M-RoPE is enabled
        target_positions: torch.Tensor,
        # [num_tokens, hidden_size]
        target_hidden_states: torch.Tensor,
        # [batch_size]
        next_token_ids: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        common_attn_metadata,
        sampling_metadata,
        mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        num_rejected_tokens_gpu: torch.Tensor | None = None,
        slot_mappings: dict[str, torch.Tensor]
        | list[dict[str, torch.Tensor]]
        | None = None,
    ) -> torch.Tensor:
        self.num_speculative_tokens = num_speculative_tokens
        self._last_draft_probs = None
        batch_size = common_attn_metadata.batch_size()

        if self.method in ("eagle3", "dflash"):
            model = self.model
            if isinstance(model, BreakableCUDAGraphWrapper):
                model = model.unwrap()
            assert isinstance(
                model,
                (
                    Eagle3LlamaForCausalLM,
                    Eagle3DeepseekV2ForCausalLM,
                    DFlashQwen3ForCausalLM,
                    Eagle3Qwen3ForCausalLM,
                ),
            )
            target_hidden_states = self.model.combine_hidden_states(
                target_hidden_states
            )
            assert target_hidden_states.shape[-1] == self.hidden_size

        num_tokens, token_indices_to_sample, common_attn_metadata = (
            self.set_inputs_first_pass(
                target_token_ids=target_token_ids,
                next_token_ids=next_token_ids,
                target_positions=target_positions,
                target_hidden_states=target_hidden_states,
                token_indices_to_sample=token_indices_to_sample,
                cad=common_attn_metadata,
                num_rejected_tokens_gpu=num_rejected_tokens_gpu,
            )
        )

        per_group_attn_metadata, per_layer_attn_metadata = (
            self.build_per_group_and_layer_attn_metadata(common_attn_metadata)
        )

        cudagraph_runtime_mode, num_input_tokens, num_tokens_across_dp = (
            self._determine_batch_execution_and_padding(num_tokens)
        )

        model_kwargs, slot_mapping_size = self.build_model_inputs_first_pass(
            num_tokens, num_input_tokens, mm_embed_inputs
        )
        # Step 0 of index_share_for_mtp_iteration: let the MTP layer
        # compute its own indices (skip_topk=False) so subsequent steps
        # can reuse them.
        if self._share_mtp_indices and hasattr(self.model.model, "set_skip_topk"):
            self.model.model.set_skip_topk(False)

        with set_forward_context(
            per_layer_attn_metadata,
            self.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            slot_mapping=self._get_slot_mapping(
                slot_mapping_size, common_attn_metadata.slot_mapping
            ),
        ):
            ret_hidden_states = self.model(**model_kwargs)
            if not self.model_returns_tuple():
                last_hidden_states = ret_hidden_states
                hidden_states = last_hidden_states
            else:
                # /-------------------- Metax Modification ---------------------\
                # Defensive unpacking: when the draft model forward returns
                # >2 values (e.g. due to compilation wrappers), take only
                # the first two instead of failing with ValueError.
                if isinstance(ret_hidden_states, (tuple, list)):
                    last_hidden_states = ret_hidden_states[0]
                    hidden_states = ret_hidden_states[1]
                else:
                    last_hidden_states = ret_hidden_states
                    hidden_states = last_hidden_states
                # \-------------------------------------------------------------/

        # After step 0: switch to reuse mode so steps 1+ skip the indexer
        # and read the indices that step 0 just wrote into the shared buffer.
        if self._share_mtp_indices and hasattr(self.model.model, "set_skip_topk"):
            self.model.model.set_skip_topk(True)

        sample_hidden_states = last_hidden_states[token_indices_to_sample]

        # No draft tokens requested (e.g. Dynamic SD decided K=0).
        # The prefill forward pass above already ran to keep the drafter
        # KV cache in sync, so just return an empty tensor.
        if self.num_speculative_tokens == 0:
            return torch.empty(
                batch_size,
                0,
                device=sample_hidden_states.device,
                dtype=torch.int64,
            )

        # Early exit if there is only one draft token to be generated.
        if self.num_speculative_tokens == 1 or self.parallel_drafting:
            draft_token_ids, draft_probs = self._sample_draft_tokens(
                sample_hidden_states, sampling_metadata
            )
            if draft_probs is not None:
                self._last_draft_probs = draft_probs.view(
                    -1, self.num_speculative_tokens, draft_probs.shape[-1]
                ).contiguous()
            return draft_token_ids.view(-1, self.num_speculative_tokens)

        if self.uses_mrope:
            positions = self.mrope_positions[:, token_indices_to_sample]
        else:
            positions = self.positions[token_indices_to_sample]
        hidden_states = hidden_states[token_indices_to_sample]

        if self.constant_draft_positions:
            # Write the sampling positions into the front of the
            # positions buffer so that subsequent loop iterations
            # (which read via _get_positions) use the correct values.
            self.positions[:batch_size] = positions

        draft_token_ids, draft_probs = self._sample_draft_tokens(
            sample_hidden_states, sampling_metadata
        )
        draft_probs_list = None if draft_probs is None else [draft_probs]

        if self.allowed_attn_types is not None:
            for group_md in per_group_attn_metadata:
                if not isinstance(group_md, self.allowed_attn_types):
                    raise ValueError(
                        f"Unsupported attention metadata type for speculative "
                        "decoding with num_speculative_tokens > 1: "
                        f"{type(group_md)}. Supported types are: "
                        f"{self.allowed_attn_types}"
                    )

        # Generate the remaining draft tokens.
        draft_token_ids_list = [draft_token_ids]

        cudagraph_runtime_mode, input_batch_size, batch_size_across_dp = (
            self._determine_batch_execution_and_padding(batch_size)
        )

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

        block_size = self.block_size
        assert block_size > 0, "block_size has not been initialized."
        for token_index in range(self.num_speculative_tokens - 1):
            # Update the inputs.
            # cast to int32 is crucial when eagle model is compiled.
            # tensor.argmax() returns int64 by default.
            input_ids = draft_token_ids_list[-1].int()

            if not self.constant_draft_positions:
                positions = self._update_positions_dependent_metadata(
                    positions,
                    common_attn_metadata,
                    batch_size,
                    input_batch_size,
                    block_size,
                )

            # Rebuild attention metadata. When draft positions are constant
            # (e.g. Gemma4 MTP), common_attn_metadata is invariant across
            # loop iterations so we build once and reuse.
            if not self.constant_draft_positions or token_index == 0:
                _, per_layer_attn_metadata = (
                    self.build_per_group_and_layer_attn_metadata(
                        common_attn_metadata, draft_index=token_index + 1
                    )
                )

            # copy inputs to buffer for cudagraph
            self.input_ids[:batch_size] = input_ids
            self.hidden_states[:batch_size] = hidden_states
            if self.supports_mm_inputs:
                self.inputs_embeds[:batch_size] = self.model.embed_input_ids(input_ids)

                input_ids = None
                inputs_embeds = self.inputs_embeds[:input_batch_size]
            else:
                input_ids = self.input_ids[:input_batch_size]
                inputs_embeds = None

            # Run the model.
            model_kwargs = {
                "input_ids": input_ids,
                "positions": self._get_positions(input_batch_size),
                "inputs_embeds": inputs_embeds,
            }
            if self.pass_hidden_states_to_model:
                model_kwargs["hidden_states"] = self.hidden_states[:input_batch_size]

            with set_forward_context(
                per_layer_attn_metadata,
                self.vllm_config,
                num_tokens=input_batch_size,
                num_tokens_across_dp=batch_size_across_dp,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                slot_mapping=self._get_slot_mapping(input_batch_size),
            ):
                ret_hidden_states = self.model(**model_kwargs)
                if not self.model_returns_tuple():
                    last_hidden_states = ret_hidden_states
                    hidden_states = ret_hidden_states
                else:
                    # /-------------------- Metax Modification ---------------------\
                    if isinstance(ret_hidden_states, (tuple, list)):
                        last_hidden_states = ret_hidden_states[0]
                        hidden_states = ret_hidden_states[1]
                    else:
                        last_hidden_states = ret_hidden_states
                        hidden_states = ret_hidden_states
                    # \-------------------------------------------------------------/

            hidden_states = hidden_states[:batch_size]
            draft_token_ids, draft_probs = self._sample_draft_tokens(
                last_hidden_states[:batch_size], sampling_metadata
            )
            if draft_probs is not None:
                assert draft_probs_list is not None
                draft_probs_list.append(draft_probs)
            draft_token_ids_list.append(draft_token_ids)

        # [batch_size, num_speculative_tokens]
        draft_token_ids = torch.stack(draft_token_ids_list, dim=1)
        if draft_probs_list is not None:
            self._last_draft_probs = torch.stack(draft_probs_list, dim=1).contiguous()
        return draft_token_ids

# Todo(hank): if this works well, remove the original
# llm_base_proposer.prepare_next_token_ids_padded to avoid confusion.
from vllm.v1.spec_decode import llm_base_proposer

llm_base_proposer.eagle_prepare_next_token_padded_kernel = (
    eagle_prepare_next_token_padded_kernel
)

SpecDecodeBaseProposer.prepare_next_token_ids_padded = (
    MacaEagleProposer.prepare_next_token_ids_padded
)

SpecDecodeBaseProposer.propose = (
    MacaEagleProposer.propose
)