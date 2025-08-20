import vllm
from vllm_metax.patch.hook_registry import register_patch
from vllm.logger import init_logger

logger = init_logger(__name__)

import torch


import tokenizers
from packaging import version
from tokenizers import Tokenizer
from tokenizers.decoders import DecodeStream
from transformers import PreTrainedTokenizerFast
from typing import Optional
from vllm.v1.engine.detokenizer import (IncrementalDetokenizer, 
                                        SlowIncrementalDetokenizer,
                                        BaseIncrementalDetokenizer)

from vllm.transformers_utils.detokenizer_utils import (
    AnyTokenizer, convert_prompt_ids_to_tokens, detokenize_incrementally)
from vllm.v1.engine import EngineCoreRequest

# Only tokenizers >= 0.21.1 supports DecodeStream used for
# FastIncrementalDetokenizer.
USE_FAST_DETOKENIZER = version.parse(
    tokenizers.__version__) >= version.parse("0.21.1")

# Error string from https://github.com/huggingface/tokenizers/blob/909fdde2a4ffedd9295206f705eb612be2a91b12/tokenizers/src/tokenizer/mod.rs#L1042
INVALID_PREFIX_ERR_MSG = "Invalid prefix encountered"

class MetaxIncrementalDetokenizer(IncrementalDetokenizer):
    @classmethod
    def from_new_request(
        cls,
        tokenizer: Optional[AnyTokenizer],
        request: EngineCoreRequest,
    ) -> "IncrementalDetokenizer":

        if tokenizer is None:
            # No tokenizer => skipping detokenization.
            return IncrementalDetokenizer()

        if USE_FAST_DETOKENIZER and isinstance(tokenizer,
                                               PreTrainedTokenizerFast):
            # Fast tokenizer => use tokenizers library DecodeStream.
            return FastIncrementalDetokenizer(tokenizer, request)

        # Fall back to slow python-based incremental detokenization.
        return SlowIncrementalDetokenizer(tokenizer, request)
    
class FastIncrementalDetokenizer(BaseIncrementalDetokenizer):

    def __init__(self, tokenizer: PreTrainedTokenizerFast,
                 request: EngineCoreRequest):
        super().__init__(request)

        sampling_params = request.sampling_params

        self.request_id = request.request_id
        self.skip_special_tokens = sampling_params.skip_special_tokens
        self.stream = DecodeStream(
            skip_special_tokens=self.skip_special_tokens)

        self.tokenizer: Tokenizer = tokenizer._tokenizer

        # Find a safe place to start.
        prompt_suffix = request.prompt_token_ids
        prompt_len = len(prompt_suffix)
        if prompt_len > 4:
            for i in range(4, min(prompt_len + 1, 24)):
                suffix = request.prompt_token_ids[-i:]
                if '�' not in self.tokenizer.decode(suffix):
                    prompt_suffix = suffix
                    break

        # Prime the stream.
        for tid in prompt_suffix:
            self._protected_step(tid)

        self.spaces_between_special_tokens = (
            sampling_params.skip_special_tokens
            or sampling_params.spaces_between_special_tokens)

        if not self.spaces_between_special_tokens:
            # Store dict of added token ids so that we can suppress
            # the spaces between them.
            if (added_token_ids := getattr(self.tokenizer, "added_token_ids",
                                           None)) is None:
                self.tokenizer.added_token_ids = added_token_ids = {
                    tid: tok.content
                    for tid, tok in
                    self.tokenizer.get_added_tokens_decoder().items()
                }

            if added_token_ids:
                self.last_special = False
                self.added_token_ids = added_token_ids
            else:
                # No added tokens.
                self.spaces_between_special_tokens = True

    def decode_next(self, next_token_id: int) -> str:
        token = self._protected_step(next_token_id)

        if not self.spaces_between_special_tokens:
            special_token = self.added_token_ids.get(next_token_id)
            is_special = special_token is not None
            if is_special and self.last_special:
                # Return raw token string without any prefixed spaces.
                token = special_token
            self.last_special = is_special

        return token or ""

    def _protected_step(self, next_token_id: int) -> Optional[str]:
        try:
            token = self.stream.step(self.tokenizer, next_token_id)
        except Exception as e:
            if str(e) != INVALID_PREFIX_ERR_MSG:
                raise e
            # Recover from edge case where tokenizer can produce non-monotonic,
            # invalid UTF-8 output, which breaks the internal state of
            # tokenizers' DecodeStream.
            # See https://github.com/vllm-project/vllm/issues/17448.
            logger.warning(
                "Encountered invalid prefix detokenization error"
                " for request %s, resetting decode stream.", self.request_id)
            self.stream = DecodeStream(self.skip_special_tokens)
            token = self.stream.step(self.tokenizer, next_token_id)
        return token

    
vllm.v1.engine.detokenizer.IncrementalDetokenizer = MetaxIncrementalDetokenizer
register_patch("vllm.v1.engine.detokenizer", "IncrementalDetokenizer", MetaxIncrementalDetokenizer)