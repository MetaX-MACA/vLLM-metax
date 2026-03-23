# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
from vllm.v1.attention.backend import AttentionMetadataBuilder


class MacaAttentionMetadataBuilder(AttentionMetadataBuilder):
    def _init_reorder_batch_threshold(
        self,
        reorder_batch_threshold: int | None = 1,
        supports_spec_as_decode: bool = False,
        supports_dcp_with_varlen: bool = False,
    ) -> None:
        self.reorder_batch_threshold = reorder_batch_threshold
        speculative_config = self.vllm_config.speculative_config
        if self.reorder_batch_threshold is not None and supports_spec_as_decode:
            # If the backend supports spec-as-decode kernels, then we can set
            # the reorder_batch_threshold based on the number of speculative
            # tokens from the config.
            if (
                speculative_config is not None
                and speculative_config.num_speculative_tokens is not None
            ):
                self.reorder_batch_threshold = max(
                    self.reorder_batch_threshold,
                    1 + speculative_config.num_speculative_tokens,
                )
        # /------------------------  Metax Modification -------------------------\
        if (
            speculative_config is not None
            and speculative_config.num_speculative_tokens is not None
        ):
            if (
                self.vllm_config.parallel_config.decode_context_parallel_size > 1
                and not supports_dcp_with_varlen
                and self.reorder_batch_threshold is not None
            ):
                self.reorder_batch_threshold = min(
                    self.reorder_batch_threshold,
                    1 + speculative_config.num_speculative_tokens,
                )
            else:
                self.reorder_batch_threshold = (
                    1 + speculative_config.num_speculative_tokens
                )
        else:
            self.reorder_batch_threshold = 1
        # \------------------------- Metax Modification -------------------------/


AttentionMetadataBuilder._init_reorder_batch_threshold = (
    MacaAttentionMetadataBuilder._init_reorder_batch_threshold
)
