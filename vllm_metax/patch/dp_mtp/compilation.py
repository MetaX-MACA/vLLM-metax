# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

# ------------------------------------------------------------------------
# Note: This file is a patch to fix Eagle IndexError, pr #29102
# ------------------------------------------------------------------------
from vllm.utils.math_utils import round_up

from vllm.config.compilation import CompilationConfig


class MacaCompilationConfig(CompilationConfig):
    def adjust_cudagraph_sizes_for_spec_decode(
        self, uniform_decode_query_len: int, tensor_parallel_size: int
    ):
        multiple_of = uniform_decode_query_len
        if tensor_parallel_size > 1 and self.pass_config.enable_sequence_parallelism:
            multiple_of = max(uniform_decode_query_len, tensor_parallel_size)
            if (
                multiple_of % uniform_decode_query_len != 0
                or multiple_of % tensor_parallel_size != 0
            ):
                raise ValueError(
                    f"Can't determine cudagraph shapes that are both a "
                    f"multiple of {uniform_decode_query_len} "
                    f"(num_speculative_tokens + 1) required by spec-decode "
                    f"and {tensor_parallel_size} (tensor_parallel_size) "
                    f"required by sequence parallelism please adjust "
                    f"num_speculative_tokens or disable sequence parallelism"
                )

        if not self.cudagraph_capture_sizes or multiple_of <= 1:
            return

        assert self.max_cudagraph_capture_size is not None
        rounded_sizes = sorted(
            set(
                round_up(size, multiple_of)
                for size in self.cudagraph_capture_sizes
                if round_up(size, multiple_of) <= self.max_cudagraph_capture_size
            )
        )

        # /------------------------  Metax Modification -------------------------\
        if len(rounded_sizes) == 0 and multiple_of <= self.max_cudagraph_capture_size:
            # if one valid but would be round_down use that
            rounded_sizes = [multiple_of]

        if len(rounded_sizes) == 0:
            raise ValueError(
                f"No valid cudagraph sizes after rounding to multiple of {multiple_of} "
                f"(num_speculative_tokens + 1 or tp if sequence parallelism is enabled)"
                f" please adjust num_speculative_tokens ({uniform_decode_query_len - 1}"
                f") or max_cudagraph_capture_size ({self.max_cudagraph_capture_size})"
                f" or cudagraph_capture_sizes ({self.cudagraph_capture_sizes})"
            )
        # \------------------------- Metax Modification -------------------------/

        self.max_cudagraph_capture_size = rounded_sizes[-1]
        self.cudagraph_capture_sizes = rounded_sizes

        # Recompute after adjusting the cudagraph sizes
        self.compute_bs_to_padded_graph_size()


CompilationConfig.adjust_cudagraph_sizes_for_spec_decode = (
    MacaCompilationConfig.adjust_cudagraph_sizes_for_spec_decode
)
