# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# --------------------------------------------------
# Monkey patch for vllm.config.vllm to ensure max_num_batched_tokens
# is included in cudagraph_capture_sizes even when it doesn't land
# on a stride-16 boundary.
# Fixes: https://github.com/vllm-project/vllm/pull/40734
# --------------------------------------------------
from vllm.config import VllmConfig, CUDAGraphMode


def _patched_set_cudagraph_sizes(self):
    if (
        self.model_config is not None
        and not self.model_config.enforce_eager
        and self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
    ):
        max_cudagraph_capture_size = self.compilation_config.max_cudagraph_capture_size

        if max_cudagraph_capture_size is None:
            decode_query_len = 1
            if (
                self.speculative_config
                and self.speculative_config.num_speculative_tokens
            ):
                decode_query_len += self.speculative_config.num_speculative_tokens

            max_cudagraph_capture_size = min(
                self.scheduler_config.max_num_seqs * decode_query_len * 2,
                512,
            )

        max_num_tokens = self.scheduler_config.max_num_batched_tokens
        max_cudagraph_capture_size = min(max_num_tokens, max_cudagraph_capture_size)

        assert max_cudagraph_capture_size >= 1, (
            "Maximum cudagraph size should be greater than or equal to 1 "
            "when using cuda graph."
        )

        if self.compilation_config.cudagraph_capture_sizes is not None:
            assert len(self.compilation_config.cudagraph_capture_sizes) > 0, (
                "cudagraph_capture_sizes should contain at least one element "
                "when using cuda graph."
            )

            dedup_sizes = list(set(self.compilation_config.cudagraph_capture_sizes))
            cudagraph_capture_sizes = [i for i in dedup_sizes if i <= max_num_tokens]
            cudagraph_capture_sizes.sort()

        else:
            if self.performance_mode == "interactivity":
                interactivity_max = min(max_cudagraph_capture_size, 32)
                cudagraph_capture_sizes = list(range(1, interactivity_max + 1))
            else:
                cudagraph_capture_sizes = [
                    i for i in [1, 2, 4] if i <= max_cudagraph_capture_size
                ]

            if max_cudagraph_capture_size >= 8:
                cudagraph_capture_sizes += list(
                    range(8, min(max_cudagraph_capture_size + 1, 256), 8)
                )

            if max_cudagraph_capture_size >= 256:
                cudagraph_capture_sizes += list(
                    range(256, max_cudagraph_capture_size + 1, 16)
                )
            # ensure max_num_tokens is captured if within max capture size---fix
            if (
                max_num_tokens <= max_cudagraph_capture_size
                and max_num_tokens not in cudagraph_capture_sizes
            ):
                cudagraph_capture_sizes.append(max_num_tokens)

            cudagraph_capture_sizes = sorted(set(cudagraph_capture_sizes))

        if (
            self.parallel_config.tensor_parallel_size > 1
            and self.compilation_config.pass_config.enable_sp
        ):
            cudagraph_capture_sizes = self.update_sizes_for_sequence_parallelism(
                cudagraph_capture_sizes
            )

        valid_max_size = cudagraph_capture_sizes[-1] if cudagraph_capture_sizes else 0

        if (
            self.compilation_config.max_cudagraph_capture_size is not None
            and self.compilation_config.max_cudagraph_capture_size != valid_max_size
        ):
            if self.compilation_config.cudagraph_capture_sizes is not None:
                raise ValueError(
                    "customized max_cudagraph_capture_size "
                    f"(={self.compilation_config.max_cudagraph_capture_size}) "
                    "should be consistent with the max value of "
                    f"cudagraph_capture_sizes (={valid_max_size})"
                )

        self.compilation_config.max_cudagraph_capture_size = valid_max_size

        if self.compilation_config.cudagraph_capture_sizes is not None and len(
            cudagraph_capture_sizes
        ) < len(self.compilation_config.cudagraph_capture_sizes):
            pass

        self.compilation_config.cudagraph_capture_sizes = cudagraph_capture_sizes

    else:
        self.compilation_config.max_cudagraph_capture_size = 0
        self.compilation_config.cudagraph_capture_sizes = []

    self.compilation_config.post_init_cudagraph_sizes()


VllmConfig._set_cudagraph_sizes = _patched_set_cudagraph_sizes
