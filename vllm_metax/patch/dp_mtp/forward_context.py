# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

# ------------------------------------------------------------------------
# Note: This file is a patch to DP4+MTP3 flashmla exceded Max_Spilts issue
# ------------------------------------------------------------------------

import vllm.forward_context
from vllm.forward_context import BatchDescriptor
from typing import NamedTuple


class MacaBatchDescriptor(NamedTuple):
    # /------------------------  Metax Modification -------------------------\
    """
    Batch descriptor for cudagraph dispatching. We should keep the num of
    items as minimal as possible to properly and uniquely describe the padded
    batch for cudagraph.
    """

    num_tokens: int
    num_reqs: int | None = None
    """
    Number of requests in the batch. Can be None for PIECEWISE cudagraphs where
    the cudagraphs can handle any number of requests.
    """
    uniform: bool = False
    """
    True if all the requests in the batch have the same number of tokens.
    """
    has_lora: bool = False
    """
    Whether this batch has active LoRA adapters.
    """

    def relax_for_mixed_batch_cudagraphs(self) -> "MacaBatchDescriptor":
        """
        Return a relaxed version of current batch descriptor that is still compatible
        with PIECEWISE cudagraphs (or mixed prefill-decode FA cudagraphs).
        """
        return MacaBatchDescriptor(
            self.num_tokens, num_reqs=None, uniform=False, has_lora=self.has_lora
        )

    # \------------------------- Metax Modification -------------------------/


vllm.forward_context.BatchDescriptor = MacaBatchDescriptor
BatchDescriptor.relax_for_mixed_batch_cudagraphs = (
    MacaBatchDescriptor.relax_for_mixed_batch_cudagraphs
)
