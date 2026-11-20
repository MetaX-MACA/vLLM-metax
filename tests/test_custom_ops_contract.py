# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import torch

from vllm_metax._dummy_ops import dummy_ops
from vllm_metax.platform import MacaPlatform


def test_gptq_schema_exposes_metax_workspace_arguments():
    MacaPlatform.import_kernels()
    schema = torch.ops._C.gptq_gemm.default._schema
    argument_names = [argument.name for argument in schema.arguments]
    assert argument_names[-3:] == ["perm_space", "temp_space", "dtype_bf16"]


def test_dummy_scheduler_metadata_contract():
    assert (
        dummy_ops.get_scheduler_metadata(
            batch_size=1,
            max_seqlen_q=1,
            max_seqlen_k=1,
            num_heads_q=1,
            num_heads_kv=1,
            headdim=64,
            cache_seqlens=torch.zeros(1, dtype=torch.int32),
        )
        is None
    )
