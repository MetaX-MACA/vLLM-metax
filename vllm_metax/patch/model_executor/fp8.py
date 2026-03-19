# SPDX-License-Identifier: Apache-2.0

# import torch
# from enum import Enum

import vllm.model_executor.layers.fused_moe.modular_kernel as mk

import vllm.model_executor.layers.fused_moe.oracle.fp8 as vllm_fp8

from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
    Fp8MoeBackend,
)

def backend_to_kernel_cls(
    backend: Fp8MoeBackend,
) -> type[mk.FusedMoEExpertsModular]:
    if backend == Fp8MoeBackend.FLASHINFER_TRTLLM:
        raise NotImplementedError

    elif backend == Fp8MoeBackend.FLASHINFER_CUTLASS:
        from vllm.model_executor.layers.fused_moe.flashinfer_cutlass_moe import (
            FlashInferExperts,
        )

        return FlashInferExperts

    elif backend == Fp8MoeBackend.DEEPGEMM:
        from vllm.model_executor.layers.fused_moe.triton_deep_gemm_moe import (
            TritonOrDeepGemmExperts,
        )

        return TritonOrDeepGemmExperts

    elif backend == Fp8MoeBackend.BATCHED_DEEPGEMM:
        from vllm.model_executor.layers.fused_moe.batched_deep_gemm_moe import (
            BatchedDeepGemmExperts,
        )

        return BatchedDeepGemmExperts

    elif backend == Fp8MoeBackend.MARLIN:
        from vllm.model_executor.layers.fused_moe.fused_marlin_moe import (
            MarlinExperts,
        )

        return MarlinExperts

    elif backend == Fp8MoeBackend.TRITON:
        # ┌------------------------  Metax Modification -------------------------┐
        from vllm_metax.model_executor.layers.fused_moe.fused_moe import (
            TritonExperts as mx_TritonExperts,
        )
        return mx_TritonExperts
        # └------------------------- Metax Modification -------------------------┘

    elif backend == Fp8MoeBackend.BATCHED_TRITON:
        from vllm.model_executor.layers.fused_moe.fused_batched_moe import (
            BatchedTritonExperts,
        )

        return BatchedTritonExperts

    elif backend == Fp8MoeBackend.AITER:
        from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (
            AiterExperts,
        )

        return AiterExperts

    elif backend == Fp8MoeBackend.VLLM_CUTLASS:
        from vllm.model_executor.layers.fused_moe.triton_cutlass_moe import (
            TritonOrCutlassExperts,
        )

        return TritonOrCutlassExperts

    elif backend == Fp8MoeBackend.BATCHED_VLLM_CUTLASS:
        from vllm.model_executor.layers.fused_moe.cutlass_moe import (
            CutlassBatchedExpertsFp8,
        )

        return CutlassBatchedExpertsFp8

    else:
        raise ValueError(f"Unknown FP8 MoE backend: {backend.value}")

vllm_fp8.backend_to_kernel_cls = backend_to_kernel_cls
