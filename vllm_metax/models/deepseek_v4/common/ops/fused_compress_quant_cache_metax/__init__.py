# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MetaX DeepSeek V4 compressor cache-store launcher.

The current profile stores the main compressed-attention cache in BF16 and the
Lightning Indexer cache in INT8. The indexer launcher also retains upstream
FP8/FP4 kernel dispatch; full FP4 model support remains gated.
"""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch

from vllm.model_executor.warmup.jit_warmup import kernel_launcher
from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    LaunchSpec,
    TritonWarmupTensor,
    VllmTritonJitKernel,
)
from vllm_metax.models.deepseek_v4.common.ops.fused_compress_quant_cache import (
    _fused_kv_compress_norm_rope_insert_sparse_attn,
    _fused_kv_compress_norm_rope_insert_indexer_attn,
    _fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn,
)
from vllm.triton_utils import triton
from vllm.utils.math_utils import round_up

from vllm_metax.models.deepseek_v4.common.ops.fused_compress_quant_cache_metax.bf16 import (
    _fused_kv_compress_norm_rope_insert_sparse_attn_bf16,
)
from vllm_metax.models.deepseek_v4.common.ops.fused_compress_quant_cache_metax.int8 import (
    _fused_kv_compress_norm_rope_insert_indexer_attn_int8,
)
from vllm_metax.models.deepseek_v4.common.ops.fused_compress_quant_cache import (
    compress_norm_rope_store_two_stage_triton,
)


def compress_norm_rope_store_triton(
    state_cache: torch.Tensor,
    num_actual: int,
    token_to_req_indices: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    state_width: int,
    cos_sin_cache: torch.Tensor,
    kv_cache: torch.Tensor,
    k_cache_metadata: Any,
    pdl_kwargs: dict,
    head_dim: int,
    rope_head_dim: int,
    compress_ratio: int,
    overlap: bool,
    use_fp4_cache: bool,
    rms_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    quant_block: int,
    token_stride: int,
    scale_dim: int,
    use_int8_cache: bool,  # used for indexer path
    use_fp8_kvcache: bool,  # used for main cache path
) -> None:
    """Shared triton launcher for the fused compress+norm+RoPE+insert path.

    Dispatches the main cache directly and the indexer through its warmup owner.
    """
    if head_dim == 512:
        if use_fp8_kvcache:
            kernel = _fused_kv_compress_norm_rope_insert_sparse_attn
            # NOTE(MetaX): v0.30.0 added SANITIZE_CACHE_NANS to the upstream
            # FP8 kernel for ROCm gfx950. MACA uses the original encoding.
            scale_kargs = {"FP8_MAX": 448.0, "SANITIZE_CACHE_NANS": False}
        else:
            kernel = _fused_kv_compress_norm_rope_insert_sparse_attn_bf16
            token_stride = kv_cache.stride(1)
            scale_kargs = {"INT8_MAX": 127.0}
        num_warps = 4
    elif head_dim == 128:
        _FUSED_KV_COMPRESS_NORM_ROPE_INSERT_INDEXER_TRITON_KERNEL(
            state_cache=state_cache,
            num_actual=num_actual,
            token_to_req_indices=token_to_req_indices,
            positions=positions,
            slot_mapping=slot_mapping,
            block_table=block_table,
            block_size=block_size,
            state_width=state_width,
            cos_sin_cache=cos_sin_cache,
            kv_cache=kv_cache,
            k_cache_metadata=k_cache_metadata,
            pdl_kwargs=pdl_kwargs,
            head_dim=head_dim,
            rope_head_dim=rope_head_dim,
            compress_ratio=compress_ratio,
            overlap=overlap,
            use_fp4_cache=use_fp4_cache,
            rms_norm_weight=rms_norm_weight,
            rms_norm_eps=rms_norm_eps,
            quant_block=quant_block,
            token_stride=token_stride,
            scale_dim=scale_dim,
            use_int8_cache=use_int8_cache,
        )
        return
    else:
        raise ValueError(f"Unsupported DeepSeek V4 cache head_dim={head_dim}.")

    kernel[(num_actual,)](
        # state cache
        state_cache,
        state_cache.stride(0),
        state_cache.stride(1),
        # metadata
        token_to_req_indices,
        positions,
        slot_mapping,
        block_table,
        block_table.stride(0),
        block_size,
        # RMSNorm
        rms_norm_weight,
        rms_norm_eps,
        # RoPE
        cos_sin_cache,
        cos_sin_cache.stride(0),
        # KV cache
        kv_cache,
        k_cache_metadata.slot_mapping,
        kv_cache.shape[1],  # paged KV cache block size (tokens per block)
        # constexprs
        HEAD_SIZE=head_dim,
        TRITON_BLOCK_SIZE=triton.next_power_of_2(head_dim),
        STATE_WIDTH=state_width,
        COMPRESS_RATIO=compress_ratio,
        OVERLAP=overlap,
        ROPE_HEAD_DIM=rope_head_dim,
        QUANT_BLOCK=quant_block,
        TOKEN_STRIDE=token_stride,
        SCALE_DIM=scale_dim,
        KV_BLOCK_STRIDE=kv_cache.stride(0),
        num_warps=num_warps,
        **scale_kargs,
        **pdl_kwargs,
    )


class FusedKVCompressNormRopeInsertIndexerTritonKernel(
    VllmTritonJitKernel["FusedKVCompressNormRopeInsertIndexerTritonKernel.CompileKey"]
):
    # NOTE(MetaX): Own the warmup contract locally. MACA defaults to INT8 and
    # binds the physical page stride, which can exceed the logical page size.
    # The FP8/FP4 JIT functions are shared with upstream; model-level FP4
    # support remains gated by the indexer backend.
    @dataclass(frozen=True)
    class CompileKey:
        dtype: torch.dtype
        use_fp4_cache: bool
        use_int8_cache: bool
        head_size: int
        triton_block_size: int
        state_width: int
        compress_ratio: int
        overlap: bool
        rope_head_dim: int
        numeric_max: float
        quant_block: int
        token_stride: int
        scale_dim: int
        block_size: int
        kv_cache_block_size: int
        kv_block_stride: int

    # Canonical argument names for launch binding; __call__ selects the kernel.
    kernel = staticmethod(_fused_kv_compress_norm_rope_insert_indexer_attn)

    def dispatch(  # type: ignore[override]
        self,
        *,
        dtype: torch.dtype,
        use_fp4_cache: bool,
        use_int8_cache: bool,
        head_dim: int,
        rope_head_dim: int,
        compress_ratio: int,
        cache_block_size: int,
        cache_alignment: int,
        runtime_state_width: int | None = None,
        runtime_quant_block: int | None = None,
        runtime_token_stride: int | None = None,
        runtime_scale_dim: int | None = None,
        runtime_kv_block_stride: int | None = None,
    ) -> CompileKey:
        overlap = compress_ratio == 4
        default_quant_block = 32 if use_fp4_cache else 128
        quant_block = (
            runtime_quant_block
            if runtime_quant_block is not None
            else default_quant_block
        )
        default_token_stride = head_dim // 2 if use_fp4_cache else head_dim
        token_stride = (
            runtime_token_stride
            if runtime_token_stride is not None
            else default_token_stride
        )
        default_scale_dim = head_dim // quant_block if use_fp4_cache else 4
        scale_dim = (
            runtime_scale_dim if runtime_scale_dim is not None else default_scale_dim
        )
        raw_kv_cache_block_size = cache_block_size // compress_ratio
        kv_cache_block_size = (
            raw_kv_cache_block_size if raw_kv_cache_block_size >= 1 else 1
        )
        state_block_size = (
            4
            if compress_ratio == 4
            else 8
            if compress_ratio == 128
            else cache_block_size
        )
        default_kv_block_stride = round_up(
            kv_cache_block_size * (token_stride + scale_dim),
            cache_alignment,
        )
        kv_block_stride = (
            runtime_kv_block_stride
            if runtime_kv_block_stride is not None
            else default_kv_block_stride
        )
        state_width = (
            runtime_state_width
            if runtime_state_width is not None
            else head_dim * (1 + overlap)
        )
        numeric_max = 127.0 if use_int8_cache else 448.0
        return self.CompileKey(
            dtype=dtype,
            use_fp4_cache=use_fp4_cache,
            use_int8_cache=use_int8_cache,
            head_size=head_dim,
            triton_block_size=triton.next_power_of_2(head_dim),
            state_width=state_width,
            compress_ratio=compress_ratio,
            overlap=overlap,
            rope_head_dim=rope_head_dim,
            numeric_max=numeric_max,
            quant_block=quant_block,
            token_stride=token_stride,
            scale_dim=scale_dim,
            block_size=state_block_size,
            kv_cache_block_size=kv_cache_block_size,
            kv_block_stride=kv_block_stride,
        )

    def get_warmup_keys(
        self,
        vllm_config: Any,
        *,
        k_cache_prefix: str,
    ) -> list[CompileKey]:
        hf_config = vllm_config.model_config.hf_config
        if hf_config is None:
            return []

        cache_block_size = vllm_config.cache_config.block_size
        from vllm_metax.v1.attention.backends.mla.indexer import (
            dsa_indexer_uses_fp4,
        )

        use_fp4_cache = dsa_indexer_uses_fp4(vllm_config)
        head_dim = int(getattr(hf_config, "index_head_dim", 0) or 0)
        rope_head_dim = int(getattr(hf_config, "qk_rope_head_dim", 0) or 0)
        if head_dim <= 0 or rope_head_dim <= 0 or cache_block_size <= 0:
            return []
        indexer_kv_dtype = vllm_config.attention_config.indexer_kv_dtype
        use_int8_cache = indexer_kv_dtype in ["auto", "int8"]

        cache_layer = vllm_config.compilation_config.static_forward_context[
            k_cache_prefix
        ]
        kv_cache = cache_layer.kv_cache
        if (
            not isinstance(kv_cache, torch.Tensor)
            or kv_cache.ndim != 3
            or kv_cache.shape[1] <= 0
        ):
            raise RuntimeError(
                f"Indexer KV cache {k_cache_prefix!r} must be bound before JIT warmup"
            )

        # NOTE(MetaX): Upstream predicts page stride from config alignment, but
        # the pooled MACA cache can have a larger physical stride. The cache
        # prefix selects this layer's bound tensor so warmup uses stride(0).
        return self._trace_dispatch(self.dispatch)(
            dtype=vllm_config.model_config.dtype,
            use_fp4_cache=use_fp4_cache,
            use_int8_cache=use_int8_cache,
            head_dim=head_dim,
            rope_head_dim=rope_head_dim,
            compress_ratio=4,
            cache_block_size=cache_block_size,
            cache_alignment=1,  # placeholder
            runtime_kv_block_stride=kv_cache.stride(0),
        )

    def warmup_inputs(self, compile_key: CompileKey) -> dict[str, Any]:
        fp32_ptr = TritonWarmupTensor(torch.float32, shape=(1, 1))
        int32_ptr = TritonWarmupTensor(torch.int32)
        int64_ptr = TritonWarmupTensor(torch.int64)
        # kv_cache supplies kv_cache_block_size (shape[1]) and KV_BLOCK_STRIDE
        # (stride(0)); reproduce both so warmup and launch specialize identically.
        kv_cache = TritonWarmupTensor(
            torch.uint8,
            shape=(1, compile_key.kv_cache_block_size, 1),
            strides=(compile_key.kv_block_stride, 1, 1),
        )
        return dict(
            state_cache=fp32_ptr,
            num_actual=1,
            token_to_req_indices=int32_ptr,
            positions=int64_ptr,
            slot_mapping=int64_ptr,
            block_table=int32_ptr,
            block_size=compile_key.block_size,
            state_width=compile_key.state_width,
            cos_sin_cache=fp32_ptr,
            kv_cache=kv_cache,
            k_cache_metadata=SimpleNamespace(slot_mapping=int64_ptr),
            pdl_kwargs={},
            head_dim=compile_key.head_size,
            rope_head_dim=compile_key.rope_head_dim,
            compress_ratio=compile_key.compress_ratio,
            overlap=compile_key.overlap,
            use_fp4_cache=compile_key.use_fp4_cache,
            rms_norm_weight=TritonWarmupTensor(
                compile_key.dtype,
                shape=(compile_key.head_size,),
            ),
            rms_norm_eps=1e-6,
            quant_block=compile_key.quant_block,
            token_stride=compile_key.token_stride,
            scale_dim=compile_key.scale_dim,
            use_int8_cache=compile_key.use_int8_cache,
        )

    @kernel_launcher
    def __call__(
        self,
        *,
        state_cache: torch.Tensor,
        num_actual: int,
        token_to_req_indices: torch.Tensor,
        positions: torch.Tensor,
        slot_mapping: torch.Tensor,
        block_table: torch.Tensor,
        block_size: int,
        state_width: int,
        cos_sin_cache: torch.Tensor,
        kv_cache: torch.Tensor,
        k_cache_metadata: Any,
        pdl_kwargs: dict,
        head_dim: int,
        rope_head_dim: int,
        compress_ratio: int,
        overlap: bool,
        use_fp4_cache: bool,
        rms_norm_weight: torch.Tensor,
        rms_norm_eps: float,
        quant_block: int,
        token_stride: int,
        scale_dim: int,
        use_int8_cache: bool,
    ) -> LaunchSpec:
        if head_dim != 128:
            raise ValueError("MetaX fused indexer requires head_dim=128")
        if use_fp4_cache and use_int8_cache:
            raise ValueError("FP4 and INT8 indexer cache modes are exclusive")
        if use_fp4_cache:
            kernel = _fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn
            numeric_max = 448.0
        elif use_int8_cache:
            kernel = _fused_kv_compress_norm_rope_insert_indexer_attn_int8
            numeric_max = 127.0
        else:
            kernel = _fused_kv_compress_norm_rope_insert_indexer_attn
            numeric_max = 448.0

        # The upstream FP8/FP4 JIT signatures retain FP8_MAX. The FP4 kernel
        # ignores that parameter; local INT8 uses INT8_MAX.
        max_kwargs = (
            {"INT8_MAX": numeric_max} if use_int8_cache else {"FP8_MAX": numeric_max}
        )
        return (num_actual,), dict(
            kernel=kernel,
            state_cache_stride0=state_cache.stride(0),
            state_cache_stride1=state_cache.stride(1),
            block_table_stride=block_table.stride(0),
            cos_sin_stride=cos_sin_cache.stride(0),
            k_cache_ptr=kv_cache,
            kv_slot_mapping_ptr=k_cache_metadata.slot_mapping,
            kv_cache_block_size=kv_cache.shape[1],
            HEAD_SIZE=head_dim,
            TRITON_BLOCK_SIZE=triton.next_power_of_2(head_dim),
            STATE_WIDTH=state_width,
            COMPRESS_RATIO=compress_ratio,
            OVERLAP=overlap,
            ROPE_HEAD_DIM=rope_head_dim,
            QUANT_BLOCK=quant_block,
            TOKEN_STRIDE=token_stride,
            SCALE_DIM=scale_dim,
            KV_BLOCK_STRIDE=kv_cache.stride(0),
            num_warps=1,
            **max_kwargs,
            **pdl_kwargs,
        )


_FUSED_KV_COMPRESS_NORM_ROPE_INSERT_INDEXER_TRITON_KERNEL = (
    FusedKVCompressNormRopeInsertIndexerTritonKernel()
)

__all__ = [
    "_FUSED_KV_COMPRESS_NORM_ROPE_INSERT_INDEXER_TRITON_KERNEL",
    "compress_norm_rope_store_triton",
]
