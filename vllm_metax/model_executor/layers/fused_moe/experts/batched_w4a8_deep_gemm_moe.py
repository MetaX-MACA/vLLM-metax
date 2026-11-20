# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""MetaX INT4 W4A8 masked experts for DeepEP low-latency dispatch."""

import torch
from vllm_metax.model_executor.layers.fused_moe import masked_int8_moe
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input
from vllm_metax.model_executor.layers.fused_moe.experts.batched_deep_gemm_moe import (
    BatchedDeepGemmExperts,
    _resize_cache,
)
from deep_gemm import m_grouped_gemm_int8_int4_bf16_nt_masked
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kInt4W4A8StaticChannelSym,
    kInt8DynamicTokenSym,
)

logger = init_logger("vllm.metax.w4a8_deepgemm")


def normalize_channel_scales(scales, e, n):
    if scales is None or scales.numel() != e * n:
        raise ValueError("W4A8 masked GEMM supports per-channel weight scales only")
    return scales.to(torch.float32).contiguous().reshape(e, n, 1)


class BatchedW4A8DeepGemmExperts(BatchedDeepGemmExperts):
    @staticmethod
    def _supports_quant_scheme(weight_key, activation_key):
        return weight_key == kInt4W4A8StaticChannelSym and activation_key in (
            None,
            kInt8DynamicTokenSym,
        )

    @staticmethod
    def _supports_activation(activation):
        return activation in (MoEActivation.SILU, MoEActivation.SWIGLUSTEP)

    @staticmethod
    def _supports_parallel_config(config):
        return config.use_deepep_ll_kernels

    def __init__(self, moe_config, quant_config, max_num_tokens, num_dispatchers):
        mk.FusedMoEExpertsModular.__init__(
            self, moe_config, quant_config, max_num_tokens, num_dispatchers
        )
        if not quant_config.use_int4_w4a8 or quant_config.block_shape is not None:
            raise ValueError(
                "DeepEP W4A8 masked experts require per-channel INT4 / per-token INT8"
            )
        if moe_config.in_dtype != torch.bfloat16:
            raise ValueError("DeepEP W4A8 masked experts require BF16 input/output")
        if self.w1_zp is not None or self.w2_zp is not None:
            raise ValueError("DeepEP W4A8 masked experts require symmetric weights")
        if self.w1_bias is not None or self.w2_bias is not None:
            raise ValueError("DeepEP W4A8 masked experts do not support bias")
        if moe_config.activation not in (MoEActivation.SILU, MoEActivation.SWIGLUSTEP):
            raise ValueError("Unsupported activation for W4A8 masked experts")
        self.is_fp8 = False
        self.is_int8 = False
        self._calls = 0
        logger.info_once(
            "Using BatchedW4A8DeepGemmExperts: DeepEP LL + native INT8xINT4 masked grouped GEMM"
        )

    def workspace_dtype(self, act_dtype):
        return torch.bfloat16

    def apply(
        self,
        output,
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        activation,
        global_num_experts,
        expert_map,
        a1q_scale,
        a2_scale,
        workspace13,
        workspace2,
        expert_tokens_meta,
        apply_router_weight_on_input,
    ):
        assert expert_tokens_meta is not None
        counts = expert_tokens_meta.expert_num_tokens
        e, m, k = hidden_states.shape
        n = w1.shape[1]
        intermediate = self.adjust_N_for_activation(n, activation)
        assert hidden_states.dtype == torch.int8 and hidden_states.is_contiguous()
        assert counts.dtype == torch.int32 and counts.shape == (e,)
        assert w1.dtype == torch.int32 and w1.shape == (e, n, k // 8)
        assert w2.dtype == torch.int32 and w2.shape == (e, k, intermediate // 8)
        assert output.shape == (e, m, k) and output.dtype == torch.bfloat16
        assert a2_scale is None
        first = _resize_cache(workspace13, (e, m, n))
        activated = _resize_cache(workspace2, (e, m, intermediate))
        fast = (
            masked_int8_moe.enabled()
            and masked_int8_moe.supports_compact_activation(
                activation, self.activation_config
            )
        )
        if not fast:
            first.zero_()
        packed = getattr(expert_tokens_meta, "packed_input", None)
        if packed is not None:
            packed = masked_int8_moe.compact_carrier(packed, counts, k)
        elif fast:
            packed = masked_int8_moe.pack_valid_rows(hidden_states, a1q_scale, counts)
        else:
            packed = masked_int8_moe.pack_per_token_int8(hidden_states, a1q_scale)
        expected = self.estimate_expected_m(global_num_experts, m, topk_ids.size(-1))
        m_grouped_gemm_int8_int4_bf16_nt_masked(
            (packed, None),
            (w1.view(torch.uint8), normalize_channel_scales(self.w1_scale, e, n)),
            first,
            counts,
            expected,
            backend="mctlassEx",
        )
        del packed
        if fast:
            packed2 = masked_int8_moe.activation_and_pack_compact(
                first, counts, self.activation_config
            )
        else:
            self.activation(
                activation, activated.view(-1, intermediate), first.view(-1, n)
            )
            q, scales = moe_kernel_quantize_input(
                activated.view(-1, intermediate),
                None,
                torch.int8,
                per_act_token_quant=True,
                block_shape=None,
            )
            q = q.reshape(e, m, intermediate)
            packed2 = masked_int8_moe.pack_per_token_int8(q, scales)
            output.zero_()
        m_grouped_gemm_int8_int4_bf16_nt_masked(
            (packed2, None),
            (w2.view(torch.uint8), normalize_channel_scales(self.w2_scale, e, k)),
            output,
            counts,
            expected,
            backend="mctlassEx",
        )
        if self._calls == 0:
            logger.info(
                "W4A8 masked GEMM: E=%d M=%d K=%d N=%d, valid_rows=%s, native_act=%s",
                e,
                m,
                k,
                n,
                fast,
                masked_int8_moe.mx_envs.VLLM_METAX_NATIVE_MASKED_ACT,
            )
        self._calls += 1
