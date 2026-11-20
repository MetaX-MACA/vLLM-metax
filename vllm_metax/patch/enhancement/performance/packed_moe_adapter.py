# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#
# -----------------------------------------------------------------------------
# Note: Unified source-packed MoE adapter supporting both W4A8 and W8A8.
#       - W4A8: directly packs at MoERunner.forward to preserve fused MHC+RMSNorm.
#       - W8A8: pairs post_attention_layernorm/ffn_norm with routed experts
#         so fused residual+RMSNorm+INT8 quant+pack feeds DeepEP dispatch.
#       Opt-in via VLLM_METAX_SOURCE_PACKED_MOE=1; legacy input is resolved in envs.py.
# -----------------------------------------------------------------------------
"""Model-independent hooks for unified source-packed (W4A8 and W8A8) MoE paths."""

import importlib
from contextlib import suppress
from functools import wraps

import torch

from vllm.config import get_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.runner import moe_runner as upstream
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_metax.model_executor.layers.fused_moe import masked_int8_moe
from vllm_metax.model_executor.layers.fused_moe import source_packed_dispatch as packed
from vllm_metax.model_executor.layers.fused_moe.experts.batched_w4a8_deep_gemm_moe import (
    BatchedW4A8DeepGemmExperts,
)
from vllm_metax.model_executor.layers.fused_moe.int8_source_pack import (
    pack_hidden,
    pack_hidden_fake,
)
from vllm_metax.patch.utils import PATCH_MARKER, patch

logger = init_logger(__name__)

if not masked_int8_moe.enabled():
    raise ValueError("Source-packed MoE requires VLLM_METAX_MASKED_INT8_MOE=1")

# ---------------------------------------------------------------------------
# Standalone source packing registration (keep the existing custom-op ABI)
# ---------------------------------------------------------------------------


with suppress(Exception):
    direct_register_custom_op(
        op_name="metax_w4a8_source_pack",
        op_func=pack_hidden,
        mutates_args=[],
        fake_impl=pack_hidden_fake,
    )

# Decoder layer classes to scan for W8A8 norm pairing, as ``module:Class``.
DECODER_LAYERS = (
    "vllm.model_executor.models.minimax_m2:MiniMaxM2DecoderLayer",
    "vllm.model_executor.models.glm4_moe:Glm4MoeDecoderLayer",
    "vllm.model_executor.models.deepseek_v2:DeepseekV2DecoderLayer",
    "vllm_metax.models.deepseek_v2:DeepseekV2DecoderLayer",  # GLM-5.x (GlmMoeDsa)
    "vllm.model_executor.models.qwen3_moe:Qwen3MoeDecoderLayer",
    "vllm_metax.models.deepseek_v4.metax.model:DeepseekV4DecoderLayer",
)

_MOE_RUNNER = "vllm.model_executor.layers.fused_moe.runner.moe_runner"
_RUNNER_CLS = upstream.MoERunner

# Packed carriers produced by a fused norm (W8A8), consumed by the next routed-MoE call.
_pending: list[tuple[upstream.MoERunner, torch.Tensor]] = []


def _original(replacement, full_target_name):
    """The attribute that ``@patch`` replaced."""
    return getattr(replacement, PATCH_MARKER)[full_target_name]


@patch(_MOE_RUNNER, "MoERunner.forward")
def forward(
    self, hidden_states, router_logits, input_ids=None, shared_experts_input=None
):
    # /-------------------- MetaX Modification --------------------\
    # Pack after the existing fused MHC+RMSNorm for INT4 masked experts.
    kernel = getattr(self._quant_method, "moe_kernel", None)
    if kernel is not None and isinstance(
        kernel.fused_experts, BatchedW4A8DeepGemmExperts
    ):
        carrier = torch.ops.vllm.metax_w4a8_source_pack(hidden_states)
        return torch.ops.vllm.metax_packed_moe(
            hidden_states,
            carrier,
            router_logits,
            input_ids,
            shared_experts_input,
            self.layer_name,
        )

    # Otherwise consume the carrier produced by the paired norm hook.
    if _pending and _pending[-1][0] is self:
        _, carrier = _pending.pop()
        return torch.ops.vllm.metax_packed_moe(
            hidden_states,
            carrier,
            router_logits,
            input_ids,
            shared_experts_input,
            self.layer_name,
        )
    # \-------------------- MetaX Modification --------------------/
    return _runner_forward(
        self, hidden_states, router_logits, input_ids, shared_experts_input
    )


@patch(_MOE_RUNNER, "MoERunner._encode_layer_name")
def _encode_layer_name(self):
    # /-------------------- MetaX Modification --------------------\
    scope = packed._scope
    if scope is not None and scope[0] is self:
        # Do not advance compilation's positional all_moe_layers counter.
        name = scope[1]
        return upstream.LayerName(name) if upstream._USE_LAYERNAME else name
    # \-------------------- MetaX Modification --------------------/
    return _runner_encode_layer_name(self)


# The originals, resolved once so the traced bodies above only do global lookups.
_runner_forward = _original(forward, f"{_MOE_RUNNER}.MoERunner.forward")
_runner_encode_layer_name = _original(
    _encode_layer_name, f"{_MOE_RUNNER}.MoERunner._encode_layer_name"
)
# The packed op calls the unpatched runner forward from inside the opaque op.
packed.original_runner_forward = _runner_forward


def _pair_layer(layer):
    """Rewire one decoder layer's norm and routed experts for W8A8, if applicable."""
    norm = getattr(layer, "post_attention_layernorm", None)
    no_residual = norm is None  # DeepSeek-V4 ffn_norm has no residual add.
    if norm is None:
        norm = getattr(layer, "ffn_norm", None)
    moe = next(
        (
            m
            for m in layer.children()
            if isinstance(getattr(m, "experts", None), _RUNNER_CLS)
        ),
        None,
    )
    if not isinstance(norm, RMSNorm) or moe is None or norm.weight is None:
        return  # dense layer or unknown structure: keep the original path

    runner = moe.experts

    # If this layer uses W4A8, source packing happens in MoERunner.forward
    # to preserve fused MHC+RMSNorm; do not hijack the norm layer.
    kernel = getattr(runner._quant_method, "moe_kernel", None)
    if kernel is not None and isinstance(
        kernel.fused_experts, BatchedW4A8DeepGemmExperts
    ):
        return

    # If MASKED_INT8_MOE is disabled, skip W8A8 norm hooking.
    if not masked_int8_moe.enabled():
        return

    vllm_config = get_current_vllm_config()
    if vllm_config.parallel_config.enable_dbo or vllm_config.lora_config is not None:
        raise ValueError("Source-packed MoE does not support DBO/LoRA")

    width = packed.carrier_width(norm.weight.numel())
    epsilon = norm.variance_epsilon
    norm_forward = norm.forward
    if no_residual:
        # 0.30 fuses FFN RMSNorm into MHC. Ask the local decoder to expose the
        # unnormalized MHC output and invoke this norm hook exactly once.
        layer._metax_source_packed_norm = True

    def forward_norm(x, residual=None):
        if residual is None and no_residual:
            packed_hidden, normed = torch.ops.vllm.metax_norm_pack_nores(
                x, norm.weight, epsilon, width
            )
            _pending.append((runner, packed_hidden))
            return normed
        if residual is None:
            return norm_forward(x)
        packed_hidden, normed = torch.ops.vllm.metax_norm_pack(
            x, residual, norm.weight, epsilon, width
        )
        _pending.append((runner, packed_hidden))
        return normed, residual

    norm.forward = forward_norm


def _patch_decoder_init(module_name, class_name):
    full_target_name = f"{module_name}.{class_name}.__init__"
    decoder_cls = getattr(importlib.import_module(module_name), class_name)

    @wraps(decoder_cls.__init__)
    def __init__(self, *args, **kwargs):
        _original(__init__, full_target_name)(self, *args, **kwargs)
        # /-------------------- MetaX Modification --------------------\
        _pair_layer(self)
        # \-------------------- MetaX Modification --------------------/

    patch(module_name, f"{class_name}.__init__")(__init__)


for _path in DECODER_LAYERS:
    try:
        _patch_decoder_init(*_path.split(":"))
    except (ImportError, AttributeError) as exc:  # model not in this vllm build
        logger.debug("Packed MoE: skip %s (%s)", _path, exc)

logger.info("Installed MetaX unified packed-MoE runtime adapter (W4A8/W8A8)")
