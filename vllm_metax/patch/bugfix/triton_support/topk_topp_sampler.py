# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

# -----------------------------------------------
# Note: Fall back to PyTorch top-k/top-p sampling to avoid Triton issues.
#
# Affected versions: v0.21.0
# Remove at: after `apply_top_k_top_p_triton` is fixed upstream.
# -----------------------------------------------

import torch
import torch.nn as nn
from vllm import envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.platforms import CpuArchEnum, current_platform
from vllm.v1.sample.ops.topk_topp_sampler import (
    apply_top_k_top_p_pytorch,
    random_sample,
)
from vllm.logger import init_logger

logger = init_logger(__name__)

_FLASHINFER_MAX_SAMPLING_ROUNDS = 32


def apply_top_k_top_p(
    logits: torch.Tensor, k: torch.Tensor | None, p: torch.Tensor | None
) -> torch.Tensor:
    if p is None and k is None:
        return logits

    # Use pytorch sort implementation for small batch sizes.
    return apply_top_k_top_p_pytorch(logits, k, p)


def _make_uniform_samples(logits: torch.Tensor) -> torch.Tensor:
    return torch.rand(
        (_FLASHINFER_MAX_SAMPLING_ROUNDS, logits.shape[0]),
        device=logits.device,
        dtype=torch.float32,
    )


def _select_failed_values(
    value: torch.Tensor | int | float | None, failed: torch.Tensor
) -> torch.Tensor | int | float | None:
    if isinstance(value, torch.Tensor):
        return value[failed]
    return value


def _replace_failed_samples(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
    samples: torch.Tensor,
    success: torch.Tensor,
) -> torch.Tensor:
    if bool(success.all().item()):
        return samples

    failed = ~success
    fallback_logits = apply_top_k_top_p(
        logits[failed].clone(),
        _select_failed_values(k, failed),
        _select_failed_values(p, failed),
    )
    fallback_probs = fallback_logits.softmax(dim=-1, dtype=torch.float32)
    fallback_samples = random_sample(fallback_probs, {})

    samples = samples.clone()
    samples[failed] = fallback_samples.to(samples.dtype)
    return samples


def flashinfer_sample(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
    generators: dict[int, torch.Generator],
) -> torch.Tensor:
    import flashinfer

    assert not generators
    assert not (k is None and p is None)

    uniform_samples = _make_uniform_samples(logits)
    if k is None:
        probs = logits.softmax(dim=-1, dtype=torch.float32)
        samples, success = flashinfer.sampling.top_p_sampling_from_probs(
            probs, uniform_samples, p, deterministic=True
        )
    elif p is None:
        probs = logits.softmax(dim=-1, dtype=torch.float32)
        samples, success = flashinfer.sampling.top_k_sampling_from_probs(
            probs, uniform_samples, k, deterministic=True
        )
    else:
        samples, success = flashinfer.sampling.top_k_top_p_sampling_from_logits(
            logits, uniform_samples, k, p, deterministic=True
        )

    samples = _replace_failed_samples(logits, k, p, samples, success)
    return samples.view(-1)


import vllm.v1.sample.ops.topk_topp_sampler

vllm.v1.sample.ops.topk_topp_sampler.apply_top_k_top_p = apply_top_k_top_p
vllm.v1.sample.ops.topk_topp_sampler.flashinfer_sample = flashinfer_sample


def _get_flashinfer_backend_cls():
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
    from vllm_metax.platform import register_attention_backends

    register_attention_backends()
    return AttentionBackendEnum.FLASHINFER.get_class()


def _supports_flashinfer_sampler_platform() -> bool:
    return current_platform.is_cuda() or (
        current_platform.is_out_of_tree() and current_platform.is_cuda_alike()
    )


def _topk_topp_sampler_init(self, logprobs_mode="raw_logprobs") -> None:
    nn.Module.__init__(self)
    self.logprobs_mode = logprobs_mode

    if (
        logprobs_mode not in ("processed_logits", "processed_logprobs")
        and _supports_flashinfer_sampler_platform()
    ):
        if envs.VLLM_USE_FLASHINFER_SAMPLER:
            flashinfer_backend = _get_flashinfer_backend_cls()
            capability = current_platform.get_device_capability()
            assert capability is not None
            if flashinfer_backend.supports_compute_capability(capability):
                logger.warning_once(
                    "Using FlashInfer for top-p & top-k sampling.",
                    scope="global",
                )
                self.forward = self.forward_cuda
            elif envs.is_set("VLLM_USE_FLASHINFER_SAMPLER"):
                capability_str = capability.as_version_str()
                raise RuntimeError(
                    "FlashInfer does not support compute capability "
                    f"{capability_str}, unset VLLM_USE_FLASHINFER_SAMPLER=1."
                )
            else:
                logger.warning_once(
                    "FlashInfer top-p/top-k sampling not supported on "
                    "compute capability %s; falling back to PyTorch-native "
                    "sampler. Set VLLM_USE_FLASHINFER_SAMPLER=0 to silence.",
                    capability.as_version_str(),
                )
                self.forward = self.forward_native
        else:
            self.forward = self.forward_native
    elif current_platform.is_cpu():
        arch = current_platform.get_cpu_architecture()
        if arch in (CpuArchEnum.RISCV, CpuArchEnum.POWERPC):
            self.forward = self.forward_native
        else:
            self.forward = self.forward_cpu
    elif current_platform.is_xpu():
        if envs.VLLM_XPU_USE_SAMPLER_KERNEL:
            self.forward = self.forward_xpu
        else:
            self.forward = self.forward_native
    elif (
        logprobs_mode not in ("processed_logits", "processed_logprobs")
        and rocm_aiter_ops.is_enabled()
    ):
        try:
            import aiter.ops.sampling  # noqa: F401

            self.aiter_ops = torch.ops.aiter
            logger.info_once(
                "Using aiter sampler on ROCm (lazy import, sampling-only)."
            )
            self.forward = self.forward_hip
        except ImportError:
            logger.warning_once(
                "aiter.ops.sampling is not available on ROCm. "
                "Falling back to forward_native implementation."
            )
            self.forward = self.forward_native
    else:
        self.forward = self.forward_native


vllm.v1.sample.ops.topk_topp_sampler.TopKTopPSampler.__init__ = (
    _topk_topp_sampler_init
)

import vllm.v1.sample.rejection_sampler

vllm.v1.sample.rejection_sampler.apply_top_k_top_p = apply_top_k_top_p

import vllm.v1.worker.gpu.sample.states

vllm.v1.worker.gpu.sample.states.apply_top_k_top_p = apply_top_k_top_p
