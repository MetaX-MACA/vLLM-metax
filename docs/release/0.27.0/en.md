# vLLM-MetaX v0.27.0 Release Notes

[中文](/release/0.27.0/zh.md){ .md-button }

Release date: 2026-09-24 · Source branch: [releases/v0.27.0](https://github.com/MetaX-MACA/vLLM-metax/tree/releases/v0.27.0)

This release aligns with upstream vLLM v0.27.0 and updates DeepSeek-V4, MiniMax sparse attention, distributed communication, and sleep mode.

## 1. Main changes

- **DeepSeek-V4**: update model, sparse MLA, and speculative decoding interfaces; fix `_gather_k_cache_kernel` stride order and disable the model's FP8 KV cache layout by default.
- **MiniMax-M3**: fix Triton shared-memory limits; disable `cooperative_topk` / `persistent_topk`, which mcoplib does not support.
- **MoE**: fix mctlass FP8, skip invalid experts, and explicitly prohibit non-batched DeepGEMM for INT8 W8A8.
- **Distributed execution and memory**: add mxmesh support, fix MCCL all-gather hangs, and add `MacaMemBackend` for sleep mode.
- **Toolchain integration**: update csrc, custom operators, and attention backends; disable `cudaLaunchAttribute` on MACA.

## 2. Versions and images

| Plugin | MACA series | PyTorch series | mcoplib | Image download page |
| --- | --- | --- | --- | --- |
| v0.27.0 | 3.8.2.x | 2.10 | 0.4.12 | [vllm-metax:0.27.0](https://developer.metax-tech.com/softnova/docker?chip_name=%E6%9B%A6%E4%BA%91C500%E7%B3%BB%E5%88%97&package_kind=AI&dimension=docker&deliver_type=%E5%88%86%E5%B1%82%E5%8C%85&ai_frame=vllm-metax&frame_version=0.27.0&arch=amd64&system=ubuntu) |

Use the matching image. See [requirements](https://github.com/MetaX-MACA/vLLM-metax/tree/releases/v0.27.0/requirements) for source dependencies.

For image deployment, environment checks, and service verification, see the [Deployment and Validation Guide](/release/common/common_en.md).

## 3. Model updates and quantization

| Model / path | Changes in this version |
| --- | --- |
| DeepSeek-V4 | Update sparse MLA, MTP, and model interfaces; fix cache gather strides |
| MiniMax-M3 | Update sparse attention and Triton top-k kernels |
| JoyAI | Update `JoyAI_LLM_Flash` integration for v0.27 |
| Quantized MoE | The INT8 W8A8 DeepGEMM implementation supports batched / EP execution only |

## 4. Upgrade notes

1. When upgrading from v0.26, use the release image paired with mcoplib `0.4.12`.
2. Source snapshot dependencies differ from the release mapping. Check the dependency files before source installation.
3. Use batched / EP execution for INT8 W8A8 DeepGEMM.
4. Sleep mode uses `MacaMemBackend`; update custom memory backend configurations.

## 5. Known issues

| Area | Notes |
| --- | --- |
| INT8 W8A8 DeepGEMM | Non-batched execution is unsupported; use batched / EP settings |
| MiniMax sparse top-k | Keep `cooperative_topk` / `persistent_topk` disabled |
