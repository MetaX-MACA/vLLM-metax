# vLLM-MetaX v0.24.0 Release Notes

[中文](/release/0.24.0/zh.md){ .md-button }

Release date: 2026-08-21 · Source branch: [releases/v0.24.0](https://github.com/MetaX-MACA/vLLM-metax/tree/releases/v0.24.0)

This release updates the MetaX backend for MoE kernels, expert parallelism, model integration, and precision debugging.

## 1. Main changes

- **MoE and quantization**: add mctlass FP8, mctlassEx W4A16 (BF16 path), and fused MoE W4A8 support; remove inplace fused experts.
- **Expert parallelism**: adapt DeepEP and `BatchedTritonExperts`, with expert filtering.
- **Attention**: update MLA prefill and FLA paths.
- **Precision and builds**: integrate msprobe; fix `compressed_tensors` INT8 W8A8 accuracy and builds with `USE_PRECOMPILED_KERNEL=1`.

## 2. Versions and images

| Plugin | MACA series | PyTorch series | mcoplib | Image download page |
| --- | --- | --- | --- | --- |
| v0.24.0 | 3.8.2.x | 2.10 | 0.4.9 | [vllm-metax:0.24.0](https://developer.metax-tech.com/softnova/docker?chip_name=%E6%9B%A6%E4%BA%91C500%E7%B3%BB%E5%88%97&package_kind=AI&dimension=docker&deliver_type=%E5%88%86%E5%B1%82%E5%8C%85&ai_frame=vllm-metax&frame_version=0.24.0&arch=amd64&system=ubuntu) |

Use the matching image. See [requirements](https://github.com/MetaX-MACA/vLLM-metax/tree/releases/v0.24.0/requirements) for source dependencies.

For image deployment, environment checks, and service verification, see the [Deployment and Validation Guide](/release/common/common_en.md).

## 3. Model updates and quantization

| Model / path | Changes in this version |
| --- | --- |
| MiniMax-M3 | Add model integration and an FP8 weight path |
| JoyAI | Register `JoyAI_LLM_Flash` |
| DeepSeek family | Adapt DeepSeek-V4; update DeepSeek-V2 and MLA paths |
| Multimodal | Update Qwen3-Omni, Qwen3-VL, and Intern integrations; fix Qwen2.5-Omni `cu_seqlens` handling |
| Quantization | Add AWQ configuration and `awq_to_gptq_4bit` conversion; fix INT8 W8A8 |

## 4. Upgrade notes

1. When upgrading from v0.23, move mcoplib from `0.4.8` to `0.4.9` and use the matching MACA environment.
2. PyTorch remains in the MetaX 2.10 series. Rebuild custom operators when the toolchain suffix changes.
3. Adapt custom MoE extensions to the non-inplace fused experts path.
4. For EP, Eagle / MTP, or custom tokenizers, check weight loading, generation accuracy, and whitespace handling.

## 5. Known issues

| Area | Notes |
| --- | --- |
| MLA prefill | Optimizations cover specific head dimensions; measure with the actual model |
| Tokenizers | Check transformers versions and compatibility patches for whitespace issues |
