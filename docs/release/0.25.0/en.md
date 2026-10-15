# vLLM-MetaX v0.25.0 Release Notes

[中文](/release/0.25.0/zh.md){ .md-button }

Release date: 2026-09-04 · Source branch: [releases/v0.25.0](https://github.com/MetaX-MACA/vLLM-metax/tree/releases/v0.25.0)

This release aligns with upstream vLLM v0.25.0 and updates DeepEP quantization, MoE weight loading, and model compatibility patches.

## 1. Main changes

- **MoE and DeepEP**: extend FP8 / INT8 paths, add `BatchedDeepGemmExperts` implementations, and adapt Modular MoE output aliasing.
- **Weight loading**: fix transposed per-channel scales in fused MoE; add MiniMax-M3 and Qwen3.5 loading patches.
- **Attention and speculative decoding**: remove backend arguments unsupported by the MetaX flashinfer wrapper and update Eagle compatibility patches.
- **Engineering**: update csrc, dependencies, and Dockerfile; sync KV offload tests.

## 2. Versions and images

| Plugin | MACA series | PyTorch series | mcoplib | Image download page |
| --- | --- | --- | --- | --- |
| v0.25.0 | 3.8.2.x | 2.10 | 0.4.10 | [vllm-metax:0.25.0](https://developer.metax-tech.com/softnova/docker?chip_name=%E6%9B%A6%E4%BA%91C500%E7%B3%BB%E5%88%97&package_kind=AI&dimension=docker&deliver_type=%E5%88%86%E5%B1%82%E5%8C%85&ai_frame=vllm-metax&frame_version=0.25.0&arch=amd64&system=ubuntu) |

Use the matching image. See [requirements](https://github.com/MetaX-MACA/vLLM-metax/tree/releases/v0.25.0/requirements) for source dependencies.

For image deployment, environment checks, and service verification, see the [Deployment and Validation Guide](/release/common/common_en.md).

## 3. Model updates and quantization

| Model / path | Changes in this version |
| --- | --- |
| Qwen3.5 | Add MoE / W8A8 weight-loading fixes |
| MiniMax-M3 | Fix INT8 W8A8 SwiGLU configuration |
| DeepSeek family | Update DeepEP and quantized expert execution paths |
| JoyAI | Retain `JoyAI_LLM_Flash` integration and update compatibility paths |
| Multimodal | Continue Qwen3-Omni, Qwen3-VL, and Intern integrations |

## 4. Upgrade notes

1. When upgrading from v0.24, move mcoplib from `0.4.9` to `0.4.10`.
2. Both v0.24 / v0.25 tags use PyTorch `2.10.0+metax3.8.2.2`. Check custom operator ABIs against the actual image.
3. For custom DeepEP / MoE flows, check expert backend selection and per-channel scale loading.
4. transformers and Eagle / MTP patches have changed. Check tokenizer behavior and speculative decoding accuracy.

## 5. Known issues

| Area | Notes |
| --- | --- |
| Transposed fused W8A8 weights | Check per-channel scale shape and orientation; retain matching loading patches |
