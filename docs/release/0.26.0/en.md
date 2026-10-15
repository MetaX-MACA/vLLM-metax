# vLLM-MetaX v0.26.0 Release Notes

[中文](/release/0.26.0/zh.md){ .md-button }

Release date: 2026-09-15 · Source branch: [releases/v0.26.0](https://github.com/MetaX-MACA/vLLM-metax/tree/releases/v0.26.0)

This release aligns with upstream vLLM v0.26.0, restructures patches and registries, and updates DeepSeek-V4, sparse attention, and quantized experts.

## 1. Main changes

- **DeepSeek-V4 / MLA**: sync sparse attention, MLA, MTP, and draft configuration interfaces; fix indexer registration and imports.
- **MoE**: update DeepEP FP8 / INT8 experts; fix mctlass FP8 and fused MoE scale loading.
- **Model compatibility**: update MiniMax-M3 top-k and SwiGLU configuration; add telechat3-yarn fixes.
- **Plugin structure**: move quantization configurations, custom operators, and enhancements into `registry/quant_config`, `registry/custom_ops`, and `patch/enhancement`; standardize `@patch` templates.
- **Environment configuration**: stop forcing some environment variable overrides by default.

## 2. Versions and images

| Plugin | MACA series | PyTorch series | mcoplib | Image download page |
| --- | --- | --- | --- | --- |
| v0.26.0 | 3.8.2.x | 2.10 | 0.4.11 | [vllm-metax:0.26.0](https://developer.metax-tech.com/softnova/docker?chip_name=%E6%9B%A6%E4%BA%91C500%E7%B3%BB%E5%88%97&package_kind=AI&dimension=docker&deliver_type=%E5%88%86%E5%B1%82%E5%8C%85&ai_frame=vllm-metax&frame_version=0.26.0&arch=amd64&system=ubuntu) |

Use the matching image. See [requirements](https://github.com/MetaX-MACA/vLLM-metax/tree/releases/v0.26.0/requirements) for source dependencies.

For image deployment, environment checks, and service verification, see the [Deployment and Validation Guide](/release/common/common_en.md).

## 3. Model updates and quantization

| Model / path | Changes in this version |
| --- | --- |
| DeepSeek-V4 | Update sparse MLA, MTP, and speculative decoding integration |
| MiniMax-M3 | Update top-k / sparse attention; fix INT8 SwiGLU configuration |
| Qwen3.5 | Retain MoE weight-loading fixes |
| JoyAI / telechat3-yarn | Fix architecture detection, tokenizer handling, and `max_model_len` inference |
| Step3.5 / MRV2 | Update model and parallel execution integration |

## 4. Upgrade notes

1. When upgrading from v0.25, move mcoplib from `0.4.10` to `0.4.11`.
2. Update custom extension imports and registrations for `registry/quant_config`, `registry/custom_ops`, and `patch/enhancement`.
3. Resolve the PyTorch and tilelang / TVM FFI dependency conflicts below before building from source.
4. Environment variables are no longer forcibly overridden by default; set required variables explicitly in custom deployments.

## 5. Known issues

| Area | Notes |
| --- | --- |
| PyTorch dependency mismatch | At snapshot `615ac62`, `build.txt` pins `2.10.0+metax3.8.0.7`, while runtime requirements pin `2.10.0+metax3.8.2.2`; align before building |
| tilelang / TVM FFI conflicts | `maca.txt` pins `0.1.9+maca` / `0.1.3.dev11`, while `maca_private.txt` pins `0.1.12+maca` / `0.1.11+maca`; use consistent versions |
