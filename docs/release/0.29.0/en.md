# vLLM-MetaX v0.29.0 Release Notes

[中文](/release/0.29.0/zh.md){ .md-button }

Release line: `vLLM-metax-0.29.0_rel3.9.0` · Upstream baseline: vLLM v0.29.0

This release aligns vLLM-MetaX with upstream vLLM v0.29.0 and completes the MACA 3.9.0 release line. It focuses on the v0.29 attention and model interfaces, DeepSeek-V4 / DSpark, MiniMax-M3, MoE quantization, MRV2 stability, and the MACA 3.9.0 dependency stack.

## 1. Main changes

- **Upstream v0.29.0 alignment**: sync the platform, model registry, attention backends, quantization configs, modular MoE, patch loading, and model-runner integration to the vLLM v0.29.0 interfaces.
- **DeepSeek-V4 / DSpark**: update MLA, MTP, DSpark, and indexer paths; route QuantFP8 group quantization through the fused MACA kernel; adapt mHC to the MACA TileLang format; and keep `main_proj` quantization controlled by the MetaX configuration.
- **Attention and distributed execution**: fix the GQA DCP kernel trap and MLA prefill `max_query_len` overstatement; use the `FLASH_ATTN_DIFFKV` backend for MiMo V2; and add workarounds for MRV2 DP + MTP graph hangs.
- **MoE and quantization**: add `filter_expert` support to the MCTlass MoE GEMM path, update AWQ layer skipping for the vLLM v0.29 `match_mode` API, and update compressed-tensors WNA16 integration.
- **Model integration**: add the initial Qwen3.8-Flash-Next integration, fix MiniMax-M3 VL / Transformers compatibility and weight loading, and restore the DeepSeek-V3.2 MTP indexer forward patch.
- **Runtime and tooling**: use the optimized Gumbel sample kernel, fix resampled-token insertion bounds, disable CUDA-graph memory estimation by default on MetaX, and sync the release dependency set.

## 2. Versions and images

| Plugin | Release line | MACA series | PyTorch series | mcoplib | Image download page |
| --- | --- | --- | --- | --- | --- |
| v0.29.0 | `vLLM-metax-0.29.0_rel3.9.0` | 3.9.0.x | 2.10 | 0.4.14 | [vllm-metax:0.29.0](https://developer.metax-tech.com/softnova/docker?chip_name=MXC500&package_kind=AI&dimension=docker&deliver_type=%E5%88%86%E5%B1%82%E5%8C%85&ai_frame=vllm-metax&frame_version=0.29.0&system=ubuntu&arch=amd64) |

The source branch pins the following release-line dependencies:

| Component | Version |
| --- | --- |
| torch | `2.10.0+metax3.9.0.8` |
| mcoplib | `0.4.14+gf3794c4.maca3.9.0.31.torch2.10` |
| deep_ep | `1.0.0+metax3.9.0.11torch2.10` |
| deep_gemm | `0.4.0+maca3.9.2.3` |
| apache-tvm-ffi | `0.1.11+maca` |
| tilelang | `0.1.12+maca` |
| mixl | `1.0.0+maca3.9.0.10` |
| transformers | `>= 5.10.4` |

Use the image and dependency bundle published on the release page. The exact image tag and driver build are defined there; do not combine this source dependency set with an unverified image.

For image deployment, environment checks, and service verification, see the [Deployment and Validation Guide](/getting_started/deployment/en.md).

## 3. Model updates and quantization

| Model / path | Changes in this version |
| --- | --- |
| DeepSeek-V4 / DeepSeek-V4-Flash | Update DSpark, MTP, MLA, indexer, QuantFP8, and mHC paths; TP must divide the checkpoint's `o_groups` |
| Qwen3.8-Flash-Next | Add the initial MetaX integration; validate with the release weights before production |
| MiniMax-M3 | Fix VL / Transformers compatibility and weight loading; QK-norm all-reduce fusion remains disabled by default |
| DeepSeek-V3.2 MTP | Restore the indexer forward patch required by the v0.29 model path |
| MiMo V2 | Use `FLASH_ATTN_DIFFKV` and apply the MRV2 DP + MTP graph-hang workaround |
| Quantized MoE | Add MCTlass `filter_expert`; update FP8 group quantization, compressed-tensors WNA16, and AWQ layer skipping |

Upstream v0.29.0 also adds or migrates several architectures. Hy4-preview, Granite SWA, NemotronH Omni, Kimi K3 NVFP4, FlexOlmo / Olmo3, and Hunyuan V1 / VL are not yet verified on MetaX in this release line. Register and validate them with available weights before publishing a supported-model claim.

## 4. Upgrade notes

1. Move to the MACA 3.9.0.x and PyTorch `2.10.0+metax3.9.0.8` toolchain and rebuild local custom operators or csrc against it.
2. Use mcoplib `0.4.14+gf3794c4.maca3.9.0.31.torch2.10`. Mixing a locally built csrc with a different mcoplib build can cause missing kernel symbols.
3. For source installations, check out `vLLM-metax-0.29.0_rel3.9.0`, install the matching `requirements/` set, and recheck `mixl` / Mooncake / LMCache versions after the connector rename.
4. Recheck runner and memory settings before comparing performance with v0.28: Mamba prefix caching and the `max_num_batched_tokens` default affect throughput and latency baselines, while CUDA-graph memory estimation is disabled by default on MetaX.
5. Validate LoRA MoE and speculative-decoding paths after upgrading. The Hunyuan Transformers backend currently needs `--enforce-eager`, and LoRA MoE small-batch runs may need `--max-lora-rank` above 64.

## 5. Known issues

| Area | Notes |
| --- | --- |
| Source dependencies | Release images and source snapshots may pin different local build numbers; use the dependency files from the exact branch or image |
| Hunyuan V1 / HunyuanOCR | CUDA-graph capture can fail in the Transformers backend; use `--enforce-eager` until the MACA-side fix is available |
| LoRA MoE | The small-batch fast path can raise an XNACK / ATU fault during capture; use `--max-lora-rank 128` as a workaround and regress accuracy |
| MLA + MTP | Some DeepSeek-R1 W4A8 / MLA + MTP configurations still fail during warmup decode with an ATU fault; do not use them in the v0.29.0 support baseline until resolved |
| FlashAttention FP8 KV cache | MACA exposes FA2 only; upstream paths that require FA3 FP8 KV cache remain unavailable |
| DeepSeek-V4 TP | TP must divide the checkpoint's `o_groups`; use TP8 x PP2 or TP8 x DP2 for Flash checkpoints with 8 groups |
| New upstream models | Models without MetaX validation or available weights are not part of the supported-model claim |
| MiniMax-M3 QK-norm fusion | The all-reduce fusion is disabled by default; enabling it requires an accuracy regression |

## 6. Related documentation

- **Deployment and validation**: [Deployment and Validation Guide](/getting_started/deployment/en.md)
- **Supported models**: [Supported Models](/models/supported_models.md)
- **Patch management**: [Patches](/developing/patches/README.md)
- **Model registration**: [Model registration guide](/developing/model/registration.md)
- **Accuracy debugging**: [msprobe guide](/developing/msprobe/msprobe_guide.md)
- **Plugin architecture**: [Contributing / plugin system](/developing/README.md)
