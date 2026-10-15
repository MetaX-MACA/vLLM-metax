# vLLM-MetaX v0.28.0 Release Notes

[中文](/release/0.28.0/zh.md){ .md-button }

Release date: 2026-10-12 · Source branch: [releases/v0.28.0](https://github.com/MetaX-MACA/vLLM-metax/tree/releases/v0.28.0)

This release aligns with upstream vLLM v0.28.0 and updates DeepSeek-V4 / DSpark, sparse attention, and multilayer MTP KV cache interfaces.

## 1. Main changes

- **DeepSeek-V4 / DSpark**: fix FlashMLA head padding with BF16 KV cache, keep DSpark `main_proj` in BF16, and sync the MLA base class.
- **Platform and attention**: force-enable the sparse attention indexer and add the `check_runner_kv_caches_multi_layer` platform interface.
- **Operators and MoE**: register OOT implementations for `SituAndMul` and `ReLUSquaredActivation`; sync Modular MoE and all-to-all; remove imports of an upstream-deleted Marlin MoE class.
- **Patch cleanup**: remove legacy Triton LoRA compatibility patches; retain version-specific telechat3-yarn adaptations.
- **Dependencies and tooling**: remove the previous `kv_connectors` dependency entries, update TVM FFI and numba, and refactor batch validation tools.
- **Upstream interface changes**: move bitsandbytes to an OOT plugin and remove `calculate_kv_scales` and `override_attention_dtype`.

## 2. Versions and images

| Plugin | MACA series | PyTorch series | mcoplib | Image download page |
| --- | --- | --- | --- | --- |
| v0.28.0 | 3.8.2.x | 2.10 | 0.4.13 | [vllm-metax:0.28.0](https://developer.metax-tech.com/softnova/docker?chip_name=%E6%9B%A6%E4%BA%91C500%E7%B3%BB%E5%88%97&package_kind=AI&dimension=docker&deliver_type=%E5%88%86%E5%B1%82%E5%8C%85&ai_frame=vllm-metax&frame_version=0.28.0&arch=amd64&system=ubuntu) |

Use the matching image. See [requirements](https://github.com/MetaX-MACA/vLLM-metax/tree/releases/v0.28.0/requirements) for source dependencies.

For image deployment, environment checks, and service verification, see the [Deployment and Validation Guide](/release/common/common_en.md).

## 3. Model updates and quantization

| Model / path | Changes in this version |
| --- | --- |
| DeepSeek-V4 | Update BF16 FlashMLA, DSpark, and MLA interfaces |
| JoyAI / telechat3-yarn | Retain model and configuration compatibility adaptations |

## 4. Upgrade notes

1. When upgrading from v0.27, use the release image paired with mcoplib `0.4.13`. Source snapshot dependencies differ from the release mapping.
2. The previous connector dependencies were removed from `requirements/kv_connectors.txt`. Configure the required components separately for KV transfer or offloading.
3. Remove old Triton LoRA patch imports and update scripts using `calculate_kv_scales` or `override_attention_dtype`.
4. Upstream moved bitsandbytes to an OOT plugin and removed its built-in quantization entry.
5. Mamba enables prefix caching by default. Fix prefix caching, `max_num_batched_tokens`, and parallel settings for performance comparisons.

## 5. Known issues

| Area | Notes |
| --- | --- |
| FlashAttention FP8 KV cache | Only FA2 is supported in this version; FP8 KV cache paths requiring FA3 are unavailable |
| DeepSeek-V4 TP | TP must divide the checkpoint's `o_groups`; TP=16 is invalid when `o_groups=8` |
