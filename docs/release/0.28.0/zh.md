# vLLM-MetaX v0.28.0 发布说明

[English](/release/0.28.0/en.md){ .md-button }

发布日期：2026-10-12 · 源码分支：[releases/v0.28.0](https://github.com/MetaX-MACA/vLLM-metax/tree/releases/v0.28.0)

本版本对齐上游 vLLM v0.28.0，更新 DeepSeek-V4 / DSpark、稀疏注意力及多层 MTP KV cache 接口。

## 1. 主要更新

- **DeepSeek-V4 / DSpark**：修复 BF16 KV cache 下 FlashMLA 的 head padding，DSpark 的 `main_proj` 保持 BF16，并同步 MLA 基类。
- **平台与注意力**：强制启用 sparse attention indexer，新增 `check_runner_kv_caches_multi_layer` 平台接口。
- **算子与 MoE**：注册 `SituAndMul` 和 `ReLUSquaredActivation` 的 OOT 实现，同步 Modular MoE 与 all-to-all；移除上游已删除的 Marlin MoE 类导入。
- **补丁整理**：移除旧版 Triton LoRA 兼容补丁；保留 telechat3-yarn 版本适配。
- **依赖与工具**：移除原 `kv_connectors` 依赖项，更新 TVM FFI 与 numba，重构批量验证工具。
- **上游接口变化**：bitsandbytes 迁移为 OOT 插件，移除 `calculate_kv_scales` 与 `override_attention_dtype`。

## 2. 版本与镜像

| 插件 | MACA 系列 | PyTorch 系列 | mcoplib | 镜像下载页 |
| --- | --- | --- | --- | --- |
| v0.28.0 | 3.8.2.x | 2.10 | 0.4.13 | [vllm-metax:0.28.0](https://developer.metax-tech.com/softnova/docker?chip_name=%E6%9B%A6%E4%BA%91C500%E7%B3%BB%E5%88%97&package_kind=AI&dimension=docker&deliver_type=%E5%88%86%E5%B1%82%E5%8C%85&ai_frame=vllm-metax&frame_version=0.28.0&arch=amd64&system=ubuntu) |

使用配套镜像；源码依赖见 [requirements](https://github.com/MetaX-MACA/vLLM-metax/tree/releases/v0.28.0/requirements)。

镜像部署、环境检查和服务验证见 [部署与验证指南](/release/common/common_zh.md)。

## 3. 模型更新与量化

| 模型 / 路径 | 本版本变化 |
| --- | --- |
| DeepSeek-V4 | 更新 BF16 FlashMLA、DSpark 与 MLA 接口 |
| JoyAI / telechat3-yarn | 保留模型及配置兼容适配 |

## 4. 升级说明

1. 从 v0.27 升级时，使用 mcoplib `0.4.13` 对应的发布镜像；源码快照依赖与发布映射不同。
2. `requirements/kv_connectors.txt` 已移除原 connector 依赖，KV 传输或卸载需另行配置对应组件。
3. 移除旧 Triton LoRA 补丁导入，并更新使用 `calculate_kv_scales`、`override_attention_dtype` 的脚本。
4. bitsandbytes 已由上游迁移为 OOT 插件，原有内置量化入口已移除。
5. Mamba 默认启用 prefix caching。对比性能时显式固定 prefix caching、`max_num_batched_tokens` 和并行参数。

## 5. 已知问题

| 项目 | 说明 |
| --- | --- |
| FlashAttention FP8 KV cache | 本版本只支持 FA2，依赖 FA3 的 FP8 KV cache 路径不可用 |
| DeepSeek-V4 TP | TP 需整除 checkpoint 的 `o_groups`；`o_groups=8` 时不能使用 TP=16 |
