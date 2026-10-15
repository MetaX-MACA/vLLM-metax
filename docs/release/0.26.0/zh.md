# vLLM-MetaX v0.26.0 发布说明

[English](/release/0.26.0/en.md){ .md-button }

发布日期：2026-09-15 · 源码分支：[releases/v0.26.0](https://github.com/MetaX-MACA/vLLM-metax/tree/releases/v0.26.0)

本版本对齐上游 vLLM v0.26.0，重构 patch 与 registry，并更新 DeepSeek-V4、稀疏注意力和量化专家路径。

## 1. 主要更新

- **DeepSeek-V4 / MLA**：同步 sparse attention、MLA、MTP 与 draft config 接口，修复 indexer 注册及导入问题。
- **MoE**：更新 DeepEP FP8 / INT8 专家路径；修复 mctlass FP8 与 fused MoE scale 加载。
- **模型兼容**：补充 MiniMax-M3 top-k、SwiGLU 配置及 telechat3-yarn 相关修复。
- **插件结构**：量化配置、自定义算子与增强逻辑迁移至 `registry/quant_config`、`registry/custom_ops` 和 `patch/enhancement`；统一 `@patch` 模板。
- **环境配置**：不再默认强制覆盖部分环境变量。

## 2. 版本与镜像

| 插件 | MACA 系列 | PyTorch 系列 | mcoplib | 镜像下载页 |
| --- | --- | --- | --- | --- |
| v0.26.0 | 3.8.2.x | 2.10 | 0.4.11 | [vllm-metax:0.26.0](https://developer.metax-tech.com/softnova/docker?chip_name=%E6%9B%A6%E4%BA%91C500%E7%B3%BB%E5%88%97&package_kind=AI&dimension=docker&deliver_type=%E5%88%86%E5%B1%82%E5%8C%85&ai_frame=vllm-metax&frame_version=0.26.0&arch=amd64&system=ubuntu) |

使用配套镜像；源码依赖见 [requirements](https://github.com/MetaX-MACA/vLLM-metax/tree/releases/v0.26.0/requirements)。

镜像部署、环境检查和服务验证见 [部署与验证指南](/release/common/common_zh.md)。

## 3. 模型更新与量化

| 模型 / 路径 | 本版本变化 |
| --- | --- |
| DeepSeek-V4 | 更新 sparse MLA、MTP 与投机解码适配 |
| MiniMax-M3 | 更新 top-k / sparse attention，补充 INT8 SwiGLU 配置修复 |
| Qwen3.5 | 保留 MoE 权重加载修复 |
| JoyAI / telechat3-yarn | 修复架构识别、tokenizer 及 `max_model_len` 推导 |
| Step3.5 / MRV2 | 更新模型与并行路径适配 |

## 4. 升级说明

1. 从 v0.25 升级时，将 mcoplib 从 `0.4.10` 升级到 `0.4.11`。
2. 自定义扩展需更新 `registry/quant_config`、`registry/custom_ops` 和 `patch/enhancement` 的导入与注册路径。
3. 源码构建前处理 PyTorch、tilelang / TVM FFI 的依赖冲突，详见下表。
4. 环境变量不再默认强制覆盖，自定义部署需显式设置所需变量。

## 5. 已知问题

| 项目 | 说明 |
| --- | --- |
| PyTorch 依赖不一致 | 快照 `615ac62` 的 `build.txt` 固定 `2.10.0+metax3.8.0.7`，运行依赖固定 `2.10.0+metax3.8.2.2`；构建前需对齐 |
| tilelang / TVM FFI 冲突 | `maca.txt` 固定 `0.1.9+maca` / `0.1.3.dev11`，`maca_private.txt` 固定 `0.1.12+maca` / `0.1.11+maca`；构建前统一版本 |
