# vLLM-MetaX v0.27.0 发布说明

[English](/release/0.27.0/en.md){ .md-button }

发布日期：2026-09-24 · 源码分支：[releases/v0.27.0](https://github.com/MetaX-MACA/vLLM-metax/tree/releases/v0.27.0)

本版本对齐上游 vLLM v0.27.0，重点更新 DeepSeek-V4、MiniMax 稀疏注意力、分布式通信与 sleep mode。

## 1. 主要更新

- **DeepSeek-V4**：更新模型、稀疏 MLA 与投机解码接口，修复 `_gather_k_cache_kernel` stride 顺序，默认关闭模型中的 FP8 KV cache layout。
- **MiniMax-M3**：修复 Triton 内核共享内存限制，禁用 mcoplib 不支持的 `cooperative_topk` / `persistent_topk`。
- **MoE**：修复 mctlass FP8，跳过无效专家；显式禁止 INT8 W8A8 使用非 batched DeepGEMM 路径。
- **分布式与内存**：增加 mxmesh 支持，修复 MCCL all-gather 卡死，新增用于 sleep mode 的 `MacaMemBackend`。
- **工具链适配**：更新 csrc、自定义算子与 attention backend，在 MACA 上禁用 `cudaLaunchAttribute`。

## 2. 版本与镜像

| 插件 | MACA 系列 | PyTorch 系列 | mcoplib | 镜像下载页 |
| --- | --- | --- | --- | --- |
| v0.27.0 | 3.8.2.x | 2.10 | 0.4.12 | [vllm-metax:0.27.0](https://developer.metax-tech.com/softnova/docker?chip_name=%E6%9B%A6%E4%BA%91C500%E7%B3%BB%E5%88%97&package_kind=AI&dimension=docker&deliver_type=%E5%88%86%E5%B1%82%E5%8C%85&ai_frame=vllm-metax&frame_version=0.27.0&arch=amd64&system=ubuntu) |

使用配套镜像；源码依赖见 [requirements](https://github.com/MetaX-MACA/vLLM-metax/tree/releases/v0.27.0/requirements)。

镜像部署、环境检查和服务验证见 [部署与验证指南](/release/common/common_zh.md)。

## 3. 模型更新与量化

| 模型 / 路径 | 本版本变化 |
| --- | --- |
| DeepSeek-V4 | 更新 sparse MLA、MTP 与模型接口，修复 cache gather stride |
| MiniMax-M3 | 更新稀疏注意力及 Triton top-k 内核 |
| JoyAI | 更新 `JoyAI_LLM_Flash` 的 v0.27 适配 |
| 量化 MoE | INT8 W8A8 的 DeepGEMM 实现仅支持 batched / EP 路径 |

## 4. 升级说明

1. 从 v0.26 升级时，使用 mcoplib `0.4.12` 对应的发布镜像。
2. 源码快照的组件版本与发布映射不同，源码安装前核对依赖文件。
3. INT8 W8A8 DeepGEMM 使用 batched / EP 路径。
4. sleep mode 使用 `MacaMemBackend`，自定义内存后端需更新配置。

## 5. 已知问题

| 项目 | 说明 |
| --- | --- |
| INT8 W8A8 DeepGEMM | 不支持非 batched 路径，使用 batched / EP 配置 |
| MiniMax 稀疏 top-k | 保持 `cooperative_topk` / `persistent_topk` 关闭 |
