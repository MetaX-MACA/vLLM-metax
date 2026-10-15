# vLLM-MetaX v0.25.0 发布说明

[English](/release/0.25.0/en.md){ .md-button }

发布日期：2026-09-04 · 源码分支：[releases/v0.25.0](https://github.com/MetaX-MACA/vLLM-metax/tree/releases/v0.25.0)

本版本对齐上游 vLLM v0.25.0，重点更新 DeepEP 量化路径、MoE 权重加载及模型兼容补丁。

## 1. 主要更新

- **MoE 与 DeepEP**：扩展 FP8 / INT8 量化路径，增加 `BatchedDeepGemmExperts` 相关实现与 Modular MoE 输出别名适配。
- **权重加载**：修复 fused MoE per-channel scale 转置处理，补充 MiniMax-M3 和 Qwen3.5 的加载补丁。
- **注意力与投机解码**：移除 MetaX flashinfer wrapper 不支持的 backend 参数，更新 Eagle 兼容补丁。
- **工程支持**：更新 csrc、依赖与 Dockerfile，并同步 KV offload 测试。

## 2. 版本与镜像

| 插件 | MACA 系列 | PyTorch 系列 | mcoplib | 镜像下载页 |
| --- | --- | --- | --- | --- |
| v0.25.0 | 3.8.2.x | 2.10 | 0.4.10 | [vllm-metax:0.25.0](https://developer.metax-tech.com/softnova/docker?chip_name=%E6%9B%A6%E4%BA%91C500%E7%B3%BB%E5%88%97&package_kind=AI&dimension=docker&deliver_type=%E5%88%86%E5%B1%82%E5%8C%85&ai_frame=vllm-metax&frame_version=0.25.0&arch=amd64&system=ubuntu) |

使用配套镜像；源码依赖见 [requirements](https://github.com/MetaX-MACA/vLLM-metax/tree/releases/v0.25.0/requirements)。

镜像部署、环境检查和服务验证见 [部署与验证指南](/release/common/common_zh.md)。

## 3. 模型更新与量化

| 模型 / 路径 | 本版本变化 |
| --- | --- |
| Qwen3.5 | 补充 MoE / W8A8 权重加载修复 |
| MiniMax-M3 | 修复 INT8 W8A8 的 SwiGLU 配置 |
| DeepSeek 系列 | 更新 DeepEP 与量化专家执行路径 |
| JoyAI | 保留 `JoyAI_LLM_Flash` 适配，更新兼容路径 |
| 多模态 | 延续 Qwen3-Omni、Qwen3-VL 与 Intern 系列适配 |

## 4. 升级说明

1. 从 v0.24 升级时，将 mcoplib 从 `0.4.9` 升级到 `0.4.10`。
2. v0.24 / v0.25 标签均使用 PyTorch `2.10.0+metax3.8.2.2`；自定义算子按实际镜像核对 ABI。
3. 自定义 DeepEP / MoE 流程需检查专家后端与 per-channel scale 加载。
4. transformers 和 Eagle / MTP 补丁有调整，升级后检查 tokenizer 及投机解码精度。

## 5. 已知问题

| 项目 | 说明 |
| --- | --- |
| 转置 fused W8A8 权重 | 检查 per-channel scale 的形状和方向，保留配套加载补丁 |
