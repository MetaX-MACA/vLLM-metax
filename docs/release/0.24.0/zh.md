# vLLM-MetaX v0.24.0 发布说明

[English](/release/0.24.0/en.md){ .md-button }

发布日期：2026-08-21 · 源码分支：[releases/v0.24.0](https://github.com/MetaX-MACA/vLLM-metax/tree/releases/v0.24.0)

本版本围绕 MoE 内核、专家并行、模型适配与精度调测完善 MetaX 后端。

## 1. 主要更新

- **MoE 与量化**：增加 mctlass FP8、mctlassEx W4A16（BF16 路径）及 fused MoE W4A8 支持；移除 inplace fused experts 机制。
- **专家并行**：适配 DeepEP 与 `BatchedTritonExperts`，补充专家过滤逻辑。
- **注意力**：更新 MLA prefill 与 FLA 路径。
- **精度与构建**：集成 msprobe；修复 `compressed_tensors` INT8 W8A8 精度问题，以及 `USE_PRECOMPILED_KERNEL=1` 构建错误。

## 2. 版本与镜像

| 插件 | MACA 系列 | PyTorch 系列 | mcoplib | 镜像下载页 |
| --- | --- | --- | --- | --- |
| v0.24.0 | 3.8.2.x | 2.10 | 0.4.9 | [vllm-metax:0.24.0](https://developer.metax-tech.com/softnova/docker?chip_name=%E6%9B%A6%E4%BA%91C500%E7%B3%BB%E5%88%97&package_kind=AI&dimension=docker&deliver_type=%E5%88%86%E5%B1%82%E5%8C%85&ai_frame=vllm-metax&frame_version=0.24.0&arch=amd64&system=ubuntu) |

使用配套镜像；源码依赖见 [requirements](https://github.com/MetaX-MACA/vLLM-metax/tree/releases/v0.24.0/requirements)。

镜像部署、环境检查和服务验证见 [部署与验证指南](/release/common/common_zh.md)。

## 3. 模型更新与量化

| 模型 / 路径 | 本版本变化 |
| --- | --- |
| MiniMax-M3 | 增加模型适配及 FP8 权重路径 |
| JoyAI | 注册 `JoyAI_LLM_Flash` |
| DeepSeek 系列 | 适配 DeepSeek-V4，更新 DeepSeek-V2 与 MLA 路径 |
| 多模态 | 更新 Qwen3-Omni、Qwen3-VL、Intern 系列适配；修复 Qwen2.5-Omni 的 `cu_seqlens` 问题 |
| 量化 | 增加 AWQ 配置与 `awq_to_gptq_4bit` 转换，修复 INT8 W8A8 路径 |

## 4. 升级说明

1. 从 v0.23 升级时，将 mcoplib 从 `0.4.8` 升级到 `0.4.9`，使用配套 MACA 环境。
2. PyTorch 仍为 MetaX 2.10 系列；工具链后缀变化时，重新编译自定义算子。
3. 自定义 MoE 扩展需适配非 inplace fused experts 路径。
4. 使用 EP、Eagle / MTP 或自定义 tokenizer 时，检查权重加载、生成精度及空格换行行为。

## 5. 已知问题

| 项目 | 说明 |
| --- | --- |
| MLA prefill | 优化覆盖特定 head dimension，性能需按模型测量 |
| Tokenizer | 空格或换行异常时，检查 transformers 版本和兼容补丁 |
