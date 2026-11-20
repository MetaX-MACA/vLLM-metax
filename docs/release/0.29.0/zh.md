# vLLM-MetaX v0.29.0 发布说明

[English](/release/0.29.0/en.md){ .md-button }

发布线：`vLLM-metax-0.29.0_rel3.9.0` · 上游基线：vLLM v0.29.0

本版本将 vLLM-MetaX 对齐到上游 vLLM v0.29.0，并完成 MACA 3.9.0 发布线。主要更新覆盖 v0.29 注意力与模型接口、DeepSeek-V4 / DSpark、MiniMax-M3、MoE 量化、MRV2 稳定性以及 MACA 3.9.0 依赖栈。

## 1. 主要更新

- **对齐上游 v0.29.0**：同步平台接口、模型注册、注意力后端、量化配置、Modular MoE、Patch 加载机制和模型运行器集成到 vLLM v0.29.0 基线。
- **DeepSeek-V4 / DSpark**：更新 MLA、MTP、DSpark 和 indexer 路径；QuantFP8 group quantization 走融合 MACA 内核；mHC 适配 MACA TileLang 格式；`main_proj` 的量化行为由 MetaX 配置控制。
- **注意力与分布式执行**：修复 GQA DCP kernel trap 和 MLA prefill `max_query_len` 超报；MiMo V2 使用 `FLASH_ATTN_DIFFKV` 后端；增加 MRV2 DP + MTP 图捕获挂起规避。
- **MoE 与量化**：MCTlass MoE GEMM 增加 `filter_expert` 支持；AWQ 适配 vLLM v0.29 的 `match_mode` 接口；更新 compressed-tensors WNA16 集成。
- **模型接入**：增加 Qwen3.8-Flash-Next 初步接入；修复 MiniMax-M3 VL / Transformers 兼容和权重加载；恢复 DeepSeek-V3.2 MTP indexer forward patch。
- **运行时与工具**：启用优化后的 Gumbel sample 内核；修复 resampled token 插入越界；MetaX 默认关闭 CUDA Graph 显存估算；同步发布依赖集合。

## 2. 版本与镜像

| 插件 | 发布线 | MACA 系列 | PyTorch 系列 | mcoplib | 镜像下载页 |
| --- | --- | --- | --- | --- | --- |
| v0.29.0 | `vLLM-metax-0.29.0_rel3.9.0` | 3.9.0.x | 2.10 | 0.4.14 | [vllm-metax:0.29.0](https://developer.metax-tech.com/softnova/docker?chip_name=MXC500&package_kind=AI&dimension=docker&deliver_type=%E5%88%86%E5%B1%82%E5%8C%85&ai_frame=vllm-metax&frame_version=0.29.0&system=ubuntu&arch=amd64) |

源码分支固定的发布线依赖如下：

| 组件 | 版本 |
| --- | --- |
| torch | `2.10.0+metax3.9.0.8` |
| mcoplib | `0.4.14+gf3794c4.maca3.9.0.31.torch2.10` |
| deep_ep | `1.0.0+metax3.9.0.11torch2.10` |
| deep_gemm | `0.4.0+maca3.9.2.3` |
| apache-tvm-ffi | `0.1.11+maca` |
| tilelang | `0.1.12+maca` |
| mixl | `1.0.0+maca3.9.0.10` |
| transformers | `>= 5.10.4` |

请使用发布页提供的配套镜像和依赖包。镜像的精确 tag 与驱动构建号以发布页为准，不要将本文的源码依赖与未经确认的镜像组合使用。

镜像部署、环境检查和服务验证见 [部署与验证指南](/getting_started/deployment/zh.md)。

## 3. 模型更新与量化

| 模型 / 路径 | 本版本变化 |
| --- | --- |
| DeepSeek-V4 / DeepSeek-V4-Flash | 更新 DSpark、MTP、MLA、indexer、QuantFP8 与 mHC 路径；TP 必须整除 checkpoint 的 `o_groups` |
| Qwen3.8-Flash-Next | 增加初步 MetaX 接入；生产使用前需用发布权重完成验证 |
| MiniMax-M3 | 修复 VL / Transformers 兼容和权重加载；QK-norm all-reduce fusion 默认仍关闭 |
| DeepSeek-V3.2 MTP | 恢复 v0.29 模型路径所需的 indexer forward patch |
| MiMo V2 | 使用 `FLASH_ATTN_DIFFKV`，并应用 MRV2 DP + MTP 图捕获挂起规避 |
| 量化 MoE | 增加 MCTlass `filter_expert`；更新 FP8 group quantization、compressed-tensors WNA16 和 AWQ layer skipping |

上游 v0.29.0 还新增或迁移了若干模型。Hy4-preview、Granite SWA、NemotronH Omni、Kimi K3 NVFP4、FlexOlmo / Olmo3 以及 Hunyuan V1 / VL 在本发布线尚未完成 MetaX 验证；在获得权重并完成验证前，不应将其列入已支持模型。

## 4. 升级说明

1. 升级到 MACA 3.9.0.x 与 PyTorch `2.10.0+metax3.9.0.8` 工具链，并基于该工具链重新编译本地自定义算子或 csrc。
2. 使用 mcoplib `0.4.14+gf3794c4.maca3.9.0.31.torch2.10`。本地 csrc 与不同构建的 mcoplib 混用可能导致内核符号缺失。
3. 源码安装时检出 `vLLM-metax-0.29.0_rel3.9.0`，安装与该分支匹配的 `requirements/`；connector 更名后需重新核对 `mixl`、Mooncake 和 LMCache 版本。
4. 与 v0.28 做性能对比前先固定模型运行器与显存参数：Mamba prefix caching 和 `max_num_batched_tokens` 默认值会影响吞吐与时延，MetaX 默认已关闭 CUDA Graph 显存估算。
5. 升级后需要回归 LoRA MoE 和推测解码路径。Hunyuan Transformers 后端当前需要 `--enforce-eager`；LoRA MoE 小批量路径可将 `--max-lora-rank` 设为大于 64 以规避快路径。

## 5. 已知问题

| 项目 | 说明 |
| --- | --- |
| 源码依赖 | 发布镜像与源码快照可能使用不同的本地构建号；请以具体分支或镜像的依赖文件为准 |
| Hunyuan V1 / HunyuanOCR | Transformers 后端在 CUDA Graph 捕获时可能失败；在 MACA 侧修复前使用 `--enforce-eager` |
| LoRA MoE | 小批量快路径可能在捕获阶段触发 XNACK / ATU fault；暂用 `--max-lora-rank 128` 规避并回归精度 |
| MLA + MTP | 部分 DeepSeek-R1 W4A8 / MLA + MTP 配置在 warmup decode 阶段仍可能触发 ATU fault；问题解决前不要纳入 v0.29.0 支持基线 |
| FlashAttention FP8 KV cache | MACA 当前只提供 FA2；依赖 FA3 的上游 FP8 KV cache 路径不可用 |
| DeepSeek-V4 TP | TP 必须整除 checkpoint 的 `o_groups`；8 组 Flash checkpoint 请使用 TP8 x PP2 或 TP8 x DP2 |
| 上游新增模型 | 未完成 MetaX 验证或缺少权重的模型不计入已支持模型 |
| MiniMax-M3 QK-norm fusion | all-reduce fusion 默认关闭；如需启用必须做精度回归 |

## 6. 相关文档

- **部署与验证**：[部署与验证指南](/getting_started/deployment/zh.md)
- **支持模型**：[支持模型列表](/models/supported_models.md)
- **Patch 管理**：[Patches](/developing/patches/README.md)
- **模型注册**：[模型注册指南](/developing/model/registration.md)
- **精度调测**：[msprobe 指南](/developing/msprobe/msprobe_guide.md)
- **插件架构**：[Contributing / 插件系统](/developing/README.md)
