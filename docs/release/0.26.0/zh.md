# vLLM-MetaX v0.26.0 使用指导手册

[English](/release/0.26.0/en.md){ .md-button }

> 本文是 vLLM-MetaX v0.26.0 版本随版发布的用户指导手册，面向在 MetaX GPU（MACA）上部署大模型的用户与交付团队。
> 从 v0.24.0 起，每个发布版本都会附一份对应的指导手册，统一记录在 `docs/releases/` 下。
>
> 仓库地址：https://github.com/MetaX-MACA/vLLM-metax

vLLM-MetaX 是 vLLM 的硬件后端插件（plugin），遵循 vLLM 的硬件可插拔 RFC（#11162、#19161），以插件形式接入上游 vLLM。本文是 MetaX 平台上的**快速上手与版本说明**，通用 vLLM 用法、从源码构建、精度调测等深入内容请参考文末「相关文档」中的对应链接。

## 版本兼容矩阵（重要）

vLLM-MetaX、mcoplib（内核组件）、MACA 驱动、torch 四者**版本强耦合**，请严格使用匹配组合，混用非对应版本可能导致编译失败、精度异常或运行期崩溃，且不保证可支持。

| plugin version | maca version | mcoplib version | docker image url |
| :---: | :---: | :---: | :---: |
| v0.26.0 | maca3.8.2.x | 0.4.11 | [vllm-metax:0.26.0](https://developer.metax-tech.com/softnova/docker?package_name=vllm-metax:0.26.0-torch2.10) |

> 注：v0.16.0 为官方跳过的版本，请勿使用。完整版本映射与镜像下载见 [Releases 总表](/getting_started/quickstart.md#releases)。
## 1. 在 Metax 平台上使用 vLLM-MetaX

你可以选择构建自己的镜像，也可以直接拉取已有的预构建镜像。本文以拉取预构建镜像为例。

### 1.1. 拉取镜像

Get your docker image [here](/getting_started/quickstart.md#releases)

### 1.2. 启动容器

```bash
# 必须通过 --device 挂载 Metax GPU 设备：--device=/dev/dri --device=/dev/mxcd
# 推理服务需要暴露端口，这里用 --network=host 方便直接访问
docker run -it --net=host --uts=host --ipc=host --privileged=true --group-add video \
    --shm-size '100gb' --ulimit memlock=-1 \
    --security-opt seccomp=unconfined --security-opt apparmor=unconfined \
    --device=/dev/dri --device=/dev/mxcd \
    -v /root/workspace:/external \
    -v /model_weights:/model_weights \
    --name vllm_metax_test \
    cr.metax-tech.com/public-ai-release/maca/vllm-metax:0.26.0-maca.ai3.8.2.2-torch2.10-py312-ubuntu22.04-amd64
```

> 更多信息请参考官方文档：https://developer.metax-tech.com
> 从源码构建、环境变量（MACA_PATH / CUCC_PATH 等）与构建注意事项，详见 [MACA 安装指南](/getting_started/installation/maca.md)。

## 2. 环境检查

### 2.1. 检查 Metax GPU 是否可用

得益于与 CUDA 的兼容性，可以像使用 NVIDIA GPU 一样检查 Metax 设备是否可用：

```python
import torch
print(torch.cuda.is_available())
# True
```

### 2.2. 检查 GPU 之间的 P2P 连接拓扑

```bash
mx-smi topo -m
# output
=================== MetaX System Management Interface Log ===================
...
Attached GPUs                                     : 8
        GPU0    GPU1    GPU2    GPU3    GPU4    GPU5    GPU6    GPU7
GPU0    X       MX      MX      MX      SYS     SYS     SYS     SYS
GPU1    MX      X       MX      MX      SYS     SYS     SYS     SYS
GPU2    MX      MX      X       MX      SYS     SYS     SYS     SYS
GPU3    MX      MX      MX      X       SYS     SYS     SYS     SYS
GPU4    SYS     SYS     SYS     SYS      X      MX      MX      MX
GPU5    SYS     SYS     SYS     SYS     MX      X       MX      MX
GPU6    SYS     SYS     SYS     SYS     MX      MX      X       MX
GPU7    SYS     SYS     SYS     SYS     MX      MX      MX      X
Legend:
X    = Self
MX   = Connection traversing MetaXLink
SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes
...
```

### 2.3. 查看 GPU 状态

```bash
mx-smi
# output
=================== MetaX System Management Interface Log ===================
Attached GPUs                                     : 8
+-------------------------+-----------------+---------------------+----------------------+
| Board Name   | GPU Persist-M | Bus-id              | GPU-Util      sGPU-M |
| Pwr:Usage/Cap| Temp   Perf  | Memory-Usage        | GPU-State            |
|-------------------------+-----------------+---------------------+----------------------|
| 0  MetaX C500 | 0  Off | 0000:0e:00.0 | 0%  Disabled |
| 57W / 350W   | 35C  P0  | 826/65536 MiB | Available            |
...
```

## 3. 运行示例

镜像内已包含匹配版本的 vLLM-MetaX 与 mcoplib，可直接启动 OpenAI 兼容推理服务。

### 3.1. 启动 vLLM 推理服务

```bash
# 以 Qwen3-8B 为例，请根据实际模型路径修改
vllm serve /model_weights/Qwen/Qwen3-8B \
    --tensor-parallel-size 1 \
    --host 0.0.0.0 --port 8000
```

服务启动成功的关键日志（节选）：

```text
INFO 03-09 15:43:01 [api_server.py] Starting vLLM API server on http://0.0.0.0:8000
INFO 03-09 15:43:05 [core.py] Initializing a V1 LLM engine (v0.26.0) with config: ...
INFO 03-09 15:43:12 [maca_platform.py] MACA backend initialized, driver maca3.8.2.2
INFO 03-09 15:43:35 [gpu_model_runner.py] Model loading took 23.41 GiB and 22.65 seconds
INFO 03-09 15:43:36 [api_server.py] Application startup complete.
```

### 3.2. 发送推理请求

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "/model_weights/Qwen/Qwen3-8B",
        "prompt": "MetaX 是一家",
        "max_tokens": 32
    }'
```

返回示例：

```json
{
  "id": "cmpl-xxxxxxxx",
  "object": "text_completion",
  "model": "/model_weights/Qwen/Qwen3-8B",
  "choices": [
    { "text": "专注于高性能 AI 计算的芯片公司，其 MACA 软件栈兼容 CUDA 生态。", "index": 0, "finish_reason": "length" }
  ],
  "usage": { "prompt_tokens": 4, "completion_tokens": 32, "total_tokens": 36 }
}
```

> 更多启动参数、并行策略（tensor/pipeline/expert parallel）、量化加载等通用用法，参考 [vLLM 官方文档](https://docs.vllm.ai/)。本手册 3.1 仅演示最小可运行示例。

### 3.3. 使用其他版本 / 从源码构建

如需构建其他版本或定制，需使用与 MACA 兼容的软件包。编译依赖 torch 2.10（metax3.8.2.2），建议使用 `--no-deps` 避免覆盖环境内已有 PyTorch：

```bash
git clone -b mx/v0.26.0-dev https://github.com/MetaX-MACA/vLLM-metax.git
cd vLLM-metax
pip install -r requirements/maca.txt -r requirements/maca_private.txt
pip install -e . --no-deps
```

环境变更后请检查 PyTorch 版本及其可用性：

```bash
pip list | grep torch
# torch                    2.10.0+metax3.8.2.2
```

```python
import torch
torch.cuda.is_available()  # True
```

> 完整构建流程（含 vllm 本体用 empty device 构建、USE_PRECOMPILED_KERNEL 含义、生产必须用 mcoplib 等注意事项）见 [MACA 安装指南](/getting_started/installation/maca.md)。

## 4. v0.26.0 本版本重点新特性

- **同步上游 vLLM v0.26.0**
  - 对齐 vLLM v0.26.0 基线，同步上游接口并重构 patch 体系（`@patch` 模板 / `patch/utils`）
- **DeepSeek-V4 / MLA**
  - `deepseek_v4` 在 v0.26.0 上完成适配与重构（sparse attention / MLA、MTP、Eagle、draft config 等）
  - 修复 SparseAttnIndexer registry 缺失、MTP 相关 import 丢失、`draft_config_overrides` import、`flashmla_sparse` 同步
  - telechat3-yarn（CCPM-9709）`max_model_len` 推导修复
  - cumem 内存分配器适配（MC3-14374）并新增 `is_cumem_allocator_available` 平台开关
- **MiniMax-M3**
  - topk / shared-mem 系列修复：backport 上游 #49149（topk buffer 布局，v0.27.0 起可移除对应 patch）、index_topk 与 sparse attention 补丁重构
  - 修复 `int8_w8a8` swiglu 配置缺失（gemm1 alpha / beta / clamp_limit）
  - `minimax_qk_norm` all-reduce fusion 默认关闭（见第 8 节）
- **MoE / 内核**
  - fused MoE scale 加载修复：mirror 上游 PR #50137，并补齐 transposed per-channel scale 归一化（落到 `int8_w8a8/load_weight.py`）
  - MCTlass fused MoE：修复 FP8 使用、跳过无效 EP experts
  - deep_ep：FP8 / int8 W8A8 量化路径（`BatchedTritonExperts` / `BatchedDeepGemmExperts`）、deepep high-throughput 与 dbo 开关
  - w8a8 int8 加载修复
- **JoyAI**：`joyai_llm_flash` 误识别为 `deepseek_mtp` 修复；tokenizer decoder 改用 BPE
- **分布式**：MxMesh low-latency all-to-all、all-gather 卡死修复、环境变量不再默认强制覆盖
- **构建 / 依赖**：maca3.8.2.2 工具链；tilelang 0.1.12+maca、apache-tvm-ffi 0.1.11+maca、xgrammar 0.2.x；mcoplib 0.4.11；tvm / tilelang 改走内部 pip 源
- **结构重构**：monkey patch 体系重构；量化配置 / 自定义算子 / 插件增强迁移到 `registry/quant_config`、`registry/custom_ops`、`patch/enhancement`

## 5. 新增与更新的模型

以下模型为 v0.26.0 新增或显著更新（部分尚未回填 `docs/models/supported_models.md`，以本手册为准）：

### 文本 / 推理模型
- **DeepSeek-V4**：在 v0.26.0 上完成适配重构（sparse attention / MLA、MTP、Eagle），并修复 telechat3-yarn 等加载与推导问题
- **MiniMax-M3**：W8A8 topk / shared-mem / sparse attention 修复，`int8_w8a8` swiglu 配置补全，`minimax_qk_norm` all-reduce fusion 默认关闭
- **DeepSeek 系列**：deep_ep FP8 / int8 量化支持（`BatchedDeepGemmExperts`）、Eagle3 / MTP、`flashmla_sparse` 更新
- **京东 JoyAI**：`JoyAI_LLM_Flash`（jd）识别与 tokenizer decode 修复，支持代码迁移至 `patch/enhancement/joyai_support`
- **MRV2 / step3p5**：EP + PCP 支持
- **GLM 系列**：延续 v0.25 适配

### 多模态模型
- v0.24 引入的 Qwen3-Omni / Qwen3-VL、Intern-S1 / InternVL 3.5 随上游 v0.26.0 基线继续支持；本版本无新增 Metax 侧多模态架构

### 量化模型支持
- FP8 / int8_w8a8 / W8A8 推理；fused MoE W4A8（自 v0.25 引入，随 registry 重构保留）；MiniMax-M3 int8 swiglu 修复

> 全部已测试模型与特性状态（LoRA、PP 等）见 [支持模型列表](/models/supported_models.md)。
> 若需接入列表之外的模型或自定义模型，参考 [模型注册指南](/developing/model/registration.md)。

## 6. 从 v0.25 升级迁移

1. **更换镜像 / 环境**：docker 用户直接切到 v0.26.0 镜像；源码用户重新 checkout `mx/v0.26.0-dev` 并按 3.3 重装依赖。
2. **torch / MACA 升级**：v0.26.0 使用 torch 2.10（metax3.8.2.2，v0.25.0 为 metax3.8.2.0）。自定义算子 / 本地编译的 csrc 需针对新工具链重新编译。
3. **mcoplib 同步升级**：v0.25.0 为 `0.4.10+g72792e0.maca3.8.0.25.torch2.10`，v0.26.0 为 `0.4.11+ge29f792.maca3.8.0.25.torch2.10`，否则内核符号不匹配。
4. **tilelang / tvm-ffi / xgrammar 组合更新**：v0.26.0 使用 tilelang 0.1.12+maca、apache-tvm-ffi 0.1.11+maca、xgrammar 0.2.x；v0.25.0 为 dsv4 兼容使用的降级组合（tilelang 0.1.9+maca、xgrammar 0.1.32）不再适用。
5. **patch / registry 结构重构**：quant config、custom ops、插件增强分别迁至 `registry/quant_config`、`registry/custom_ops`、`patch/enhancement`；基于旧路径（`patch/plugin_enhancement`、`customized/*`、`transformers_utils` 等）的自定义扩展需要改注册路径。
6. **fused MoE scale 加载修复位置变化**：PR #50137 相关修复与 transposed scale 归一化统一在 `patch/bugfix/int8_w8a8/load_weight.py`。
7. **MiniMax-M3 补丁独立化**：topk / indexer / sparse 相关修复在 0.26 为独立实现（#49149 backport），vLLM 依赖升到 v0.27.0 后应移除。
8. **投机解码**：eagle / MTP 继续走 patch 与 draft config 覆盖，升级后需回归测试 eagle 提议精度。

## 7. Metax 与 NVIDIA CUDA 的差异

Metax 在大部分接口上与 NVIDIA 对齐，但在某些软件行为和环境变量上存在差异。

### 7.1. MACA_MPS_MODE（多进程共享 GPU）

默认情况下，MACA 不允许多个进程共享同一块 GPU。如果 GPU 已被占用，则无法启动新进程。

如需启用类似 MPS（Multi-Process Service）的功能，需设置：`MACA_MPS_MODE=1`

```bash
export MACA_MPS_MODE=1
vllm serve /model_weights/Qwen/Qwen3-8B --tensor-parallel-size 1
```

### 7.2. 多节点部署

vLLM 分布式（tensor/pipeline/expert parallel）跨节点时，建议设置以下环境变量以确保节点间通信正常：
- `MCCL_SOCKET_IFNAME`：用于 MCCL 通信的网络接口
- `GLOO_SOCKET_IFNAME`：用于 GLOO 通信的网络接口
- `MCCL_IB_HCA`：指定使用的 InfiniBand 设备

可通过 `ifconfig` 和 `mx-smi topo -n` 确定所用网卡和 IB 设备（参考 verl / Megatron 文档中的拓扑示例）。推荐设置：

```bash
export MCCL_SOCKET_IFNAME=ens20f0np0
export GLOO_SOCKET_IFNAME=ens20f0np0
export MCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3
```

> ⚠️ 请根据自己机器的 `ifconfig` 和 `mx-smi topo -n` 输出进行相应调整，切勿盲目复制上述命令。

## 8. 已知问题与规避

| 问题 | 影响 | 规避 / 状态 |
|------|------|------------|
| MiniMax-M3 topk buffer 布局（upstream #49147，backport #49149） | v0.24.0–v0.26.0 均受影响，混合 decode + prefill 时可能触发非法内存访问 | 已修复（indexer / sparse_attention patch）；vLLM 依赖升到 v0.27.0 后移除 |
| `minimax_qk_norm` all-reduce fusion 默认关闭 | 该融合算子默认不启用，相关模型性能略低 | 如需开启请修改 `rms_norm_tp` 相关配置并回归精度 |
| deep_ep FP8 / int8、W4A8 等量化路径 | 新引入 / 重构路径，需回归验证精度 | 精度异常时用 msprobe 抓取对比（[指南](/developing/msprobe/msprobe_guide.md)） |
| transformers v5 下部分 tokenizer（Moonlight / Kimi 等） | 旧 import 路径（`bytes_to_unicode`）在 transformers v5 被移除 | 已重新注入兼容实现；仍异常时锁定 transformers 版本 |
| 本地编译 csrc 与预编译 mcoplib 差异 | 本地编译需与 mcoplib 0.4.11 的内核符号匹配 | 生产环境建议使用预编译 mcoplib 组合 |
| tilelang / tvm-ffi / xgrammar 强匹配 | 混用组合可能导致内核加载或 grammar 异常 | 严格按 `requirements/` 中的组合安装 |
| v0.16.0 被官方跳过 | 无此版本 | 请勿使用 v0.16.x |

## 9. 相关文档

- **版本映射总表**：[Releases](/getting_started/quickstart.md#releases)
- **支持模型列表**：[Supported Models](/models/supported_models.md)
- **从源码安装 / 构建**：[MACA 安装指南](/getting_started/installation/maca.md)
- **精度调测（msprobe）**：[msprobe 精度调测指南](/developing/msprobe/msprobe_guide.md)
- **Patch 管理（monkey / git patches）**：[Patches](/developing/patches/README.md)
- **接入自定义 / 新模型**：[模型注册指南](/developing/model/registration.md)
- **插件架构与贡献**：[Contributing / 插件系统](/developing/README.md)
- mcoplib 仓库：https://github.com/MetaX-MACA/mcoplib
- 插件主页 / 源码：https://github.com/MetaX-MACA/vLLM-metax
- 开发者社区 / 镜像下载：https://developer.metax-tech.com
- vLLM 官方文档：https://docs.vllm.ai/

---
*发布日期：TODO（填入实际发布日期）｜维护：PDE/AI vLLM-MetaX 团队*
