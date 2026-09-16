# vLLM-MetaX v0.25.0 使用指导手册

[English](/release/0.25.0/en.md){ .md-button }

> 本文是 vLLM-MetaX v0.25.0 版本随版发布的用户指导手册，面向在 MetaX GPU（MACA）上部署大模型的用户与交付团队。
> 从 v0.24.0 起，每个发布版本都会附一份对应的指导手册，统一记录在 `docs/releases/` 下。
>
> 仓库地址：<https://github.com/MetaX-MACA/vLLM-metax>

vLLM-MetaX 是 vLLM 的硬件后端插件（plugin），遵循 vLLM 的硬件可插拔 RFC（#11162、#19161），以插件形式接入上游 vLLM。本文是 MetaX 平台上的**快速上手与版本说明**，通用 vLLM 用法、从源码构建、精度调测等深入内容请参考文末「相关文档」中的对应链接。

## 版本兼容矩阵（重要）

vLLM-MetaX、mcoplib（内核组件）、MACA 驱动、torch 四者**版本强耦合**，请严格使用匹配组合，混用非对应版本可能导致编译失败、精度异常或运行期崩溃，且不保证可支持。

| plugin version | maca version | mcoplib version | docker image url |
| :---: | :---: | :---: | :---: |
| v0.23.0 | maca3.8.0.x | 0.4.8 | [vllm-metax:0.23.0](https://developer.metax-tech.com/softnova/docker?package_name=vllm-metax:0.23.0-torch2.10) |
| v0.24.0 | maca3.8.2.x | 0.4.9 | [vllm-metax:0.24.0](https://developer.metax-tech.com/softnova/docker?package_name=vllm-metax:0.24.0-torch2.10) |
| v0.25.0 | maca3.8.2.x | 0.4.10 | [vllm-metax:0.25.0](https://developer.metax-tech.com/softnova/docker?package_name=vllm-metax:0.25.0-torch2.10) |

> 注：v0.16.0 为官方跳过的版本，请勿使用。完整版本映射与镜像下载见 [Releases 总表](/getting_started/quickstart.md#releases)。镜像 URL 模板：
> `https://developer.metax-tech.com/softnova/docker?package_name=vllm-metax:0.25.0-torch2.10`

## 1. 在 Metax 平台上使用 vLLM-MetaX

你可以选择构建自己的镜像，也可以直接拉取已有的预构建镜像。本文以拉取预构建镜像为例。

### 1.1. 拉取镜像

```bash
docker pull cr.metax-tech.com/public-ai-release/maca/vllm-metax:0.25.0-maca.ai3.8.2.0-torch2.10-py310-ubuntu22.04-amd64
# 示例 tag 规则（以实际发布页为准）：
#   0.25.0            -> 插件版本
#   maca.ai3.8.2.0    -> MACA 驱动版本
#   torch2.10         -> torch 版本
#   py310-ubuntu22.04 -> 基础环境
```

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
    cr.metax-tech.com/public-ai-release/maca/vllm-metax:0.25.0-maca.ai3.8.2.0-torch2.10-py310-ubuntu22.04-amd64
```

> 更多信息请参考官方文档：<https://developer.metax-tech.com>
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
INFO 03-09 15:43:05 [core.py] Initializing a V1 LLM engine (v0.25.0) with config: ...
INFO 03-09 15:43:12 [maca_platform.py] MACA backend initialized, driver maca3.8.2.0
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

如需构建其他版本或定制，需使用与 MACA 兼容的软件包。编译依赖 torch 2.10（metax3.8.x），建议使用 `--no-deps` 避免覆盖环境内已有 PyTorch：

```bash
git clone -b mx/v0.25.0-dev https://github.com/MetaX-MACA/vLLM-metax.git
cd vLLM-metax
pip install -r requirements/maca.txt -r requirements/maca_private.txt
pip install -e . --no-deps
```

环境变更后请检查 PyTorch 版本及其可用性：

```bash
pip list | grep torch
# torch                    2.10.0+metax3.8.2.0
```

```python
import torch
torch.cuda.is_available()  # True
```

> 完整构建流程（含 vllm 本体用 empty device 构建、USE_PRECOMPILED_KERNEL 含义、生产必须用 mcoplib 等注意事项）见 [MACA 安装指南](/getting_started/installation/maca.md)。

## 4. v0.25.0 本版本重点新特性

- **同步上游 vLLM v0.25.0**
    - 对齐 vLLM v0.25.0 基线（PR #326）
    - cherry-pick 回 v0.24.0 分支的修复/特性，并 backport 上游 PR #50137（fused 权重 scale 加载）
- **MoE 内核增强**
    - deep_ep 在 maca 上的支持落地（原为 NVIDIA-only），并新增 **FP8**（`BatchedDeepGemmExperts`）与 **int8** 两条量化路径
    - 新增 fused MoE **W4A8** 支持
    - 修复 `mctlass` fused MoE FP8 使用问题
    - 新增 Modular MoE 输出别名（MetaX 专用 patch）
    - 修复 fused MoE per-channel scales 转置加载前的归一化
    - backport 上游 #36701（`get_supported_kernel_block_sizes` 不再调用 `get_current_vllm_config`）
- **flashinfer**：移除 metax flashinfer wrapper 不支持的 backend 参数
- **投机解码**：eagle 相关改动以 patch 形式回归（v0.24 曾回退，见第 8 节）
- **模型 patch**：新增 Qwen3.5 W8A8 权重加载修复；MiniMax-M3 修复 `int8_w8a8` swiglu 配置
- **构建系统**：csrc 同步 mcoplib v0.25.0 适配；修复本地编译 csrc 在 MACA 上运行、`USE_PRECOMPILED_KERNEL=1` 构建报错；requirements 更新支持 v0.25.0 自动安装（auto-install）；Dockerfile 更新

## 5. 新增与更新的模型

以下模型为 v0.25.0 新增或显著更新（部分尚未回填 `docs/models/supported_models.md`，以本手册为准）：

### 文本/推理模型

- **Qwen3.5 系列**：新增支持，含 W8A8 权重加载修复
- **MiniMax-M3**：修复 `int8_w8a8` swiglu 配置缺失（FP8 路径正常，int8 路径此前报错）；`minimax_qk_norm` all-reduce fusion 默认关闭（见第 8 节）
- **DeepSeek 系列**：deep_ep 新增 FP8 / int8 量化支持（`BatchedDeepGemmExperts`）
- **京东 JoyAI**：`JoyAI_LLM_Flash`（jd）随 flashinfer 修复继续完善
- **GLM 系列**：延续 v0.24 适配

### 多模态模型

- v0.24 引入的 Qwen3-Omni / Qwen3-VL、Intern-S1 / InternVL 3.5 继续支持，本版本无新增多模态架构

### 量化模型支持

- FP8 推理；fused MoE **W4A8**（新增）；int8_w8a8（MiniMax-M3 swiglu 修复）；W8A8（Qwen3.5 加载修复）

> 全部已测试模型与特性状态（LoRA、PP 等）见 [支持模型列表](/models/supported_models.md)。
> 若需接入列表之外的模型或自定义模型，参考 [模型注册指南](/developing/model/registration.md)。

## 6. 从 v0.24 升级迁移

1. **更换镜像 / 环境**：docker 用户直接切到 v0.25.0 镜像；源码用户重新 checkout `releases/v0.25.0` 并按 3.3 重装依赖。
2. **mcoplib 需同步升级**到与 v0.25.0 匹配的版本（csrc 已同步 mcoplib v0.25.0 适配；v0.24 为 `0.4.9+maca3.8.0.25.torch2.10`，v0.25.0 对应版本见上方版本矩阵），否则内核符号不匹配。
3. **torch / MACA 版本不变**：v0.25.0 与 v0.24.0 同为 torch 2.10（metax3.8.2.0），自定义算子 / patch 无需因 torch 重新编译（仍需按第 2 条同步 mcoplib）。
4. **transformers patch 移除**：v0.25.0 移除了冗余的 transformers patch（`model_arch_config_convertor`）。若此前依赖该 patch 的模型出现异常，请检查 transformers 版本兼容性。
5. **投机解码路径变化**：eagle / MTP 相关改动在 v0.24 曾回退，v0.25.0 以 patch 形式回归，升级后需回归测试 eagle 提议精度。
6. **deep_ep 新增量化路径**：FP8 / int8 权重加载走 `BatchedDeepGemmExperts`，若自定义了 deep_ep 量化流程，需按新路径适配。

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
| ------ | ------ | ------------ |
| `minimax_qk_norm` all-reduce fusion 默认关闭（v0.25.0 起） | 该融合算子默认不启用，相关模型性能略低 | 如需开启请修改 `rms_norm_tp` 相关配置并回归精度 |
| deep_ep FP8 / int8 为新支持路径 | 新引入，需回归验证精度 | 精度异常时用 msprobe 抓取对比（[指南](/developing/msprobe/msprobe_guide.md)） |
| transformers v5 下部分 tokenizer 空格/换行异常 | v0.24 修复的 Llama 类问题依赖 patch；v0.25 移除 transformers patch 后其他架构仍可能受影响 | 锁定 transformers 版本或应用对应 patch |
| v0.16.0 被官方跳过 | 无此版本 | 请勿使用 v0.16.x |
| eagle / MTP 投机解码相关改动以 patch 形式回归 | 投机解码路径不稳定窗口 | 升级后回归 eagle 提议精度 |

---
发布日期：2026.9.4｜维护：PDE/AI vLLM-MetaX 团队
