# vLLM-MetaX v0.24.0 使用指导手册

[English](/release/0.25.0/en.md){ .md-button }

> 本文是 vLLM-MetaX v0.24.0 版本随版发布的用户指导手册，面向在 MetaX GPU（MACA）上部署大模型的用户与交付团队。
> 从本版本起，每个发布版本都会附一份对应的指导手册，统一记录在 `docs/releases/` 下。
>
> 仓库地址：<https://github.com/MetaX-MACA/vLLM-metax>

vLLM-MetaX 是 vLLM 的硬件后端插件（plugin），遵循 vLLM 的硬件可插拔 RFC（#11162、#19161），以插件形式接入上游 vLLM。本文是 MetaX 平台上的**快速上手与版本说明**，通用 vLLM 用法、从源码构建、精度调测等深入内容请参考文末「相关文档」中的对应链接。

## 版本兼容矩阵（重要）

vLLM-MetaX、mcoplib（内核组件）、MACA 驱动、torch 四者**版本强耦合**，请严格使用匹配组合，混用非对应版本可能导致编译失败、精度异常或运行期崩溃，且不保证可支持。

| plugin 版本 | MACA 驱动 | torch (metax) | mcoplib | Docker 镜像 |
| :-----------: | :---------: | :-------------: | :-------: | :----------- |
| **v0.24.0** | **maca3.8.2.5** | **2.10.0** | **0.4.9** | [vllm-metax:0.24.0](https://developer.metax-tech.com/softnova/docker?chip_name=%E6%9B%A6%E4%BA%91C500%E7%B3%BB%E5%88%97&package_name=vllm-metax:0.24.0-maca.ai3.8.2.5-torch2.10-py310-ubuntu22.04-amd64) |

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
    cr.metax-tech.com/public-ai-release/maca/vllm-metax:0.24.0-maca.ai3.8.2.5-torch2.10-py310-ubuntu22.04-amd64
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
INFO 03-09 15:43:05 [core.py] Initializing a V1 LLM engine (v0.24.0) with config: ...
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

如需构建其他版本或定制，需使用与 MACA 兼容的软件包。编译依赖 2.10.0+metax3.8.2.5，建议使用 `--no-deps` 避免覆盖环境内已有 PyTorch：

```bash
git clone -b releases/v0.24.0 https://github.com/MetaX-MACA/vLLM-metax.git
cd vLLM-metax
pip install -r requirements/maca.txt -r requirements/maca_private.txt
pip install -e . --no-deps
```

环境变更后请检查 PyTorch 版本及其可用性：

```bash
pip list | grep torch
# torch                    2.10.0+metax3.8.2.5
```

```python
import torch
torch.cuda.is_available()  # True
```

> 完整构建流程（含 vllm 本体用 empty device 构建、USE_PRECOMPILED_KERNEL 含义、生产必须用 mcoplib 等注意事项）见 [MACA 安装指南](/getting_started/installation/maca.md)。

## 4. v0.24.0 本版本重点新特性

- **MoE 内核全面升级**
    - 新增 `mctlass` fused MoE FP8
    - 新增 `mctlassEx` W4A16 fused MoE（当前仅 bf16 路径）
    - 新增 fused MoE **W4A8** 支持
    - 移除 inplace fused experts 机制，降低精度风险
- **专家并行（EP）增强**：deep_ep 支持 `BatchedTritonExperts`；EP 路径新增 `filter_expert`
- **注意力优化**：MLA prefill 对部分 head_dim 优化；切换 FLA 路径
- **精度调测工具**：集成 `msprobe` 精度调试工具，便于定位数值偏差（用法见 [msprobe 精度调测指南](/developing/msprobe/msprobe_guide.md)）
- **构建系统**：requirements 重构支持自动安装（auto-install）；修复 `USE_PRECOMPILED_KERNEL=1` 构建报错

## 5. 新增与更新的模型

以下模型为 v0.24.0 新增或显著更新（部分尚未回填 `docs/models/supported_models.md`，以本手册为准）：

### 文本/推理模型

- **MiniMax-M3**：新增 `minimax-m3`，并新增 FP8 权重路径 `minimax-m3-fp8`
- **京东 JoyAI**：新增 `JoyAI_LLM_Flash`（jd）
- **DeepSeek-V4（dsv4）**：新增支持，含相关 MoE/attention 清理
- **DeepSeek 系列**：同步 `deepseek_v2` 与 v0.24.0 改动
- **GLM 系列**：粒度补充 GLM / GLM-4.x 适配

### 多模态模型

- **Qwen3-Omni / Qwen3-VL 系列**：新增对应架构
- **Intern-S1 / InternVL 3.5**：新增架构支持
- **Qwen2.5-Omni**：修复 `cu_seqlens` 导致的 CUDA error

### 量化模型支持

- FP8 推理；AWQ（`auto_awq` 与 `awq_to_gptq_4bit`）；`compressed_tensor` int8_w8a8 精度修复

> 全部已测试模型与特性状态（LoRA、PP 等）见 [支持模型列表](/models/supported_models.md)。
> 若需接入列表之外的模型或自定义模型，参考 [模型注册指南](/developing/model/registration.md)。

## 6. 从 v0.23 升级迁移

1. **更换镜像 / 环境**：docker 用户直接切到 v0.24.0 镜像；源码用户重新 checkout `releases/v0.24.0` 并按 3.3 重装依赖。
2. **mcoplib 必须同步升级**到 `0.4.9`（v0.23 为 `0.4.2`），否则内核符号不匹配。
3. **torch 大版本跃迁**：v0.23 为 torch 2.8（metax3.5.3.9），v0.24 为 2.10.0+metax3.8.2.5，自定义算子 / patch 需重新编译。
4. **EP/DeepEP 行为变化**：deep_ep 改用 `BatchedTritonExperts`，若此前依赖旧的 EP 实现，需回归测试 MoE 模型吞吐与精度。
5. **已移除 inplace fused experts**：MoE 权重加载 / 量化配置若依赖该机制，需适配新的非 inplace 路径。
6. **Tokenizer**：transformers v5 下 `LlamaTokenizerFast` 空格/换行乱码已修复；如自定义 tokenizer patch 与上游冲突，需清理冗余 patch（patch 管理机制见 [Patches](/developing/patches/README.md)）。

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
| transformers v5 下部分 tokenizer 空格/换行异常 | Llama 类已修复；其它架构仍可能受影响 | 锁定 transformers 版本或应用对应 patch |
| `minimax_qk_norm_fusion` 等新融合算子 | 新引入，需回归验证精度 | 精度异常时用 msprobe 抓取对比（[指南](/developing/msprobe/msprobe_guide.md)） |
| MLA prefill 优化仅覆盖部分 head_dim | 非覆盖维度回退通用路径 | 性能略低，不影响正确性 |
| v0.16.0 被官方跳过 | 无此版本 | 请勿使用 v0.16.x |
| eagle / MTP 投机解码相关改动曾回退并重新与上游同步 | 投机解码路径不稳定窗口 | 升级后回归 eagle 提议精度 |

## 9. 相关文档

- **版本映射总表**：[Releases](/getting_started/quickstart.md#releases)
- **支持模型列表**：[Supported Models](/models/supported_models.md)
- **从源码安装 / 构建**：[MACA 安装指南](/getting_started/installation/maca.md)
- **精度调测（msprobe）**：[msprobe 精度调测指南](/developing/msprobe/msprobe_guide.md)
- **Patch 管理（monkey / git patches）**：[Patches](/developing/patches/README.md)
- **接入自定义 / 新模型**：[模型注册指南](/developing/model/registration.md)
- **插件架构与贡献**：[Contributing / 插件系统](/developing/README.md)
- mcoplib 仓库：<https://github.com/MetaX-MACA/mcoplib>
- 插件主页 / 源码：<https://github.com/MetaX-MACA/vLLM-metax>
- 开发者社区 / 镜像下载：<https://developer.metax-tech.com>
- vLLM 官方文档：<https://docs.vllm.ai/>

---
发布日期：2026.8.21｜维护：PDE/AI vLLM-MetaX 团队
