# 部署与验证指南

[English](/release/common/common_en.md){ .md-button }

本文介绍各版本通用的部署操作。版本依赖和已知问题见对应发布说明。

## 1. 获取镜像

前往 [MetaX 开发者社区](https://developer.metax-tech.com/softnova/docker?chip_name=%E6%9B%A6%E4%BA%91C500%E7%B3%BB%E5%88%97&package_kind=AI&dimension=docker&deliver_type=%E5%88%86%E5%B1%82%E5%8C%85&ai_frame=vllm-metax&arch=amd64&system=ubuntu)，选择所需版本，按页面提供的命令登录并拉取镜像。

## 2. 启动容器

以下以 v0.28.0 为例。请替换为已拉取的镜像，并将宿主机模型目录挂载到 `/model_weights`。

```bash
docker run -it --net=host --uts=host --ipc=host \
    --privileged=true --group-add video \
    --shm-size 100gb --ulimit memlock=-1 \
    --security-opt seccomp=unconfined --security-opt apparmor=unconfined \
    --device=/dev/dri --device=/dev/mxcd \
    -v /model_weights:/model_weights \
    --name vllm_metax_test \
    cr.metax-tech.com/public-ai-release/maca/vllm-metax:0.28.0-maca.ai3.8.2.401-torch2.10-py310-ubuntu22.04-amd64 \
    /bin/bash
```

## 3. 检查环境

进入容器后，检查 GPU 状态、P2P 拓扑和 PyTorch 可用性：

```bash
mx-smi
mx-smi topo -m
python -c 'import torch; print(torch.__version__); print(torch.cuda.is_available())'
```

`torch.cuda.is_available()` 应返回 `True`。拓扑中的 `MX` 表示 MetaXLink，`SYS` 表示跨 PCIe / NUMA 的连接。

## 4. 启动推理服务

以 Qwen3-8B 单卡推理为例，按实际情况调整模型路径、并行度和端口：

```bash
vllm serve /model_weights/Qwen/Qwen3-8B \
    --served-model-name Qwen3-8B \
    --tensor-parallel-size 1 \
    --host 0.0.0.0 --port 8000
```

## 5. 发送请求

服务启动后，在另一个终端发送请求：

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen3-8B",
        "prompt": "Hello, my name is",
        "max_tokens": 32
    }'
```

## 6. MACA 环境配置

预构建镜像已配置基础环境。从源码构建或使用自定义环境时，可按安装路径设置：

- `MACA_PATH`：MACA SDK 安装目录。
- `CUCC_PATH`：cu-bridge 工具目录。
- `CUDA_PATH`：cu-bridge 提供的 CUDA 兼容目录。
- `CUCC_CMAKE_ENTRY`：cu-bridge CMake 构建入口。
- `USE_PRECOMPILED_KERNEL`：`1` 使用配套 mcoplib；`0` 使用本地 csrc，需在构建和运行时保持一致。
- `PATH` / `LD_LIBRARY_PATH`：工具和动态库搜索路径。

```bash
export MACA_PATH=/opt/maca
export CUCC_PATH="${MACA_PATH}/tools/cu-bridge"
export CUDA_PATH="${HOME}/cu-bridge/CUDA_DIR"
export CUCC_CMAKE_ENTRY=2
export USE_PRECOMPILED_KERNEL=1

export PATH="${MACA_PATH}/mxgpu_llvm/bin:${MACA_PATH}/bin:${CUCC_PATH}/tools:${CUCC_PATH}/bin:${PATH}"
export LD_LIBRARY_PATH="${MACA_PATH}/lib:${MACA_PATH}/ompi/lib:${MACA_PATH}/mxgpu_llvm/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
```

## 7. 多进程与多节点配置

### 7.1. 多进程共享 GPU

需要多个进程共享同一块 GPU 时，在启动服务前启用 `MACA_MPS_MODE`：

```bash
export MACA_MPS_MODE=1
```

### 7.2. 多节点部署

vLLM 的 tensor / pipeline / expert parallel 跨节点运行时，建议设置以下环境变量：

- `MCCL_SOCKET_IFNAME`：用于 MCCL 通信的网络接口。
- `GLOO_SOCKET_IFNAME`：用于 GLOO 通信的网络接口。
- `MCCL_IB_HCA`：指定使用的 InfiniBand 设备。

先检查本机网卡和 IB 拓扑：

```bash
ifconfig
mx-smi topo -n
```

再在各节点启动服务的终端中配置，设备名按本机实际情况修改：

```bash
export MCCL_SOCKET_IFNAME=ens20f0np0
export GLOO_SOCKET_IFNAME=ens20f0np0
export MCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3
```

## 8. 从源码构建

检出对应的 `releases/v<版本号>` 分支，以下以 v0.28.0 为例：

```bash
git clone --branch releases/v0.28.0 https://github.com/MetaX-MACA/vLLM-metax.git
cd vLLM-metax
```

按 [MACA 安装指南](/getting_started/installation/maca.md) 完成插件安装与 vLLM 的 `VLLM_TARGET_DEVICE=empty` 构建，依赖使用所选发布分支中的版本。

## 9. 相关文档

- [版本映射](/getting_started/quickstart.md#releases)
- [支持模型](/models/supported_models.md)
- [msprobe 精度调测](/developing/msprobe/msprobe_guide.md)
- [Patch 管理](/developing/patches/README.md)
