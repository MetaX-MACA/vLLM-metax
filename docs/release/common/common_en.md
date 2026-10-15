# Deployment and Validation Guide

[中文](/release/common/common_zh.md){ .md-button }

This guide covers deployment steps shared by all releases. See the corresponding release notes for dependencies and known issues.

## 1. Get the image

Visit the [MetaX developer community](https://developer.metax-tech.com/softnova/docker?chip_name=%E6%9B%A6%E4%BA%91C500%E7%B3%BB%E5%88%97&package_kind=AI&dimension=docker&deliver_type=%E5%88%86%E5%B1%82%E5%8C%85&ai_frame=vllm-metax&arch=amd64&system=ubuntu), select a version, and follow the provided commands to log in and pull the image.

## 2. Start a container

This example uses v0.28.0. Replace it with your downloaded image and mount the host model directory at `/model_weights`.

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

## 3. Check the environment

Inside the container, check GPU status, P2P topology, and PyTorch availability:

```bash
mx-smi
mx-smi topo -m
python -c 'import torch; print(torch.__version__); print(torch.cuda.is_available())'
```

`torch.cuda.is_available()` should return `True`. In the topology, `MX` denotes MetaXLink and `SYS` denotes a path across PCIe / NUMA.

## 4. Start the inference service

This example serves Qwen3-8B on one GPU. Adjust the model path, parallel size, and port as needed:

```bash
vllm serve /model_weights/Qwen/Qwen3-8B \
    --served-model-name Qwen3-8B \
    --tensor-parallel-size 1 \
    --host 0.0.0.0 --port 8000
```

## 5. Send a request

After the server starts, send a request from another terminal:

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen3-8B",
        "prompt": "Hello, my name is",
        "max_tokens": 32
    }'
```

## 6. Configure the MACA environment

Prebuilt images include the basic environment configuration. For source builds or custom environments, set these variables for your installation:

- `MACA_PATH`: MACA SDK installation directory.
- `CUCC_PATH`: cu-bridge tool directory.
- `CUDA_PATH`: CUDA compatibility directory provided by cu-bridge.
- `CUCC_CMAKE_ENTRY`: cu-bridge CMake build entry.
- `USE_PRECOMPILED_KERNEL`: `1` uses matching mcoplib; `0` uses local csrc and must be set at both build time and runtime.
- `PATH` / `LD_LIBRARY_PATH`: tool and shared library search paths.

```bash
export MACA_PATH=/opt/maca
export CUCC_PATH="${MACA_PATH}/tools/cu-bridge"
export CUDA_PATH="${HOME}/cu-bridge/CUDA_DIR"
export CUCC_CMAKE_ENTRY=2
export USE_PRECOMPILED_KERNEL=1

export PATH="${MACA_PATH}/mxgpu_llvm/bin:${MACA_PATH}/bin:${CUCC_PATH}/tools:${CUCC_PATH}/bin:${PATH}"
export LD_LIBRARY_PATH="${MACA_PATH}/lib:${MACA_PATH}/ompi/lib:${MACA_PATH}/mxgpu_llvm/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
```

## 7. Multiple processes and nodes

### 7.1. Share a GPU between processes

To share a GPU between multiple processes, enable `MACA_MPS_MODE` before starting the service:

```bash
export MACA_MPS_MODE=1
```

### 7.2. Multiple nodes

For tensor / pipeline / expert parallelism across nodes, configure:

- `MCCL_SOCKET_IFNAME`: network interface for MCCL communication.
- `GLOO_SOCKET_IFNAME`: network interface for GLOO communication.
- `MCCL_IB_HCA`: InfiniBand devices to use.

Inspect the node's network interfaces and IB topology:

```bash
ifconfig
mx-smi topo -n
```

Then set these variables in the service terminal on each node. Replace the example device names with the actual local devices:

```bash
export MCCL_SOCKET_IFNAME=ens20f0np0
export GLOO_SOCKET_IFNAME=ens20f0np0
export MCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3
```

## 8. Build from source

Check out the corresponding `releases/v<version>` branch. For example, for v0.28.0:

```bash
git clone --branch releases/v0.28.0 https://github.com/MetaX-MACA/vLLM-metax.git
cd vLLM-metax
```

Follow the [MACA Installation Guide](/getting_started/installation/maca.md) to install the plugin and build vLLM with `VLLM_TARGET_DEVICE=empty`. Use dependencies from the selected release branch.

## 9. Related documentation

- [Release mapping](/getting_started/quickstart.md#releases)
- [Supported models](/models/supported_models.md)
- [Precision debugging with msprobe](/developing/msprobe/msprobe_guide.md)
- [Patch management](/developing/patches/README.md)
