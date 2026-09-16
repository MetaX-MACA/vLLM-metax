# vLLM-MetaX v0.24.0 User Guide

[中文](/release/0.25.0/zh.md){ .md-button }

> This is the per-release user guide shipped with vLLM-MetaX v0.24.0, for users and delivery teams deploying large models on MetaX GPU (MACA).
> Starting from this version, each release ships with a matching guide, kept together under `docs/releases/`.
>
> Repository: <https://github.com/MetaX-MACA/vLLM-metax>

vLLM-MetaX is a hardware backend plugin for vLLM, following vLLM's hardware-pluggable RFCs (#11162, #19161), and plugs into upstream vLLM as a plugin. This document is a **quick-start and release-notes** guide for the MetaX platform. For general vLLM usage, source builds, and precision debugging, follow the links in "Related Documentation" at the end.

## Version Compatibility Matrix (Important)

vLLM-MetaX, mcoplib (the kernel component), the MACA driver, and torch are **tightly coupled by version**. Always use a matching combination; mixing non-matching versions may cause build failures, accuracy issues, or runtime crashes, and is not supported.

| plugin version | MACA driver | torch (metax) | mcoplib | Docker image |
| :--------------: | :-----------: | :-------------: | :-------: | :------------ |
| **v0.24.0** | **maca3.8.2.5** | **2.10.0** | **0.4.9** | [vllm-metax:0.24.0](https://developer.metax-tech.com/softnova/docker?chip_name=%E6%9B%A6%E4%BA%91C500%E7%B3%BB%E5%88%97&package_name=vllm-metax:0.24.0-maca.ai3.8.2.5-torch2.10-py310-ubuntu22.04-amd64) |

## 1. Using vLLM-MetaX on the MetaX Platform

You can either build your own image or pull a prebuilt one. This guide uses a prebuilt image as the example.

### 1.1. Pull the image

Get your docker image [here](/getting_started/quickstart.md#releases)

### 1.2. Launch the container

```bash
# Metax GPU devices must be mounted via --device: --device=/dev/dri --device=/dev/mxcd
# The serving port must be exposed; --network=host is used here for convenience
docker run -it --net=host --uts=host --ipc=host --privileged=true --group-add video \
    --shm-size '100gb' --ulimit memlock=-1 \
    --security-opt seccomp=unconfined --security-opt apparmor=unconfined \
    --device=/dev/dri --device=/dev/mxcd \
    -v /root/workspace:/external \
    -v /model_weights:/model_weights \
    --name vllm_metax_test \
    cr.metax-tech.com/public-ai-release/maca/vllm-metax:0.24.0-maca.ai3.8.2.5-torch2.10-py310-ubuntu22.04-amd64
```

> See <https://developer.metax-tech.com> for more. For source builds, environment variables (MACA_PATH / CUCC_PATH, etc.) and build caveats, see the [MACA Installation Guide](/getting_started/installation/maca.md).

## 2. Environment Check

### 2.1. Check MetaX GPU availability

Thanks to CUDA compatibility, you can check the device the same way as on NVIDIA GPU:

```python
import torch
print(torch.cuda.is_available())
# True
```

### 2.2. Check P2P topology between GPUs

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

### 2.3. Check GPU status

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

## 3. Run an Example

The image already contains matching versions of vLLM-MetaX and mcoplib, so you can start the OpenAI-compatible serving endpoint directly.

### 3.1. Start the vLLM serving endpoint

```bash
# example with Qwen3-8B; adjust the model path as needed
vllm serve /model_weights/Qwen/Qwen3-8B \
    --tensor-parallel-size 1 \
    --host 0.0.0.0 --port 8000
```

Key logs on a successful startup (excerpt):

```text
INFO 03-09 15:43:01 [api_server.py] Starting vLLM API server on http://0.0.0.0:8000
INFO 03-09 15:43:05 [core.py] Initializing a V1 LLM engine (v0.24.0) with config: ...
INFO 03-09 15:43:12 [maca_platform.py] MACA backend initialized, driver maca3.8.2.0
INFO 03-09 15:43:35 [gpu_model_runner.py] Model loading took 23.41 GiB and 22.65 seconds
INFO 03-09 15:43:36 [api_server.py] Application startup complete.
```

### 3.2. Send an inference request

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "/model_weights/Qwen/Qwen3-8B",
        "prompt": "MetaX is a",
        "max_tokens": 32
    }'
```

Example response:

```json
{
  "id": "cmpl-xxxxxxxx",
  "object": "text_completion",
  "model": "/model_weights/Qwen/Qwen3-8B",
  "choices": [
    { "text": " company focused on high-performance AI computing, whose MACA software stack is CUDA-compatible.", "index": 0, "finish_reason": "length" }
  ],
  "usage": { "prompt_tokens": 4, "completion_tokens": 32, "total_tokens": 36 }
}
```

> For more launch options, parallelism strategies (tensor/pipeline/expert parallel), and quantized weight loading, see the [vLLM documentation](https://docs.vllm.ai/). Section 3.1 only demonstrates a minimal runnable example.

### 3.3. Other versions / build from source

To build another version or customize, use MACA-compatible packages. The build depends on torch 2.10 (metax3.8.x); use `--no-deps` to avoid overwriting an existing PyTorch in the environment:

```bash
git clone -b releases/v0.24.0 https://github.com/MetaX-MACA/vLLM-metax.git
cd vLLM-metax
pip install -r requirements/maca.txt -r requirements/maca_private.txt
pip install -e . --no-deps
```

After any environment change, verify the PyTorch version and availability:

```bash
pip list | grep torch
# torch                    2.10.0+metax3.8.2.5
```

```python
import torch
torch.cuda.is_available()  # True
```

> Full build flow (building vllm itself with the empty device, the meaning of USE_PRECOMPILED_KERNEL, and the requirement to use mcoplib in production) is in the [MACA Installation Guide](/getting_started/installation/maca.md).

## 4. v0.24.0 Highlights

- **MoE kernel overhaul**
    - New `mctlass` fused MoE FP8
    - New `mctlassEx` W4A16 fused MoE (bf16 path only for now)
    - New fused MoE **W4A8** support
    - Removed the inplace fused experts mechanism to reduce accuracy risk
- **Expert Parallelism (EP) enhancements**: deep_ep now supports `BatchedTritonExperts`; EP path adds `filter_expert`
- **Attention optimizations**: MLA prefill optimized for some head dims; switched the FLA path
- **Precision debugging**: integrated the `msprobe` precision-debugging tool (see the [msprobe guide](/developing/msprobe/msprobe_guide.md))
- **Build system**: requirements restructured for auto-install support; fixed the `USE_PRECOMPILED_KERNEL=1` build error

## 5. New and Updated Models

These models are new or significantly updated in v0.24.0 (some are not yet reflected in `docs/models/supported_models.md` — this guide is authoritative):

### Text / reasoning models

- **MiniMax-M3**: new `minimax-m3`, plus an FP8 weight path `minimax-m3-fp8`
- **JD JoyAI**: new `JoyAI_LLM_Flash` (jd)
- **DeepSeek-V4 (dsv4)**: new support, with related MoE/attention cleanup
- **DeepSeek series**: synced `deepseek_v2` with v0.24.0 changes
- **GLM series**: incremental GLM / GLM-4.x adaptation

### Multimodal models

- **Qwen3-Omni / Qwen3-VL series**: new corresponding architectures
- **Intern-S1 / InternVL 3.5**: new architecture support
- **Qwen2.5-Omni**: fixed the `cu_seqlens` CUDA error

### Quantized model support

- FP8 inference; AWQ (`auto_awq` and `awq_to_gptq_4bit`); `compressed_tensor` int8_w8a8 accuracy fix

> The full list of tested models and feature status (LoRA, PP, etc.) is in [Supported Models](/models/supported_models.md).
> To onboard a model not in the list or a customized model, see the [Model Registration Guide](/developing/model/registration.md).

## 6. Upgrading from v0.23

1. **Switch image / environment**: docker users switch directly to the v0.24.0 image; source users re-checkout `releases/v0.24.0` and reinstall deps as in 3.3.
2. **Upgrade mcoplib to `0.4.9`** (v0.23 was `0.4.2`), otherwise kernel symbols will not match.
3. **Major torch bump**: v0.23 used torch 2.8 (metax3.5.3.9), v0.24 uses 2.10.0+metax3.8.2.5 — custom operators / patches must be recompiled.
4. **EP/DeepEP behavior change**: deep_ep now uses `BatchedTritonExperts`; if you previously relied on the old EP implementation, regression-test MoE throughput and accuracy.
5. **inplace fused experts removed**: MoE weight loading / quant configs that depended on it must move to the new non-inplace path.
6. **Tokenizer**: the `LlamaTokenizerFast` space/newline garbling under transformers v5 is fixed; if your custom tokenizer patch conflicts with upstream, clean up the redundant patch (patch management is in [Patches](/developing/patches/README.md)).

## 7. Differences from NVIDIA CUDA

MetaX aligns with NVIDIA on most interfaces, but differs in some software behaviors and environment variables.

### 7.1. MACA_MPS_MODE (multi-process GPU sharing)

By default, MACA does not allow multiple processes to share one GPU. If a GPU is occupied, a new process cannot start.

To enable an MPS-like (Multi-Process Service) feature, set: `MACA_MPS_MODE=1`

```bash
export MACA_MPS_MODE=1
vllm serve /model_weights/Qwen/Qwen3-8B --tensor-parallel-size 1
```

### 7.2. Multi-node deployment

When vLLM runs distributed (tensor/pipeline/expert parallel) across nodes, set the following environment variables to keep inter-node communication healthy:

- `MCCL_SOCKET_IFNAME`: network interface for MCCL
- `GLOO_SOCKET_IFNAME`: network interface for GLOO
- `MCCL_IB_HCA`: the InfiniBand devices to use

Determine the NIC and IB devices via `ifconfig` and `mx-smi topo -n` (see the topology examples in the verl / Megatron docs). Recommended settings:

```bash
export MCCL_SOCKET_IFNAME=ens20f0np0
export GLOO_SOCKET_IFNAME=ens20f0np0
export MCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3
```

> ⚠️ Adjust according to your own machine's `ifconfig` and `mx-smi topo -n` output; do not blindly copy the commands above.

## 8. Known Issues and Workarounds

| Issue | Impact | Workaround / status |
| ------- | -------- | --------------------- |
| Some tokenizers miss-handle spaces/newlines under transformers v5 | Fixed for Llama-class; other archs may still be affected | Pin the transformers version or apply the matching patch |
| New fusion ops such as `minimax_qk_norm_fusion` | Newly introduced; regression-test accuracy | On accuracy deviation, capture and compare with msprobe ([guide](/developing/msprobe/msprobe_guide.md)) |
| MLA prefill optimization covers only some head dims | Uncovered dims fall back to the generic path | Slightly lower perf, correctness unaffected |
| v0.16.0 officially skipped | No such version | Do not use v0.16.x |
| eagle / MTP speculative decoding changes were reverted and re-synced with upstream | Unstable window for the speculative path | Regression-test eagle proposal accuracy after upgrade |

## 9. Related Documentation

- **Version map**: [Releases](/getting_started/quickstart.md#releases)
- **Supported models**: [Supported Models](/models/supported_models.md)
- **Install / build from source**: [MACA Installation Guide](/getting_started/installation/maca.md)
- **Precision debugging (msprobe)**: [msprobe Guide](/developing/msprobe/msprobe_guide.md)
- **Patch management (monkey / git patches)**: [Patches](/developing/patches/README.md)
- **Onboard custom / new models**: [Model Registration Guide](/developing/model/registration.md)
- **Plugin architecture & contributing**: [Contributing / Plugin System](/developing/README.md)
- mcoplib repo: <https://github.com/MetaX-MACA/mcoplib>
- Plugin homepage / source: <https://github.com/MetaX-MACA/vLLM-metax>
- Developer community / image downloads: <https://developer.metax-tech.com>
- vLLM documentation: <https://docs.vllm.ai/>

---
Release date: 2026.8.21 | Maintained by: PDE/AI vLLM-MetaX team
