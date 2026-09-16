# vLLM-MetaX v0.25.0 User Guide

[中文](zh.md){ .md-button }

> This is the per-release user guide shipped with vLLM-MetaX v0.25.0, for users and delivery teams deploying large models on MetaX GPU (MACA).
> Since v0.24.0, each release ships with a matching guide, kept together under `docs/releases/`.
>
> Repository: https://github.com/MetaX-MACA/vLLM-metax

vLLM-MetaX is a hardware backend plugin for vLLM, following vLLM's hardware-pluggable RFCs (#11162, #19161), and plugs into upstream vLLM as a plugin. This document is a **quick-start and release-notes** guide for the MetaX platform. For general vLLM usage, source builds, and precision debugging, follow the links in "Related Documentation" at the end.

## Version Compatibility Matrix (Important)

vLLM-MetaX, mcoplib (the kernel component), the MACA driver, and torch are **tightly coupled by version**. Always use a matching combination; mixing non-matching versions may cause build failures, accuracy issues, or runtime crashes, and is not supported.

| plugin version | maca version | mcoplib version | docker image url |
| :---: | :---: | :---: | :---: |
| v0.23.0 | maca3.8.0.x | 0.4.8 | [vllm-metax:0.23.0](https://developer.metax-tech.com/softnova/docker?package_name=vllm-metax:0.23.0-torch2.10) |
| v0.24.0 | maca3.8.2.x | 0.4.9 | [vllm-metax:0.24.0](https://developer.metax-tech.com/softnova/docker?package_name=vllm-metax:0.24.0-torch2.10) |
| v0.25.0 | maca3.8.2.x | 0.4.10 | [vllm-metax:0.25.0](https://developer.metax-tech.com/softnova/docker?package_name=vllm-metax:0.25.0-torch2.10) |

> Note: v0.16.0 was officially skipped — do not use it. The full version map and image downloads are in the [Releases table](../getting_started/quickstart.md#releases). Image URL template:
> `https://developer.metax-tech.com/softnova/docker?package_name=vllm-metax:0.25.0-torch2.10`

## 1. Using vLLM-MetaX on the MetaX Platform

You can either build your own image or pull a prebuilt one. This guide uses a prebuilt image as the example.

### 1.1. Pull the image

```bash
docker pull cr.metax-tech.com/public-ai-release/maca/vllm-metax:0.25.0-maca.ai3.8.2.0-torch2.10-py310-ubuntu22.04-amd64
# tag convention (subject to the actual release page):
#   0.25.0            -> plugin version
#   maca.ai3.8.2.0    -> MACA driver version
#   torch2.10         -> torch version
#   py310-ubuntu22.04 -> base environment
```

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
    cr.metax-tech.com/public-ai-release/maca/vllm-metax:0.25.0-maca.ai3.8.2.0-torch2.10-py310-ubuntu22.04-amd64
```

> See https://developer.metax-tech.com for more. For source builds, environment variables (MACA_PATH / CUCC_PATH, etc.) and build caveats, see the [MACA Installation Guide](../getting_started/installation/maca.md).

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
INFO 03-09 15:43:05 [core.py] Initializing a V1 LLM engine (v0.25.0) with config: ...
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
git clone -b mx/v0.25.0-dev https://github.com/MetaX-MACA/vLLM-metax.git
cd vLLM-metax
pip install -r requirements/maca.txt -r requirements/maca_private.txt
pip install -e . --no-deps
```

After any environment change, verify the PyTorch version and availability:

```bash
pip list | grep torch
# torch                    2.10.0+metax3.8.2.0
```

```python
import torch
torch.cuda.is_available()  # True
```

> Full build flow (building vllm itself with the empty device, the meaning of USE_PRECOMPILED_KERNEL, and the requirement to use mcoplib in production) is in the [MACA Installation Guide](../getting_started/installation/maca.md).

## 4. v0.25.0 Highlights

- **Aligned with upstream vLLM v0.25.0**
  - Baseline synced to vLLM v0.25.0 (PR #326)
  - cherry-picked v0.24.0 fixes/features back, plus upstream PR #50137 backport (fused weight scale loading)
- **MoE kernel enhancements**
  - deep_ep support landed on MACA (previously NVIDIA-only), with new **FP8** (`BatchedDeepGemmExperts`) and **int8** quantization paths
  - New fused MoE **W4A8** support
  - Fixed `mctlass` fused MoE FP8 usage
  - New Modular MoE output alias (MetaX-specific patch)
  - Fixed fused MoE per-channel scales normalization before transposed loading
  - Backported upstream #36701 (`get_supported_kernel_block_sizes` no longer calls `get_current_vllm_config`)
- **flashinfer**: dropped the unsupported backend argument for metax flashinfer wrappers
- **Speculative decoding**: eagle changes returned as a patch (previously reverted in v0.24; see Section 8)
- **Model patches**: new Qwen3.5 W8A8 weight loading fix; MiniMax-M3 `int8_w8a8` swiglu configuration fix
- **Build system**: csrc synced with mcoplib v0.25.0 adaptation; fixed locally-built csrc running on MACA and the `USE_PRECOMPILED_KERNEL=1` build error; requirements updated for v0.25.0 auto-install; Dockerfile updated

## 5. New and Updated Models

These models are new or significantly updated in v0.25.0 (some are not yet reflected in `docs/models/supported_models.md` — this guide is authoritative):

### Text / reasoning models
- **Qwen3.5 series**: new support, including the W8A8 weight loading fix
- **MiniMax-M3**: fixed the missing `int8_w8a8` swiglu configuration (the FP8 path works; the int8 path previously failed); `minimax_qk_norm` all-reduce fusion now disabled by default (see Section 8)
- **DeepSeek series**: deep_ep gained FP8 / int8 quantization support (`BatchedDeepGemmExperts`)
- **JD JoyAI**: `JoyAI_LLM_Flash` (jd) further improved with the flashinfer fix
- **GLM series**: continued from the v0.24 adaptation

### Multimodal models
- Qwen3-Omni / Qwen3-VL and Intern-S1 / InternVL 3.5 introduced in v0.24 remain supported; no new multimodal architectures in this release

### Quantized model support
- FP8 inference; fused MoE **W4A8** (new); int8_w8a8 (MiniMax-M3 swiglu fix); W8A8 (Qwen3.5 loading fix)

> The full list of tested models and feature status (LoRA, PP, etc.) is in [Supported Models](../models/supported_models.md).
> To onboard a model not in the list or a customized model, see the [Model Registration Guide](../developing/model/registration.md).

## 6. Upgrading from v0.24

1. **Switch image / environment**: docker users switch directly to the v0.25.0 image; source users re-checkout `releases/v0.25.0-dev` and reinstall deps as in 3.3.
2. **Upgrade mcoplib** to the version matching v0.25.0 (csrc is already synced with the mcoplib v0.25.0 adaptation; v0.24 was `0.4.9+maca3.8.0.25.torch2.10`, the v0.25.0 version is in the matrix above), otherwise kernel symbols will not match.
3. **torch / MACA unchanged**: v0.25.0 keeps torch 2.10 (metax3.8.2.0) as in v0.24.0, so custom operators / patches need no recompile for torch itself (mcoplib per item 2 still applies).
4. **transformers patch removed**: v0.25.0 removed the redundant transformers patch (`model_arch_config_convertor`). If a model that relied on the patch misbehaves, check transformers version compatibility.
5. **Speculative decoding path change**: eagle / MTP changes were reverted in v0.24 and returned as a patch in v0.25.0 — regression-test eagle proposal accuracy after upgrade.
6. **New deep_ep quantization paths**: FP8 / int8 weight loading goes through `BatchedDeepGemmExperts`; if you customized the deep_ep quantization flow, adapt it to the new path.

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
|-------|--------|---------------------|
| `minimax_qk_norm` all-reduce fusion disabled by default (since v0.25.0) | Fusion ops off by default, slightly lower perf on affected models | To enable, change the `rms_norm_tp` config and regression-test accuracy |
| deep_ep FP8 / int8 are new paths | Newly introduced; regression-test accuracy | On accuracy deviation, capture and compare with msprobe ([guide](../developing/msprobe/msprobe_guide.md)) |
| Some tokenizers miss-handle spaces/newlines under transformers v5 | The v0.24 Llama-class fix relied on a patch; after the patch removal other archs may still be affected | Pin the transformers version or apply the matching patch |
| v0.16.0 officially skipped | No such version | Do not use v0.16.x |
| eagle / MTP speculative decoding changes returned as a patch | Unstable window for the speculative path | Regression-test eagle proposal accuracy after upgrade |

## 9. Related Documentation

- **Version map**: [Releases](../getting_started/quickstart.md#releases)
- **Supported models**: [Supported Models](../models/supported_models.md)
- **Install / build from source**: [MACA Installation Guide](../getting_started/installation/maca.md)
- **Precision debugging (msprobe)**: [msprobe Guide](../developing/msprobe/msprobe_guide.md)
- **Patch management (monkey / git patches)**: [Patches](../developing/patches/README.md)
- **Onboard custom / new models**: [Model Registration Guide](../developing/model/registration.md)
- **Plugin architecture & contributing**: [Contributing / Plugin System](../developing/README.md)
- mcoplib repo: https://github.com/MetaX-MACA/mcoplib
- Plugin homepage / source: https://github.com/MetaX-MACA/vLLM-metax
- Developer community / image downloads: https://developer.metax-tech.com
- vLLM documentation: https://docs.vllm.ai/

---
*Release date: 2026.9.4 | Maintained by: PDE/AI vLLM-MetaX team*
