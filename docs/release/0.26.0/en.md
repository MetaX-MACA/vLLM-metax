# vLLM-MetaX v0.26.0 User Guide

[中文](/release/0.26.0/zh.md){ .md-button }

> This is the per-release user guide shipped with vLLM-MetaX v0.26.0, for users and delivery teams deploying large models on MetaX GPU (MACA).
> Since v0.24.0, each release ships with a matching guide, kept together under `docs/releases/`.
>
> Repository: https://github.com/MetaX-MACA/vLLM-metax

vLLM-MetaX is a hardware backend plugin for vLLM, following vLLM's hardware-pluggable RFCs (#11162, #19161), and plugs into upstream vLLM as a plugin. This document is a **quick-start and release-notes** guide for the MetaX platform. For general vLLM usage, source builds, and precision debugging, follow the links in "Related Documentation" at the end.

## Version Compatibility Matrix (Important)

vLLM-MetaX, mcoplib (the kernel component), the MACA driver, and torch are **tightly coupled by version**. Always use a matching combination; mixing non-matching versions may cause build failures, accuracy issues, or runtime crashes, and is not supported.

| plugin version | maca version | mcoplib version | docker image url |
| :---: | :---: | :---: | :---: |
| v0.26.0 | maca3.8.2.x | 0.4.11 | [vllm-metax:0.26.0](https://developer.metax-tech.com/softnova/docker?package_name=vllm-metax:0.26.0-torch2.10) |

> Note: v0.16.0 was officially skipped — do not use it. The full version map and image downloads are in the [Releases table](/getting_started/quickstart.md#releases). Image URL template:
> `https://developer.metax-tech.com/softnova/docker?package_name=vllm-metax:0.26.0-torch2.10`

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
    cr.metax-tech.com/public-ai-release/maca/vllm-metax:0.26.0-maca.ai3.8.2.2-torch2.10-py312-ubuntu22.04-amd64
```

> See https://developer.metax-tech.com for more. For source builds, environment variables (MACA_PATH / CUCC_PATH, etc.) and build caveats, see the [MACA Installation Guide](/getting_started/installation/maca.md).

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
INFO 03-09 15:43:05 [core.py] Initializing a V1 LLM engine (v0.26.0) with config: ...
INFO 03-09 15:43:12 [maca_platform.py] MACA backend initialized, driver maca3.8.2.2
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

To build another version or customize, use MACA-compatible packages. The build depends on torch 2.10 (metax3.8.2.2); use `--no-deps` to avoid overwriting an existing PyTorch in the environment:

```bash
git clone -b mx/v0.26.0-dev https://github.com/MetaX-MACA/vLLM-metax.git
cd vLLM-metax
pip install -r requirements/maca.txt -r requirements/maca_private.txt
pip install -e . --no-deps
```

After any environment change, verify the PyTorch version and availability:

```bash
pip list | grep torch
# torch                    2.10.0+metax3.8.2.2
```

```python
import torch
torch.cuda.is_available()  # True
```

> Full build flow (building vllm itself with the empty device, the meaning of USE_PRECOMPILED_KERNEL, and the requirement to use mcoplib in production) is in the [MACA Installation Guide](/getting_started/installation/maca.md).

## 4. v0.26.0 Highlights

- **Aligned with upstream vLLM v0.26.0**
  - Baseline synced to vLLM v0.26.0; upstream interfaces synced and the patch system refactored (`@patch` templates / `patch/utils`)
- **DeepSeek-V4 / MLA**
  - `deepseek_v4` adapted and refactored on v0.26.0 (sparse attention / MLA, MTP, Eagle, draft config, etc.)
  - Fixed missing SparseAttnIndexer registry, lost MTP import, `draft_config_overrides` import, and synced `flashmla_sparse`
  - Fixed telechat3-yarn (CCPM-9709) `max_model_len` derivation
  - cumem allocator adaptation (MC3-14374) plus a new `is_cumem_allocator_available` platform override
- **MiniMax-M3**
  - topk / shared-mem fixes: upstream #49149 backport (topk buffer layout; patches removable once on v0.27.0), index_topk and sparse-attention patch refactors
  - Fixed missing `int8_w8a8` swiglu configuration (gemm1 alpha / beta / clamp_limit)
  - `minimax_qk_norm` all-reduce fusion disabled by default (see Section 8)
- **MoE / kernels**
  - Fused MoE scale-loading fix: mirrors upstream PR #50137 plus transposed per-channel scale normalization (landed in `int8_w8a8/load_weight.py`)
  - MCTlass fused MoE: fixed FP8 usage and skip invalid EP experts
  - deep_ep: FP8 / int8 W8A8 quantization paths (`BatchedTritonExperts` / `BatchedDeepGemmExperts`), deepep high-throughput and dbo toggles
  - w8a8 int8 loading fix
- **JoyAI**: fixed `joyai_llm_flash` being misrecognized as `deepseek_mtp`; tokenizer decoder now uses BPE
- **Distributed**: MxMesh low-latency all-to-all, all-gather hang fix, environment variables no longer force-overridden by default
- **Build / dependencies**: maca3.8.2.2 toolchain; tilelang 0.1.12+maca, apache-tvm-ffi 0.1.11+maca, xgrammar 0.2.x; mcoplib 0.4.11; tvm / tilelang installed from the internal pip source
- **Structure refactor**: monkey-patch system refactored; quant config / custom ops / plugin enhancements moved to `registry/quant_config`, `registry/custom_ops`, and `patch/enhancement`

## 5. New and Updated Models

These models are new or significantly updated in v0.26.0 (some are not yet reflected in `docs/models/supported_models.md` — this guide is authoritative):

### Text / reasoning models
- **DeepSeek-V4**: adapted and refactored on v0.26.0 (sparse attention / MLA, MTP, Eagle) with loading / derivation fixes such as telechat3-yarn
- **MiniMax-M3**: W8A8 topk / shared-mem / sparse-attention fixes, `int8_w8a8` swiglu configuration completed, `minimax_qk_norm` all-reduce fusion disabled by default
- **DeepSeek series**: deep_ep FP8 / int8 quantization support (`BatchedDeepGemmExperts`), Eagle3 / MTP, `flashmla_sparse` update
- **JD JoyAI**: `JoyAI_LLM_Flash` (jd) recognition and tokenizer-decoder fixes; support moved to `patch/enhancement/joyai_support`
- **MRV2 / step3p5**: EP + PCP support
- **GLM series**: continued from the v0.25 adaptation

### Multimodal models
- Qwen3-Omni / Qwen3-VL and Intern-S1 / InternVL 3.5 introduced in v0.24 remain supported on the v0.26.0 upstream baseline; no new Metax-side multimodal architecture in this release

### Quantized model support
- FP8 / int8_w8a8 / W8A8 inference; fused MoE W4A8 (introduced in v0.25, kept under the registry refactor); MiniMax-M3 int8 swiglu fix

> The full list of tested models and feature status (LoRA, PP, etc.) is in [Supported Models](/models/supported_models.md).
> To onboard a model not in the list or a customized model, see the [Model Registration Guide](/developing/model/registration.md).

## 6. Upgrading from v0.25

1. **Switch image / environment**: docker users switch directly to the v0.26.0 image; source users re-checkout `mx/v0.26.0-dev` and reinstall deps as in 3.3.
2. **Upgrade torch / MACA**: v0.26.0 uses torch 2.10 (metax3.8.2.2; v0.25.0 was metax3.8.2.0). Custom operators and locally built csrc must be rebuilt against the new toolchain.
3. **Upgrade mcoplib**: v0.25.0 was `0.4.10+g72792e0.maca3.8.0.25.torch2.10`; v0.26.0 is `0.4.11+ge29f792.maca3.8.0.25.torch2.10` — otherwise kernel symbols will not match.
4. **tilelang / tvm-ffi / xgrammar combination updated**: v0.26.0 uses tilelang 0.1.12+maca, apache-tvm-ffi 0.1.11+maca and xgrammar 0.2.x; the v0.25.0 downgrade combo for dsv4 (tilelang 0.1.9+maca, xgrammar 0.1.32) no longer applies.
5. **Patch / registry structure refactor**: quant config, custom ops, and plugin enhancements moved to `registry/quant_config`, `registry/custom_ops`, and `patch/enhancement`. Custom extensions based on the old paths (`patch/plugin_enhancement`, `customized/*`, `transformers_utils`, etc.) must be re-registered.
6. **Fused MoE scale-loading fix relocated**: PR #50137-related fixes and transposed-scale normalization now live in `patch/bugfix/int8_w8a8/load_weight.py`.
7. **MiniMax-M3 patches made standalone**: topk / indexer / sparse fixes are independent implementations in 0.26 (#49149 backport) and should be removed once the vLLM dependency reaches v0.27.0.
8. **Speculative decoding**: eagle / MTP continues via patches and draft-config overrides — regression-test eagle proposal accuracy after upgrade.

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
| MiniMax-M3 topk buffer layout (upstream #49147, backport #49149) | Affects v0.24.0–v0.26.0; mixing decode with prefill can trigger illegal memory access | Fixed (indexer / sparse_attention patches); remove once the vLLM dependency reaches v0.27.0 |
| `minimax_qk_norm` all-reduce fusion disabled by default | Fusion ops off by default, slightly lower perf on affected models | To enable, change the `rms_norm_tp` config and regression-test accuracy |
| deep_ep FP8 / int8, W4A8 and other quantization paths | New / refactored paths; regression-test accuracy | On accuracy deviation, capture and compare with msprobe ([guide](/developing/msprobe/msprobe_guide.md)) |
| Some tokenizers under transformers v5 (Moonlight / Kimi, etc.) | Old import path (`bytes_to_unicode`) removed in transformers v5 | Compatibility shim re-injected; pin the transformers version if issues persist |
| Locally built csrc vs prebuilt mcoplib | Local builds must match mcoplib 0.4.11 kernel symbols | Prefer the prebuilt mcoplib combination in production |
| tilelang / tvm-ffi / xgrammar tightly coupled | Mixing combinations may break kernel loading or grammar | Install strictly per `requirements/` |
| v0.16.0 officially skipped | No such version | Do not use v0.16.x |

## 9. Related Documentation

- **Version map**: [Releases](/getting_started/quickstart.md#releases)
- **Supported models**: [Supported Models](/models/supported_models.md)
- **Install / build from source**: [MACA Installation Guide](/getting_started/installation/maca.md)
- **Precision debugging (msprobe)**: [msprobe Guide](/developing/msprobe/msprobe_guide.md)
- **Patch management (monkey / git patches)**: [Patches](/developing/patches/README.md)
- **Onboard custom / new models**: [Model Registration Guide](/developing/model/registration.md)
- **Plugin architecture & contributing**: [Contributing / Plugin System](/developing/README.md)
- mcoplib repo: https://github.com/MetaX-MACA/mcoplib
- Plugin homepage / source: https://github.com/MetaX-MACA/vLLM-metax
- Developer community / image downloads: https://developer.metax-tech.com
- vLLM documentation: https://docs.vllm.ai/

---
*Release date: TODO (fill in the actual date) | Maintained by: PDE/AI vLLM-MetaX team*
