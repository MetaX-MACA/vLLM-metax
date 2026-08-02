## Summary

This PR adds **three additional fused-MoE Triton tuned configurations** for the MetaX C500 (`MXC500`) backend, expanding coverage from small MoE variants to the most widely used large MoE models today.

| Model | Shape (H, E, N) | top_k | Max speedup |
|-------|----------------|-------|-------------|
| DeepSeek-V2 | (5120, 64, 1536) | 6 | **3.38x** @ M=2048 |
| Mixtral-8x22B TP8 | (6144, 8, 2048) | 2 | **2.72x** @ M=2048 |
| DeepSeek-V3 / DeepSeek-R1 TP8 | (7168, 256, 256) | 8 | **2.45x** @ M=2048 |

Combined with the existing three configs (Qwen1.5-MoE / DeepSeek-V2-Lite / Qwen3-30B-A3B), the contribution now covers **six popular MoE shapes** on MXC500.

## What is being changed

- **New staged JSON configs** under `vllm_metax/model_executor/layers/fused_moe/configs/`:
  - `H=5120,E=64,N=1536,device_name=MXC500.json`
  - `H=6144,E=8,N=2048,device_name=MXC500.json`
  - `H=7168,E=256,N=256,device_name=MXC500.json`
- **Updated `config_list.txt`** to document the new configs.
- **No code changes** — pure configuration additions, minimal risk.

## Methodology

The configs were tuned directly against the **actual MACA fused-MoE Triton kernel** shipped in `vllm_metax` (OOT-registered, not the upstream `fused_experts`), using a micro-benchmark with random weights so no model download is required.

- Search space: `BLOCK_SIZE_M/N/K`, `GROUP_SIZE_M`, `num_warps`, `num_stages`
- Timing: CUDA events + `torch.cuda.synchronize()`, warmup + median of 15 iterations
- Packaging: flat best tiles converted to MetaX two-stage schema (`stage1`/`stage2`)

Raw tuning data and logs are preserved in the companion repository:  
`https://github.com/LindseyMei/vllm_metax/tree/master/moe_tuning/results`

## Performance details

### DeepSeek-V2 (H=5120, E=64, N=1536, top_k=6)

| M | default (ms) | tuned (ms) | speedup |
|---:|---:|---:|:--:|
| 128 | 3.183 | 2.381 | **1.34x** |
| 256 | 3.582 | 2.647 | **1.35x** |
| 512 | 4.678 | 2.972 | **1.57x** |
| 1024 | 9.488 | 3.831 | **2.48x** |
| 2048 | 24.488 | 7.248 | **3.38x** |

### Mixtral-8x22B TP8 (H=6144, E=8, N=2048, top_k=2)

| M | default (ms) | tuned (ms) | speedup |
|---:|---:|---:|:--:|
| 16 | 0.806 | 0.647 | **1.25x** |
| 64 | 0.904 | 0.709 | **1.27x** |
| 256 | 1.770 | 1.017 | **1.74x** |
| 512 | 2.855 | 1.375 | **2.08x** |
| 1024 | 5.299 | 2.174 | **2.44x** |
| 2048 | 9.727 | 3.573 | **2.72x** |

### DeepSeek-V3 / R1 TP8 (H=7168, E=256, N=256, top_k=8)

| M | default (ms) | tuned (ms) | speedup |
|---:|---:|---:|:--:|
| 512 | 3.470 | 2.919 | **1.19x** |
| 1024 | 4.648 | 3.421 | **1.36x** |
| 2048 | 11.342 | 4.624 | **2.45x** |

Small-M (decode) entries intentionally keep the default or near-default tile to avoid latency regressions; large-M (prefill / large batch) entries deliver the major speedups.

## Verification

All six configs (the three existing + three new) pass:

- **Correctness**: `torch.allclose(out_default, out_tuned, rtol=2e-2, atol=2e-2) == True`, typical `max|Δ| ≈ 1e-3`
- **Loader pickup**: `get_moe_configs(E, N, None, 0, 0, H)` returns all 10 M-entries for each shape

Verified with:

```bash
python3 moe_tuning/tools/moe_verify.py --config-dir moe_tuning/configs
# ALL PASS
```

## Environment

- GPU: MetaX C500 (sGPU slice: 16 GB / 25% compute)
- MACA: 3.3.0.15
- torch: 2.8.0+metax3.3.0.2
- vllm / vllm_metax / mcoplib: 0.13.0 / 0.13.0 / 0.3.1

## Honest boundaries

- Tuning and verification were performed on a **C500 sGPU slice (16 GB / 25% compute)**. Tile shapes (BLOCK_SIZE / warps / stages) are per-SM properties and are expected to transfer to a full C500; grid-level parameters (`GROUP_SIZE_M=1`, `SPLIT_K=1`) are conservative.
- **Full-card C500 optimality has not been verified.** Maintainers are welcome to re-run `moe_tuning/tools/moe_tune.py` on a full card to refine tiles.
- Because of the 16 GB limit, full-model E2E benchmarks for the larger variants were not run; evidence is kernel-level, consistent with standard MoE config PRs.

## Related

- Original MoE tuning PR: https://github.com/MetaX-MACA/vLLM-metax/pull/323
- Competition record repository: https://www.gitlink.org.cn/LindseyMei/vllm_metax
