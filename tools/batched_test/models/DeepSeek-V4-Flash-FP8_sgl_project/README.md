# DeepSeek V4 Flash FP8: five-layer dummy model

Restored on 2026-09-23 using the same reduction as the earlier artifact.
The supplied source metadata matched the original backup byte for byte.
This directory contains no checkpoint weights and requires `--load-format dummy`.
Dummy loading still allocates parameters; output does not establish model accuracy.

## Reduction and coverage

Only two configuration fields changed:

- `num_hidden_layers`: 43 -> 5.
- `compress_ratios`: 44 entries -> `[0, 0, 4, 128, 4, 0]`.
  Retain original backbone layers 0–4, and move the original prediction-layer slot
  43 to slot 5. Current MetaX attention uses effective ratio 1 for MTP; the final
  zero preserves the original metadata convention.

| New layer | Original layer | Routing | Compression ratio |
| --- | --- | --- | --- |
| 0 | 0 | Hash MoE | 0 (effective 1) |
| 1 | 1 | Hash MoE | 0 (effective 1) |
| 2 | 2 | Hash MoE | 4 |
| 3 | 3 | Learned MoE | 128 |
| 4 | 4 | Learned MoE | 4 |

Preserve `num_hash_layers=3`, `num_nextn_predict_layers=1`, hidden_size=4096,
256 routed experts, top-k=6, one shared expert, all projection/head dimensions,
and vocabulary size 129280. Preserve `expert_dtype="fp8"` and the complete
quantization configuration: dynamic FP8, e4m3, ue8m0 scales, and 128x128 blocks.
No W8A8/BF16 substitution or width reduction was performed.

This prefix preserves hash and learned routing with the original hash boundary,
and uncompressed, 4x, and 128x attention. Deeper repetitions are omitted.
MTP remains optional and requires a separate speculative decoding test; ordinary
serving does not execute it. The source has no DSpark configuration.

## Files and backup

Required model assets are `config.json`, `tokenizer.json`, `tokenizer_config.json`,
and `generation_config.json` (6,369,741 bytes in total). This README and
`run_serve.sh` provide usage instructions.

Removed `model.safetensors.index.json`, which references absent real-weight shards,
and `configuration.json`, which contains ModelScope framework/task metadata unused
by this local Hugging Face configuration. This saves 5,371,445 bytes.
No symlinks or custom model/tokenizer code are required by the supplied assets.
Original configuration and removed files remain backed up outside the model directory:

`/workspace/model_test/DeepSeek-V4-Flash-FP8_sgl_project-before-trim-lymdzqdk/`

The backup is not needed for serving. There is no chat template in the source;
use `/v1/completions` or explicitly supply a verified compatible chat template.

## Startup

Select free GPUs and run from the repository root:

```bash
CUDA_VISIBLE_DEVICES=0 TP=1 \
  bash tools/batched_test/models/DeepSeek-V4-Flash-FP8_sgl_project/run_serve.sh
```

The script resolves its own model directory, enables offline dummy loading, and
uses localhost:8000, context length 1024, one sequence, eager execution, and a 0.8
GPU memory budget. Override `PORT`, `TP`, and `GPU_MEMORY_UTILIZATION` as needed.
Ensure any inherited MACA visibility matches CUDA visibility. Additional arguments
are forwarded to vLLM. TP=1 is an example, not a measured memory/support guarantee.
Stop the service and its workers after testing.

```bash
curl --fail-with-body --max-time 10 http://127.0.0.1:8000/health
curl --fail-with-body --max-time 120 http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4-fp8-dummy","prompt":"Hello","max_tokens":4,"temperature":0,"ignore_eos":true}'
```

Confirm successful responses and multiple generated tokens. For sparse selection,
use more than 512 prompt tokens counted by this tokenizer, cross compression/window
boundaries, and reserve context for output. Short requests cannot establish that coverage.
Random outputs do not establish accuracy, representative routing, or full-model performance.

## Validation scope

The restoration checks compare against the original backup and require exactly the
two intended configuration changes. Offline validation loads the installed vLLM
`DeepseekV4Config` directly by file path, then loads AutoTokenizer and GenerationConfig,
checks vocabulary bounds and matching EOS IDs, and encodes a sample prompt.
The launch script is checked with `bash -n`.

The earlier normal vLLM configuration import failed because
`torchaudio/lib/libtorchaudio.so` could not resolve
`_ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_ib`.
That environment failure has not been rechecked or repaired during restoration.
Direct configuration-class loading bypasses the affected import chain and is not
proof of service startup. GPU construction, effective FP8 backend, memory usage,
prefill/decode, TP, MTP, and graph execution remain untested.
