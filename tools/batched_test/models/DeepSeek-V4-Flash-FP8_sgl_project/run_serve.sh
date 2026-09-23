#!/usr/bin/env bash
set -euo pipefail
MODEL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to available GPU IDs before launching}"
export MACA_VISIBLE_DEVICES="${MACA_VISIBLE_DEVICES:-$CUDA_VISIBLE_DEVICES}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
exec vllm serve "$MODEL_DIR" \
  --served-model-name deepseek-v4-fp8-dummy \
  --host 127.0.0.1 --port "${PORT:-8000}" \
  --load-format dummy --tensor-parallel-size "${TP:-1}" \
  --max-model-len 1024 --max-num-seqs 1 --max-num-batched-tokens 1024 \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.8}" \
  --enforce-eager "$@"
