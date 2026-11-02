#!/usr/bin/env bash
set -euo pipefail
MODEL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
GPU_IDS="${GPU_IDS:-0}"
TP="${TP:-1}"
PORT="${PORT:-8043}"
INDEXER_DTYPE="${INDEXER_DTYPE:-int8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-4096}"
export CUDA_VISIBLE_DEVICES="$GPU_IDS" MACA_VISIBLE_DEVICES="$GPU_IDS"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec /opt/venv/bin/python -m vllm.entrypoints.cli.main serve "$MODEL_DIR" \
  --served-model-name glm53-dummy --host 127.0.0.1 --port "$PORT" \
  --load-format dummy --tensor-parallel-size "$TP" \
  --max-model-len "$MAX_MODEL_LEN" --max-num-seqs 1 --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
  --gpu-memory-utilization 0.85 --enforce-eager \
  --limit-mm-per-prompt '{"image":1,"video":1}' \
  --attention-config "{\"indexer_kv_dtype\":\"$INDEXER_DTYPE\"}" "$@"
