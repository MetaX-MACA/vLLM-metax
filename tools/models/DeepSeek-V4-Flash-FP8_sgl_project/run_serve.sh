#!/usr/bin/env bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
vllm serve tools/models/DeepSeek-V4-Flash-FP8_sgl_project \
    --max-model-len 1024 \
    --load-format dummy \
    --served-model-name deepseek-v4-fp8-dummy \
    --host 127.0.0.1 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --max-num-seqs 1 \
    --max-num-batched-tokens 1024 \
    --gpu-memory-utilization 0.8 \
    --enforce-eager \
    "$@"
