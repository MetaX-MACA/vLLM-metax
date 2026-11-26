#!/usr/bin/env bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
vllm serve tools/models/Qwen3.8-Flash-Next-dummy-3layers \
    --max-model-len 4096 \
    --load-format dummy \
    --served-model-name qwen-flash-dummy \
    --host 127.0.0.1 \
    --port 8000 \
    --dtype bfloat16 \
    --tensor-parallel-size 4 \
    --pipeline-parallel-size 1 \
    --max-num-seqs 1 \
    --max-num-batched-tokens 4096 \
    --gpu-memory-utilization 0.8 \
    --no-enable-prefix-caching \
    --enforce-eager \
    "$@"
