#!/usr/bin/env bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
vllm serve tools/models/GLM-5.3-Flash-dummy-3layers-w8a8 \
    --max-model-len 4096 \
    --load-format dummy \
    --served-model-name glm53-dummy \
    --host 127.0.0.1 \
    --port 8043 \
    --tensor-parallel-size 1 \
    --max-num-seqs 1 \
    --max-num-batched-tokens 4096 \
    --gpu-memory-utilization 0.85 \
    --enforce-eager \
    --limit-mm-per-prompt '{"image":1,"video":1}' \
    --attention-config '{"indexer_kv_dtype":"int8"}' \
    "$@"
