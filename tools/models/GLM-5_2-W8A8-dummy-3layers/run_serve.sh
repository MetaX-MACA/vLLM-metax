#!/usr/bin/env bash
vllm serve tools/models/GLM-5_2-W8A8-dummy-3layers \
    --max-model-len 4096 \
    --load-format dummy \
    --tensor-parallel-size 1 \
    --max-num-seqs 1 \
    --max-num-batched-tokens 4096 \
    --gpu-memory-utilization 0.8 \
    --enforce-eager \
    "$@"
