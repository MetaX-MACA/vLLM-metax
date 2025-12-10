#!/bin/bash

vllm bench sweep serve \
    --serve-cmd 'vllm serve /mnt/share/models/Qwen/Qwen3-4B -tp 2 -pp 1 -dcp 1 --trust-remote-code --max-model-len 4096 --swap-space 16 --gpu-memory-utilization 0.82' \
    --bench-cmd 'vllm bench serve --model /mnt/share/models/Qwen/Qwen3-4B --save-result --dataset-name random --num-prompts 32 --max-concurrency 32 --ignore-eos  --save-result --result-dir /workspace' \
    --bench-params /workspace/bench.json \
    -o /workspace/results

vllm bench sweep serve \
    --serve-cmd 'vllm serve /path/to/example-model --port 8000 -tp 99 -pp 99 -dp 99 --trust-remote-code --gpu-memory-utilization 0.9 --swap-space 16 --max-model-len 4096 --distributed-executor-backend ray -mtp --chat-template chat_template/tool_chat_template_deepseekr1.jinja' \
    --bench-cmd 'vllm bench serve --model /path/to/example-model --port 8000 --dataset-name random --ignore-eos' \
    --bench-params /root/dummy_home/vllm-metax/tools/batched_test/bench_params/bench_1.json \
    --output-dir /workspace/model_test_2025_12_10_14:40/performance/example-model_tp1_pp1_dp1

