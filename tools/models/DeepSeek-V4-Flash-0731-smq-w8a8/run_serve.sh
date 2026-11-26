vllm serve tools/models/DeepSeek-V4-Flash-0731-smq-w8a8 \
        --max-model-len 4096 \
        --load-format dummy \
        --gpu-memory-utilization 0.9 \
        --profiler-config '{"profiler": "torch", "torch_profiler_dir": "/tmp/vllm_profiler"}' \
        --speculative-config '{"method":"dspark","num_speculative_tokens":2}'

vllm bench serve --backend vllm --model tools/models/DeepSeek-V4-Flash-0731-smq-w8a8 \
    --dataset-name random --profile --num-prompts 2