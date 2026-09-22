#export CUDA_VISIBLE_DEVICES=2
VLLM_USE_V2_MODEL_RUNNER=0 VLLM_METAX_USE_FP8_WO_A=0 \
vllm serve ./temp/DeepSeek-V4-Flash-0731-smq-w8a8 \
        --max-model-len 1024 \
	--max-num-seqs 2 \
        --load-format dummy \
	--gpu-memory-utilization 0.85 \
        --speculative-config '{"method":"dspark","num_speculative_tokens":2}'
