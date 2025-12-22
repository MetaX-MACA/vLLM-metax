# -*- coding: utf-8 -*-
"""
Requirements:
- vLLM: v0.11.0 or higher
- vLLM-metax: v0.11.0 or higher
- MACA SDK: 3.2.x.x or higher
"""
#  以Qwen/Qwen-7B为例
# 模型下载可以使用modelscope和huggingface-cli
# Note:部分模型需要登陆huggingface官网，使用access token即可
# modelscope download Qwen/Qwen-7B
# huggingface-cli download Qwen/Qwen-7B


#离线推理代码
from vllm import LLM, SamplingParams
import os
MODEL_PATH = "./models/Qwen/Qwen-7B" # 可修改
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=200
)
if __name__ == '__main__':
    llm = LLM(
        model=MODEL_PATH,
        trust_remote_code=True,  
        tensor_parallel_size=1,  # 可修改实现多卡并行
        gpu_memory_utilization=0.8,  
        disable_log_stats=True,
        load_format="auto",  
        dtype="auto"  
    )
    # 测试提示词
    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "请简单介绍 Qwen-7B 模型的特点",
    ]
    # 生成结果
    outputs = llm.generate(prompts, sampling_params)
    # 格式化输出
    for i, output in enumerate(outputs, 1):
        prompt = output.prompt
        generated_text = output.outputs[0].text.strip()
        print(f"\n===== 第 {i} 条结果 =====")
        print(f"Prompt: {prompt!r}")
        print(f"Generated text: {generated_text!r}")

#在线推理命令：
#模型部署
# vllm serve ./models/Qwen/Qwen-7B --trust-remote-code --tensor-parallel-size=1 --port=8000 --gpu_memory_utilization=0.8 &

#在线推理
# curl http://localhost:8000/v1/completions -H "Content-Type: application/json" -d '{"model":"Qwen/Qwen-7B","prompt":"introduce yourself","max_tokens":200,"temperature":0.6}'
