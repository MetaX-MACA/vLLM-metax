# -*- coding: utf-8 -*-
"""
Requirements:
- vLLM: v0.11.0 or higher
- vLLM-metax: v0.11.0 or higher
- MACA SDK: 3.2.x.x or higher
"""
#  以nie3e/sentiment-polish-gpt2-small为例
# 模型下载可以使用modelscope和huggingface-cli
# Note:部分模型需要登陆huggingface官网，使用access token即可
# modelscope download nie3e/sentiment-polish-gpt2-small
# huggingface-cli download nie3e/sentiment-polish-gpt2-small

#离线推理代码
from vllm import LLM
if __name__ == '__main__':
    llm = LLM(model="nie3e/sentiment-polish-gpt2-small", task="classify")
    (output,) = llm.classify("Hello, my name is")
    probs = output.outputs.probs
    print(f"Class Probabilities: {probs!r} (size={len(probs)})")

#在线推理命令：
#模型部署
# vllm serve nie3e/sentiment-polish-gpt2-small --task classify --trust-remote-code  --port 8000

#在线推理
# curl http://localhost:8000/classify -H "Content-Type: application/json" -d '{"input":["Hello, my name is"],"model":"nie3e/sentiment-polish-gpt2-small"}'