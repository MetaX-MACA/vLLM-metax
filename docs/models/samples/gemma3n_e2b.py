# -*- coding: utf-8 -*-
"""
This example demonstrates how to use gemma-3n-E2B-it model with vLLM.
Requirements:
- vLLM: v0.11.0 or higher
- vLLM-metax: v0.11.0 or higher
- MACA SDK: 3.2.x.x or higher
- timm: run `pip install timm`
"""

from vllm import LLM, SamplingParams

if __name__ == "__main__":
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=200
    )

    llm = LLM(model="google/gemma-3n-E2B-it", trust_remote_code=True)

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
