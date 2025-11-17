# -*- coding: utf-8 -*-
"""
Requirements:
- vLLM: v0.11.0 or higher
- vLLM-metax: v0.11.0 or higher
- MACA SDK: 3.2.x.x or higher
"""

import argparse
import sys
import torchvision
torchvision.disable_beta_transforms_warning()

from vllm import LLM, SamplingParams

PROMPTS = [
    "Describe vLLM's core advantages in one paragraph.",
    "Give me 3 engineering recommendations to improve large language model throughput, and briefly explain why.",
    "Explain what tensor parallelism is and when to use it.",
    "Translate this sentence into Chinese: We are using vLLM to perform offline inference validation testing.",
    "Write a Python snippet (no more than 5 lines) showing how to use vLLM to call a local model to generate text.",
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ibm-granite/granite-4.0-tiny-preview", help="Model name or path.")
    parser.add_argument("--max-tokens", type=int, default=128, help="Max tokens.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Temp.")
    args = parser.parse_args()

    sampling = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        n=1,
    )

    print(f"[INFO] Loading model: {args.model}")
    llm = LLM(
        model=args.model,
        dtype="auto",
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,
    )

    print(f"[INFO] Generating {len(PROMPTS)} completions")
    outputs = llm.generate(PROMPTS, sampling)

    for i, (prompt, out) in enumerate(zip(PROMPTS, outputs), 1):
        text = out.outputs[0].text.strip() if out.outputs else ""
        print(f"\n[{i}]:\nUser: <<< {prompt}\nAssistant: >>> {text}")

    success = any(o.outputs and o.outputs[0].text.strip() for o in outputs)
    sys.exit(0 if success else 2)

if __name__ == "__main__":
    main()