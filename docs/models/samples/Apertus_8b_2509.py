# -*- coding: utf-8 -*-
"""
This example demonstrates how to use apertus-8b-2509 model with vLLM.
This model weight is randomly initialized and is only for testing purposes.

Requirements:
- vLLM: v0.11.0 or higher
- vLLM-metax: v0.11.0 or higher
- MACA SDK: 3.2.x.x or higher
"""

import argparse
import os
import sys
from typing import List
import torchvision
torchvision.disable_beta_transforms_warning()  # silence annoying warning

from vllm import LLM, SamplingParams

DEFAULT_PROMPTS: List[str] = [
    "Describe vLLM's core advantages in one paragraph.",
    "Give me 3 engineering recommendations to improve large language model throughput, and briefly explain why.",
    "Explain what tensor parallelism is and when to use it.",
    "Translate this sentence into Chinese: We are using vLLM to perform offline inference validation testing.",
    "Write a Python snippet (no more than 5 lines) showing how to use vLLM to call a local model to generate text.",
]

def load_prompts(args) -> List[str]:
    prompts: List[str] = []
    if args.prompt:
        prompts.extend([p.strip() for p in args.prompt if p.strip()])
    if args.prompt_file:
        path = args.prompt_file
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    p = line.strip()
                    if p:
                        prompts.append(p)
        else:
            print(f"[WARN] Prompt file does not exist: {path}")
    if not prompts:
        prompts = DEFAULT_PROMPTS.copy()
    return prompts

def main():
    # Add some args to make program configurable from command line
    parser = argparse.ArgumentParser(description="Minimal offline inference with vLLM (batch prompts, simplified)")
    parser.add_argument("--model", default="swiss-ai/Apertus-8B-2509",
                        help="Model name or local directory. If not provided, a small demo model is used. You can replace with e.g. meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("-p", "--prompt", action="append",
                        help="Repeatable; provide multiple prompts by specifying multiple times")
    parser.add_argument("--prompt-file",
                        help="Load prompts from a file, one per line")
    parser.add_argument("--max-tokens", type=int, default=128,
                        help="Maximum number of generated tokens")
    parser.add_argument("--temperature", type=float, default=0.2,
                        help="Sampling temperature")
    args = parser.parse_args()

    # Inference configures
    sampling = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    prompts = load_prompts(args)
    print(f"[INFO] Loading model: {args.model}")
    llm = LLM(
        model=args.model,          # online or local
        dtype="auto",
        trust_remote_code=True,    # allow custom model code from online repo
        tensor_parallel_size=1,    # adjustable by number of GPUs
        gpu_memory_utilization=0.90,
    )

    print(f"[INFO] Batch generation, total {len(prompts)} prompts")
    outputs = llm.generate(prompts, sampling)

    print("\n===== OUTPUT =====")
    for i, (inp, out) in enumerate(zip(prompts, outputs), 1):
        text = out.outputs[0].text.strip() if out.outputs else ""
        print(f"\n--- #{i} ---")
        print(f"Prompt:\n<< {inp}")
        print(f"Output:\n>> {text}")

    success = any(o.outputs and (o.outputs[0].text.strip() != "") for o in outputs)
    sys.exit(0 if success else 2)

if __name__ == "__main__":
    main()