# -*- coding: utf-8 -*-
"""
This example demonstrates how to use glm-4-9b-chat-hf model with vLLM.
Requirements:
- vLLM: v0.11.0 or higher
- transformers: 4.37.0 or higher
"""

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

import os
# Set multiprocessing method for worker processes
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

def prepare_inputs_for_vllm(messages, tokenizer):
    """
    Prepare input data for vLLM model inference.
    
    Args:
        messages (list): A list of messages containing text content.
        tokenizer (AutoTokenizer): The model tokenizer for tokenizing and formatting inputs.
        
    Returns:
        dict: A dictionary containing the formatted prompt.
    """
    # Apply chat template to generate text prompt
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Return formatted inputs for vLLM
    return {
        'prompt': text,
        'multi_modal_data': None
    }

def process_glm4_messages(messages):
    """
    Process messages specifically for GLM-4 format.
    This function is kept for consistency but simplified for text-only inputs.
    """
    processed_messages = []
    
    for message in messages:
        role = message['role']
        content = message['content']
        
        # If content is a list (mixed content), extract only text
        if isinstance(content, list):
            text_content = []
            for item in content:
                if item['type'] == 'text':
                    text_content.append(item['text'])
            # Join all text items
            processed_content = ' '.join(text_content) if text_content else ""
            processed_messages.append({'role': role, 'content': processed_content})
        else:
            processed_messages.append({'role': role, 'content': content})
    
    return processed_messages

if __name__ == '__main__':
    # Define input messages for GLM-4
    # Example 1: Text-only conversation
    messages = [
        {
            "role": "user",
            "content": "Please explain the basic concepts of machine learning."
        }
    ]

    checkpoint_path = "ZhipuAI/glm-4-9b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

    # Process messages for GLM-4 format if needed
    processed_messages = process_glm4_messages(messages)

    # Prepare inputs for the model
    inputs = [prepare_inputs_for_vllm(processed_messages, tokenizer)]

    # Initialize vLLM model
    llm = LLM(
        model=checkpoint_path,
        trust_remote_code=True,              # GLM models often require trust_remote_code
        max_model_len=8192,                  # Maximum sequence length
        gpu_memory_utilization=0.8,          # GPU memory utilization ratio
        tensor_parallel_size=torch.cuda.device_count(),  # Number of GPUs for tensor parallelism
        seed=42,                             # Random seed for reproducibility
        dtype=torch.float16                  # Use float16 for better performance
    )

    # Configure sampling parameters
    sampling_params = SamplingParams(
        temperature=0.7,        # Balanced creativity and determinism
        top_p=0.9,              # Nucleus sampling
        max_tokens=1024,        # Maximum number of generated tokens
        top_k=50,               # Top-k sampling
        stop_token_ids=[],      # No specific stop tokens
    )

    # Print input prompts for debugging
    for i, input_ in enumerate(inputs):
        print()
        print('=' * 60)
        print(f"Inputs[{i}]: {input_['prompt']=!r}")
    print('\n' + '>' * 60)

    # Generate outputs using the model
    outputs = llm.generate(inputs, sampling_params=sampling_params)
    
    # Print generated text
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        print()
        print('=' * 60)
        print(f"Generated text: {generated_text}")
        print('=' * 60)
    
    # Test with more deterministic parameters
    deterministic_params = SamplingParams(
        temperature=0.1,
        top_p=0.5,
        max_tokens=512,
    )
    
    deterministic_outputs = llm.generate(inputs, sampling_params=deterministic_params)
    for output in deterministic_outputs:
        print()
        print('=' * 40)
        print(f"Deterministic output: {output.outputs[0].text}")