# SPDX-License-Identifier: Apache-2.0
"""
This example demonstrates how to use Qwen-3-VL-2B-Instruct model with vLLM.
Requirements:
- vLLM: v0.11.0 or higher
- vLLM-metax: v0.11.0 or higher
- MACA SDK: 3.2.x.x or higher
"""

import os

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

# Set multiprocessing method for worker processes
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'


def prepare_inputs_for_vllm(messages, processor):
    """
    Prepare input data for vLLM model inference.
    
    Args:
        messages (list): A list of messages containing text and/or image/video content.
        processor (AutoProcessor): The model processor for tokenizing and formatting inputs.
        
    Returns:
        dict: A dictionary containing the formatted prompt and multimodal data.
    """
    # Apply chat template to generate text prompt
    text = processor.apply_chat_template(messages,
                                         tokenize=False,
                                         add_generation_prompt=True)

    # Process vision information (images/videos) from messages
    # Requires qwen_vl_utils version 0.0.14 or higher
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True)
    print(f"video_kwargs: {video_kwargs}")

    # Prepare multimodal data dictionary
    mm_data = {}
    if image_inputs is not None:
        mm_data['image'] = image_inputs
    if video_inputs is not None:
        mm_data['video'] = video_inputs

    # Return formatted inputs for vLLM
    return {
        'prompt': text,
        'multi_modal_data': mm_data,
        'mm_processor_kwargs': video_kwargs
    }


if __name__ == '__main__':
    # Define input messages - currently using an image recognition task
    # Alternative: video input (commented out)
    # messages = [
    #     {
    #         "role": "user",
    #         "content": [
    #             {
    #                 "type": "video",
    #                 "video": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-VL/space_woaudio.mp4",
    #             },
    #             {"type": "text", "text": "这段视频有多长"},
    #         ],
    #     }
    # ]

    messages = [{
        "role":
        "user",
        "content": [
            {
                "type":
                "image",
                "image":
                "https://ofasys-multimodal-wlcb-3-toshanghai.oss-accelerate.aliyuncs.com/wpf272043/keepme/image/receipt.png",
            },
            {
                "type": "text",
                "text": "Read all the text in the image."
            },
        ],
    }]

    # Load model processor
    checkpoint_path = "Qwen/Qwen3-VL-2B-Instruct"
    processor = AutoProcessor.from_pretrained(checkpoint_path)

    # Prepare inputs for the model
    inputs = [
        prepare_inputs_for_vllm(message, processor) for message in [messages]
    ]

    # Initialize vLLM model
    llm = LLM(
        model=checkpoint_path,
        mm_encoder_tp_mode="data",  # Multimodal encoder tensor parallel mode
        enable_expert_parallel=False,  # Disable expert parallelism
        max_model_len=4096,  # Maximum sequence length
        gpu_memory_utilization=0.6,  # GPU memory utilization ratio
        tensor_parallel_size=torch.cuda.device_count(
        ),  # Number of GPUs for tensor parallelism
        seed=0  # Random seed for reproducibility
    )

    # Configure sampling parameters
    sampling_params = SamplingParams(
        temperature=0,  # Deterministic output (greedy decoding)
        max_tokens=1024,  # Maximum number of generated tokens
        top_k=-1,  # Disable top-k sampling
        stop_token_ids=[],  # No specific stop tokens
    )

    # Print input prompts for debugging
    for i, input_ in enumerate(inputs):
        print()
        print('=' * 40)
        print(f"Inputs[{i}]: {input_['prompt']=!r}")
    print('\n' + '>' * 40)

    # Generate outputs using the model
    outputs = llm.generate(inputs, sampling_params=sampling_params)

    # Print generated text
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        print()
        print('=' * 40)
        print(f"Generated text: {generated_text!r}")
