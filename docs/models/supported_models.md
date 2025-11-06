# Supported Models

You could refer to [vllm's docs](https://docs.vllm.ai/en/stable/models/supported_models.html) for more details.

Here the plugin would list all the **tested** model on Maca.

## Feature Status Legend

- ✅︎ indicates that the feature is supported for the model.

- 🚧 indicates that the feature is planned but not yet supported for the model.

- ⚠️ indicates that the feature is available but may have known issues or limitations.

## List of Text-only Language Models

### Text Generative Models

| Architecture | Models | Example HF Models | [LoRA](https://docs.vllm.ai/en/stable/features/lora.html) | [PP](https://docs.vllm.ai/en/stable/serving/parallelism_scaling.html) |
|--------------|--------|-------------------|----------------------|---------------------------|
| `AquilaForCausalLM` | Aquila, Aquila2 | `BAAI/Aquila-7B`, `BAAI/AquilaChat-7B`, etc. | ✅︎ | ✅︎ |
| `BaiChuanForCausalLM` | Baichuan2, Baichuan | `baichuan-inc/Baichuan2-13B-Chat`, `baichuan-inc/Baichuan-7B`, etc. | ✅︎ | ✅︎ |
| `BloomForCausalLM` | BLOOM, BLOOMZ, BLOOMChat | `bigscience/bloom`, `bigscience/bloomz`, etc. | | ✅︎ |
| `ChatGLMModel`, `ChatGLMForConditionalGeneration` | ChatGLM | `zai-org/chatglm2-6b`, `zai-org/chatglm3-6b`, etc. | ✅︎ | ✅︎ |
| `DeepseekForCausalLM` | DeepSeek | `deepseek-ai/deepseek-llm-7b-chat`, etc. | ✅︎ | ✅︎ |
| `DeepseekV2ForCausalLM` | DeepSeek-V2 | `deepseek-ai/DeepSeek-V2`, `deepseek-ai/DeepSeek-V2-Chat`, etc. | ✅︎ | ✅︎ |
| `DeepseekV3ForCausalLM` | DeepSeek-V3 | `deepseek-ai/DeepSeek-V3`, `deepseek-ai/DeepSeek-R1`, `deepseek-ai/DeepSeek-V3.1`, etc. | ✅︎ | ✅︎ |
| `Ernie4_5ForCausalLM` | Ernie4.5 | `baidu/ERNIE-4.5-0.3B-PT`, etc. | ✅︎ | ✅︎ |
| `Ernie4_5_MoeForCausalLM` | Ernie4.5MoE | `baidu/ERNIE-4.5-21B-A3B-PT`, `baidu/ERNIE-4.5-300B-A47B-PT`, etc. |✅︎| ✅︎ |
| `FalconForCausalLM` | Falcon | `tiiuae/falcon-7b`, `tiiuae/falcon-40b`, `tiiuae/falcon-rw-7b`, etc. | | ✅︎ |
| `GlmForCausalLM` | GLM-4 | `zai-org/glm-4-9b-chat-hf`, etc. | ✅︎ | ✅︎ |
| `Glm4ForCausalLM` | GLM-4-0414 | `zai-org/GLM-4-32B-0414`, etc. | ✅︎ | ✅︎ |
| `Glm4MoeForCausalLM` | GLM-4.5, GLM-4.6 | `zai-org/GLM-4.5`, etc. | ✅︎ | ✅︎ |
| `GPTBigCodeForCausalLM` | StarCoder, SantaCoder, WizardCoder | `bigcode/starcoder`, `bigcode/gpt_bigcode-santacoder`, `WizardLM/WizardCoder-15B-V1.0`, etc. | ✅︎ | ✅︎ |
| `GPTNeoXForCausalLM` | GPT-NeoX, Pythia, OpenAssistant, Dolly V2, StableLM | `EleutherAI/gpt-neox-20b`, `EleutherAI/pythia-12b`, `OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5`, `databricks/dolly-v2-12b`, `stabilityai/stablelm-tuned-alpha-7b`, etc. | | ✅︎ |
| `InternLMForCausalLM` | InternLM | `internlm/internlm-7b`, `internlm/internlm-chat-7b`, etc. | ✅︎ | ✅︎ |
| `InternLM2ForCausalLM` | InternLM2 | `internlm/internlm2-7b`, `internlm/internlm2-chat-7b`, etc. | ✅︎ | ✅︎ |
| `InternLM3ForCausalLM` | InternLM3 | `internlm/internlm3-8b-instruct`, etc. | ✅︎ | ✅︎ |
| `LlamaForCausalLM` | Llama 3.1, Llama 3, Llama 2, LLaMA, Yi | `meta-llama/Meta-Llama-3.1-405B-Instruct`, `meta-llama/Meta-Llama-3.1-70B`, `meta-llama/Meta-Llama-3-70B-Instruct`, `meta-llama/Llama-2-70b-hf`, `01-ai/Yi-34B`, etc. | ✅︎ | ✅︎ |
| `MixtralForCausalLM` | Mixtral-8x7B, Mixtral-8x7B-Instruct | `mistralai/Mixtral-8x7B-v0.1`, `mistralai/Mixtral-8x7B-Instruct-v0.1`, `mistral-community/Mixtral-8x22B-v0.1`, etc. | ✅︎ | ✅︎ |
| `MPTForCausalLM` | MPT, MPT-Instruct, MPT-Chat, MPT-StoryWriter | `mosaicml/mpt-7b`, `mosaicml/mpt-7b-storywriter`, `mosaicml/mpt-30b`, etc. | | ✅︎ |
| `QWenLMHeadModel` | Qwen | `Qwen/Qwen-7B`, `Qwen/Qwen-7B-Chat`, etc. | ✅︎ | ✅︎ |
| `Qwen2ForCausalLM` | QwQ, Qwen2 | `Qwen/QwQ-32B-Preview`, `Qwen/Qwen2-7B-Instruct`, `Qwen/Qwen2-7B`, etc. | ✅︎ | ✅︎ |
| `Qwen2MoeForCausalLM` | Qwen2MoE | `Qwen/Qwen1.5-MoE-A2.7B`, `Qwen/Qwen1.5-MoE-A2.7B-Chat`, etc. | ✅︎ | ✅︎ |
| `Qwen3ForCausalLM` | Qwen3 | `Qwen/Qwen3-8B`, etc. | ✅︎ | ✅︎ |
| `Qwen3MoeForCausalLM` | Qwen3MoE | `Qwen/Qwen3-30B-A3B`, etc. | ✅︎ | ✅︎ |
| `Qwen3NextForCausalLM` | Qwen3NextMoE | `Qwen/Qwen3-Next-80B-A3B-Instruct`, etc. | ✅︎ | ✅︎ |

## List of Multimodal Language Models

The following modalities are supported depending on the model:

- **T**ext
- **I**mage
- **V**ideo
- **A**udio

Any combination of modalities joined by `+` are supported.

- e.g.: `T + I` means that the model supports text-only, image-only, and text-with-image inputs.

On the other hand, modalities separated by `/` are mutually exclusive.

- e.g.: `T / I` means that the model supports text-only and image-only inputs, but not text-with-image inputs.

See [this page](https://docs.vllm.ai/en/stable/features/multimodal_inputs.html) on how to pass multi-modal inputs to the model.


### Text Generative Models

| Architecture | Models | Inputs | Example HF Models | [LoRA](https://docs.vllm.ai/en/stable/features/lora.html) | [PP](https://docs.vllm.ai/en/stable/serving/parallelism_scaling.html) |
|--------------|--------|--------|-------------------|----------------------|---------------------------|
| `DeepseekVLV2ForCausalLM`<sup>^</sup> | DeepSeek-VL2 | T + I<sup>+</sup> | `deepseek-ai/deepseek-vl2-tiny`, `deepseek-ai/deepseek-vl2-small`, `deepseek-ai/deepseek-vl2`, etc. | | ✅︎ |
| `DeepseekOCRForCausalLM` | DeepSeek-OCR | T + I<sup>+</sup> | `deepseek-ai/DeepSeek-OCR`, etc. | | ✅︎ |
| `Ernie4_5_VLMoeForConditionalGeneration` | Ernie4.5-VL | T + I<sup>+</sup>/ V<sup>+</sup> | `baidu/ERNIE-4.5-VL-28B-A3B-PT`, `baidu/ERNIE-4.5-VL-424B-A47B-PT` | | ✅︎ |
| `GLM4VForCausalLM`<sup>^</sup> | GLM-4V | T + I | `zai-org/glm-4v-9b`, `zai-org/cogagent-9b-20241220`, etc. | ✅︎ | ✅︎ |
| `Glm4vForConditionalGeneration` | GLM-4.1V-Thinking | T + I<sup>E+</sup> + V<sup>E+</sup> | `zai-org/GLM-4.1V-9B-Thinking`, etc. | ✅︎ | ✅︎ |
| `Glm4vMoeForConditionalGeneration` | GLM-4.5V | T + I<sup>E+</sup> + V<sup>E+</sup> | `zai-org/GLM-4.5V`, etc. | ✅︎ | ✅︎ |
| `InternS1ForConditionalGeneration` | Intern-S1 | T + I<sup>E+</sup> + V<sup>E+</sup> | `internlm/Intern-S1`, `internlm/Intern-S1-mini`, etc. | ✅︎ | ✅︎ |
| `InternVLChatModel` | InternVL 3.5, InternVL 3.0, InternVideo 2.5, InternVL 2.5, Mono-InternVL, InternVL 2.0 | T + I<sup>E+</sup> + (V<sup>E+</sup>) | `OpenGVLab/InternVL3_5-14B`, `OpenGVLab/InternVL3-9B`, `OpenGVLab/InternVideo2_5_Chat_8B`, `OpenGVLab/InternVL2_5-4B`, `OpenGVLab/Mono-InternVL-2B`, `OpenGVLab/InternVL2-4B`, etc. | ✅︎ | ✅︎ |
| `InternVLForConditionalGeneration` | InternVL 3.0 (HF format) | T + I<sup>E+</sup> + V<sup>E+</sup> | `OpenGVLab/InternVL3-1B-hf`, etc. | ✅︎ | ✅︎ |
| `LlavaForConditionalGeneration` | LLaVA-1.5, Pixtral (HF Transformers) | T + I<sup>E+</sup> | `llava-hf/llava-1.5-7b-hf`, `TIGER-Lab/Mantis-8B-siglip-llama3` (see note), `mistral-community/pixtral-12b`, etc. | | ✅︎ |
| `LlavaNextForConditionalGeneration` | LLaVA-NeXT | T + I<sup>E+</sup> | `llava-hf/llava-v1.6-mistral-7b-hf`, `llava-hf/llava-v1.6-vicuna-7b-hf`, etc. | | ✅︎ |
| `LlavaNextVideoForConditionalGeneration` | LLaVA-NeXT-Video | T + V | `llava-hf/LLaVA-NeXT-Video-7B-hf`, etc. | | ✅︎ |
| `LlavaOnevisionForConditionalGeneration` | LLaVA-Onevision | T + I<sup>+</sup> + V<sup>+</sup> | `llava-hf/llava-onevision-qwen2-7b-ov-hf`, `llava-hf/llava-onevision-qwen2-0.5b-ov-hf`, etc. | | ✅︎ |
| `Qwen2_5_VLForConditionalGeneration` | Qwen2.5-VL | T + I<sup>E+</sup> + V<sup>E+</sup> | `Qwen/Qwen2.5-VL-3B-Instruct`, `Qwen/Qwen2.5-VL-72B-Instruct`, etc. | ✅︎ | ✅︎ |
| `Qwen2_5OmniThinkerForConditionalGeneration` | Qwen2.5-Omni | T + I<sup>E+</sup> + V<sup>E+</sup> + A<sup>+</sup> | `Qwen/Qwen2.5-Omni-3B`, `Qwen/Qwen2.5-Omni-7B` | ✅︎ | ✅︎ |
| `Qwen3VLForConditionalGeneration` | Qwen3-VL | T + I<sup>E+</sup> + V<sup>E+</sup> | `Qwen/Qwen3-VL-4B-Instruct`, etc. | ✅︎ | ✅︎ |
| `Qwen3VLMoeForConditionalGeneration` | Qwen3-VL-MOE | T + I<sup>E+</sup> + V<sup>E+</sup> | `Qwen/Qwen3-VL-30B-A3B-Instruct`, etc. | ✅︎ | ✅︎ |
| `Qwen3OmniMoeThinkerForConditionalGeneration` | Qwen3-Omni | T + I<sup>E+</sup> + V<sup>E+</sup> + A<sup>+</sup> | `Qwen/Qwen3-Omni-30B-A3B-Instruct`, `Qwen/Qwen3-Omni-30B-A3B-Thinking` | ✅︎ | ✅︎ |

