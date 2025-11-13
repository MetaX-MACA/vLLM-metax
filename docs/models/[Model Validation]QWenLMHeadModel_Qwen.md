## Purpose

Qwen/Qwen-7B, , etc.Qwen/Qwen-7B-Chat模型验证

## Test Plan

1.使用ModelScope 下载模型

2.离线推理验证 

离线推理验证代码：

```
from vllm import LLM, SamplingParams
import os

MODEL_PATH = "./models/Qwen/Qwen-7B-Chat"

sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=200
)

if __name__ == '__main__':
    llm = LLM(
        model=MODEL_PATH,
        trust_remote_code=True,  
        tensor_parallel_size=1,  
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
```

3.在线推理验证 

部署模型启动 vLLM 服务器

```
nohup vllm serve ./models/Qwen/Qwen-7B --trust-remote-code --tensor-parallel-size=1 --port=8000 --gpu_memory_utilization=0.8 &
```

## Test Result

### 离线推理验证：

#### Qwen/Qwen-7B

![image-20251113163214186](C:\Users\王家玉\AppData\Roaming\Typora\typora-user-images\image-20251113163214186.png)

![image-20251113163230635](C:\Users\王家玉\AppData\Roaming\Typora\typora-user-images\image-20251113163230635.png)

#### Qwen/Qwen-7B-Chat

![image-20251113163247163](C:\Users\王家玉\AppData\Roaming\Typora\typora-user-images\image-20251113163247163.png)

![image-20251113163258989](C:\Users\王家玉\AppData\Roaming\Typora\typora-user-images\image-20251113163258989.png)

### 在线推理验证截图

#### Qwen/Qwen-7B

![image-20251113163321670](C:\Users\王家玉\AppData\Roaming\Typora\typora-user-images\image-20251113163321670.png)

#### Qwen/Qwen-7B-Chat 

![image-20251113163332396](C:\Users\王家玉\AppData\Roaming\Typora\typora-user-images\image-20251113163332396.png)