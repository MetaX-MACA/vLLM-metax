# GLM-5.2 W8A8 三层 dummy 模型

来源：`/mxstorage/pde_ai/models/llm/ChatGLM/GLM-5_2-W8A8`。
此目录不包含权重，也不依赖源目录的软链接，可整体复制到没有原权重的机器。
必须使用 `--load-format dummy`：仍然创建并分配模型参数，只跳过权重读取，使用随机参数。
适合验证加载、算子和执行路径；输出没有原模型的语言能力，MoE 路由和性能也不代表原模型。

## 裁剪内容

层号从 0 开始，保留原主干层的三种组合：

| 新层号 | 原层号（结构参考） | MLP | DSA indexer |
| --- | --- | --- | --- |
| 0 | 0 | Dense | full，计算索引 |
| 1 | 6 | MoE | full，计算索引 |
| 2 | 7 | MoE | shared，复用前层索引 |

仅裁剪层数，hidden_size=6144、256 个路由专家、1 个共享专家、每 token 选择 8 个专家、head/LoRA 维度与 W8A8 配置均保留。

`config.json` 的修改：

- `num_hidden_layers`: 78 → 3。
- `first_k_dense_replace`: 3 → 1；`moe_layer_freq=1` 保持。
- `layer_types`: 3 个 `deepseek_sparse_attention`。
- `mlp_layer_types`: `["dense", "sparse", "sparse"]`。
- `indexer_types`: `["full", "full", "shared"]`。
- `index_topk_pattern`: null → `"FFS"`。MetaX 的 `deepseek_v2.py` 优先读取这个显式模式；原 `index_topk_freq=4` 和 `index_skip_topk_offset=3` 保留但不再控制主干调度。
- `compression_config.ignore`: 按原层号 0→0、6→1、7→2、78→3 重映射，删除未保留层的条目；保持路由 gate、indexer.weights_proj 和 lm_head 的非量化设置。
- `num_nextn_predict_layers=1` 保留，原 MTP 层 78 的量化排除项迁到新层 3。普通启动不运行 MTP；须显式开启 speculative decoding 才会创建和执行它。

不应仅把 `num_hidden_layers` 改为 3：那会只留下 Dense 层；也不应只改 `indexer_types`，因为本版本 MetaX 实现使用 `index_topk_pattern` / freq / offset 决定是否创建 indexer。

## 文件需求

| 文件 | 用途 |
| --- | --- |
| `config.json` | 模型结构及内嵌 compressed-tensors W8A8 配置，必须 |
| `tokenizer.json` | 完整 tokenizer 数据，常规文本输入需要 |
| `tokenizer_config.json` | tokenizer 类、特殊 token 和行为配置，配套保留 |
| `chat_template.jinja` | 聊天消息格式，使用 chat 接口时保留 |
| `generation_config.json` | 原模型 EOS 和默认生成参数；可省略，但保留更完整 |

`*.safetensors` 和 `model.safetensors.index.json` 不需要；`llmc_qconfig.yaml` 是量化工具配置，本模型 dummy 加载不需要。
此模型依赖运行环境内置的 `glm_moe_dsa` 配置类和 MetaX 注册的 `GlmMoeDsaForCausalLM`，源目录没有需要复制的自定义 Python 文件。
其他模型若使用 `auto_map` / 自定义 tokenizer，或者把量化配置放在独立文件中，则还需带上对应代码/配置，不能一概只复制这五个文件。

## 启动

在已安装对应 vLLM 和 vllm_metax 的环境中运行。根据空闲情况选择 GPU；示例使用 2、3。

```bash
cd /root/dummy_home/workspace/vLLM-metax-0.27.0-mx
MODEL="$PWD/temp/GLM-5_2-W8A8-dummy-3layers"

# 单卡
CUDA_VISIBLE_DEVICES=2 MACA_VISIBLE_DEVICES=2 vllm serve "$MODEL" \
  --load-format dummy --tensor-parallel-size 1 \
  --max-model-len 4096 --max-num-seqs 1 --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.8 --enforce-eager

# 双卡，张量并行
CUDA_VISIBLE_DEVICES=2,3 MACA_VISIBLE_DEVICES=2,3 vllm serve "$MODEL" \
  --load-format dummy --tensor-parallel-size 2 \
  --max-model-len 4096 --max-num-seqs 1 --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.8 --enforce-eager
```

需要覆盖 MTP 路径时，在启动命令末尾添加：

```bash
--speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

使用 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` 可要求完全离线。
若缓存目录只读，可设 `FLASHINFER_WORKSPACE_BASE=/tmp/glm-dummy-cache XDG_CACHE_HOME=/tmp/glm-dummy-cache`。

按原 safetensors 文件头的张量字节数计算，三层主干连同 embedding/lm_head 共约 22.32 GiB；一个 MTP 层额外约 9.35 GiB。这是原存储张量的大小估算，不是实测运行显存：布局转换、KV cache、临时缓冲区和 TP 的复制参数都会影响实际占用。
原配置支持 1048576 上下文，但启动时先限制为 4096，避免长上下文压测带来的显存需求。单卡能否运行还取决于可用显存和内核支持，不能只按卡数保证。
短输入可验证启动和 decode；要验证超过 `index_topk=2048` 的稀疏 token 选择场景，应另加长度大于 2048 的输入。
