# Qwen3.8-Flash-Next：三层 dummy 模型

源目录：`/mxstorage/pde_ai/models/llm/Qwen/Qwen3.8-Flash-Next`。源文件未修改。
本目录只包含配置和配套资源，必须使用 `--load-format dummy`，没有复制权重或权重索引。
dummy 仍会创建完整尺寸的参数；随机输出不能用于准确率、真实路由分布或性能评估。

## 最小结构与层映射

文本主干 **48 → 3 层**，视觉编码器 **27 → 1 层**，MTP **1 层保持**。
层号除特别说明外均从 0 开始。

| 新层 | 原层 | 保留的执行路径 |
| --- | --- | --- |
| 文本 0 | 文本 0 | 无 PLE 的 Gated DeltaNet 线性注意力、MoE、HyperConnection |
| 文本 1 | 文本 1 | PLE n-gram 查表及短卷积、线性注意力、MoE、HyperConnection |
| 文本 2 | 文本 3 | QSA 索引器、压缩键缓存及稀疏注意力、MoE、HyperConnection |
| 视觉 0 | 视觉 0 | 完整视觉 block，保留 patch embedding、位置编码、merger，供图像和视频共用 |
| MTP 运行时层 3 | MTP 运行时层 48 | 保留一个预测层及 HC/注意力/MoE；checkpoint 自有命名空间仍是 `mtp.layers.0` |

三层是**保持原先顺序、覆盖三种原有 block 组合**的最小主干。
两层虽然可以包含 PLE 和 QSA，但会失去无 PLE 的线性注意力 block，或者使 PLE
移到第一层，跳过其处理前一层 pending HC 状态的路径。本方案还保留两个相邻线性层的状态传递。
视觉 blocks 同构，`deepstack_visual_indexes=[]`，因此只需一个 block；没有跨深度特征选择需要迁移。

`ple_layer_ids=[2]` 使用 **1-based** 索引，仍挂在新文本层 1。
其 dense PLE 编号仍为 0，因此 n-gram 哈希种子、词表布局和表尺寸保持；未删除查表模块。
原配置没有启用 Dense FFN、DeepStack 或 DSpark，不额外引入这些功能。

## 配置修改

仅修改以下四个字段，其他源 JSON 字段全部保留：

| 字段 | 原值 | 新值 |
| --- | --- | --- |
| `text_config.num_hidden_layers` | 48 | 3 |
| `text_config.layer_types` | 48 个条目 | `[linear_attention, linear_attention, full_attention]` |
| `text_config.full_attention_interval` | 4 | 3 |
| `vision_config.depth` | 27 | 1 |

构造器直接读取 `layer_types[layer_idx]`；周期字段同步为 3，使重新推导时也得到相同的层模式。
`mtp_num_hidden_layers=1`、`mtp.num_hidden_layers=1`、`mtp.layer_types=[full_attention]`
及 `mtp_use_hidden_state_from_layer=null` 保留；MTP 起始层由新的主干深度自动推导。
没有量化路径、每层量化排除项或其它跨层引用需要重映射，原模型为 BF16。

保持 hidden_size=2560、512 个路由专家、top-k=10、共享专家宽度 640、MoE 宽度 640、
24 个 Q heads / 2 个 KV heads、head_dim=256、线性注意力 16 个 K heads / 48 个 V heads、
hc_count=4、hc_lowrank=320、PLE/n-gram 全部尺寸、RoPE、词表和原始最大上下文 262144。
不设置 `language_model_only`，保留图像和视频路径。

## 资源与并行

仅检查原 safetensors 文件头，没有读取权重 payload。
按保留张量的原存储字节数估算：

| 模块 | GiB |
| --- | ---: |
| PLE（包含查表、投影和卷积） | 95.4291 |
| 其它文本主干、embedding、lm_head | 16.8632 |
| 单层视觉编码器及公共视觉模块 | 0.0982 |
| 普通主模型小计 | 112.3905 |
| MTP 额外张量 | 4.8562 |

PLE 的 128 个**存储分片**合计 320001536 行、每行 160 个 BF16 元素；分片数不等于 TP。
dummy 模式无需这些源分片，但依然分配对应的参数表。详情见 `source_tensor_inventory.json`。
以上不是实测 GPU 显存，尚未计入缓存、临时张量、并行复制及布局转换等开销。

建议先用空闲的 4 张 64 GiB 卡进行 TP4 验证。
TP4/TP8 都满足保留维度的基本整除关系；QSA 的 KV heads 在 TP 大于 2 时复制，索引器为复制投影。
具体内核支持和实际显存仍需运行验证。减少层数不要求减少 TP，三层也不需要被 TP 整除。

## 启动和功能触发

在已安装配套 vLLM、vllm_metax 和 Transformers 的环境中，从仓库根目录执行。
默认使用 TP4，需要选择四张空闲 GPU。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 MACA_VISIBLE_DEVICES=0,1,2,3 \
    bash tools/models/Qwen3.8-Flash-Next-dummy-3layers/run_serve.sh

# 单独验证 MTP，先停止上一个服务。
CUDA_VISIBLE_DEVICES=0,1,2,3 MACA_VISIBLE_DEVICES=0,1,2,3 \
    bash tools/models/Qwen3.8-Flash-Next-dummy-3layers/run_serve.sh \
    --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

启动脚本使用 dummy、BF16、PP1、4096 上下文、单请求和 eager，关闭 prefix caching，
保留视觉编码器。可在脚本路径后追加 vLLM 参数调整端口、TP 和显存预算。
更改 TP 时同步选择对应数量的卡。

```bash
curl --fail-with-body --max-time 10 http://127.0.0.1:8000/health
curl --fail-with-body --max-time 120 http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen-flash-dummy","prompt":"Hello, world!","max_tokens":8,"temperature":0,"ignore_eos":true}'
```

| 目标 | 请求/配置触发条件 | 本次状态 |
| --- | --- | --- |
| 线性注意力 prefill/decode、MoE、HC、PLE 短卷积 | 多 token 输入并实际生成至少 4 个 token；较长输入覆盖 9-token PLE 卷积历史 | 结构保留，GPU 未验证 |
| PLE n-gram 跨步上下文 | 至少 3-token 输入和多步 decode；跨 chunk、EOS 边界需另测 | 结构保留，GPU 未验证 |
| QSA 长输入筛选及压缩 | 用实际 tokenizer 构造约 3000-token prompt，超过 budget=2048；compress_ratio=4，block top-k=512 | 结构保留，GPU 未验证 |
| 图像 | chat 请求包含有效 image_url；例如 256×256 图像 | CPU 预处理检查，GPU 未验证 |
| 视频 | 运行时支持的视频输入，例如至少 4 帧 256×256 视频，保留 temporal_patch_size=2 | CPU 预处理检查，GPU 未验证 |
| MTP | `--speculative-config`，多 token 请求并确认 speculative 日志/指标 | 配置保留，GPU 未验证 |

普通服务不会执行 MTP。随机路由不保证每个专家都被访问；保留配置也不能证明所有内核分支已运行。
TP/PP/EP、HC sequence parallel、图执行、prefix caching 等部署分支须在配套实现支持时另测；未为覆盖它们修改模型。

## 验证记录与环境限制（2026-10-27，Asia/Shanghai）

参考上游：`/home/m01016/dummy_home/vllm_workspace/vllm/vllm/models/qwen4_exp/nvidia/model.py`，
workspace commit `5df63a7203cdb9328d824a27b682e77c7b9928be`。
同时检查 MetaX `models/qwen4_exp/metax/{model,ple_layer,mtp}.py`、配置及注册。
MetaX workspace commit：`703172700abbd84143e530c7c4a723e06619327a`。

源目录使用 `full_attention` 标签；已检查的 MetaX 实现在存在 `indexer_n_heads` 时选择 QSA，
不是普通 full-attention fallback。较新的上游 NVIDIA 实现接受 `indexed_attention` /
`qwen_sparse_attention`，依赖新版 Transformers 的 Qwen4 配置契约；跨版本使用时必须核对归一化结果。
此裁剪保留源配置标签，没有借修改标签绕过运行时兼容性问题。

本次观察到运行环境有多套安装：最初 vLLM 来自用户 site-packages、运行时版本为
`0.29.1.dev0+g98dff2a81.d20260929`；最终检查分发元数据为 `0.32.0+empty`，
实际模块来自 `/opt/conda/lib/python3.12/site-packages/vllm`。
Python 为 `/opt/conda/bin/python` 3.12.11，Transformers 5.15.1，PyTorch 2.10.0+metax3.8.2.2；
MetaX 分发元数据为 `0.30.0+gf2fcc5.d20260929.maca3.8.2.6.torch2.10.0`。
脚本显式使用 `PYTHONPATH` 指向本 MetaX workspace，避免同名已安装包遮蔽；实际导入路径记录在日志和 JSON 中。

- JSON 字段依赖、文件 SHA256、离线 tokenizer、chat template、GenerationConfig、图像/视频 processor 检查见 `validation_cpu.json` 和 `logs/config_preflight.log`。
- 精确的 Qwen4 AutoConfig 归一化受到当前环境缺少相应配置模块阻碍，单独标记 BLOCKED；tokenizer 的通用配置回退不代表 Qwen4 模型构造成功。
- 完整 MetaX 插件导入失败：`ImportError: cannot import name 'DEFERRABLE_TAGS' from 'vllm.device_allocator'`，见 `logs/runtime_plugin.log`。失败发生于模型加载之前。
- 8 张 C500 均已占用约 60–62 GiB / 64 GiB，见 `logs/gpu_occupancy.log`。本次未启动服务或分配 GPU，没有需要清理的 GPU worker。
- startup、generation、GPU 图像/视频及 MTP 均为 **UNVERIFIED**。交付的是结构裁剪结果，尚不能声称在当前 shell 中可成功启动。

复查 CPU 资源（当前环境会因配置模块缺失返回 2，并保留独立资源检查结果）：

```bash
PYTHONPATH=/home/m01016/dummy_home/vllm_workspace/vllm-metax \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python tools/models/Qwen3.8-Flash-Next-dummy-3layers/validate_config.py
```

## 文件与复现

`config.json`、tokenizer 数据/词表/merges、chat template、generation 配置、图像和视频预处理配置均为实体文件。
`configuration.json` 是源目录的附加框架元数据，`LICENSE` 随资源保留。
没有 `auto_map` 自定义代码依赖；没有权重索引发现依赖。目录可以整体复制，无源目录软链接。
源 README 的模型说明未复制为本目录说明，避免与裁剪结果混淆。

`trim_manifest.json` 记录映射、修改值和资源 SHA256。
`reproduce_trim.py` 可向**新目录**重新生成相同的模型配置和资源，不会覆盖已有目录：

```bash
python tools/models/Qwen3.8-Flash-Next-dummy-3layers/reproduce_trim.py --output /tmp/Qwen3.8-Flash-Next-dummy-recreated
```
