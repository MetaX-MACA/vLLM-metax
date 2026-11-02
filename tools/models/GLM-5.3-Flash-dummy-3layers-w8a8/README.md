# GLM-5.3-Flash 最少层数 INT8 W8A8 dummy 模型

来源：[ZhipuAI/GLM-5.3-Flash](https://modelscope.cn/models/ZhipuAI/GLM-5.3-Flash/files)。
仅包含配置及非权重资产，使用 `--load-format dummy`；随机参数仍占用 GPU 显存。
此模型用于验证构造、算子和功能分支，不用于原模型精度、输出质量或性能评估。

## 裁剪与覆盖

在结构上保留三种原始文本层组合，因此主干最少为 3 层。两层无法同时保留这三种组合。
下表的“保留路径”表示配置与模块结构覆盖，GPU 实际执行状态见文末。

| 新层 | 原层 | 保留路径 | 验证输入 |
| --- | --- | --- | --- |
| 文本 0 | 文本 0 | KDA + Dense，mHC 首层展开 | 普通文本 prefill + 多步 decode |
| 文本 1 | 文本 3 | DSA/MLA + MoE、共享专家、kpool 压缩与尾缓存 | 大于 2048 token 的 prefill + 至少 8 token decode |
| 文本 2 | 文本 44 | KDA + MoE，mHC 层间融合与末层收缩 | 普通文本 prefill + 多步 decode |
| MTP 3 | MTP 45 | DSA + MoE、预测头与索引复用配置 | 显式开启 MTP speculative decoding |
| 视觉 0 | 视觉 0 | 图像/视频 patch embedding、attention、MLP、downsample、merger | 图像请求；至少 2 帧的视频请求 |

视觉塔为相同 block 的循环，没有中间层选择或跨层引用，因此深度由 24 减为 1；
不删除视觉塔或改变视觉输出维度。文本长度由 45 减为 3，MTP 层数仍为 1。
原始层序列、连续重复层数量、输出精度不在 dummy 覆盖声明内。

显式修改如下；完整前后值见 `trim_manifest.json`：

- `text_config.num_hidden_layers`: 45 → 3。
- `text_config.first_k_dense_replace`: 3 → 1。
- `text_config.layer_types`: `[linear_attention, deepseek_sparse_attention, linear_attention]`。
  Transformers 5.19 将 DSA 名称归一化为 `indexed_attention`，仍由模型的非 KDA 分支执行 MLA。
- `text_config.mlp_layer_types`: `[dense, sparse, sparse]`。
- `text_config.indexer_types`: `[full, full, full]`。
- `text_config.linear_attn_config.kda_layers`: `[0,2]`；`full_attn_layers`: `[1]`。
  当前模型主要按 `layer_types` 派发；同时更新显式层号，避免残留越界描述。
- `vision_config.depth`: 24 → 1。
- `quantization_config.modules_to_not_convert`: 1509 → 127 条。
  精确重映射 `model.layers.{0,3,44,45}` 和 `visual.blocks.0`；删除已裁剪层的排除项。
  排除项在转换后保留于 `compression_config.ignore`。

hidden_size=4096、词表=154880、288 个路由专家、1 个共享专家、top-8、
head/LoRA/卷积维度、mHC 参数、kpool=4、topk=2048、图像/视频几何参数均保持原值。

用户明确要求将 dummy 配置改为 MetaX 支持的 INT8 W8A8，因此将官方 FP8 的活动配置
替换为 `compression_config`：`quant_method=compressed-tensors`、`format=int-quantized`、
`targets=[Linear]`；权重为静态、对称、per-channel INT8，激活为动态、对称、per-token INT8。
沿用原始量化排除模块并重映射层号，保留其 BF16 路径。
删除活动 `quantization_config` 避免 FP8 优先级冲突；完整源 FP8 元数据另存
`source_fp8_quantization_config.json`。这只改变随机 dummy 参数的构造方案，
未将真实 FP8 checkpoint 转换为 INT8，亦不声明存在可加载的真实 INT8 权重。
MACA 的实际算子选择和参数后处理以启动日志为准。

## 可移植资产与复现

包括 `config.json`、tokenizer、`processor_config.json`、`generation_config.json`、
`chat_template.jinja`、`configuration.json`、许可证和源模型卡 `SOURCE_MODEL_CARD.md`。
不包含权重和权重索引，不依赖源模型目录软链接。
`source_assets_manifest.json` 记录官方文件列表中的逐文件 revision、下载 URL、SHA256 和大小。
源 config SHA256：`bb8f01c42cb92a52ca72e65afb4d5bd8d11aef083cd210e8de25dfb904f23e9f`。

本 checkout 已有模型样例在 `tools/models/`，因此新模型放在同一位置。
复现时提供保持原始内容的官方资产目录；脚本先检查资产哈希，再深拷贝 JSON 并做上述字段和量化修改：

```bash
cd /root/workspace/vLLM-metax
/opt/venv/bin/python tools/models/GLM-5.3-Flash-dummy-3layers/trim_model.py \
  --source .metax-upgrade/glm5next-review-20261102/model-source
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /opt/venv/bin/python \
  tools/models/GLM-5.3-Flash-dummy-3layers/validate_cpu.py
/opt/venv/bin/python tools/models/GLM-5.3-Flash-dummy-3layers/validate_contract.py \
  --source .metax-upgrade/glm5next-review-20261102/model-source
```

环境确认 fingerprint：`cb371766ab652e185f98dede9fdb4a3d41c55a434a50d9c50b283f5981189e39`。
Python `/opt/venv/bin/python`；Transformers 5.19.0；vLLM `0.32.0+empty`，
实际源 `/root/workspace/vllm/vllm`；MetaX `0.32.0+g360a93.d20261020.maca3.9.1.6.torch2.10.0`，
实际源 `/root/workspace/vLLM-metax/vllm_metax`。独立检查见 `validation_cpu.json`。

## 启动与分支验证

先选择空闲 GPU。以下命令默认 TP1、4096 上下文、eager、DCP1；使用已写入配置的 INT8 W8A8 权重量化。
`run_serve.sh` 允许用环境变量设置 GPU、TP、端口、indexer dtype、上下文和 batch token 预算：

```bash
GPU_IDS=0 TP=1 PORT=8043 INDEXER_DTYPE=int8 \
  bash tools/models/GLM-5.3-Flash-dummy-3layers/run_serve.sh

# 独立复测 BF16 indexer；先关闭前一个服务并检查 GPU 释放。
GPU_IDS=0 TP=1 PORT=8043 INDEXER_DTYPE=bf16 \
  bash tools/models/GLM-5.3-Flash-dummy-3layers/run_serve.sh
```

服务启动后检查 `/health`，再测试 `/v1/completions`、`/v1/chat/completions`。
使用 `ignore_eos=true` 并请求至少 8 个输出 token，以覆盖 kpool=4 的尾缓存和完整 pool 写入。
长输入需要用该 tokenizer 实测大于 2048 token（例如 2305），为输出保留上下文空间；
短输入会走跳过 sparse scoring 的路径，不能证明 INT8/BF16 稀疏 logits/top-k 可用。
图像可用 112×112 RGB，视频可用两个 112×112 RGB 帧；两者分别验证，CPU preprocessing 成功不能代替 GPU encoder 执行。

MTP 参数配置和量化路径已保留，但当前 MetaX 注册仅覆盖 target/base 两个架构，
上游 `Glm5NextMTPModel` 的注册指向 `vllm.models.glm5next`。
普通服务不执行 MTP，不能据此声称本地 MTP 已验证。
需要另测 `--speculative-config '{"method":"mtp","num_speculative_tokens":2}'`，
并核对实际 draft 导入来自何处。DCP 依适配限制维持关闭。

## 验证状态

`validation_cpu.json` 分别记录离线 config 归一化、tokenizer/chat template、generation config、
图像/视频 processor、实际惰性注册和 compressed-tensors 配置构造结果。这里不构造 GPU 模型。
离线配置、tokenizer/chat template、generation config、112×112 图像和两个 112×112 帧的视频
preprocessing 均通过；target/base 的惰性注册指向 MetaX，MTP 架构的惰性注册仍指向上游。
惰性注册可见不能证明对应模型模块成功导入或构造。
W8A8 的 `CompressedTensorsConfig.from_config` 构造检查通过，确认权重/激活位数、
类型、对称性、channel/token 策略与动态/静态字段以及 127 条排除项。
`validate_contract.py` 另行确认所有非裁剪、非用户授权量化字段均等于官方原值，
图像/视频等资产保持官方 SHA256，且源 FP8 元数据完整保留。

本次初始官方 FP8 配置的正常 CLI 启动在 GPU 模型分配前被确认环境拒绝：
`fp8 quantization is currently not supported in maca`。
平台默认 `VLLM_METAX_SUPPORTS_FP8` 为 false，原始启动证据保存在仓库根目录
`.metax-upgrade/glm5next-review-20261102/runtime-int8/`；该证据只对应此前的 FP8 配置，
不代表当前用户授权的 W8A8 配置已启动或失败。
当前 W8A8 配置已重新经过正常 CLI 启动：配置校验通过，实际选择
`MctlassScaledMMLinearKernel` 的 `CompressedTensorsW8A8Int8` 线性层和 `TRITON` INT8 MoE，
日志报告模型加载占用 17.48 GiB。随后 GPU profiling 的 `_dummy_run` 在
`vllm_metax/models/glm5next/common/attention.py:447` 抛出
`UnboundLocalError: cannot access local variable 'q_fp8' where it is not associated with a value`。
这证明已越过原 FP8 配置限制并构造、分配 W8A8 参数，但服务尚未就绪。
当前运行证据位于 `.metax-upgrade/glm5next-review-20261102/runtime-w8a8-int8-final/`，
完整命令和进程退出状态见其中 `result.json`，算子选择和异常见 `serve.log`。
未收到健康响应或发送生成请求，因此生成、长输入稀疏选择、图像/视频 GPU encoder、
INT8/BF16 indexer 完整执行和 MTP 仍未验证。切换 indexer dtype 与权重量化是两个独立配置。

服务启动、生成、长输入、图像、视频、BF16/INT8 和 MTP 结果由本次审查的运行证据单独记录；
未执行的分支保持未验证。dummy 成功不代表真实 checkpoint 加载或原模型精度通过。
