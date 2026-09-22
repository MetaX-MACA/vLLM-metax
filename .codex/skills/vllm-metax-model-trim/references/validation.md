# Configuration and Service Validation

## Configuration Preflight

Use the target runtime and record versions and import origins. Load AutoConfig and
AutoTokenizer from the output directory offline with `local_files_only=True`. Check
GenerationConfig when present and AutoProcessor for relevant modalities. Custom configurations
may require vLLM/MetaX registration or remote code first. If standalone AutoConfig reports an
unknown model, inspect the service's actual configuration-loading entrypoint before declaring it unsupported.

Check normalized text/submodel configurations, per-layer arrays and mappings, index ranges,
sharing producers, expert routing, sharding, and quantization alignment. Encode a sample
with the tokenizer and check the chat template for chat tasks. Successful configuration
parsing does not establish successful parameter allocation or quantization postprocessing.

## Startup Example

Run this template from the repository root. Replace MODEL_DIR, GPU_IDS, TP, and PORT and
adjust budgets to measured free memory. Some models require a larger minimum context or
specific parallelism; example parameters are not architectural limits. If both
`CUDA_VISIBLE_DEVICES` and `MACA_VISIBLE_DEVICES` are set, they should select the same physical GPUs.

```bash
MODEL_DIR="$PWD/tools/batched_test/models/<model-name>-dummy-<N>layers"
GPU_IDS=0
TP=1
PORT=8000
CUDA_VISIBLE_DEVICES="$GPU_IDS" MACA_VISIBLE_DEVICES="$GPU_IDS" \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
vllm serve "$MODEL_DIR" --served-model-name dummy-smoke \
  --host 127.0.0.1 --port "$PORT" --load-format dummy \
  --tensor-parallel-size "$TP" --max-model-len 1024 \
  --max-num-seqs 1 --max-num-batched-tokens 1024 \
  --gpu-memory-utilization 0.8 --enforce-eager
```

Start with eager execution to reduce graph-capture overhead. If graph execution is a target,
validate it separately without `--enforce-eager`. Add verified remote-code, quantization,
backend, environment, or speculative options only as needed by the model. Preserve source
quantization semantics and confirm effective quantization/backend selection in startup logs;
avoid inadvertently selecting another path through CLI overrides. For version-dependent
options, inspect the target environment's `vllm serve --help` and implementation.

Use a process manager or a test script with process groups to save stdout/stderr, PID/process
group, and the complete command. Set finite startup and request timeouts, for example 600
and 120 seconds, adjusting for initial compilation when needed. While polling health, also
check process exit status and error logs rather than waiting a fixed interval. Stop only the
service process group started for this task and wait for workers to exit; escalate termination
signals for that group if necessary.

## Requests and Pass Criteria

Once the service is ready, run this text-generation smoke request in another terminal,
using the same PORT:

```bash
PORT=8000
curl --fail-with-body --max-time 10 "http://127.0.0.1:$PORT/health"
curl --fail-with-body --max-time 120 \
  "http://127.0.0.1:$PORT/v1/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"dummy-smoke","prompt":"Hello","max_tokens":4,"temperature":0,"ignore_eos":true}'
```

- Check HTTP status, absence of errors, choices/usage, and token counts. Confirm multiple
  generated tokens before claiming multiple decode steps. Investigate stop settings,
  generation_config, or other causes of early termination before adjusting the request.
  Do not require correct answers or nonempty decoded text; randomly generated special tokens may be hidden.
- Add `/v1/chat/completions` requests with messages when testing chat. Use supported endpoints
  and inputs for encoder-decoder, multimodal, or embedding/pooling models. Do not impose
  decode checks on models that do not generate tokens.
- Give branch tests explicit triggers. Count long-input tokens with the actual tokenizer,
  cross sparse top-k/compression-block/sliding-window boundaries, and reserve context for
  output tokens. Increase max-model-len accordingly. Use small, valid multimodal samples.
  Record TP/PP/EP, MTP/DSpark, and graph-mode results separately.
- Successful generation alone does not prove that every kernel branch ran. Combine effective
  configuration, logs, or necessary instrumentation to distinguish retained structure,
  executed trigger inputs, and confirmed execution of the target branch.
- Do not evaluate semantic accuracy from dummy outputs. Still investigate NaNs, invalid
  indices, and crashes, including possible dummy initialization/quantization postprocessing
  limitations; random outputs do not justify ignoring execution failures.

## batched_test Integration

Generate a separate YAML only when batch-test configuration is requested. For a file under
`tools/batched_test/configs/`, use this example:

```yaml
- name: example-dummy-smoke
  model_path: ../models/<model-name>-dummy-<N>layers
  serve_config:
    tp: 1
    pp: 1
    dp: 1
    gpu_memory_utilization: 0.8
    max_model_len: 1024
    extra_args:
      --load-format: dummy
      --max-num-seqs: 1
      --max-num-batched-tokens: 1024
      --enforce-eager: null
  infer_type:
    - text-only
  tags:
    - dummy-smoke
```

Preview selection with
`python -m tools.batched_test.launch --model-config <yaml-path> --dry-run`.
See `tools/batched_test/README.md` for path resolution rules. The current argument converter
uses null for switches without values; true would add an unwanted `True` argument.
The current `--infer` suite evaluates answer accuracy, so its accuracy result cannot directly
judge dummy functionality. Use the independent HTTP smoke workflow above or user-specified
functional cases; do not change existing accuracy-test pass criteria. The batch GPU allocator
sets CUDA visibility; check for conflicts with inherited MACA visibility settings.

## Delivery Evidence

Document source paths/versions, changed fields, layer mappings, copied assets, actual startup
commands, GPUs/TP/quantization/backend, log paths, service and request outcomes, measured
memory when available, untested branches, and failure causes in the output directory.
Parameter-size estimates do not include all scales, layout-conversion peaks, KV/SSM state,
activations, communication, or graph caches. Checkpoint bytes, even divided by TP, are not
a promise of actual runtime memory usage.
