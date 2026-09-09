# Batched Test Guide

Used for batched inference tests and performance benchmarks.

## Basic Usage

Run from `tools/batched_test`:

```bash
cd tools/batched_test

# Preview selected models.
python launch.py --dry-run

# Inference tests.
python launch.py --infer --concurrency 2

# Performance benchmarks.
python launch.py --perf --concurrency 1

# Filter models and run both tests.
python launch.py --infer --perf --gpus 1,2,4,8 --tag moe

# Resume inference tests.
python launch.py --infer --resume-csv /path/to/inference_results.csv
```

From the repository root, use `python -m tools.batched_test.launch`.
From any other directory, use the absolute path to `launch.py`.

## Arguments

| Argument | Description |
| --- | --- |
| `--work-dir` | Result directory. Default: `/workspace/model_test`. |
| `--model-config` | Model YAML. Default: `configs/model.yaml`. |
| `--cluster-config` | Cluster YAML for multi-node tests. |
| `--infer` | Run inference correctness tests. |
| `--perf` | Run performance benchmarks. |
| `--text-case`, `--image-case`, `--embedding-case` | Override the corresponding case YAML. |
| `--long-text-case` | Add one long case per text suite; default `max_tokens` is 1024. |
| `--resume-csv` | Resume from a previous inference CSV. |
| `--gpus` | Filter by required GPU count (`tp * pp * dp`). Default: `1,2,4,8`. |
| `--tag` | Filter by any of the comma-separated tags. Models without `tags` default to `dense`. |
| `--concurrency` | Maximum concurrent models. Default: local GPU count; cluster mode runs serially. |
| `--model-timeout` | Execution timeout after GPU allocation, in seconds. Default: 3600. |
| `--dry-run` | Print selected models without running tests. |
| `--dump-selected` | Save the selected model configurations to YAML and continue. |

Use `python launch.py -h` for all options.

## Model Config

```yaml
- name: example-model
  model_path: /path/to/model
  timeout: 1200  # Inference service startup timeout, in seconds.
  serve_config:
    tp: 1
    pp: 1
    dp: 1
    distributed_executor_backend: mp
    gpu_memory_utilization: 0.8
    max_model_len: 4096
    extra_args:
      --chat-template: /path/to/template.jinja
  infer_type:
    - text-only
  benchmark:
    bench_param: configs/bench_params/bench_default.json
    dataset_name: random
    ignore_eos: true
    sweep_num_runs: 1
  extra_env:
    EXAMPLE_ENV_VAR: value
  tags:
    - dense
```

- `infer_type`: supports `text-only`, `single-image`, and `embedding`. All listed types are tested; the model must support them.
- For embedding models, set `serve_config.task: embed` and provide `--embedding-case`.
- `extra_args`: accepts a mapping, argument list, or command-line string.
- `sweep_num_runs`: number of benchmark repetitions per parameter combination.
- Inference passes only when all cases pass; lower accuracy is recorded as `ACCURACY_FAILED`.

## Paths

- CLI relative paths use the directory where the command is run. This also applies to `--work-dir` and `--dump-selected`.
- Paths inside YAML/JSON use the directory containing that file, including benchmark parameters, templates, images, and SSH keys.
- Input prefixes `configs/`, `chat_template/`, and `assets/` refer to the corresponding directories under `tools/batched_test`. Use `./configs/`, `./chat_template/`, or `./assets/` to refer to directories beside your own config instead. Output paths and SSH keys do not use these aliases.
- Absolute paths, URLs, and model repository IDs such as `org/model` are supported. Prefix relative local model paths with `./` or `../`.

## Cluster Usage

Run on the **first node** in the cluster configuration:

```bash
python launch.py --infer --cluster-config /path/to/cluster.yaml --gpus 16,32
```

```yaml
- ssh:
    hostname: host1
    port: 22
    user: root
    auth_type: key
    private_key: /path/to/id_rsa
  nic: eth0

- ssh:
    hostname: host2
    port: 22
    user: root
    auth_type: key
    private_key: /path/to/id_rsa
  nic: eth0
```

Password authentication is also supported with `auth_type: password` and `password`.
Each node must have a compatible `vllm` environment and the model/template files at
the same paths. Cluster tests use the mp backend and currently assume 8 GPUs per node.

## Results and Resume

Results are saved under `<work-dir>/<YYYYMMDD_HHMM>/`:

```text
inference/
  inference_results.csv
  inference_results.manifest.json
  <model-config-id>/
    serve.log
    <suite>_inference.log
    result.json
performance/
  bench_tasks_result.csv
  bench_tasks_result.manifest.json
  <model-config-id>/
    sweep.log
    bench_params.json
    result.json
    sweep/
```

For resume, keep `inference_results.csv` and its adjacent `.manifest.json` together.
Only matching successful results with 100% accuracy are skipped. Changed model
settings or cases are rerun. Old CSVs without a manifest are accepted, but cannot
detect configuration or case changes.
