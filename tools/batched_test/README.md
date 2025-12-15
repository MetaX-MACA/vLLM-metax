# Batched Test Guide

Used for batched e2e *inference* and *performance* benchmark.

***All the relative path are defaultly based on `tools/batched_test` folder.***

## Basic Usage

```bash
python launch.py [-h] [--work-dir WORK_DIR] [--model-config CONFIG_YAML_FILE] [--infer] [--inference-case LM_CASE_FILE] [--perf]
```

- `--work-dir`: 
    Folder for saving tests result.
    If not specified, default using: </workspace/model_test>

- `--model-config`
    Model config file path. 
    If not specified, default to: <configs/model.yaml>,

- `--infer`
    Specify to run inference tests for all models in `--model-config`

- `--text-case`
    Cases for text-only inference.
    If not specified, default to: <configs/inference/text_case.yaml>

- `--text-case`
    Cases for image inference.
    If not specified, default to: <configs/inference/image_case.yaml>

- `--perf`
    Specify to run performance benchmark for all models in `--model-config`

Note: set UV=1 if you are using uv instead of pip.

## Model Config Template

```yaml
- name: "example-model"
  model_path: "/path/to/example-model"
  serve_config:
    tp: 99
    pp: 99
    dp: 99
    distributed_executor_backend: "ray"  # default
    gpu_memory_utilization: 0.9  # default
    swap_space: 16  # default
    max_model_len: 4096  # default
    # Optional extra arguments for vllm serve command
    # won't overwrite the default args
    # won't check the validity of these args
    extra_args:
      --chat-template: "/relative/path/to/script"
      --hf-overrides: "'{\"architectures\": [\"GLM4VForCausalLM\"]}'"
  infer_type: # one of the following types (required)
    - text-only  # supported
    - single-image # supported
    - multi-image # TODO(hank): not supported yet
    - multi-modal # TODO(hank): not supported yet
    - video # TODO(hank): not supported yet
    - audio # TODO(hank): not supported yet
  benchmark:
    bench_param: "/relative/path/to/script" # default
    dataset_name: "random"  # default
    ignore_eos: true  # default
    sweep_num_runs: 3  # default
  extra_env:
    MACA_QUEUE_SCHEDULE_POLICY: 1
```

- sweep_num_runs: Number of runs per parameter combination.
    

