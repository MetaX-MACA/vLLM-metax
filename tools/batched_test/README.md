# Batched Test Guide

Used for batched e2e *inference* and *performance* benchmark.

***All the relative path are defaultly based on `tools/batched_test` folder.***

## Basic Usage

```bash
[UV=1] python launch.py [-h] [--work-dir WORK_DIR] [--model-config CONFIG_YAML_FILE] [--cluster-config CONFIG_YAML_FILE] [--infer] [--text-case LM_CASE_FILE] [--image-case IMAGE_CASE_FILE] [--resume-csv RESUME_CSV] [--perf] [--gpus] [--tag] [--dry-run] [--dump-selected]
```

- `--work-dir`: 
    Folder for saving tests result.
    If not specified, default using: </workspace/model_test>

- `--model-config`
    Model config file path. 
    If not specified, default to: <configs/model.yaml>,

- `--cluster-config`
    Cluster config file path. 
    If specified:
    - disable concurrency for all the tests
    - use ray node for testing 
    - ***the script must run in one of these nodes!***

- `--infer`
    Specify to run inference tests for all models in `--model-config`

- `--text-case`
    Overwrite cases for text-only inference.
    If not specified, default to: <configs/inference/text_case.yaml>

- `--resume-csv`
    Resume from the failed case in specified inference_result.csv (need to keep the `model_config` the same)

- `--text-case`
    Overwrite cases for image inference.
    If not specified, default to: <configs/inference/image_case.yaml>

- `--perf`
    Specify to run performance benchmark for all models in `--model-config`

- `--gpus`
    Filter Models by Required GPU Count
    If not specified, no GPU-count filtering is applied.

- `--tag`
    Filter models using the tag field defined in model.yaml.
    If a model does not define tag, it is treated as: 
    tag:
    - dense
    MoE models or other models should explicitly define:
    tag:
    - moe (other tag)

- `--dry-run`
    Run model selection only, without executing inference or performance tests.

- `--dump-selected`
    Dump the currently selected model subset into a new YAML config file.

> Note: set **UV=1** if you are using uv instead of pip.

## Enable cluster

```
<cluster template>
- ssh:
    hostname: host1 # SSH hostname
    port: 22  # SSH port
    user: root  # SSH user
    auth_type: password  # Authentication type: "password" or "key"
    private_key: ~/.ssh/id_rsa  # SSH key path (used when auth_type="key")
    password: "1"  # SSH password (used when auth_type="password")
  ray:
    nic: eth0  # Network Interface Card (used for GLOO_SOCKET_IFNAME)

- ssh:
    hostname: host2 # SSH hostname
    port: 22  # SSH port
    user: root  # SSH user
    auth_type: password  # Authentication type: "password" or "key"
    private_key: ~/.ssh/id_rsa  # SSH key path (used when auth_type="key")
    password: "1"  # SSH password (used when auth_type="password")
  ray:
    nic: eth0  # Network Interface Card (used for GLOO_SOCKET_IFNAME)
```

> *The node is assumed to be a docker or the environments of which vllm could directly run on.*

If using ray for multi-node tests, you need specify `--cluster-config` for scripts. The script would try to allocate nodes according to the model config's requirement by `TP * DP * PP`. And you need specify `--gpus` to filter the multi-node model. For example:`--gpus: 8, 16, 32`

If not using ray for multi-node tests, you need not specify `--cluster-config` and not specify  `--gpus` also for scripts, run models requiring {1,2,4,8} GPUs by default.

For example, if a model needs :
- 8 cards, script would trying allocated 1 nodes and launch ray on it.
- And if 16 cards, script would trying allocated 2 nodes and launch ray on both of them.
- And 32 cards for 4 nodes with ray launched on all of them.

*Note!: the script must run in one of the nodes and must be **the first node** in the cluster config file*


## Model Config Template

```yaml
- name: "example-model"
  model_path: "/path/to/example-model"
  timeout: 600 # timeout for waiting vllm serve start
  serve_config:
    tp: 99
    pp: 99
    dp: 99
    distributed_executor_backend: "ray"  # default
    gpu_memory_utilization: 0.8  # default
    swap_space: 16  # default
    max_model_len: 4096  # default

    # Optional extra arguments for vllm serve command
    # won't overwrite the default args
    # won't check the validity of these args
    extra_args:
      --chat-template: "/relative/path/to/chat_template.py"
  infer_type: # one of the following types (required)
    - text-only  # supported
    - single-image # supported
    - multi-image # TODO(hank): not supported yet
    - multi-modal # TODO(hank): not supported yet
    - video # TODO(hank): not supported yet
    - audio # TODO(hank): not supported yet
  benchmark:
      bench_param: "configs/bench_params/bench_default.yaml" # default
      dataset_name: "random"  # default
      ignore_eos: true  # default
      sweep_num_runs: 1  # default
  extra_env:
    EXAMPLE_ENV_VAR: "value"
  tag:
    - "define"
    - "your"
    - "tags"
    - "here"
```

- tag: User defined tags for the model. If not defined, the model tag is default to be `dense`.
- sweep_num_runs: determine how many times for running on each combination in bench_param.


## Structure folder

### --infer

Inference test folder structures are showed below:

```
# tree -L 6 /your/work/dir
model_test
`-- 20260113_1507
    |-- collect_env.txt
    `-- inference
        |-- deepseek-v2-lite-chat[tp8pp1dp1]_serve.log
        |-- deepseek-v2-lite-chat[tp8pp1dp1]_text_only_inference.log
        `-- inference_results.csv
```

### --perf

Perf test folder structures are showed below:

```
# tree -L 6 /your/work/dir
model_test
`-- 20260113_1721
    |-- collect_env.txt
    `-- performance
        |-- deepseek-v2-lite-chat_tp8_pp1_dp1
        |   `-- 20260113_172158
        |       |-- BENCH--max_concurrency=1-num_prompts=5-random_input_len=128-random_output_len=128
        |       |   |-- run=0.json
        |       |   `-- summary.json
        |       |-- BENCH--max_concurrency=16-num_prompts=80-random_input_len=128-random_output_len=128
        |       |   |-- run=0.json
        |       |   `-- summary.json
        |       |-- BENCH--max_concurrency=4-num_prompts=20-random_input_len=1024-random_output_len=1024
        |       |   |-- run=0.json
        |       |   `-- summary.json
        |       `-- summary.csv
        `-- deepseek-v2-lite-chat_tp8_pp1_dp1_serve.log
```

### --dry-run

Selection preview only
