# msprobe Precision Debugging Guide

[msprobe](https://gitee.com/ascend/mstt/tree/master/debug/accuracy_tools/msprobe) (MindStudio Probe)
is a precision-debugging toolkit that dumps intermediate tensors, weights, and
activations so a "problem side" run can be compared against a "benchmark
side" run to locate where numerical divergence starts. vllm-metax wires it up
using the same `--additional-config` schema
[vllm-ascend already established](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/performance_and_debug/msprobe_guide.html),
so a dump captured on Maca can be compared directly against one captured on
Ascend or upstream CUDA using the same msprobe config.

## Installation

```bash
pip install mindstudio-probe
```

## Usage

msprobe relies on Python-level `nn.Module` hooks, so the engine must run in
eager mode:

```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --enforce-eager \
  --additional-config '{
    "dump_config": {
      "task": "statistics",
      "level": "L1",
      "dump_path": "/data/msprobe_dump",
      "statistics": {"list": []}
    }
  }'
```

`dump_config` is an inline dict that gets materialized to a temporary
`config.json` and handed to msprobe's `PrecisionDebugger`. You can instead
point to an existing config file with `dump_config_path`:

```bash
--additional-config '{"dump_config_path": "/path/to/msprobe_config.json"}'
```

Passing neither field keeps the stock `vllm.v1.worker.gpu_worker.Worker` in
use with no extra overhead; the msprobe integration only activates when a
dump is requested.

Dump files land under `dump_path/step{N}/rank{ID}/`, containing
`dump_tensor_data/`, `dump.json`, and (for `level: L0`/`mix`) `construct.json`.

### `dump_config` field reference

| Field | Values | Meaning |
| --- | --- | --- |
| `task` | `"statistics"`, `"tensor"`, `"overflow_check"`, `"free_benchmark"`, `"run_ut"`, `"grad_probe"`, `"structure"` | What kind of data to collect. `"statistics"` records only summary stats (min/max/mean/L2 norm) per op — small and fast, good for a first pass. `"tensor"` dumps the full raw tensor (`.npy`/`.bin`) — large and slow, needed once you've narrowed down a suspect range and want an exact element-wise diff. |
| `level` | `"L0"`, `"L1"`, `"mix"` | Dump granularity. `L0` is module-level (`nn.Module` boundaries; also emits `construct.json` used to rebuild the graph for `graph_visualize`). `L1` is API/operator-level (every individual torch op call — finer-grained). `mix` records both. |
| `dump_path` | directory path | Root output directory. Files are written to `dump_path/step{N}/rank{ID}/`; `step` increments once per `execute_model` call, `rank` matches the TP/DP worker rank. |
| `statistics.list` | list of module/API names, or `[]` | Filter restricting the dump to specific names; `[]` means no filter (dump everything at the configured `level`). |

> **Version caveat**: the field name above (`statistics.list`) matches what
> works against the `mindstudio-probe` version this was validated with. Newer
> `mstt` source renames it to `tensor_list` in some branches, so if you
> upgrade `mindstudio-probe` and the config gets rejected, check the field
> names against your installed copy:
> `python -c "import msprobe, os; print(os.path.dirname(msprobe.__file__))"`
> then `grep -n "list\|scope" <that_path>/pytorch/pt_config.py`.

## Comparing two runs

msprobe ships two complementary comparison workflows.

### Quantitative comparison (`msprobe compare`)

This is the direct way to get per-op accuracy metrics and works with both
`task: statistics` and `task: tensor` dumps. Capture a dump from the run
under investigation and a dump from a known-good baseline (a different
branch, framework version, or hardware backend — inputs and sampling steps
just need to line up), then write a `compare.json`:

```json
{
  "npu_path": "/tmp/msprobe_dump_metax/step0",
  "bench_path": "/tmp/msprobe_dump_baseline/step0"
}
```

`npu_path`/`bench_path` are msprobe's (Ascend-legacy) field names for
"side under test" vs. "trusted baseline" — neither side needs to actually be
an NPU.

```bash
msprobe -f pytorch compare -i compare.json -o /tmp/msprobe_compare_out
```

This produces `compare_result_{timestamp}.xlsx` with one row per op call,
including Cosine similarity, max absolute/relative error, and an "Accuracy
Reached or Not" column. The first row marked `No` (e.g. Cosine < 0.99 or
MaxAbsError over threshold) from the top is the first op where the two runs
diverge.

### Visual comparison (`graph_visualize`)

Useful once you want to see *where in the model* a divergence sits. Requires
dumps taken with `level: L0` or `level: mix` (needs `construct.json`):

```bash
msprobe graph_visualize -tp <problem_dump_path> -gp <benchmark_dump_path> -o <output_dir>
```

Open the resulting `.vis.db` in TensorBoard with the `tb_graph_ascend` plugin
to inspect layer-by-layer where the two runs diverge. Use this after
`compare` has told you roughly where to look.

## Implementation notes

- `vllm_metax/utils/msprobe_debug.py` reads `dump_config`/`dump_config_path`
  from `additional_config` and builds the `PrecisionDebugger`.
- `vllm_metax/v1/worker/gpu_worker.py::MacaWorker` starts/stops/steps the
  debugger around each `execute_model` call.
- `MacaPlatformBase.check_and_update_config` only swaps `worker_cls` to
  `MacaWorker` when a dump was requested, and raises early if
  `--enforce-eager` was not also passed.