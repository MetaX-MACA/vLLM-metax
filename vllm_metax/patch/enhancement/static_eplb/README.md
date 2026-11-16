# Static EPLB ownership and upstream hooks

Reviewed target: vLLM v0.33.0, `4ba787ec9959e566d3b587a796dff07673324da5`.

`placement.py` owns JSON/PT serialization, payload selection and pure map validation.
`StaticEplbRecorder` owns configuration, the attached EPLB state, weight import,
load collection, policy evaluation and placement export. `static_eplb.py` installs
four thin hooks. The package initializer activates them through the normal plugin.

## Required hook boundaries

Upstream paths below are relative to the vLLM source checkout.

| Hook | Native caller and timing | Static action and reason |
| --- | --- | --- |
| `vllm.v1.worker.gpu_model_runner.GPUModelRunner.load_model` | MRV1 worker loading; registers draft first when present, then the main MoE model. | Prepare the recorder before native loading; attach after loading returns, when every model is registered. Validate every model map before the first weight transfer. Forward the entire native call and return value. |
| `vllm.v1.worker.gpu.model_runner.GPUModelRunner.load_model` | MRV2 worker loading calls `EPLBController.prepare_load`, `maybe_register_speculator`, `maybe_register_model`, then `maybe_start_async_loop`. | Use the same recorder lifecycle. MRV2 exposes `eplb_state` through its controller property, so the manager needs no runner-specific state implementation. |
| `vllm.distributed.eplb.eplb_state.EplbState.start_async_loop` | Both native load methods may start the async thread **before** the outer load wrapper returns. | Suppress startup when static environment configuration is active. Applying this only after recorder attachment would allow background weight transfers to race with import. Ordinary calls delegate every argument to native EPLB. |
| `vllm.distributed.eplb.eplb_state.EplbState.step` | MRV1 `GPUModelRunner.eplb_step` and MRV2 `EPLBController.step` both reach this shared method. | The hook forwards all step flags to the attached recorder. The recorder owns profiling through native `state.rearrange(is_profile=True)` and discards profile load afterward. All unattached states delegate to native EPLB. |

The target has no static placement file lifecycle callback. Policy registration
only replaces placement selection; it cannot apply imported weights after all
models register or own one-shot file export. These wrappers retain native model
loading and native transfer/map-commit algorithms.

`EplbState.add_model` is too early: a draft may register before the main model,
and importing there would lose all-model prevalidation. `prepare_forward` remains
native because it updates the real-token masks used by the routing kernels.
MRV1's runner-specific `eplb_step` does not cover MRV2. Patching the controller
would conversely miss MRV1. Leaving both guards native preserves MRV2's
`EPLBController.suppress` during `Worker.compile_or_warm_up_model`, graph-memory
profiling, and calls with `skip_eplb=True`.

## Runtime lifecycle

1. With neither path configured, native loading and EPLB run normally.
2. Dummy loading (`load_dummy_weights=True`, positional or keyword) bypasses
   recorder preparation. This differs from a normal load using dummy checkpoint
   initialization through `load_format`, which still follows normal serving setup.
3. Prepare and validate static configuration and read any input file before model
   loading. Static mode requires native EPLB support and rejects elastic EP.
4. Native loading builds EPLB state for all supported main/draft models. The async
   startup hook keeps that placement fixed.
5. Attach the recorder. Validate all imported maps, migrate real expert weights,
   then commit physical/logical maps and replica counts using native functions.
   Mixed-layout layers use matching receive buffers; NIXL uses Gloo for these
   temporary layouts because its original buffer addresses are registered.
6. Export mode accumulates `int64` physical-expert load. At the configured step,
   native EP all-reduce combines ranks; scatter-add merges replicas into logical
   expert demand; the configured native policy computes the exported placement.
7. Only EP rank zero writes the file atomically. Export leaves live weights/maps
   unchanged and disables recording. A later import applies the proposed layout.

The shared state holds one `_metax_static_eplb_recorder` reference. There is no
separate static-mode boolean or second owner for collection counters/buffers.
The recorder handles profile steps itself, reusing `state.rearrange(is_profile=True)` for native communication-buffer reservation without recursively calling the patched `state.step`. Profile steps do not advance collection. Static mode does not schedule native periodic balancedness logging; `log_stats` is accepted by the shared step interface. Unsuppressed dummy DP iterations advance
the synchronized counter with zero load, keeping all EP ranks aligned.

| Environment variable | Meaning |
| --- | --- |
| `VLLM_METAX_EPLB_IMPORT_PATH` | Import a JSON/PT placement or logical-load payload. |
| `VLLM_METAX_EPLB_EXPORT_PATH` | Write a policy-generated JSON/PT placement once. |
| `VLLM_METAX_EPLB_EXPORT_STEPS` | Number of synchronized EPLB steps; default 1000, positive. |

Import and export can be combined with distinct paths. Placement rows identify
MoE layers; columns identify global physical expert slots; values identify logical
experts. Multi-model files preserve registration order and verify model names.
The exported file contains placements, without checkpoint weights or raw counts.

## Validation scope

Model-free regression cases cover the four bindings, load forwarding and dummy
bypass, configuration rejection, import ordering, profile/dummy/suppression
behavior, replica aggregation and JSON/PT file round-trips. Two-GPU component
checks exercise real native transfers and map commits for BF16, mixed FP16/INT8,
redundant experts, mixed-layout Gloo fallback, and multi-model export/reload.
NIXL fallback dispatch uses a type sentinel; native NIXL RDMA and complete-model
inference are separate validation scopes. No FP8/FP4 arithmetic is exercised on
MXC500. Use `USE_PRECOMPILED_KERNEL=0` for the selected source-kernel environment.
