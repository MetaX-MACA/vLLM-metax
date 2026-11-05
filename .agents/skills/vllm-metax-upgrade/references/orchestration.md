# Ordered framework, model and patch adaptation

Common rules and CLI mechanics live in [common-sense](../../common-sense/SKILL.md).
Keep one full run and one owner per shared file. Roles may run sequentially; parallel
agents are optional when authorized. The phase sequence is strict, with recorded late
framework repairs and early blocking-patch exceptions described by the coordinator.

## Platform first

Compare target `vllm/platforms/cuda.py`, the base platform interface and their callers
with `vllm_metax/platform.py`. Review new/removed APIs, signatures/defaults, dispatch,
backend selection, device capabilities, compilation/graph policy, cache configuration,
quantization, distributed settings and environment rewrites. Follow CUDA's semantic
role while checking actual MACA SDK/runtime/component compatibility; a CUDA flag is
not a MACA capability. Follow common's chip-validation policy rather than adding a
speculative chip capability table during an upgrade.

## Registry and attention

Assign [registry-upgrade](../../registry-upgrade/SKILL.md), then
[attention-upgrade](../../attention-upgrade/SKILL.md). Include their transitive local
wrappers, oracles, experts, modular MoE kernels, prepare/finalize and collectives in the
owning phase. Record explicit ownership for shared helpers rather than treating a
registration decorator as complete adaptation of its execution path.

Attention source review must cover quantized attention/indexer KV cache, scales,
physical block layout and alignment, insertion/gather/read consumers and separate
prefill/decode forwards, including mixed batches. Model-private attention stays with
the model phase; shared interface contracts must already be adapted.

## Native sources under both kernel providers

Assign [csrc-upgrade](../../csrc-upgrade/SKILL.md) all csrc sources, headers and native
build contracts. This is mandatory with or without mcoplib; using the image library
never exempts local code from upgrade. Adapt required native dependencies in the earliest
consuming phase; complete remaining csrc before opening models. In mcoplib mode, verify
image API compatibility and separately build/test the upgraded local source without
replacing the production installation. Source mode must build/run with
`USE_PRECOMPILED_KERNEL=0` and prove independence from mcoplib, including Python/Triton
helpers and patches that may ignore the flag. The selected provider is an initial-plan
choice, not an automatic fallback or a new source phase.

## Sweep the remaining framework

Use the complete inventory, not a list of recently edited files. Review every remaining
Python file: plugin initialization, model registration glue, distributed communication,
workers, allocator, environment utilities, custom/dummy ops, shared model-executor
kernels/layers, third-party local code and inactive branches. Also account for native
build/ABI, requirements and configuration dependencies. Map each file to retain, update,
migrate, remove, platform-only or explicitly unsupported with concrete evidence.

Trace import/startup, registration, call, data/state and conditional dependency edges.
Collapse cycles into coordinated adaptation units. Verify that every earlier-phase
file is source-ready before opening models; a skipped file is an incomplete framework.

## Models, then removal-first patches

[model-upgrade](../../model-upgrade/SKILL.md) owns every model, its local implementation
dependencies and its model-directory spec. Models must not depend on upstream concrete model implementations, directly or via
local helpers. Public framework APIs, including shared model utilities/interfaces, remain upstream.
New framework dependencies discovered here need a tracked repair and consumer recheck.

Only after all MetaX module source adaptation, perform the complete
[patch-upgrade](../../patch-upgrade/SKILL.md) pass. Re-review early blocking repairs.
Consider upstream fixes, supported hooks/registrations and already adapted local modules
before retaining a patch. If a patch removal needs a local module repair, track it and
revalidate affected consumers before opening model validation.

## Final model matrix

Declare exact chip/quantization/cache/parallel/model configurations and required cases.
Bind every supported row to normal-entry e2e and batched inference correctness evidence;
include prefill/decode, real loading, multimodal, MTP, distributed and graph obligations
where supported and requested. Every inventory file needs reviewed disposition and
current acceptance, even when its outcome is no code change or removal.

Run model services only in final validation. Capture actual child/worker dispatch,
configuration, source and extension origins, raw logs, batched result/manifests and
frozen numerical results. Current-chip limits remain visible as unverified branches;
passing a narrower configuration must never be relabeled as broader coverage.
