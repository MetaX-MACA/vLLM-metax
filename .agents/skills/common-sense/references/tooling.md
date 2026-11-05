# MetaX upgrade run tools

Use the repository skill `$vllm-metax-upgrade` for a full upgrade, or a specialist
skill for a limited component. These standard-library tools persist the environment
gate, worklist and evidence; the agent still reviews contracts and implements code.
They do not run an autonomous model service. Installation helpers belong to environment-check.

The portable coordinator instructions are in
[vllm-metax-upgrade.md](../../../agents/vllm-metax-upgrade.md).
An agent can read that Markdown file or invoke the top-level skill directly.
Native subagent registration, when needed, is configured by the client using these
instructions and the session's model and permissions. See the repository's
[AGENTS.md](../../../../AGENTS.md) for skill discovery and scope.

Invoke every command with the **selected validation interpreter** from the repository
root. For example, replace the interpreter and target paths below as needed:

```bash
/opt/venv/bin/python .agents/skills/common-sense/scripts/metax_upgrade/upgrade.py init \
  --run-dir .metax-upgrade/release-upgrade \
  --vllm-source /root/workspace/vllm \
  --metax-source /root/workspace/vLLM-metax \
  --runtime-cwd /root/workspace/vLLM-metax \
  --scope 'Requested models, quantization and parallel configurations' --mode full
```

The command only prepares a run. Review `ENVIRONMENT.json`, the immutable JSON
under `environment/`, the original text in `ENVIRONMENT.txt`, and `TODO.md` with the
user. `.agents/skills/common-sense/scripts/metax_upgrade/environment.py` independently collects SDK/GPU identities,
the selected interpreter's package metadata, `requirements/maca_private.txt`
comparisons, source/runtime correspondence and native-file identities. It does not
import or require a structured API from `vllm_metax/collect_env.py`.

The normal collector keeps its original `main -> get_pretty_env_info -> pretty_str`
flow and section style, takes no command-line options, and prints all installed
packages by default. Use shell redirection to save it:

```bash
vllm_collect_env > environment.txt
# Or use an explicitly selected interpreter and checkout:
/opt/venv/bin/python vllm_metax/collect_env.py > environment.txt
```

The toolkit optionally runs that plain script in a child with automatic plugin
activation disabled and records its sanitized text, command, timestamp and status.
Each full collection runs the script afresh; specialists collect their own current
facts rather than reusing another skill's output. Saved reports remain historical
evidence, not a source of live environment data. `refresh --run-dir ...` collects a
new full snapshot. Collector failures retain the independently collected environment
facts and their limitations; they do not establish compatibility or silently approve
anything. Native stat/RECORD identities likewise do not prove ABI or numerical support.

Full upgrades first apply [environment-check](../../environment-check/SKILL.md).
Approve its complete preparation plan once, install/verify both frameworks, then refresh
this run and call `assert-ready`. The plan's user approval and the machine's final snapshot
verification are separate durable records; no second snapshot confirmation is requested.
New runs and all full upgrades reject the legacy `confirm` command. Older component-only runs without a
preparation plan retain their existing explicit snapshot-approval path.

Before modifying a unit, register its files, configuration and required evidence:

```bash
/opt/venv/bin/python .agents/skills/common-sense/scripts/metax_upgrade/upgrade.py task-add \
  --run-dir .metax-upgrade/release-upgrade --id attention-layout \
  --title 'Adapt layout contract' --files vllm_metax/actual_module.py \
  --phase attention --owner attention-upgrade \
  --requires static --requires review --requires component_numeric --requires closure --requires e2e --requires batched_test \
  --configuration bf16-single-gpu --next-action 'Compare the exact upstream function'
```

Repeat `--files`, `--requires` and `--depends-on` as needed. Dependencies name existing
tasks; combine cycles into a coordinated task. Use one task per exact configuration.
Every task needs a checked mapping before it can become `done`. Register complete
file coverage with `mapping-put --id TASK --file mapping.json`; task notes alone do
not satisfy this requirement. A mapping document has this shape (replace all example
paths, revisions and hashes with actual source identities):

```json
{
  "mappings": [{
    "local": "vllm_metax/actual_module.py",
    "upstream": {
      "revision": "TARGET_REVISION",
      "path": "vllm/path/to/module.py",
      "symbol": "actual_function",
      "sha256": "ACTUAL_TARGET_CONTENT_SHA256"
    },
    "previous": {
      "revision": "PREVIOUS_UPSTREAM_REVISION",
      "path": "vllm/path/to/module.py",
      "symbol": "actual_function"
    },
    "disposition": "update",
    "metax_semantics": "State the verified previous MACA behavior to preserve",
    "reason": "Explain why this target still needs that difference",
    "removal_condition": "State the concrete condition for removing the difference"
  }]
}
```

Dispositions are `retain`, `update`, `migrate`, `remove`, `unsupported` and
`platform_only`. An intentionally absent upstream or previous counterpart uses
`{"absent": true, "reason": "Concrete absence explanation"}`. Do not invent a symbol
for platform-only additions or removed paths. The tool verifies mapped identities;
the agent still reviews whether the mapping and MACA rationale are correct. `sha256`
identifies the entire referenced source file; `symbol` must exist in it (`<module>`
can represent the whole module). References default to the selected vLLM checkout;
`previous.source` may identify a different prior-source checkout when needed. Reviewable
target-vs-local diffs and mapping records are retained under the run's `review/` directory.

Persist semantic producer/consumer dependencies separately. `contract-put --file contract.json`
accepts:

```json
{
  "id": "cache-layout",
  "upstream": "TARGET_REVISION:path:Symbol",
  "summary": "Producer and consumer share the physical cache layout",
  "invariants": ["Specify concrete dtype, axes, strides and index units"],
  "producer": "cache-task",
  "consumers": ["attention-layout"]
}
```

Register the support matrix with `matrix-put --file matrix.json`. This replaces the
whole matrix, so retain all intended rows when updating it:

```json
{
  "rows": [{
    "id": "bf16-single-gpu",
    "status": "supported",
    "configuration": {"dtype": "bfloat16", "tensor_parallel_size": 1},
    "requires": ["component_numeric", "e2e", "batched_test"],
    "tasks": ["attention-layout"]
  }]
}
```

Each row declares its exact configuration, required evidence kinds and linked task
IDs. The task's `--configuration` is a label; row membership comes from `row.tasks`,
while `row.configuration` is the expected actual dictionary reported by the child.
Do not bind one task to rows with conflicting configuration dictionaries.
Full runs need at least one supported row and current `e2e` and `batched_test` plus every other required
kind for **each supported row**. An `unsupported` row needs `rejection_reason` and
an actual contract/production rejection result. A label or exclusion note cannot
stand in for the rejection check. Component numerical and production checks also
need matrix binding in component mode; a component run makes only that scoped claim.

Before numerical checks, register the frozen contract with
`numeric-put --id TASK --file numerical.json`:

```json
{
  "reference": {"description": "Independent reference implementation", "revision": "REFERENCE_REVISION"},
  "seed": 0,
  "tolerances": {"atol": 0.001, "rtol": 0.001},
  "nan_policy": "forbid",
  "inf_policy": "forbid",
  "cases": ["prefill-short", "decode-multi-token"]
}
```

Changing an existing numerical contract requires `--note TEXT` explaining the
technical basis. The previous contract remains in `acceptance_history` and affected
evidence becomes stale. Do not loosen tolerances solely to make a failing test pass.

The reference must be independent of the kernel under test. NaN/Inf policies are
`forbid` or `match_reference`; use the documented child-result matching fields when
special values are allowed: the relevant case must report `nan_matches_reference: true`
or `inf_matches_reference: true` after comparing actual patterns and Inf signs. Changing mappings, contracts or matrix obligations makes
affected evidence stale. It does not turn older results into evidence for the new plan.

The agent copies the exact target upstream module/function before reapplying verified
MACA semantics. Preserve upstream structure and semantic roles; annotate necessary
differences concisely with `NOTE(metax):`. Do not optimize during adaptation.

Useful commands, all taking `--run-dir`:

| Command | Purpose |
| --- | --- |
| `scope-set --scope TEXT` | Update the confirmed run scope without reinitializing; records history and invalidates task evidence. |
| `task-set --id ID --status STATUS --note TEXT --next-action TEXT` | Save progress/blockers; done requires current evidence and done dependencies. |
| `run-check --execution-scope component\|model --id ID --kind KIND --configuration CONFIG --input-file PATH --timeout SECONDS -- COMMAND ARGS` | Run a fresh bounded subprocess and retain input hashes, output, provenance and case results; repeat `--input-file` for indirect inputs. |
| `checkpoint --note TEXT --next-action TEXT` | Save a handoff and invalidate evidence affected by source changes. |
| `status` | Reconcile source hashes and regenerate the visible TODO. |
| `resume` | Check environment/source identity, recover interrupted work and clean recorded checks. |
| `refresh` | Collect a new full environment snapshot; changed identity invalidates verification until checked against the approved preparation plan. |
| `matrix-put --file PATH` | Replace the structured support matrix and invalidate affected evidence. |
| `mapping-put --id ID --file PATH` | Register source mappings and MACA dispositions for every task file. |
| `numeric-put --id ID --file PATH [--note TEXT]` | Freeze numerical acceptance; changing an existing contract needs a technical reason. |
| `migrate --note TEXT` | Back up an old schema and migrate it with prior evidence marked stale. |
| `accept-policy-change --note TEXT` | Record reviewed tool/rule changes before new checks; old evidence becomes stale. |
| `phase-next --note TEXT` | Advance exactly one source phase after complete current inventory/source acceptance. |
| `coverage-put --file FILE` | Record unavailable-chip branches as explicitly unverified; see hardware-validation.md. |
| `finish` | Check current acceptance; full mode requires final validation, every inventory file verified and every supported row's e2e/batched/other obligations. |

Evidence kinds: `static`, `contract`, `component_numeric`, `closure`, `weight_loading`,
`e2e`, `batched_test`, `distributed`, `graph`, `review`. The command must actually assert the expected
behavior; exit zero by itself cannot establish adequate coverage. A later failure
supersedes an earlier pass. Contract/source/environment changes invalidate evidence.
Declare scripts, reference implementations, fixtures and indirect data/config inputs
using repeatable `--input-file`; their content hashes are checked again before evidence
can count. Direct command scripts and PATH executables are hashed too, including
extensionless scripts; retargeting a declared symlink invalidates its old evidence.
Each check saves `acceptance.json` with the matrix, mapping, numerical contract and
tool/rule identity used for that invocation. Evidence records its policy fingerprint.

For runs with a preparation plan, production origins must also include the actual
`kernel_backend` provider, string flag and loaded mcoplib module list as described in
[validation.md](validation.md). The runner fixes the approved flag in child processes.
Source-mode mcoplib imports fail acceptance; switching providers invalidates environment
identity and cannot be silently inferred from installed packages.

Production kinds require actual child module origins and selected dispatch in JSON
written to `METAX_UPGRADE_ORIGINS_FILE`. `component_numeric` requires actual installed
callable/extension origins. Both require the actual structured configuration. Numerical
results are required for a registered numerical contract; `component_numeric` always
requires such a contract. The runner sets
`METAX_UPGRADE_NUMERICAL_CONTRACT_FILE` to the persisted contract and
`METAX_UPGRADE_NUMERICAL_CONTRACT_DIGEST` to its digest. Read these in the validating
child; do not supply an invented digest or prewrite its report in the coordinator.

Numerical results contain `numerical.contract_digest` and a `cases` array. Every
registered case must report `id`, `max_abs`, `reference_scale`, `nan_count`, `inf_count`
and `passed: true`. The tool independently checks
`max_abs <= atol + rtol * reference_scale` and the frozen NaN/Inf policy; `passed` alone
cannot approve a result. Report measured error and reference scale from the independent
comparison, not constants selected to satisfy the formula. The following production
report illustrates a task with a numerical contract; all paths, dispatch and numeric
values must come from the actual validating child:

```json
{
  "vllm": "/actual/runtime/vllm/__init__.py",
  "vllm_metax": "/actual/runtime/vllm_metax/__init__.py",
  "dispatch": {"backend": "actual.module.SelectedBackend"},
  "kernel_backend": {"provider": "source", "use_precompiled_kernel": "0", "loaded_mcoplib_modules": []},
  "configuration": {"dtype": "bfloat16", "tensor_parallel_size": 1},
  "numerical": {
    "contract_digest": "VALUE_OF_METAX_UPGRADE_NUMERICAL_CONTRACT_DIGEST",
    "cases": [
      {"id": "prefill-short", "max_abs": 0.0002, "reference_scale": 1.0, "nan_count": 0, "inf_count": 0, "passed": true},
      {"id": "decode-multi-token", "max_abs": 0.0003, "reference_scale": 1.1, "nan_count": 0, "inf_count": 0, "passed": true}
    ]
  }
}
```

A component numerical report uses `callables` with the actual callable/extension
identities instead of requiring MetaX package imports solely for provenance, and
retains the same configuration/numerical fields. Every declared configuration field
must match; the child may report additional observed fields. A deliberate unsupported-row check
writes `rejected: true`, a nonempty `rejection_reason` and the matching actual
`configuration` to the same child report file. It must assert the actual expected
capability rejection; simply writing an expected message is insufficient.

Review commands can check a persisted disposition artifact; also record the review
judgment and limitations in task notes. Arbitrary commands use argument vectors,
without an implicit shell. Only production kinds retain the normal plugin configuration.

Initialization persists its recovery phase before archiving the initial dirty worktree
in `baseline/metax-source.zip` with its manifest. The archive is built atomically;
resume an interrupted baseline phase before adapting code, rather than replacing the
original snapshot with a later edited tree.
`state.json` is authoritative and atomically replaced; `TODO.md` is a recoverable view.
Runs under `.metax-upgrade/` are ignored by Git but remain visible in the workspace.
Keep them on durable storage when moving machines or handing work to another AI.
State uses a Linux advisory lock. Checks run under a supervisor that kills their process
group if the coordinator dies or their timeout expires; a launch handshake records
process birth identity before starting the workload. Programs that deliberately detach
into another session must manage and record their own cleanup.

`--mode component` limits completion to declared component obligations. It makes no
full-repository claim. Full mode requires every inventoried production/build/requirements
file to have a disposition task and a mapping. Each supported matrix row must pass
normal-entry e2e, batched inference correctness and every other declared evidence kind. The agent must still review
whether the matrix covers the intended model/configuration scope and whether numerical
references, case selection and measured reports are semantically valid.

Older run schemas must use `migrate --note TEXT`; the command preserves the original
state backup and retains old evidence as stale. Fill missing mappings/matrix/numerical
contracts and rerun required checks. When current tool or shared-rule identity changes,
review the difference and use `accept-policy-change --note TEXT` before new checks.
This records a workflow-version acknowledgment and invalidates old evidence; it is
separate from user approval of the preparation plan. Fresh unchanged environment
facts retain verification; drift must be checked against that plan without repeated prompts.

Offline helper validation (independent of the repository test fixtures). Keep
`PYTHONPATH` local to the test command so child test processes can import the package:

```bash
PYTHONPATH="$PWD/.agents/skills/common-sense/scripts" \
  python -m unittest discover -s .agents/skills/common-sense/scripts/metax_upgrade/tests -p 'test_*.py' -v
ruff check .agents/skills/common-sense/scripts/metax_upgrade
ruff format --check .agents/skills/common-sense/scripts/metax_upgrade
```

The toolkit and workflow files moved into `.agents/`; existing `.metax-upgrade/`
run directories remain in place. Their saved workflow identity will differ. Review
the relocation and use `accept-policy-change --note TEXT` when resuming a run;
previous evidence is retained as stale under the existing workflow rules.

## Ordered source adaptation (schema 4)

Full runs advance `platform → registry → attention → framework → models → patch →
validation`. The inventory is initialized before adaptation and generated in `FILES.md`
and `FILES.json`; TODO shows owners, source readiness and unverified hardware coverage.
Review task phase defaults and use `--phase` / `--owner` for transitive helper ownership.
Each full-run source task requires static and review evidence plus its local obligations.

```bash
python .agents/skills/common-sense/scripts/metax_upgrade/upgrade.py run-check \
  --run-dir .metax-upgrade/release-upgrade --id attention-layout \
  --execution-scope component --kind static --configuration bf16-single-gpu \
  --input-file .metax-upgrade/release-upgrade/check_layout.py \
  -- python .metax-upgrade/release-upgrade/check_layout.py
# Run the recorded review and any other required local checks, then:
python .agents/skills/common-sense/scripts/metax_upgrade/upgrade.py task-set \
  --run-dir .metax-upgrade/release-upgrade --id attention-layout --status implemented
python .agents/skills/common-sense/scripts/metax_upgrade/upgrade.py phase-next \
  --run-dir .metax-upgrade/release-upgrade --note 'Every file through this phase has current source acceptance'
```

Do not advance just because this example task passed; the tool checks **all** inventory
entries through the current phase. `implemented` records source readiness; `done`
requires final configured evidence. Early blocking patch repairs use `--exception-note`
and still need a fresh formal patch-phase review. Late framework repairs record their
reason and invalidate affected consumers; they do not open early model execution.

`--execution-scope model` is mandatory for complete-model checks, including dummy smoke
recorded as `closure`. Such checks require the final validation phase and current source
readiness for every framework, model and patch file. A component declaration must never
hide a model constructor. Standalone component runs do not claim full framework readiness;
a specialist assigned to a full run cannot switch modes to bypass its phase gates.

For `batched_test`, use the report described in [validation.md](validation.md): actual
configuration/dispatch and worker origins, passing measured cases and content-hashed
result.json plus fingerprint manifest. Keep source/config/test inputs and output logs
with the run. A benchmark or a launcher exit code cannot satisfy this evidence kind.

Migrate schema 1, 2 or 3 using `migrate --note TEXT`. Migration backs up old state, preserves
prior evidence as stale, initializes the file ledger and starts ordered source review
at platform; it does not manufacture source readiness or final verification.

Legacy tasks which span incompatible source phases are retained with `needs_replan`.
Use `task-replan --id ID --phase PHASE --owner SKILL --files PATH --note TEXT` to
assign a valid subset, then create tasks for the remaining files. Old assignments,
mappings and stale evidence remain in history; supply a fresh checked mapping before
continuing. Unassigned inventory entries still block advancement and completion.

## Contract-scoped source and model evidence

Source signatures bind owned files, model spec/import closure, source mapping, local
check policy, numerical reference and only contracts where the task is producer or
consumer. Final model evidence additionally binds its matrix/acceptance. A new unrelated
contract does not invalidate earlier framework tasks. A relevant contract change requeues
its participants and transitive consumers; matrix-only changes invalidate model evidence.

Before adaptation use `analysis-put --file PLAN.json`: `files` lists the complete file
inventory; `models` lists every baseline sample; `model_contracts` maps each sample ID to
nonempty `framework_interfaces`, `cache_layout`, `quantization` observations; `edges`
contains producer/consumer file paths and contract summaries; `cycles` gives file groups
and coordinated strategies; `summary` records the review. Inspect all local model needs
read-only here, before framework code changes. Additions require updating the plan.

Use `checks-put --id TASK --file CHECKS.json` before local checks. The document has
`changes` (api, dispatch, lifecycle, layout, quantization, indexing, numerics,
communication, or none), `reason` and required `cases`. The tool derives mandatory
contract/numerical obligations. A justified `defer_numeric` (feature, chip, reason) must
match an explicit unavailable-chip record covering this exact task/configuration.

Use `model-review --id TASK --entry LOCAL_MODEL.py --spec LOCAL_SPEC.md`, repeating
entries/specs as needed. It runs the model-adaptation auditor and binds its source/spec
hashes before static/review evidence. Specs must be owned files in the model directory.

`model-bind --file MODELS.json` accepts `{"models": [{"id": "tools/models/SAMPLE",
"spec": "vllm_metax/models/FAMILY/SPEC.md", "matrix_rows": ["ROW"]}]}`. Each row
identifies the matching `model_sample`, its actual `configuration.model_path`, and a
source task with a current model review. Initialize/synchronize discovers all samples;
unbound samples prevent finish. `MODELS.json` and TODO expose results.

`model-skip --id tools/models/SAMPLE --reason-code insufficient_gpus|insufficient_vram
--required N --available N --evidence-file PATH --note TEXT` records a genuine resource
deficit in final validation. Use GPU counts or VRAM bytes consistently. Required must
exceed available. It preserves missing model evidence as explicitly skipped, not passed;
all local source/component obligations still apply. Source/config/environment/artifact
changes invalidate the skip. Missing weights, implementation errors and omitted tests
are not resource skips. Final reports must enumerate every skipped model and limitation.

Model bindings require explicit `configuration.validation_mode` (`dummy_smoke` for dummy
samples, `accuracy` for real weights); dummy configurations also declare `load_format`.
Full runs still need e2e and batched evidence for each bound sample, interpreted under
that mode. Resource skips remain separate from successful execution and from semantic
accuracy. The batched task manifest includes validation modes and source/environment/
checkpoint/input identities, and the actual report names its `model_tag`.
