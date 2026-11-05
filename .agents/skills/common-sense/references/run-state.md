# Durable run state and handoff

Use the repository's `.agents/skills/common-sense/scripts/metax_upgrade/upgrade.py` and its `--help` and `.agents/skills/common-sense/references/tooling.md` for the
installed CLI. Keep runs under a durable workspace directory such as
`.metax-upgrade/<target>-<name>/`, outside protected skill/config directories and `/tmp`.
The run directory may be untracked; do not automatically commit evidence or logs.
`init --mode full` is the default full-repository workflow; use `--mode component` for
limited standalone domain/review work. TODO displays the mode. Full completion requires
passing normal-entry e2e and batched inference checks for every supported matrix row and disposition
coverage of the baseline production/build/requirements inventory; component completion
covers only its declared obligations. Full runs need at least one supported row.

Current collection and approval rules live in [environment-preflight.md](environment-preflight.md);
validation and defect review rules live in [validation.md](validation.md).

## Source of truth

The structured state is authoritative. `TODO.md` is generated for the user and other
agents; never infer completion from a checkbox or edit it instead of state. Use the
CLI's atomic writes and durable snapshots. Do not hand-edit state to bypass environment
approval or evidence checks. After a crash, recover the latest valid state/snapshot
through the CLI, regenerate TODO if needed, and inspect unfinished source changes.
Initialization first persists recoverable state, then builds the baseline archive
atomically. An interrupted baseline phase must be completed through recovery before
adaptation or validation; do not overwrite the original baseline with later edits.

Every task has a stable ID, scope/files and upstream mapping, dependency/contract IDs,
status, acceptance obligations, evidence references, blocker and next action. Use:

| Status | Meaning |
| --- | --- |
| `todo` | Work recorded and not started. |
| `in_progress` | An owner is inspecting, adapting or verifying the unit. |
| `implemented` | Source changes exist; required evidence is still incomplete. |
| `blocked` | A specific prerequisite currently prevents the remaining obligation. |
| `done` | All required evidence is current and dependencies satisfy its scope. |

A blocked verification obligation need not stop the task's independent work. Record
which obligation is blocked and the precise resume condition. Missing/failed installation
or environment preparation closes adaptation gates; independent source edits cannot
bypass that prerequisite. A `done` task must be
reopened if evidence becomes stale; never equate a code edit with validation.

## File inventory and phase gates

`state.json` contains the complete inventory; `FILES.json` and `FILES.md` are generated
views alongside TODO. Every Python file, including initializers and third-party local
code, and relevant production/build/requirements file has a phase, owner, task binding,
current/initial hash, present/deleted flag, disposition and one of the three file labels
from common. A source-adapted file awaiting final evidence stays 正在适配中 with
`source_ready: true`. A final completed label is limited to the declared chip scope.

Task `--phase` and `--owner` record the actual assignment. Defaults classify files by
directory; review them and explicitly assign transitive framework helpers to the
consuming phase when needed. One shared file must keep one phase and owning skill
across configuration tasks. A file with no task or disposition cannot pass a phase.
New files are discovered on synchronization; deleted files remain as tombstones.

Use `task-set --status implemented` after current static/review and required local
checks and a checked source mapping. Use `phase-next --note TEXT` after all source
through the current phase is ready. Final `done` still requires every acceptance
obligation and done dependencies. Full runs cannot start models before all framework
files are source-ready; complete-model checks remain blocked until validation even
when one model's source is ready.

Late framework dependency repairs are permitted with a task note. An early blocking
patch task requires `--exception-note TEXT`; it still needs a fresh review in the
formal patch phase before final source readiness. A changed source/contract or stale
local check closes the corresponding source gate again.

`coverage-put --file coverage.json` records `{"unverified": [...]}`. Each entry has
`feature`, `chip`, `files`, `reason`, `resume_when` and `status: unverified`. These are
unavailable-hardware branches, not passing support rows. Current-chip acceptance can
finish with them explicitly listed; required checks within the declared supported
scope cannot be waived this way.

## Schema and workflow-version recovery

Use `migrate --run-dir ... --note ...` for old state schemas. Migration retains the original state
backup and prior evidence, marks old checks stale and exposes acceptance fields needing
new records. Do not hand-edit the schema number or import old pass results as current.

On resume, compare the recorded tool/rule identity with the current files. Review the
difference and record `accept-policy-change --run-dir ... --note ...` before running
new checks. Preserve superseded evidence as stale and rerun current obligations.
Current state uses schema 4; migration from schema 1, 2 or 3 does not preserve old passing
checks as current support evidence. This
acknowledges reviewed workflow changes; it is separate from the one-time user approval
of the environment preparation plan. Machine verification records the prepared result;
fresh unchanged facts retain it, while drift is checked against the same approved plan.
Do not repeatedly ask for environment confirmation during the run.

## Checkpoint before work can be lost

Persist the task and intended files **before** edits. At each completed unit, meaningful
discovery, validation result and handoff, save current hashes, decisions and next action.
Record actual source hashes for evidence, not only Git HEAD, since worktrees can be dirty.
The initial `baseline/source-manifest.json` and `baseline/metax-source.zip` preserve the
worktree before adaptation, including initial dirty changes. Do not overwrite them.
Use this baseline to recover prior MACA semantics; expected adaptation edits update
execution fingerprints rather than masquerading as environment changes.

On resume:

1. Read authoritative state, generated TODO, environment approval and last handoff.
2. Recheck current environment identity and approval through the preflight workflow;
   historical reports cannot substitute for current observations.
3. Compare recorded source fingerprints with current files and inspect local diffs.
   Treat an interrupted `in_progress` unit as unfinished; a written file is not proof
   of completed intent or successful validation.
4. Invalidate changed-unit evidence and evidence of consumers of changed contracts.
   Use dependency tracking; if precise impact is unknown, invalidate the conservative
   affected closure. Do not relabel old logs as current evidence.
5. Choose the earliest available obligation and checkpoint the next executable step.

## Evidence and handoff

Each evidence item records task/contract IDs, evidence type, exact command and inputs,
expected and observed result, source/environment and tool/rule fingerprints, recorded
test-input hashes, interpreter/cwd/import origins, relevant component/binary identity,
limitations and persistent artifact paths. Numerical evidence also binds the frozen
contract and actual per-case metrics; full-run evidence binds its support-matrix row.
Keep failed and superseded evidence visible with status/reason. Distinguish checks
which ran, checks which passed and checks which remain required.

The handoff identifies the approved preparation plan/reply, installation status and
verification record, current environment fingerprint, target revision and source/runtime
plan; modified units and upstream mappings; current contracts and blocked dependencies;
last successful and failed commands; exact next command/action; and cleanup of any
processes started for validation. Keep secrets out of state, command logs and dumps.
Show the user the TODO path and a brief completed/remaining/blocked count after meaningful
milestones. Before an expected quota stop, write a handoff; atomic checkpoints support
recovery even when there is no opportunity for a final message.

## CLI work cycle

Use the selected interpreter for every invocation. The exact path is discovered by
environment collection; `python` below denotes that interpreter, not an arbitrary shell
Python. Run `assert-ready` first, then create a task before editing:

```bash
python .agents/skills/common-sense/scripts/metax_upgrade/upgrade.py assert-ready --run-dir .metax-upgrade/release-upgrade
python .agents/skills/common-sense/scripts/metax_upgrade/upgrade.py task-add \
  --run-dir .metax-upgrade/release-upgrade --id attention-layout \
  --title "Adapt the attention layout contract" \
  --files vllm_metax/path/to/affected.py \
  --phase attention --owner attention-upgrade \
  --requires static --requires review --requires closure \
  --configuration bf16-single-gpu \
  --next-action "Copy the target function, then reapply verified MACA layout changes"
python .agents/skills/common-sense/scripts/metax_upgrade/upgrade.py task-set \
  --run-dir .metax-upgrade/release-upgrade --id attention-layout \
  --status in_progress --note "Upstream counterpart and consumers recorded"
```

Replace the example file/task/configuration with the real unit. Add repeatable
`--depends-on` edges. A task represents one exact configuration; split configurations
into separate tasks rather than letting one passing case cover others. The task's string
configuration is a label; matrix rows link it through `tasks` and define the exact actual
configuration dictionary to match in child reports. Do not attach conflicting row
configurations to the same task. Persist contracts
with `contract-put --run-dir ... --file ...`; its JSON requires `id`, `upstream`, `summary`,
`invariants`, `producer` and `consumers` (the latter two identify task IDs). Keep typed and
conditional dependency details in those contracts and linked run artifacts. Contract
updates conservatively invalidate existing evidence; checkpoint and revalidate consumers.
Before adapting a task, persist `mapping-put --id ... --file ...` with coverage for all
of its files: exact target and prior source identities, symbols/content hashes, MACA
reason and disposition. Explicit platform-only/removed records must not fabricate an
upstream counterpart. Persist `numeric-put --id ... --file ...` before numerical checks
and `matrix-put --file ...` for the support matrix. The latter replaces the entire
`{"rows": [...]}` document. Numerical/production checks need row binding in component
mode too; static/review-only component tasks do not make numerical support claims. The tooling guide documents the exact
accepted JSON shapes. Their updates invalidate dependent evidence. Changing an existing
numerical contract needs `numeric-put --note TEXT` with its technical basis; old and new
contracts remain in the acceptance history. Each check saves its frozen definition in
`acceptance.json` and records the policy fingerprint.

A task's `--requires` list is the acceptance vector;
include `contract`, `component_numeric`, `weight_loading`, `e2e`, `batched_test`, `distributed`, `graph`
and/or `review` when required by its capability, not just cheap static checks.

Use `run-check --execution-scope component|model --id ... --kind ... --configuration ... --input-file ... --timeout ... -- COMMAND ARGS`
for actual validation commands, with `--run-dir` as above. Repeat `--input-file` for
scripts, fixtures, reference data and other command inputs; later changes invalidate
the evidence. Command text alone does not identify the contents of an executed file. Commands are argument vectors,
not shell snippets. Record origins from the actual child as required by the CLI for
production evidence; do not fabricate reports in the coordinator. Follow the supported
child report shape in `.agents/skills/common-sense/references/tooling.md`, including actual configuration,
numerical contract digest, per-case metrics and results where required.

After source edits, use `checkpoint --note ... --next-action ...`; it synchronizes
fingerprints and invalidates stale affected evidence. Use `task-set --status implemented`
while validation is pending, `blocked` with a precise reason when a prerequisite is
missing, and `done` only after current required evidence exists. Use `status` to render
the TODO, `resume` to reconcile interrupted work and `finish` to check the recorded
completion gates. Every command takes `--run-dir`.

The CLI guards recorded approval, hashes, dependencies and evidence obligations. It
cannot prove that an agent discovered every dependency, chose sufficient numerical
cases, identified the semantic role correctly or copied a real user's reply. Those
remain explicit review responsibilities; never describe the tool as proving universal
compatibility from its exit status.

## Child reports and acceptance

The precise mapping, numerical-contract, matrix and child-report schemas live in
`.agents/skills/common-sense/references/tooling.md`. Use actual selected implementation origins and assert
dispatch in the validating child. Report the actual configuration and measured numerical
case results under the registered contract; do not write expected values in place of
measurements or prewrite the report in the coordinator. Component-only checks can
report their actual installed callable/extension without importing unfinished MetaX.

Provenance-only reports from earlier versions cannot satisfy the strengthened numerical
and per-configuration obligations. The tool validates registered structure, fingerprints,
case coverage and metrics, while the agent must still establish reference independence,
correct dispatch assertions and adequate semantic coverage.

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
