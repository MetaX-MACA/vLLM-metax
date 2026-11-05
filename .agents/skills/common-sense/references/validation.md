# Shared validation and defect review

Read this for compatibility reviews and adaptations before selecting tests or making
correctness claims. Domain skills supply the concrete cases; persistence and CLI
provenance formats live in [run-state.md](run-state.md).

## Execution phases

Apply the coordinator's strict source phase order in full runs. Before final validation,
allow only model-free static/review checks, framework/operation unit tests, component
numerical checks and startup/dispatch closures. A complete-model constructor, real
checkpoint load, dummy/trimmed smoke, token generation or batched inference is **model
execution**, regardless of its test filename or use of mocks. It must wait until every
module and patch is source-ready and `phase-next` reaches `validation`.

Every full-run `run-check` declares `--execution-scope component` or `model`. The
`closure` kind may describe either scope and therefore is not an automatic exemption.
`weight_loading`, `e2e`, `batched_test`, `distributed` and `graph` are reserved for the
final model scope; an isolated collective/graph unit test uses `contract` or
`component_numeric`. Model validation cannot be relabeled as a static/unit check.
The tool checks recorded declarations and common entry points; the reviewing agent
must also inspect what an arbitrary script actually executes.

Read [hardware-validation.md](hardware-validation.md) before selecting quantization
cases. Source adaptation includes unavailable formats; runtime acceptance is scoped to
the declared current-chip configurations, with other branches recorded as unverified.

## Final repository lint acceptance

Run `pre-commit run --all-files` from the MetaX repository root and fix **all current
hook failures**, including pre-existing formatting issues. Review automatic fixes and
rerun the full command until it exits zero with no pending hook changes. Individual
Ruff checks, staged-file-only checks or a clean diff check do not replace this command.
Keep the configured hooks enabled; bypassing hooks or weakening their configuration
does not satisfy acceptance. Missing hook dependencies or an incomplete run remain blockers.

Run the full hooks after source adaptation and before final model validation to catch
fixes early, then run them again on the final code before `finish`. Source changes from
hooks invalidate affected evidence under the normal source/dependency rules: reconcile
the file ledger and rerun the required checks for changed files and their consumers,
including model checks when required. After any further fixes, the final full-hook run
must pass again. Keep changes necessary to satisfy the hooks; preserve upstream layout,
comments and blank lines wherever the repository's required formatting permits.

Save the command, repository cwd, interpreter/tool identity, exit status, stdout/stderr
and final source identity in the durable run. Report the successful log path at handoff.
A prior passing log for different source cannot establish final lint success. Ensure new
upgrade files are covered too: `--all-files` selects Git-tracked files, so additionally
run `pre-commit run --files ...` for new untracked upgrade files when necessary.
Do not mark the upgrade complete while repository-wide checks still fail.

## Validate components while dependencies are unfinished

Begin with self-contained isolated scripts written from the fixed target contracts.
Do not trust repository tests, fixtures/conftest or old `/tmp` files before reviewing
their dependencies. Keep reproducible scripts and raw logs in the durable run directory.
Use repository integration/regression tests once the relevant fixtures and real
production dependency closures are ready.

Run candidates in fresh subprocesses with the selected interpreter, actual runtime
cwd and explicit process-local import plan. Record actual package/plugin, callable
and extension origins from the validating child; a metadata version or parent-process
resolution is insufficient. Candidate execution discovers failures without making
unrelated MetaX modules trusted. Recheck source/runtime correspondence after changes
to dependencies, checkouts, interpreter, cwd or import plan.

Stubs, file-based loading, optional-dependency bypasses and `--confcutdir` can isolate
a contract. Record exactly what they bypass, retain the real function under review,
and rerun the production path when ready. A bypassed initializer or mocked dependency
does not establish startup, production reachability, installed-kernel support or e2e
correctness. Do not silently alter the environment or remove required coverage to
turn a blocked check into a pass.

## Kernel provider evidence

Use the provider explicitly selected in environment-check. Build/runtime flags must
agree: mcoplib uses `USE_PRECOMPILED_KERNEL=1`, source uses `0`. The shared runner
propagates it; verify effective values in actual server/distributed workers as well.
Production origins include `kernel_backend` with `provider`, string
`use_precompiled_kernel`, and `loaded_mcoplib_modules` observed from that child's
`sys.modules`. Keep actual callable/extension origins with the dispatch record. Source
mode must report an empty mcoplib list and pass with mcoplib unavailable/imports rejected;
flag presence alone does not establish independence. Do not report parent-process values
as worker observations.

All csrc sources need independent build/contract/numerical evidence regardless of the
production provider. In mcoplib mode, isolate local-source component builds/checks and
record their distinct build directory, override and native origins; never substitute
mcoplib output for validation of newly ported local code. These isolated checks may
exercise source mode under their declared component context without changing the selected
production provider. They do not permit early model execution.

## Match evidence to the claim

Select acceptance obligations per configuration before editing. Evidence is a set of
independent obligations, not a single ascending score:

| Evidence | What it establishes |
| --- | --- |
| Static | Source/signature/AST/contract reasoning against the fixed target. |
| Isolated contract | The asserted behavior within recorded stubs or bypass boundaries. |
| Component numerical | The real installed component agrees with an independent reference for specified cases. |
| Real closure | Actual adapter startup, registration, dispatch, wrappers and asserted behavior for the exercised dependency/configuration closure. |
| Weight loading / e2e (final phase only) | Normal-entry construction, real checkpoint loading and execution for the declared cases. |
| Batched inference (final phase only) | Actual inference cases all pass, with result files and configuration-fingerprint manifests. |
| Distributed / graph (final model phase) | The exercised multi-process/multi-GPU or capture/replay behavior. |
| Performance | Measurements under a recorded benchmark configuration. |

Use actual installed kernels for numerical claims. The reference must not reuse the
same potentially faulty kernel. Imports/signatures cannot prove numerical behavior;
a raw kernel check cannot prove wrapper dispatch; a dummy/trimmed model proves only
the paths it preserves, not full-model accuracy or real-weight loading. Upstream flags
and support in individual dimensions do not establish their joint MetaX configuration.

Before numerical execution, persist the numerical contract: independent reference and
its version/source, fixed seeds, absolute/relative tolerance, NaN/Inf policy and required
cases. Declare scripts, fixtures, reference files and other test inputs as hashed check
inputs. The validating child must report its actual configuration, contract digest,
per-case results and error metrics; its measured values must meet the frozen policy.
A command's success or a nonempty provenance object cannot substitute for these results.
Changing the contract or any recorded input invalidates affected evidence.
An existing numerical contract can change only with a recorded technical reason;
retain its previous version. Do not relax tolerances solely to pass a failing case.

Choose the least expensive checks that expose the changed behavior, including boundary
and rejected cases. Run relevant syntax/format/lint/type/diff checks. For comment or
docstring-only code changes, an AST comparison excluding docstrings can verify that
executable code is unchanged without requiring a model suite. Once appropriate checks
pass, expand only for new changes, failures or unresolved concerns. Numerical success
does not establish speed, distributed support or graph correctness without those checks.

Record task/contract IDs, exact command/inputs/configuration, expected and observed
results, source/environment fingerprints, actual runtime origins, component/binary
identity, bypasses and limitations. Separate checks run, checks passed and obligations
still pending; keep failed and superseded evidence visible. Missing hardware, checkpoints
or components remain explicit unverified/blocked obligations.

## Attribute failures and re-review

For each suspected defect, inspect the equivalent path in the exact target upstream
revision, including dispatch and relevant dependencies. Where feasible run the same
discriminating reproduction. Classify adaptation regression, existing MetaX defect,
shared upstream defect, installed component/ABI issue, environment mismatch or blocked
prerequisite. Distinguish source-based suspicion from a reproduced upstream failure
and limit upstream claims to the revision actually inspected. Specialist-specific
repair boundaries still apply; attribution alone does not authorize broader changes.

For a suspected installed-component defect, isolate a direct call with valid inputs
at that component's boundary. Distinguish its result from wrapper transformations
before attributing the failure to the component or adapter.

Verify behavioral claims in comments against implementation and executable checks.
After edits, revisit every prior finding as fixed, partially fixed, still present or
masked by an earlier failure/capability rejection. A first exception can hide later
bugs; do not stop at fixing it. Recheck for concurrent source changes before final
reporting and invalidate stale evidence for changed units and affected consumers.

## Final batched inference

Run `tools/batched_test` inference correctness using the selected model/config/case YAML,
exact GPUs, chip-supported quantization and full current sources. Hash model and case
configs, templates, test inputs and wrapper scripts as check inputs. Save result JSON,
CSV, adjacent fingerprint manifest, service/client logs and actual server/worker origins.
Use `--infer`; `--perf` is additional only when requested. Do not accept a dry run,
benchmark-only run, startup-only check or success exit code as correctness evidence.

Record kind `batched_test` separately from `e2e`. Its actual child report includes
`batched_test.cases` with unique IDs, `status: passed` and `correct_ratio: 1.0`, plus
`batched_test.artifacts` with absolute `path` and `sha256` for result.json and the
`.manifest.json` file. The tool verifies case outcomes and output content identities.
Use the current fingerprint manifest to justify reused successful results; a legacy CSV
without source/config correspondence cannot establish current upgrade acceptance.

## Mandatory component obligations and source identity

Before local checks, record `checks-put` with semantic change categories, reason and
cases. Contract tests are mandatory for API/dispatch/lifecycle changes. Layout,
quantization, indexing, numerical and communication changes also require an independent
component numerical reference, frozen numerical contract and actual installed components.
No complete model is needed for these checks. Unavailable-chip numerical deferral needs
matching explicit unverified coverage; it cannot exempt supported BF16/FP16/INT8 cases.

Final batched reuse must bind the current MetaX/vLLM source identity, selected environment,
checkpoint content and test inputs. The scheduler captures this during planning and writes
schema-2 manifests. Legacy CSV/manifests, unbound tasks, mutable remote inputs and model
IDs without a locally verified checkpoint identity are never reused. Local checkpoint
weights are streamed for full content hashes; this can take time for large models.
Hashing is explicit planning work, never an eager dataclass/property side effect.

The upgrade runner supplies `METAX_UPGRADE_VALIDATION_IDENTITY_FILE` to the validating
child. Its source/environment fields must appear in the batched manifest; mismatched or
legacy manifests cannot establish current upgrade acceptance. Actual server/worker
origins and dispatch are still required; a coordinator declaration alone is insufficient.

## Dummy execution and real-weight accuracy

The baseline tools/models samples use dummy initialization. Run their startup, prefill,
at least two decode tokens and concurrent request stability cases in the final phase.
Use `validation_mode: dummy_smoke` with the actual dummy load format. Text is checked for
valid nonempty generation and reported completion-token counts, not answer keywords;
embedding smoke checks finite valid vectors. Record server-side batch/dispatch metadata
when claiming a particular batch shape or mixed prefill/decode branch; concurrent HTTP
requests alone do not prove that exact kernel batch occurred.

Batched CSV/JSON/manifest distinguish DUMMY_SMOKE_PASSED from semantic accuracy. Dummy
rows report Correct Ratio as N/A and a separate Smoke Pass Ratio. Real weights use
accuracy cases and retain the existing scoring. Reports for dummy rows include
`batched_test.validation_mode: dummy_smoke`, `semantic_accuracy_claim: false` and the
matching `model_tag` from the manifest. The actual configuration has matching
`validation_mode` and load format. Never reinterpret dummy output as model accuracy.
