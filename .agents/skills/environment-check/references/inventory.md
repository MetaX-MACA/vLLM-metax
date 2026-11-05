# Independent environment inventory

Read this for every new or resumed upgrade run, including standalone specialist work.
Collection is read-only environment preparation. Follow [preparation.md](preparation.md)
for one-time plan authorization and successful installation before adaptation.

## Select inputs without importing MetaX

Discover the intended Python, both source checkouts and validation cwd from the user
and workspace. Inspect `sys.executable`, `sys.prefix`, `sys.base_prefix`, `pyvenv.cfg`
and Conda metadata through that executable. `VIRTUAL_ENV`, `CONDA_PREFIX` and the current
shell prompt are hints, not authority. A uv venv can legitimately use a Conda base;
resolving a Python symlink does not identify the selected package environment.

Record discovered venv/Conda alternatives and identify which is selected. Dump installed
packages from that selected interpreter, including missing/private packages; do not
silently substitute global Conda's package list. If another environment will execute
workers or tests, collect it separately and include that mapping in the confirmation.

Use the selected interpreter for the CLI. For example, replace the executable and paths
below with discovered choices, and choose a durable run directory:

```bash
/opt/venv/bin/python .agents/skills/common-sense/scripts/metax_upgrade/upgrade.py init \
  --run-dir .metax-upgrade/release-upgrade \
  --vllm-source /workspace/vllm \
  --metax-source /workspace/vLLM-metax \
  --runtime-cwd /workspace/vLLM-metax \
  --scope "requested model and configuration matrix" --mode full
```

`init` collects immutable environment artifacts, preserves the initial source manifest
and MetaX worktree archive, initializes state and renders TODO. Use `--mode full` for
repository upgrades or `--mode component` for a deliberately limited standalone domain
or review. Component completion makes no full-repository/e2e support claim.
Use `refresh --run-dir ...` to collect an updated snapshot; use `resume --run-dir ...`
when handing a run to another session. Read [shared tooling](../../common-sense/references/tooling.md) and CLI `--help` for current arguments.
Do not create a new run to conceal existing partial work or avoid a failed gate.

Each skill request independently collects current environment facts, even if another
skill or earlier session already produced a report. Historical ENVIRONMENT reports,
JSON snapshots and diagnostic text remain audit evidence; do not reuse them as current
facts. The agent's independent standard-library inventory is
`.agents/skills/common-sense/scripts/metax_upgrade/environment.py`; it owns SDK/GPU/Python/package/source discovery,
requirements comparison and the structured JSON snapshot. It must work before MetaX
modules are adapted and without importing Torch/vLLM/MetaX. The existing
`.agents/skills/common-sense/scripts/probe_environment.py` is a backwards-compatible wrapper around the tool's
source-probe mode. Do not make this inventory depend on `collect_env.py` output or API.

Keep the selected MetaX checkout's `vllm_metax/collect_env.py`, exposed as
`vllm_collect_env`, as the original human-readable supplement. Its entry follows
`main -> get_pretty_env_info -> pretty_str`, accepts no CLI arguments, and prints to
stdout. Activate the selected environment and use its matching console entry point:

```bash
vllm_collect_env > environment.txt
```

If that entry point belongs to another installation, run the exact selected file:

```bash
/opt/venv/bin/python /workspace/vLLM-metax/vllm_metax/collect_env.py > environment.txt
```

Run this supplemental diagnostic in a separate subprocess. Record its command,
interpreter/cwd, stdout, stderr and exit status. It may import runtime dependencies
and fail while MetaX is unfinished; preserve that failure and continue the independent
inventory. Never substitute upstream `vllm/collect_env.py`, parse this text as a required
structured schema, or interpret successful collection as compatibility validation.

Collect supplemental stdout or failure afresh for the current collection; no cache
or cross-skill reuse is permitted. Ordinary identity checks use fresh independent facts
and do not force execution of the supplemental collector. Agent `init`/`refresh` perform
fresh full collection and save new immutable structured reports with that invocation's
supplemental stdout or failure. Identity-only checks recollect current facts without
running `collect_env.py`. Historical reports remain available for auditing and comparison only. Previously given
confirmation may remain valid when newly collected identity and mappings are unchanged;
this preserves approval semantics without treating old environment facts as current.

## Required snapshot

| Item | Required evidence |
| --- | --- |
| MACA | SDK installation/version, compiler/toolchain, runtime and driver versions/origins where discoverable; missing fields and probe failures. |
| GPU | Exact model, device identifiers/count, visibility and available-device evidence; record validation device selections. |
| Python | Executable, version, prefix/base prefix, venv/Conda identity, discovered alternatives, cwd and relevant path overrides. |
| Packages | Full selected-environment metadata inventory; installed versions/origins compared with every requirement in `requirements/maca_private.txt`, including included files and markers if present. |
| Local vLLM | Confirmed released target revision, repository/package paths, dirty state and source fingerprints. |
| Installed vLLM | All distribution candidates, version, metadata/package location, editable/install provenance and effective resolution. |
| Local MetaX | Initial revision/index/worktree snapshot, source fingerprints and checkout to edit. |
| Installed MetaX | All distribution candidates, effective origins and source/wheel differences; this is unverified code. |
| Components | Torch/Triton, attention/GEMM/communication and every private requirement's package identities, extension origins and ABI evidence when available. |
| Collection details | This request's independent JSON snapshot and probe provenance; supplemental `collect_env.py` source/interpreter/cwd and newly collected stdout/stderr/status when executed. |
| Runtime plan | Comparison source, edited checkout, actual validation/worker imports and any proposed process-local path adjustments. |

Parse requirements as data; never execute requirements or pip-install during collection.
Do not limit package inventory to a hard-coded list: `maca_private.txt` is the source
of required private-package coverage. Pin mismatches are explicit issues for confirmation
and later capability checks. Approved ordinary dependency changes must still preserve all image/private locks.
Unknown or unparsable requirements remain visible as gaps.

A missing or failed supplemental collector does not block metadata/hardware/source
inventory. Include the failure and remaining unknowns in the confirmation report.
The complete preparation plan must be approved once and both installations verified
before adaptation; approving known gaps does not establish the missing capability.
Do not semantically import unfinished MetaX modules into the coordinator. Environment
probes are the only runtime collection allowed before approval.

## Inspect correspondence

Compare source trees with installed distributions, not just versions or `__init__.py`.
Multiple versions, user site, `.pth` files, editable finders, symlinks, cwd shadowing and
`PYTHONPATH` can make metadata describe a different package from actual imports.
Local egg-info is not proof an installed wheel matches. `direct_url.json` can identify
a build source without proving current runtime correspondence; its absence alone is
not a failure. Separate generated version files from substantive source differences.

The probe models `python -c`/`python -m` from the runtime cwd; it cannot prove console
script, pytest, Ray or engine-worker origins. Relative `PYTHONPATH` or cwd-dependent
startup customization requires launching from the actual runtime cwd. Confirm effective
origins inside real validation processes after the gate. Mixed Python and extension
origins need explicit ABI evidence; Python tree equality does not establish binary match.

Keep relevant path overrides only. Do not dump unrestricted environment variables,
credentials or credential-bearing index/provenance URLs. Preserve sanitized raw logs
and report any redaction; do not conceal mismatches through redaction.

## Plan authorization and final verification

Follow [preparation.md](preparation.md). Use this inventory to assemble all pending
choices before asking once. Save the actual reply against the complete plan digest.
Installation and approved Git/dependency actions are covered by that plan. After setup,
collect fresh facts and verify automatically; never ask for another fingerprint approval
or pretend that the user saw and confirmed the post-install snapshot.

The shared `assert-ready` gate checks the verified preparation record, current facts and
source mapping. Missing/failed installs close adaptation gates. Reuse the same plan and
run across specialists. New environment choices outside the plan are recorded blockers,
not repeated prompts during adaptation. Environment diagnostics remain allowed.

## Resume and stale evidence

Environment identity includes SDK/GPU, selected interpreter/packages, requirements,
released target and runtime mapping. Fresh unchanged identity retains the plan's
verification. Expected MetaX adaptation edits update source fingerprints separately;
they do not force environment reapproval after every edit. Drift requires automatic
verification against the original plan and invalidates relevant evidence; incompatible
choices remain blocked. Source/contract changes invalidate affected tasks/consumers.

Resume first, inspect the current report and source diffs, then choose an unfinished
obligation. Never use a historic passing log after its environment or source changed.
