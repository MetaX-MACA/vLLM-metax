---
name: environment-check
description: Prepare and verify the Python environment, target upstream checkout and source installations for vLLM-MetaX upgrades. Gather one complete preparation plan for user approval before installation or adaptation; verify the installed result without repeated confirmation.
---

# Upgrade environment preparation

Use before a full framework upgrade, or when a domain upgrade needs environment
preparation. Maintaining skills/tooling does not itself start this workflow.
Shared source and validation rules remain in [common-sense](../common-sense/SKILL.md).
Read [preparation.md](references/preparation.md) for the plan schema, commands and
installation procedure, and [inventory.md](references/inventory.md) for independent
SDK/GPU/package/source collection and diagnostic provenance.

## Gather all decisions before work

Require an explicit upstream branch/tag, resolve its full commit and public version,
and identify the previous release for the review interval. If these depend on the user
supplying the target in the initial bundled answer, resolve them read-only afterward
under that approved selection and freeze them before installation. Do not infer a target from
the installed package or silently follow a moving branch.

Discover Python through the actual interpreter and `pyvenv.cfg`. Prefer a detected
default uv environment. If none is found, include these conditional choices in the
single confirmation list: an undetected uv environment and its path; otherwise create
project `.venv`; otherwise use the current global environment (usually Conda).
New uv environments use the approved image base Python and `--system-site-packages`
after checking ABI and package identities. Never mix a different Conda environment's
private packages with the selected uv environment.

Inventory installed vLLM/MetaX and ask whether to reuse the environment. Reuse retains
Python/image dependencies but conditionally reinstalls framework packages that do not
match the target/source. Declining reuse uninstalls and reinstalls both framework
packages in the selected environment. MetaX must always be editable from the project
owning this skill. Package presence, version equality or cwd shadowing alone is insufficient.

Include a mandatory question in the initial list: **does this upgrade depend on mcoplib?**
Record `kernel_provider: mcoplib` or `source`; package presence is not a selection.
mcoplib uses the locked environment installation and explicitly sets
`USE_PRECOMPILED_KERNEL=1` for build/runtime. Source mode explicitly sets it to `0`
for both, requires rebuilding MetaX and prohibits runtime fallback to mcoplib, including
Python/Triton helpers. Keep an installed but unused mcoplib unchanged. Both choices
require **all csrc sources to be fully upgraded**, using
[csrc-upgrade](../csrc-upgrade/SKILL.md); include isolated local-source component checks
in the plan even when production uses mcoplib.

Inspect `pre-commit` availability and the configured hooks' runtime/dependency needs
before the initial approval. Include any needed lint-tool preparation under the approved
dependency bounds, keeping image/private packages locked. The final upgrade must pass
`pre-commit run --all-files`; missing lint prerequisites must not first surface only
when the long adaptation run is otherwise complete.

Ask for an existing upstream checkout or approve a shallow clone from
`https://github.com/vllm-project/vllm.git`. MetaX defaults to this project on `master`.
Preserve dirty files, index and local commits; select a clean upstream alternative if
switching would conflict. Include branch-switch/clone paths and conditional reinstalls
in the plan. Capture source baselines before persistent preparation edits.

**Present all pending decisions and known gaps together, and obtain the user's
explicit approval of the complete preparation plan once before mutations.** Include
target ref/commit/version and previous release, Python/ABI/base, both checkouts and Git
actions, installed-package decisions, kernel provider/build/runtime flag, full csrc
coverage, image locks, ordinary dependency changes, temporary
file restoration, chip/resource scope and recovery limits. Save the actual reply against
the immutable resolved plan digest. If the reply fills a conditional choice (such as an
undetected uv path or target ref), preserve the displayed proposal and actual reply, then
materialize only those explicitly authorized choices without another confirmation. Never
claim the user saw a resolved digest they were not shown. Existing explicit session
answers count; do not ask again.
Missing required answers keep preparation pending; silence is not approval.

## Execute the approved preparation

Create/select the environment and prepare checkouts as approved. Initial clone is
restricted to the selected ref; try to retain the previous-release-to-target history
during initial preparation, using bounded shallow fetch/deepening when needed. Do not
fetch all branches or the entire history.

Synchronize only `requirements/common.txt` byte-for-byte from target upstream and the
fixed public base version in `setup.py`; preserve MetaX SCM/MACA/Torch suffix logic.
Keep other requirements unchanged in the final source diff.

Install target vLLM with `VLLM_TARGET_DEVICE=empty`, no build isolation and no dependencies.
For MetaX, source `env.sh`, run `use_existing_metax.py`, then install editable without
build isolation. Run the cleaner inside the recoverable temporary-file wrapper: restore
all affected requirements and `pyproject.toml` after success/failure, including pre-existing
dirty contents. Never restore with Git checkout/reset. Recover interrupted cleanup first.

Bind every installation to the selected interpreter. Use `python -m pip` when available,
otherwise `uv pip --python <selected executable>`. Permit ordinary common/build dependency
changes under exact locks for installed Torch, Triton and **all** image/private requirements,
including packages without `+metax`/`+maca` suffixes. No private-package replacement,
parent-environment uninstall or dependency conflict workaround is implicitly authorized.

## Verify before adaptation

Collect fresh facts and automatically verify the prepared result against the approved
plan. Keep the user's plan authorization separate from the machine's final environment
verification. Record actual installed provenance, package/private/native identities,
source correspondence and final environment fingerprint. Do not ask for a second snapshot
confirmation or invent a new user reply after installation.

**Both frameworks must be successfully installed and verified before compatibility
decisions, adaptation edits or adaptation tests begin.** An old plugin that cannot install
on target vLLM is a preparation blocker; do not start source adaptation to repair it.
Environment diagnostics and already-approved installation retries may continue. Never
construct/load a complete model as an installation check; all model execution still
waits for the coordinator's final phase.

Use the same plan across specialists/resume. Recollect facts on every request. Unchanged
verified identity retains approval; drift requires automatic checking against the same
plan, not another user prompt. Out-of-plan changes block dependent work and are recorded
with logs/next actions. Do not broaden authorization or repeatedly ask during the long run;
continue only independent work whose gates remain satisfied. Report an unresolved
preparation blocker instead of bypassing it.
