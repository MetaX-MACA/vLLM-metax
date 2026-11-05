# Preparation plan and installation

## Plan record

The helper is `../scripts/prepare_environment.py` relative to this reference. From the
project root use `.agents/skills/environment-check/scripts/prepare_environment.py`.
All helper actions are explicit; none fetches repositories or changes environments
merely by reading the skill. Run mutation/verification actions with the selected Python.

Create a JSON plan using fresh independent inventory. Required fields:

| Field | Contents |
| --- | --- |
| `schema_version` | `1` |
| `target` | `ref`, resolved full `commit`, public base `version`, `previous_release`, `history: "previous-release-range"` |
| `python` | Exact `executable` (preserve the venv path), `prefix`, `base_prefix`, full `sys.version`, `base_executable`, `selection` (`detected_uv`, `provided_uv`, `create_uv`, `global`), `system_site_packages` |
| `sources` | Absolute `vllm`, `metax`, `runtime_cwd`; `metax_branch: "master"`, frozen `metax_commit` |
| `kernel_provider` | Mandatory `mcoplib` or `source`, explicitly selected in the initial user reply; both require full csrc upgrade. |
| `environment_choice` | `reuse` or `reinstall` |
| `installs` | Boolean `vllm` and `metax`; both true when declining reuse; otherwise true for missing/nonconforming packages. Source mode always requires `metax: true` to establish build provenance. |
| `host_identity` | Exact `sdk`, `gpu`, `system` from the approved inventory's `identity` |
| `private_packages` | Full selected-image package metadata rows returned by `metax_environment.preparation.private_locks(report)` |
| `private_native_identity` | `identity.native_files.distributions` restricted to the locked package names |
| `git_policy` | `preserve-worktree` |
| `ordinary_dependencies` | `allow-with-private-locks` |
| `known_issues` | Disclosed inventory gaps, pin/build conflicts and any unavailable probes |
| `validation_scope` | Nonempty current-chip/resource scope, unsupported quantization branches and model sample scope |

For a new inherited uv environment, derive its intended Python fields from the approved
base interpreter: version/base prefix stay bound, executable/prefix become `.venv` paths.
The inherited package/native rows must still match the actual post-create inventory.
Capture these rows from the base selected for inheritance, not a different active venv.
If required image packages are missing or duplicate, resolve the selection in the initial
confirmation; do not invent package rows or silently choose a private build.

Present a compact self-contained table plus the preparation plan and known gaps. Explain
this skill's one-time approval requirement and link its `SKILL.md`. Ask once
in the user's language. The reply must actually settle the conditional choices; the agent
must not manufacture an approval. The framework's target must be explicitly supplied.
If the answer supplies a previously unknown target ref or hidden uv path, save the exact
displayed conditional proposal and the actual reply, then resolve those authorized
values read-only into the canonical plan. Do not ask again merely to display a newly
computed digest; do not claim the user saw that digest. Values not determined by the
reply or the displayed deterministic rules remain blockers.

```bash
python .agents/skills/environment-check/scripts/prepare_environment.py plan \
  --run-dir .metax-upgrade/release-upgrade --file preparation-plan.json
python .agents/skills/environment-check/scripts/prepare_environment.py approve \
  --run-dir .metax-upgrade/release-upgrade --plan-digest RESOLVED_APPROVED_PLAN_DIGEST \
  --user-response-file actual-user-reply.txt --proposal-file displayed-preparation.txt
```

`plan` writes an immutable content-addressed plan and the review view. `approve` stores
the actual reply and, when conditional choices were resolved from it, the displayed
proposal separately. The agent must verify that every resolved value is authorized
by that proposal/reply; the helper cannot infer the meaning of an arbitrary reply.
`PREPARATION.json` tracks awaiting_approval/preparing/blocked/ready. Never replace a plan or reply to hide a failed setup. A task genuinely
outside the plan is a blocker to report, not permission to ask repeatedly during the run.

## Checkouts, history and uv

Existing checkout: inspect dirty state, current branches/worktrees and local commits
before attempting a safe target-ref switch. Resolve and freeze the exact ref commit.
Use this skill's owning MetaX project, default `master`; do not pull/reset automatically.

Without a checkout, clone only the selected branch/tag from the official remote:

```bash
git clone --depth 1 --single-branch --branch "$TARGET_REF" \
  https://github.com/vllm-project/vllm.git "$UPSTREAM_CHECKOUT"
```

During this initial preparation, try to include the entire previous-release-to-target
interval, rather than deferring history collection until adaptation. When release dates
are reliable, a bounded `--shallow-since` clone may replace depth 1. Otherwise fetch only
the necessary refs and deepen the selected history in bounded increments. Verify that
the previous release is reachable and that the requested review interval has no shallow
boundary gaps. Record the range/commands/SHAs; inability to obtain it remains an upstream
review blocker. Never use an unrestricted full clone, `fetch --all` or `--unshallow`.

When the plan calls for a new uv environment:

```bash
uv venv --python "$IMAGE_PYTHON" --system-site-packages --no-python-downloads .venv
```

Do not overwrite an existing `.venv`. Verify Python ABI, prefixes and inherited image
package/native identities before proceeding. A detected uv environment does not need to
be recreated. A hidden uv path supplied in the initial answers takes priority over an
inferred global Conda environment. Do not activate one environment and invoke another's pip.

Once approved checkouts and Python exist, run the shared `upgrade.py init` before changing
common.txt/setup.py so the baseline archive includes original dirty contents. Keep one
durable run for both preparation and adaptation. `init` may collect an incomplete installed
environment; this does not open the adaptation gate.

## Source synchronization and packages

```bash
"$SELECTED_PYTHON" .agents/skills/environment-check/scripts/prepare_environment.py sync-sources \
  --run-dir .metax-upgrade/release-upgrade
```

This copies only common.txt and changes the AST-identified fixed version string in
`fixed_version_scheme`, retaining suffix generation and other source formatting.
Review any existing changes to those two files against the baseline; their approved
synchronization is an intended upgrade diff for the eventual MetaX commit.

Install missing ordinary common/build dependencies under the generated image-package
constraints before the source installs when needed. Include indirect dependencies in the
resolver check. Never resolve a conflict by upgrading/downgrading Torch, Triton, private
kernels or another image package, or by adding `--no-deps` to the MetaX install to conceal
missing ordinary dependencies. Build isolation stays disabled. Record unavailable build
tools as blockers. Constraints must match installed image versions, not blindly copy
potentially stale pins from `maca_private.txt`.

The helper writes `preparation/private-constraints.txt` during install. The exact lock rows
are already in the approved plan, so the same constraints can be written beforehand for
ordinary dependency preparation. Native identities are checked before each framework
install and again in final verification. No build/installer return code substitutes for
the final private-package comparison.

The initial confirmation must select whether mcoplib is a dependency. In mcoplib mode,
require its installed image package in the private locks; do not install a different
version or switch providers when an API is missing. In source mode mcoplib may remain
installed, but execution must be independent of it. Rebuild even an otherwise matching
MetaX editable installation: version/source equality does not establish how its old native
binaries were built. Old plans missing this choice are incomplete, not implicitly mcoplib.

All csrc files must be fully upgraded with [csrc-upgrade](../../csrc-upgrade/SKILL.md)
under either choice. The initial native install prerequisite still applies: compilation
failure blocks adaptation, rather than authorizing early csrc compatibility edits.

Install target vLLM first, then MetaX:

```bash
"$SELECTED_PYTHON" .agents/skills/environment-check/scripts/prepare_environment.py install \
  --run-dir .metax-upgrade/release-upgrade --package vllm
"$SELECTED_PYTHON" .agents/skills/environment-check/scripts/prepare_environment.py install \
  --run-dir .metax-upgrade/release-upgrade --package metax
```

The helper fixes `USE_PRECOMPILED_KERNEL` to `1` for mcoplib or `0` for source in both
installation processes, and records it with the installation receipt. Final collection
and shared run-check subprocesses use the same approved value. Explicitly propagate it
into servers, Ray/distributed workers and any commands outside the shared runner; inspect
the effective flag and native callable origins inside those actual children.

The selected-interpreter pip/uv equivalents are:

```bash
# In upstream vLLM; this assignment is confined to this install.
VLLM_TARGET_DEVICE=empty "$SELECTED_PYTHON" -m pip install -v . --no-build-isolation --no-deps
# In MetaX, within the helper's temporary-file transaction and private constraints:
source env.sh
# Set KERNEL_FLAG from the approved provider: 1=mcoplib, 0=source.
export USE_PRECOMPILED_KERNEL="$KERNEL_FLAG"
"$SELECTED_PYTHON" use_existing_metax.py
"$SELECTED_PYTHON" -m pip install -v -e . --no-build-isolation
```

The canonical filename is `use_existing_metax.py`. Bootstrap processes disable automatic
plugin activation with process-local `VLLM_PLUGINS=""`; normal-entry validation later
uses the normal plugin configuration. Do not leak `VLLM_TARGET_DEVICE=empty` into the
MetaX build or later tests. If the selected Python has no pip, the helper uses
`uv pip --python <selected executable>` with equivalent options. Reuse skips an install
only after correspondence checks pass. Declining reuse removes both selected-environment
framework distributions before their respective reinstalls; it never removes a parent's
inherited image installation through a child venv. Such a conflict must have an explicit
initial plan resolution or remain blocked.

## Temporary dependency edits and recovery

The wrapper journals every `requirements/*.txt` and `pyproject.toml` before invoking the
cleaner, preserving bytes, mode and pre-existing dirty contents. Persistent common.txt
and version synchronization happens first; restoration retains those approved changes.
Success, cleaner failure and installer failure all restore the saved files in `finally`.
Never use `git checkout`, `reset` or a clean-tree copy for restoration. The user's cleaner
implementation is reused, not rewritten.

After interruption, inspect `preparation/temporary-requirements.json`, preserve any new
concurrent edits before recovery, then run:

```bash
"$SELECTED_PYTHON" .agents/skills/environment-check/scripts/prepare_environment.py recover \
  --run-dir .metax-upgrade/release-upgrade
```

No new install/verification may ignore a journal left in temporary state. Do not edit
these files concurrently with installation. The helper records each installation command,
status and log. Sanitize credential-bearing installer diagnostics before sharing reports.

## Final machine verification

```bash
"$SELECTED_PYTHON" .agents/skills/environment-check/scripts/prepare_environment.py verify \
  --run-dir .metax-upgrade/release-upgrade
"$SELECTED_PYTHON" .agents/skills/common-sense/scripts/metax_upgrade/upgrade.py refresh \
  --run-dir .metax-upgrade/release-upgrade
"$SELECTED_PYTHON" .agents/skills/common-sense/scripts/metax_upgrade/upgrade.py assert-ready \
  --run-dir .metax-upgrade/release-upgrade
```

Verification requires both installations, target version/source correspondence, current
project MetaX editable provenance/effective origin, common/version synchronization,
unchanged Python/host/image identities and recovered temporary files. Required source
installs need successful receipts belonging to this plan. Installed vLLM wheel/source
comparison allows only generated version-file differences; it is not forced editable.
Verification also checks the approved kernel flag and required installation receipts
from that same provider. Production children report the actual flag, loaded mcoplib
modules and callable/extension origins; source-mode mcoplib use fails acceptance.
The final report and fingerprint are machine evidence linked to the original plan approval,
not a second user confirmation. New runs, including component runs, cannot use the legacy `confirm` command to
bypass preparation. Existing full runs also require a preparation plan when resumed. Existing component-only snapshot approvals remain readable for older
limited runs; use this workflow when preparing a new environment.

Missing install, dependency conflict or build failure closes **all adaptation** gates.
Do not fix compatibility source before both packages install successfully; retain logs and
report the blocker. Already-approved setup retries and independent environment diagnostics
can continue. No model construction/load/generation is an installation check.

On resume collect fresh facts. If final identity is unchanged, retain the original plan
authorization and its machine verification. For drift, use `verify` against the same plan
and then refresh; incompatible/out-of-plan changes stay blocked. Do not reinterpret the old
reply as approval of different choices, expand the plan silently, or keep prompting users
during a long adaptation run.
