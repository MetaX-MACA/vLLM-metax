# Independent branch-port environment

Use this workflow after the read-only audit and identity approval. It prepares the exact
target MetaX worktree and a matching upstream vLLM checkout without invoking the
full-upgrade coordinator.

## Required bindings

Before installation, freeze:

```text
metax_worktree
metax_branch
metax_target_sha
metax_public_version
vllm_checkout
vllm_ref
vllm_commit
vllm_public_version
vllm_previous_release
selected_python
host_identity
private_package_locks
kernel_provider
```

Do not infer `vllm_ref` or `vllm_commit` from the MetaX branch name. Read the mapping from
the approved matrix edge.

## Version correspondence gate

The target MetaX public version and target vLLM public version must be equal:

```text
metax_public_version == vllm_public_version
```

Allow build-local suffixes after the public version only where the version scheme defines
them. For example, a MetaX development build may be
`0.32.0+maca<version>.torch<version>` while vLLM is `0.32.0`; this passes the public
base-version check. A MetaX `0.30.0` worktree paired with vLLM `0.32.0` does not pass.

Resolve the expected version before installation and verify the actual installed
distribution versions after both installs. Also verify that the target MetaX
`requirements/common.txt` matches the selected vLLM target. If either check fails, mark
the edge `BLOCKED`; do not auto-synchronize source files during branch-port preparation.

## Target worktree

Create the worktree from the approved, re-fetched target SHA before creating the
environment plan. Install and import `vllm_metax` from this worktree. Do not install an
editable package from the repository's master checkout as a substitute.

Keep the worktree free of installation-only source edits. If a pre-existing installed
`vllm_metax` belongs to the selected environment but another source path, remove that
environment-local distribution before the new editable install. Refuse to uninstall an
inherited package from a parent/image environment.

## Preflight

1. Collect a fresh inventory of the selected interpreter, SDK/GPU, host identity,
   installed packages and private native files.
2. Verify the installed `torch`, `triton`, `mcoplib` choice and every locked private
   package against the approved plan.
3. Before cleaning requirements, verify installed `torch` matches the Metax torch
   requirement in the unsanitized `requirements/build.txt` and `pyproject.toml`.
4. Resolve all `requirements/*.txt` include files and compare MetaX build requirements
   with the selected vLLM checkout's build requirements. The branch-port agent must
   install their required public union; it must not assume one file is always a superset.
5. Confirm build tools are available or obtain approval to install them under private
   package constraints. Record any unavailable compiler/CMake dependency as a blocker.

## Temporary requirement transaction

Journal the exact bytes and modes of every `requirements/*.txt` and `pyproject.toml`
before invoking the cleaner. Preserve pre-existing user edits. A journal left in
`temporary` state blocks any new installation attempt until recovery.

Run the repository cleaner to remove private/Metax dependency entries, including entries
for `+metax`, `+maca`, `mcpy` and `-r maca_private.txt`. Record the cleaned diff and verify
that public dependencies needed for build and runtime remain visible.

The cleaned `requirements/build.txt` no longer proves the Metax torch version because its
torch line is removed. The preflight verification from the unsanitized file is therefore
mandatory.

Restore the journals after MetaX installation even when cleaning or installation fails.
The target worktree must have no installation-only requirement or `pyproject.toml` diff
after recovery.

## Installation order

Use `<selected-python> -m pip` when pip is available. Otherwise use
`uv pip --python <selected-python>` with equivalent arguments. Apply private constraints
through `PIP_CONSTRAINT`/`UV_CONSTRAINT` and explicit `--constraint` where applicable.

### 1. Public build dependencies

Within the temporary requirement transaction, after private dependencies are removed:

```bash
<selected-python> -m pip install -r requirements/build.txt \
  --constraint <private-constraints.txt>
```

Add missing public build requirements from the selected vLLM target. Do not install any
private Metax wheel as part of this step.

### 2. MetaX editable package

```bash
cd <metax_worktree>
VLLM_PLUGINS="" \
USE_PRECOMPILED_KERNEL=<0-or-1> \
<selected-python> -m pip install -v -e . \
  --no-build-isolation \
  --constraint <private-constraints.txt>
```

Do not set `VLLM_TARGET_DEVICE` for the MetaX build. Private dependencies remain supplied
by the locked image/base environment. Public runtime dependencies may resolve under the
approved constraints; do not hide a missing dependency with `--no-deps`.

Restore all temporary files immediately after this command finishes, then verify the
target worktree has no installation-only diff.

### 3. Matching upstream vLLM

```bash
cd <vllm_checkout>
VLLM_TARGET_DEVICE=empty \
VLLM_PLUGINS="" \
USE_PRECOMPILED_KERNEL=<0-or-1> \
<selected-python> -m pip install -v . \
  --no-build-isolation \
  --no-deps
```

The selected target source and caches must be clean before installation. vLLM may be
installed from source rather than editable; its actual package tree must still match the
selected source except for generated version files.

## Final verification

Verify all of the following from a fresh process using the selected interpreter:

- both framework distributions resolve to the approved sources;
- installed public versions both equal the frozen target version;
- `vllm_metax` is editable from the target worktree;
- vLLM matches the selected upstream checkout;
- common requirements and build/runtime dependencies are complete;
- private package versions and native identities are unchanged;
- the approved kernel provider matches both build and runtime;
- temporary requirement files are restored;
- normal vLLM plugin discovery can load `vllm_metax` after bootstrap-only
  `VLLM_PLUGINS=""` is removed;
- the final environment fingerprint and matrix version binding are recorded.

If any step fails, keep the edge `BLOCKED`. Do not repair build or compatibility source
outside the approved commit set merely to make installation pass.
