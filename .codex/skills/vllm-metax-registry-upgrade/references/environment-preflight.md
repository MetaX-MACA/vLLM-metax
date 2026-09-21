# Environment and Source Correspondence

Run `scripts/probe_environment.py` with the selected interpreter. Paths below are
examples from the development environment, not universal defaults:

```bash
/opt/venv/bin/python .codex/skills/vllm-metax-registry-upgrade/scripts/probe_environment.py \
  --vllm-source /workspace/vllm \
  --metax-source /workspace/vLLM-metax \
  --runtime-cwd /workspace/vLLM-metax \
  > /tmp/metax-registry-environment.json
```

The standard-library-only probe collects metadata, Git state, Python source hashes,
import resolution hints and component package records. It does not import GPU packages,
validate binaries, install dependencies, or edit either checkout. Exit 0 means collection
succeeded, not that the environment is compatible.

Review all of these, including differences that a matching version string hides:

| Location | Evidence |
| --- | --- |
| Python/venv | Executable, prefix/base prefix, pyvenv.cfg, Python version and cwd. A uv venv can use a Conda base; a resolved executable symlink does not identify the active venv. |
| Local vLLM | Git root, commit, dirty state, package tree and relevant module contents. |
| Installed vLLM | All matching distributions, versions, metadata/package/extension locations, editable or wheel provenance, changed/missing/install-only Python files. |
| Local vllm_metax | Git root, commit, dirty state and whether the subject is the index or worktree. |
| Installed vllm_metax | Distribution and package locations, version, provenance and correspondence with the intended source. |
| Components | All matching compressed-tensors, DeepGEMM, FlashAttention, FlashMLA, FlashInfer, MCOPLIB, Torch and Triton records, plus selected top-level module origins. |

Version files, wheel names and `direct_url.json` do not prove source identity. A build
path does not make a non-editable installation track subsequent edits. Stale installed
Python modules may mask removed upstream modules. Local egg-info is not an installed
wheel. Record unknown provenance rather than inventing an installation history.

The probe models `python -c/-m` path precedence; console scripts, pytest, workers and
subprocesses may resolve differently. Run from the actual cwd when startup customization
or relative PYTHONPATH matters. Reconfirm package `__file__`, callable module/source,
loaded extension paths and the MetaX plugin origin inside the actual validation process.
A mixed local-Python/installed-binary run is legitimate when intentional, but must be
reported as such rather than called a fully matched build.

Present the detected mapping and ask for confirmation as described in SKILL.md, reusing
an unchanged session confirmation. Resolve ambiguous target choices before dependent
compatibility decisions. Do not silently reinstall packages or change global paths to
make a mismatch disappear. For staged reviews, explicitly inject the exported index
snapshot into the test process and prove it was imported; use the worktree for testing
new edits. Record any process-local path or optional-dependency workaround.

The probe does not inspect API signatures. After establishing the target, inspect actual
callables in a fresh process with the selected Python, for example using
`inspect.signature(make_wna16_moe_kernel)` and
`inspect.getsourcefile(selected_experts_cls)`. Follow wrapper aliases to the selected
raw callable. Capture symbol availability and import failures; never treat them as
successful numerical validation. Do not dump unrestricted environment variables or
credential-bearing installation URLs into reports.
