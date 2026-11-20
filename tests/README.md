# Upstream test ports

The baseline is vLLM `v0.33.0`, commit
`4ba787ec9959e566d3b587a796dff07673324da5`. The
[port ledger](upstream_test_ports.toml) records each reviewed upstream test
file, its Git blob, the MetaX implementation it exercises, and coverage gaps
or the reason it does not apply.

## Port a file

1. Run the inventory from the repository root. Pick an upstream file and
   inspect its test functions, parameter cases, fixtures, and helpers at the
   pinned revision.

   ```bash
   python tools/upstream_test_ports.py --upstream ../vllm inventory --prefix tests/kernels
   git -C ../vllm show v0.33.0:tests/kernels/test_concat_mla_q.py
   ```

2. Follow each case to the actual MetaX implementation and its call sites.
   When the plugin overrides a kernel, backend, or custom op, run the plugin
   implementation or its registered dispatch with the upstream oracle where
   applicable. Keep the upstream file path and applicable test case names,
   adapting device handling, data layout, and provider requirements as needed.
3. Record the upstream blob from the inventory, the implementation file paths,
   and the coverage decision under that exact path in the ledger. Use
   `ported` when all applicable cases have coverage, `partial` when applicable
   cases remain, and `not_applicable` when the full file has no equivalent
   execution path. Explain omitted cases in the notes. An unrecorded upstream
   file appears as `pending`.
4. Run the adapted test in the supported MetaX environment, followed by the
   related suite. Use the source kernel provider.

   ```bash
   USE_PRECOMPILED_KERNEL=0 python -m pytest -q tests/kernels/test_concat_mla_q.py
   USE_PRECOMPILED_KERNEL=0 python -m pytest -q tests/kernels
   USE_PRECOMPILED_KERNEL=0 python -m pytest -q tests
   python tools/upstream_test_ports.py --upstream ../vllm check
   ```

The two-rank MCCL test requires two devices. The generation tests build tiny
local Qwen2 and DeepSeek V2 checkpoints and compare against Transformers.

## Update the upstream baseline

Generate the changed-file list against the ledger's pinned revision. For
example, use the following commands when reviewing vLLM `v0.34.0`.

```bash
python tools/upstream_test_ports.py --upstream ../vllm changes v0.34.0
git -C ../vllm diff v0.33.0 v0.34.0 -- tests/kernels/test_top_k_per_row.py
git -C ../vllm ls-tree v0.34.0 -- tests/kernels/test_top_k_per_row.py
```

Each row contains the Git change status, file type, review status, and path.
Renames include both paths. Inspect `review_again` rows in the existing ports,
and review the dependent ports of changed fixtures, data, and helpers marked
`review_dependents`. New and previously unreviewed test files appear as
`pending`. A `--prefix tests/kernels` filter retains changes to ancestor
`conftest.py` files.

Move or remove ledger entries for reviewed renames or deletions, and update
each changed blob after reviewing its cases. Once those entries are reconciled,
set `version` and `revision` to the new baseline, then run `check` and the
affected tests. `check` flags stale blobs, missing plugin tests, and local
tests that share an upstream path without a ledger record.

When a MetaX implementation changes, look up its path in the ledger and
revisit affected case decisions even if the upstream blobs are unchanged.

## Check kernel operator coverage

Start with the actual plugin call sites and follow each operator to its source
binding, Triton or TileLang implementation, or vendor entry point. Add a
`tests/kernels` test with an independent numerical or behavioral reference
for each operator. Keep MetaX-only cases in files dedicated to those
operators when the baseline has no matching test file.

For dispatcher operators, run the coverage reporter in the supported MetaX
environment. It records the source extensions and associates operator names
with the kernel tests that executed them.

```bash
USE_PRECOMPILED_KERNEL=0 python tools/kernel_operator_coverage.py \
  --report /tmp/metax-kernel-operators.json -- -q tests/kernels
```

An `observed` result means the profiler saw the operator execute in a
passing test. Review the linked assertion to establish numerical coverage.
`expected_failure` records executed regressions such as the persistent
Top-K case. `unobserved` marks a registered implementation with no execution
observed in a passing test or expected failure. `schema_only` identifies
schemas without a CPU, GPU or composite implementation, and `not_loaded`
identifies optional Python registrations absent in the selected process.
For Triton and TileLang entry points without a dispatcher registration,
use the corresponding test and implementation paths in the port ledger.
