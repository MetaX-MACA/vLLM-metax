# Mandatory Environment and Source Preflight

Complete this preflight at the start of a patch upgrade. The goal is to establish
which source is being reviewed, which copy will be edited, and which code tests will
actually execute. A local source tree and its installed wheel are separate artifacts.

## Discover explicit inputs

1. Identify the requested Python executable. Inspect `sys.executable`, `sys.prefix`,
   `sys.base_prefix`, and `pyvenv.cfg` through that executable. `VIRTUAL_ENV` and
   `CONDA_PREFIX` are hints, not authority. A uv venv may legitimately use a Conda
   base interpreter; resolving the Python symlink does not determine the active venv.
2. Locate the local vLLM and vllm_metax repositories using user-provided paths or the
   workspace. Confirm the expected package directories and Git roots. Do not assume
   a sibling directory or the current working directory is the intended checkout.
3. Choose the working directory used for the intended Python test or launch command.
   Preserve the user's import-path configuration while inspecting it.

Run the probe with the selected Python. The paths below are examples, not defaults:

```bash
/opt/venv/bin/python .codex/skills/vllm-metax-patch-upgrade/scripts/probe_environment.py \
  --vllm-source /workspace/vllm \
  --metax-source /workspace/vLLM-metax \
  --runtime-cwd /workspace/vLLM-metax \
  > /tmp/metax-patch-environment.json
```

The probe uses only the standard library and does not import vLLM or MetaX, load GPU
libraries, modify packages, install dependencies, or change either repository. It
prints metadata, Git state, top-level import resolution, and Python source hashes.
Exit status 0 means the report was collected, not that every source matches.

For import resolution it models `python -c`/`python -m` with the specified working
directory first on `sys.path`. This is not proof of resolution in a console script,
pytest with another import mode, a Ray worker, or an engine subprocess. Confirm the
effective paths inside the actual validation process once imports are available.
Other `sys.path` entries come from probe startup. If `PYTHONPATH` contains relative
paths or startup customization depends on cwd, launch the probe from the actual
runtime directory, using the script's absolute path, rather than relying on cwd modeling.

## Inspect the correspondence table

Record one row per package plus an interpreter row:

| Item | Required evidence |
| --- | --- |
| Python/venv | Executable, prefix/base prefix, version, venv configuration, cwd and relevant path overrides. |
| Local vLLM | Repository and package paths, commit, dirty/untracked state, source manifest fingerprint. |
| Installed vLLM | Distribution version, metadata location, installed package directory, installer, editable status and installation provenance where available. |
| Local vllm_metax | Repository and package paths, commit, dirty/untracked state, source manifest fingerprint. |
| Installed vllm_metax | Distribution version, metadata location, installed package directory, installer, editable status and installation provenance where available. |
| Effective imports | Resolved origins/search paths for both packages under the intended invocation, checked again in the actual process. |
| Source correspondence | Matching, changed, missing, and install-only Python files; generated version files reported separately; explicit explanation for intentional differences. |

Inspect all distributions found, not just the first metadata match. Multiple versions,
user-site packages, `.pth` files, editable finders, `PYTHONPATH`, working-directory
shadowing, and symlinks can cause metadata and imports to describe different copies.
The report's distribution candidates can include local `.egg-info` as well as installed
`.dist-info`; matching a local egg-info tree is not evidence that the wheel matches it.
Do not dump unrestricted environment variables or credential-bearing provenance URLs.

`direct_url.json` can point to the directory used to build a non-editable wheel. That
does not mean runtime imports use that directory or that the installed snapshot still
matches it. An editable install can have metadata in site-packages and imports in the
checkout, with no package tree next to its metadata. Verify that mapping explicitly.
An absent `direct_url.json` or missing historical wheel filename is not itself a failure;
installed paths and content evidence are more useful than reconstructing an archive name.

## Confirm the environment with the user

After collecting and interpreting the report, proactively ask the user to confirm
the environment, even when the detected paths appear consistent. Present a compact
table with the five locations, package versions and checkout revisions, effective
import origins, and any material source/wheel differences or unresolved fields.
State which vLLM copy is the comparison target, which MetaX checkout will be edited,
and which copies validation will import under the proposed interpreter and cwd.
Label proposed import-path adjustments as planned, not already verified.

Ask one bundled question in the user's language, for example:
"Are these the intended environment and source versions for this patch upgrade?
Confirm the proposed setup, or specify which paths or target revision to change."
Use an available user-input tool that supports this clarification; otherwise ask
in the final response. Make the question self-contained by including the detected
paths and chosen target, rather than relying only on a collapsed progress message.
Explain that this skill's environment-confirmation step prevents reviewing one copy
while editing or testing another; link this skill's `SKILL.md` and quote its relevant
confirmation instruction when requesting the confirmation.

Until the user responds, continue only independent read-only work such as listing
patches and reading repository requirements. Wait before compatibility decisions,
patch edits, or runtime validation. Silence, elapsed time, a preselected option, and
probe exit status 0 are not confirmation. This confirms the intended environment;
it does not authorize package installation or other changes outside the task.

If the user corrects a path or target, re-run the relevant detection and present the
updated plan for confirmation. Do not ask again when the user already explicitly
confirmed the same mapping in this session. Recheck after environment changes and
ask again only if the confirmed interpreter, source target, or runtime mapping changes
or becomes uncertain; expected edits within the confirmed checkout do not invalidate
confirmation. Follow an explicit user instruction to skip this question, while still
performing the technical checks and reporting discrepancies.

## Interpret mismatches before adapting patches

- **Upstream checkout versus installed vLLM differs:** identify whether the user wants
  the installed revision or the checkout revision. Compare the exact patched modules,
  helpers, and callers in that target. Do not silently read one and test the other.
- **Local MetaX changes versus an older installed wheel:** this may be intentional.
  Record that tests must import the edited checkout. If they load the wheel instead,
  the test does not validate the changes. Use an explicit process-local source path
  when appropriate and recheck origins; installing a wheel is not required just to
  run source-level tests.
- **Source import shadows installed metadata:** list both. Local `_version.py` and
  installed distribution metadata may describe different builds. A matching package
  `__init__.py` does not establish that all patched modules match.
- **Missing source, distribution, or unresolved editable mapping:** report unknowns.
  Continue independent inventory work, but do not assert compatibility against that
  artifact. Include unresolved fields in the environment confirmation and ask for
  missing inputs when available evidence cannot establish intent.

The probe compares Python trees and separately lists installed extension libraries.
It does not establish binary compatibility. When local Python uses extensions from
another build, record the extension origin, Torch/MACA ABI evidence, and appropriate
runtime checks. Do not label mixed source/binary execution as a fully matched build.

Keep the resulting table, the chosen comparison/runtime plan, and the user's
confirmation or explicit instruction to skip confirmation in the audit record.
Re-run the preflight after path, interpreter, checkout, or installation changes.
