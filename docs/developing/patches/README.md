# Patches
---

## What's it for?

Monkey patching is a technique used to modify or extend the behavior of code at runtime, which is ***not stable and not reliable***. 

This directory contains patches for the vllm-metax codebase. Each patch is designed to fix a specific issue:

- vllm upstream bugs, 
- add new features, 
- optimization or improve performance

without having to modify the vllm source code directly.

## Principles:

Patches are **not recommended** for general use and it was out of necessity. Before applying a patch, please make sure that you have a clear understanding of the issue it addressed and the changes it introduces. Always try to figure out if there is already a way that vllm offers to achieve the same functionality without patching. 

E.g., the registry system for `CustomOp` and `PluggableLayer`:

- `customized/**`

or the registry system for `AttentionBackend`:
- `v1/attention/backends/**` (impl)
- `platform.py::register_attention_backends()` (registry)

or modify the *serve CLI args* or *model configuration*:

- `current_platform.check_and_update_config()`
- `current_platform.pre_register_and_update()`

could all be done without patching. If you are not sure, please ask for help in the vllm-metax team.

## Code Structure

### `bugfix/`: 
This subdirectory contains patches that address specific bugs in the codebase.

### `plugin_enhancement/`: 
This subdirectory contains patches that enhance the functionality of plugins.

### `performance/`: 
This subdirectory contains patches that optimize the performance of the codebase. 

### `torch_fix/`:
torch fix needs to be manually applied in `vllm_metax/__init__.py` to fix specific PyTorch 
issues with `torch+metax`, and it's must be applied ahead of all other patches.

## Requirements for Patches

When submitting a patch, please provide the following information at the beginning of your patch file:
1. A clear description of the issue being addressed.
2. Affected versions.
3. When could the patch be removed (e.g., after the next vllm release, or after a specific bug is fixed upstream).

```
# -----------------------------------------------
# Note: The reason for this patch request. Make 
#       it as clear and concise as possible.
#
# Affected versions: List the versions of vllm 
#               that are affected by this issue.
#
# Remove at: Specify when this patch can be removed 
#           (e.g., after the next vllm release, or 
#             after a specific bug is fixed upstream). 
# -----------------------------------------------
```

***significant***: you must make eye-catching comment to highlight the part of your modification.

```
def vllm_func():
    ... original code ...

def patch_func():
    ... original code ...
    # /-------------------- Metax Modification ---------------------\
    ... patch modification ...
    # \-------------------------------------------------------------/
    ... original code ...

module.vllm_func = patch_func
```