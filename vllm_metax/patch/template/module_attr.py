# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#
# -----------------------------------------------------------------------------
# Note: TODO: Explain why this module attribute needs to be replaced.
#
# Affected versions: TODO: Use an explicit version or version range.
#
# Remove at: TODO: Give a verifiable removal condition or upstream issue.
# -----------------------------------------------------------------------------
"""Templates for module functions, classes, Triton kernels, and imported objects.

The target attribute is inferred when it matches the replacement name::

    @patch("vllm.some_module")
    def target_function(...):
        ...

Equivalent to:

    vllm.some_module.target_function = target_function

    -----------------------------------------------------------------------------

Pass it explicitly when the names differ::

    @patch("vllm.some_module", "TargetClass")
    class MetaXTargetClass: ...

Equivalent to:

    vllm.some_module.TargetClass = MetaXTargetClass

    -----------------------------------------------------------------------------

An implementation imported from another module can be installed directly::

    from vllm_metax.some_module import metax_function

    metax_function = patch("vllm.some_module", "target_function")(metax_function)

Equivalent to:

    vllm.some_module.target_function = metax_function

    ------------------------------------------------------------------------------

Stack ``@patch`` decorators if one implementation replaces multiple import
locations. Use ``allow_missing=True`` only when intentionally adding an attribute;
existing targets are required by default to catch misspelled paths.

For a Triton kernel, keep ``@patch`` outermost so it receives and installs the
final JIT/autotune object::

    @patch("vllm.some_module", "target_kernel")
    @triton.heuristics({...})
    @triton.autotune(configs=[...], key=[...])
    @triton.jit
    def target_kernel(...):
        ...

Equivalent to:

    vllm.some_module.target_kernel = target_kernel

Do not put ``@triton.jit`` outside ``@patch``; doing so can install the undecorated
Python function instead of the launchable Triton kernel.

    ------------------------------------------------------------------------------
"""

from typing import Any

from vllm_metax.patch.utils import patch


@patch("vllm_metax.module")
def to_be_replaced_function(*args: Any, **kwargs: Any) -> Any:
    # /-------------------- MetaX Modification --------------------\
    # TODO: Make the smallest possible MetaX change.
    # \-------------------- MetaX Modification --------------------/
    raise NotImplementedError("Complete the module attribute patch first")


# Triton kernel template.
@patch("vllm.some_module", "target_triton_kernel")
# @triton.heuristics({...})
# @triton.autotune(configs=[...], key=[...])
# @triton.jit
def target_triton_kernel():
    raise NotImplementedError("Complete the module attribute patch first")
