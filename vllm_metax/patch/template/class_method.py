# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#
# -----------------------------------------------------------------------------
# Note: TODO: Explain why this class method needs to be replaced.
#
# Affected versions: TODO: Use an explicit version or version range.
#
# Remove at: TODO: Give a verifiable removal condition or upstream issue.
# -----------------------------------------------------------------------------
"""Templates for instance methods and built-in method descriptors.

Pass the target module and a dotted class attribute path. The class is resolved
inside the patch utility and does not need to be imported here. Keep ``@patch``
outermost.

Static, class, and property variants::

    @patch("vllm.some_module", "TargetClass.create")
    @staticmethod
    def create(config): ...


    @patch("vllm.some_module", "TargetClass.from_config")
    @classmethod
    def from_config(cls, config): ...


    @patch("vllm.some_module", "TargetClass.value")
    @property
    def value(self): ...

The descriptor line may be omitted because ``patch`` preserves the descriptor of
an existing class target. Keeping it is recommended for a clean upstream diff.
"""

from typing import Any

from vllm_metax.patch.utils import patch


@patch("vllm_metax.module", "TargetClass.method")
def method(self, *args: Any, **kwargs: Any) -> Any:
    # TODO: Copy all unchanged upstream logic.

    # /-------------------- MetaX Modification --------------------\
    # TODO: Make the smallest possible MetaX change.
    # \-------------------- MetaX Modification --------------------/
    raise NotImplementedError("Complete the class method patch first")
