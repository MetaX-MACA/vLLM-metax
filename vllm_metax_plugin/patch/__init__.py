# SPDX-License-Identifier: Apache-2.0
"""
Top-level initializer for the entire patch system.

Responsibilities:
1. Dynamically discover and import all category-level patches under patch/ 
   (e.g. attention, worker, models).
2. For each category, load version-specific patches (hard-coded to patch_0_8_5).
3. Install the RedirectFinder into sys.meta_path to intercept all future imports.
4. Retroactively apply module redirects, function/method/variable patches,
   and exec-hooks to any modules already imported.
"""

import pkgutil
import sys
import importlib
import pprint


from .hooks import install_hooks, uninstall_hooks, apply_all_patches

from . import before_all
from . import distributed
from . import device_allocator
from . import attention
from . import model_executor
from . import worker
from . import engine
from . import after_all

from .hooks import install_hooks

# 公共API
__all__ = [
    'redirect_module',
    'register_patch',
    'install_hooks',
    'uninstall_hooks',
    'apply_all_patches'
]

install_hooks()
