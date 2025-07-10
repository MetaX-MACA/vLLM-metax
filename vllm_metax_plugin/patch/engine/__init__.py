# SPDX-License-Identifier: Apache-2.0
"""
Category-level attention patches loader.

Responsibilities:
1. Import versioned sub-packages under patch.attention.
"""

# import pkgutil

# for finder, name, ispkg in pkgutil.iter_modules(__path__):
#     __import__(f"{__name__}.{name}")

from . import patch_0_8_5
