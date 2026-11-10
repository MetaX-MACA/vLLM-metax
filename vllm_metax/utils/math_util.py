# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.


def is_power_of_2(n: int) -> bool:
    """Return whether n is a positive power of two, including 1."""
    return n > 0 and (n & (n - 1)) == 0
