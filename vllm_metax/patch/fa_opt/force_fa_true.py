# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# vllm_metax/patch/force_fa_true.py

import vllm.attention.layer as attn


def always_true_check(dtype):
    return True


attn.check_upstream_fa_availability = always_true_check
