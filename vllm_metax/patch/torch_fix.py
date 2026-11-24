# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import torch

if torch.distributed.all_gather_single is not None:
    torch.distributed.all_gather_into_tensor = torch.distributed.all_gather_single
