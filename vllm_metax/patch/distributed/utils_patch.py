# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

import torch
import vllm
from vllm import envs, logger
from vllm.logger import init_logger

from vllm_metax.utils.mccl import find_mccl_library
from vllm_metax.utils import import_pymxsml

vllm.utils.find_nccl_library = find_mccl_library
vllm.utils.import_pynvml = import_pymxsml
