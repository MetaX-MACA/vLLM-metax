# SPDX-License-Identifier: Apache-2.0
# 2025 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved. 

import sys

import vllm
from vllm.logger import init_logger

from vllm_metax import _custom_ops as _metax_custom_ops

logger = init_logger(__name__)

vllm._custom_ops = _metax_custom_ops
