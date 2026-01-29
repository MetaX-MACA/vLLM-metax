# SPDX-License-Identifier: Apache-2.0

import vllm

from vllm_metax.utils.mccl import find_mccl_library
from vllm_metax.utils import import_pymxsml

import vllm.utils.nccl

vllm.utils.nccl.find_nccl_library = find_mccl_library

import vllm.utils.import_utils

vllm.utils.import_utils.import_pynvml = import_pymxsml
