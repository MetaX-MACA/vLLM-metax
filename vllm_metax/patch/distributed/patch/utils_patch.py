# SPDX-License-Identifier: Apache-2.0

import vllm
from vllm import envs, logger
from vllm.logger import init_logger
import torch

from vllm_metax.utils import import_pymxml, find_mccl_library


vllm.utils.find_nccl_library = find_mccl_library
vllm.utils.import_pynvml = import_pymxml




