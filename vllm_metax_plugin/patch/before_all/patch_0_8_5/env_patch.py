# SPDX-License-Identifier: Apache-2.0

import vllm
from vllm.logger import init_logger
from typing import TYPE_CHECKING, Any, Callable
import os

from vllm.envs import environment_variables # modify point

logger = init_logger(__name__)

if TYPE_CHECKING:
    MACA_VLLM_USE_TN_2_NN: bool = True
    MACA_VLLM_PG_OPT: bool = True

metax_env: dict[str, Callable[[], Any]] = {
    # if set, enable loading weight by transpose
    "MACA_VLLM_USE_TN_2_NN":
    lambda: os.environ.get("MACA_VLLM_USE_TN_2_NN", "1") == "1",
    
    # if set, enable page attention backend
    "MACA_VLLM_PG_OPT":
    lambda: os.environ.get("MACA_VLLM_PG_OPT", "0") == "1",

    # Metax Modification (override the default value of VLLM_USE_FLASHINFER_SAMPLER)
    # FIXME: some models hang, change VLLM_USE_FLASHINFER_SAMPLER default to False, V1 does not use flashinfo top-k top-p sampling.
    "VLLM_USE_FLASHINFER_SAMPLER":
    lambda: bool(int(os.environ["VLLM_USE_FLASHINFER_SAMPLER"]))
    if "VLLM_USE_FLASHINFER_SAMPLER" in os.environ else False,
}

vllm.envs.environment_variables = environment_variables | metax_env