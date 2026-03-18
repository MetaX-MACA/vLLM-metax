# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# -------------------------------------------------------
# Note: This patch is for compatibility on Metax platform,
#       replaced some libraries' interface with maca's
# -------------------------------------------------------

from vllm.engine.arg_utils import EngineArgs
from dataclasses import MISSING, dataclass, fields, is_dataclass
from vllm.utils.argparse_utils import FlexibleArgumentParser

_original_add_cli_args = EngineArgs.add_cli_args

@dataclass
class MACAEngineArgs(EngineArgs):
    @staticmethod
    def add_cli_args(parser: FlexibleArgumentParser) -> FlexibleArgumentParser:
        parser = _original_add_cli_args(parser)
        # set to False, disable async scheduling 
        parser.set_defaults(async_scheduling=False)
        return parser

EngineArgs.add_cli_args = MACAEngineArgs.add_cli_args
