# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# -------------------------------------------------------
# Note: This patch is for compatibility on Metax platform,
#       replaced some libraries' interface with maca's
# -------------------------------------------------------

r"""Benchmark online serving throughput.

On the server side, run one of the following commands
to launch the vLLM OpenAI API server:
    vllm serve <your_model> <engine arguments>

On the client side, run:
    vllm bench serve \
        --backend <backend or endpoint type. Default 'openai'> \
        --label <benchmark result label. Default using backend> \
        --model <your_model. Optional, defaults to first model from server> \
        --dataset-name <dataset_name. Default 'random'> \
        --input-len <general input length. Optional, maps to dataset-specific args> \
        --output-len <general output length. Optional, maps to dataset-specific args> \
        --request-rate <request_rate. Default inf> \
        --num-prompts <num_prompts. Default 1000>
"""

import argparse
import asyncio
import contextlib
import importlib.util
import json
import os
import random
import shutil
import time
import uuid
import warnings
from collections.abc import AsyncGenerator, Iterable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Literal

import aiohttp
import numpy as np
from tqdm.asyncio import tqdm

from vllm.benchmarks.datasets import SampleRequest, add_dataset_parser, get_samples

from vllm.benchmarks.lib.endpoint_request_func import (
    ASYNC_REQUEST_FUNCS,
    OPENAI_COMPATIBLE_BACKENDS,
    RequestFuncInput,
    RequestFuncOutput,
)

def maca_add_cli_args(parser: argparse.ArgumentParser):
    add_dataset_parser(parser)
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="The label (prefix) of the benchmark results. If not specified, "
        "the value of '--backend' will be used as the label.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="openai",
        choices=list(ASYNC_REQUEST_FUNCS.keys()),
        help="The type of backend or endpoint to use for the benchmark.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Server or API base url if not using http host and port.",
    )
    # Use 127.0.0.1 here instead of localhost to force the use of ipv4
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--endpoint",
        type=str,
        default="/v1/completions",
        help="API endpoint.",
    )
    parser.add_argument(
        "--header",
        metavar="KEY=VALUE",
        nargs="*",
        help="Key-value pairs (e.g, --header x-additional-info=0.3.3) "
        "for headers to be passed with each request. These headers override "
        "per backend constants and values set via environment variable, and "
        "will be overridden by other arguments (such as request ids).",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help="Maximum number of concurrent requests. This can be used "
        "to help simulate an environment where a higher level component "
        "is enforcing a maximum number of concurrent requests. While the "
        "--request-rate argument controls the rate at which requests are "
        "initiated, this argument will control how many are actually allowed "
        "to execute at a time. This means that when used in combination, the "
        "actual request rate may be lower than specified with --request-rate, "
        "if the server is not processing requests fast enough to keep up.",
    )

    parser.add_argument(
        "--model",
        type=str,
        required=False,
        default=None,
        help="Name of the model. If not specified, will fetch the first model "
        "from the server's /v1/models endpoint.",
    )
    parser.add_argument(
        "--input-len",
        type=int,
        default=None,
        help="General input length for datasets. Maps to dataset-specific "
        "input length arguments (e.g., --random-input-len, --sonnet-input-len). "
        "If not specified, uses dataset defaults.",
    )
    parser.add_argument(
        "--output-len",
        type=int,
        default=None,
        help="General output length for datasets. Maps to dataset-specific "
        "output length arguments (e.g., --random-output-len, --sonnet-output-len). "
        "If not specified, uses dataset defaults.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help="Name or path of the tokenizer, if not using the default tokenizer.",  # noqa: E501
    )
    parser.add_argument(
        "--tokenizer-mode",
        type=str,
        default="auto",
        help="""Tokenizer mode:\n
        - "auto" will use the tokenizer from `mistral_common` for Mistral models
        if available, otherwise it will use the "hf" tokenizer.\n
        - "hf" will use the fast tokenizer if available.\n
        - "slow" will always use the slow tokenizer.\n
        - "mistral" will always use the tokenizer from `mistral_common`.\n
        - "deepseek_v32" will always use the tokenizer from `deepseek_v32`.\n
        - Other custom values can be supported via plugins.""",
    )
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument(
        "--logprobs",
        type=int,
        default=None,
        help=(
            "Number of logprobs-per-token to compute & return as part of "
            "the request. If unspecified, then either (1) if beam search "
            "is disabled, no logprobs are computed & a single dummy "
            "logprob is returned for each token; or (2) if beam search "
            "is enabled 1 logprob per token is computed"
        ),
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Number of requests per second. If this is inf, "
        "then all the requests are sent at time 0. "
        "Otherwise, we use Poisson process or gamma distribution "
        "to synthesize the request arrival times.",
    )
    parser.add_argument(
        "--burstiness",
        type=float,
        default=1.0,
        help="Burstiness factor of the request generation. "
        "Only take effect when request_rate is not inf. "
        "Default value is 1, which follows Poisson process. "
        "Otherwise, the request intervals follow a gamma distribution. "
        "A lower burstiness value (0 < burstiness < 1) results in more "
        "bursty requests. A higher burstiness value (burstiness > 1) "
        "results in a more uniform arrival of requests.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code from huggingface",
    )
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="Specify to disable tqdm progress bar.",
    )
    parser.add_argument(
        "--num-warmups",
        type=int,
        default=0,
        help="Number of warmup requests.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Use vLLM Profiling. --profiler-config must be provided on the server.",
    )
    parser.add_argument(
        "--save-result",
        action="store_true",
        help="Specify to save benchmark results to a json file",
    )
    parser.add_argument(
        "--save-detailed",
        action="store_true",
        help="When saving the results, whether to include per request "
        "information such as response, error, ttfts, tpots, etc.",
    )
    parser.add_argument(
        "--append-result",
        action="store_true",
        help="Append the benchmark result to the existing json file.",
    )
    parser.add_argument(
        "--metadata",
        metavar="KEY=VALUE",
        nargs="*",
        help="Key-value pairs (e.g, --metadata version=0.3.3 tp=1) "
        "for metadata of this run to be saved in the result JSON file "
        "for record keeping purposes.",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default=None,
        help="Specify directory to save benchmark json results."
        "If not specified, results are saved in the current directory.",
    )
    parser.add_argument(
        "--result-filename",
        type=str,
        default=None,
        help="Specify the filename to save benchmark json results."
        "If not specified, results will be saved in "
        "{label}-{args.request_rate}qps-{base_model_id}-{current_dt}.json"  # noqa
        " format.",
    )
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="Set ignore_eos flag when sending the benchmark request."
        "Warning: ignore_eos is not supported in deepspeed_mii and tgi.",
    )
    parser.add_argument(
        "--percentile-metrics",
        type=str,
        default=None,
        help="Comma-separated list of selected metrics to report percentiles. "
        "This argument specifies the metrics to report percentiles. "
        'Allowed metric names are "ttft", "tpot", "itl", "e2el". '
        'If not specified, defaults to "ttft,tpot,itl" for generative models '
        'and "e2el" for pooling models.',
    )
    parser.add_argument(
        "--metric-percentiles",
        type=str,
        default="99",
        help="Comma-separated list of percentiles for selected metrics. "
        'To report 25-th, 50-th, and 75-th percentiles, use "25,50,75". '
        'Default value is "99".'
        'Use "--percentile-metrics" to select metrics.',
    )
    parser.add_argument(
        "--goodput",
        nargs="+",
        required=False,
        help='Specify service level objectives for goodput as "KEY:VALUE" '
        "pairs, where the key is a metric name, and the value is in "
        'milliseconds. Multiple "KEY:VALUE" pairs can be provided, '
        "separated by spaces. Allowed request level metric names are "
        '"ttft", "tpot", "e2el". For more context on the definition of '
        "goodput, refer to DistServe paper: https://arxiv.org/pdf/2401.09670 "
        "and the blog: https://hao-ai-lab.github.io/blogs/distserve",
    )
    parser.add_argument(
        "--request-id-prefix",
        type=str,
        required=False,
        default=f"bench-{uuid.uuid4().hex[:8]}-",
        help="Specify the prefix of request id.",
    )

    sampling_group = parser.add_argument_group("sampling parameters")
    sampling_group.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Top-p sampling parameter. Only has effect on openai-compatible backends.",
    )
    sampling_group.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k sampling parameter. Only has effect on openai-compatible backends.",
    )
    sampling_group.add_argument(
        "--min-p",
        type=float,
        default=None,
        help="Min-p sampling parameter. Only has effect on openai-compatible backends.",
    )

    # /------------------------  Metax Modification -------------------------\
    sampling_group.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature sampling parameter. Only has effect on "
        "openai-compatible backends.",
    )
    # \------------------------- Metax Modification -------------------------/

    sampling_group.add_argument(
        "--frequency-penalty",
        type=float,
        default=None,
        help="Frequency penalty sampling parameter. Only has effect on "
        "openai-compatible backends.",
    )
    sampling_group.add_argument(
        "--presence-penalty",
        type=float,
        default=None,
        help="Presence penalty sampling parameter. Only has effect on "
        "openai-compatible backends.",
    )
    sampling_group.add_argument(
        "--repetition-penalty",
        type=float,
        default=None,
        help="Repetition penalty sampling parameter. Only has effect on "
        "openai-compatible backends.",
    )

    parser.add_argument(
        "--served-model-name",
        type=str,
        default=None,
        help="The model name used in the API. "
        "If not specified, the model name will be the "
        "same as the `--model` argument. ",
    )

    parser.add_argument(
        "--lora-modules",
        nargs="+",
        default=None,
        help="A subset of LoRA module names passed in when "
        "launching the server. For each request, the "
        "script chooses a LoRA module at random.",
    )

    parser.add_argument(
        "--ramp-up-strategy",
        type=str,
        default=None,
        choices=["linear", "exponential"],
        help="The ramp-up strategy. This would be used to "
        "ramp up the request rate from initial RPS to final "
        "RPS rate (specified by --ramp-up-start-rps and "
        "--ramp-up-end-rps.) over the duration of the benchmark.",
    )
    parser.add_argument(
        "--ramp-up-start-rps",
        type=int,
        default=None,
        help="The starting request rate for ramp-up (RPS). "
        "Needs to be specified when --ramp-up-strategy is used.",
    )
    parser.add_argument(
        "--ramp-up-end-rps",
        type=int,
        default=None,
        help="The ending request rate for ramp-up (RPS). "
        "Needs to be specified when --ramp-up-strategy is used.",
    )
    parser.add_argument(
        "--ready-check-timeout-sec",
        type=int,
        default=0,
        help="Maximum time to wait for the endpoint to become ready "
        "in seconds. Ready check will be skipped by default.",
    )

    parser.add_argument(
        "--extra-body",
        help="A JSON string representing extra body parameters to include "
        "in each request."
        'Example: \'{"chat_template_kwargs":{"enable_thinking":false}}\'',
        type=json.loads,
        default=None,
    )


import vllm.entrypoints.cli.benchmark.serve as cli_serve
cli_serve.add_cli_args = maca_add_cli_args
