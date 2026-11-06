# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Benchmark equivalent windowed policies, not upstream full-history penalties.

Run ``python -m benchmarks.benchmark_windowed_penalties`` from the repository root.
Wall timing includes dispatch and synchronization; reset copies are excluded.
"""

import argparse
from collections import Counter
from datetime import UTC, datetime
from functools import partial
import itertools
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import time

import torch

from vllm_metax.v1.sample.windowed_penalties import (
    SparsePenaltyBatch,
    WindowedPenaltyConfig,
    WindowedPenaltyState,
)


class DensePenaltyBatch:
    """Equivalent dense reference, not vLLM's standard penalty implementation."""

    def __init__(self, size, vocab, config):
        self.shape = (size, vocab)
        self.config = config
        self.counts = torch.empty(self.shape, device="cuda")
        self.seen = torch.empty(self.shape, device="cuda", dtype=torch.bool)

    def update(self, requests):
        host = torch.zeros(self.shape, dtype=torch.float32, pin_memory=True)
        array = host.numpy()
        for row, state in requests.items():
            for token, count in Counter(
                state.output_token_ids[-state.config.window :]
            ).items():
                if 0 <= token < self.shape[1] and state.config.includes(token):
                    array[row, token] = count
        self.counts = host.to("cuda", non_blocking=True)
        self.seen = self.counts > 0

    def apply(self, logits):
        cfg = self.config
        original = logits.float()
        penalized = torch.where(
            original > 0, original / cfg.repetition, original * cfg.repetition
        )
        shift = self.counts * cfg.frequency + cfg.presence
        logits.copy_(
            torch.where(self.seen, penalized - shift, original).to(logits.dtype)
        )
        return logits


def measure(fn, logits, initial, iterations):
    samples = []
    for i in range(iterations + 5):
        logits.copy_(initial)
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        fn()
        torch.cuda.synchronize()
        if i >= 5:
            samples.append((time.perf_counter_ns() - start) / 1000)
    return samples


def update_and_apply(batch, requests, logits, *, torch_reference=False, apply_fn=None):
    batch.update(requests)
    if apply_fn is not None:
        return apply_fn(logits)
    if torch_reference:
        return batch.apply_torch(logits)
    return batch.apply(logits)


def summarize(samples):
    ordered = sorted(samples)
    return {
        "median_us": statistics.median(samples),
        "p95_us": ordered[int(0.95 * (len(ordered) - 1))],
        "min_us": min(samples),
        "samples_us": samples,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32, 128])
    parser.add_argument("--vocab-sizes", type=int, nargs="+", default=[32000, 151936])
    parser.add_argument("--windows", type=int, nargs="+", default=[32, 128])
    parser.add_argument(
        "--dtype", choices=["float32", "float16", "bfloat16"], default="float32"
    )
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if (
        args.iterations < 1
        or args.repeats < 1
        or args.cpu_threads < 1
        or any(n < 1 for n in args.batch_sizes + args.vocab_sizes)
    ):
        parser.error(
            "iterations, repeats, vocab sizes and batch sizes must be positive"
        )
    torch.set_num_threads(args.cpu_threads)
    root = Path(__file__).resolve().parents[1]
    sources = [
        Path(__file__).resolve(),
        *sorted((root / "vllm_metax/v1/sample").glob("*.py")),
    ]
    report = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "base_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "device": torch.cuda.get_device_name(),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch": torch.__version__,
        "dtype": args.dtype,
        "iterations": args.iterations,
        "cpu_threads": torch.get_num_threads(),
        "source_sha256": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources
        },
        "repeats": args.repeats,
        "timing": "wall time including dispatch and synchronization; reset excluded",
        "baseline": "equivalent dense windowed policy, NOT upstream full-history penalties",
        "execution_scope": "production module imports",
        "completed": False,
        "rows": [],
    }
    rng = random.Random(42)
    torch.manual_seed(42)
    for size, vocab, window in itertools.product(
        args.batch_sizes, args.vocab_sizes, args.windows
    ):
        config = WindowedPenaltyConfig.parse(
            {"window": window, "repetition": 1.2, "frequency": 0.1, "presence": 0.2}
        )
        requests = {
            i: WindowedPenaltyState(
                config, [rng.randrange(vocab) for _ in range(window * 2)]
            )
            for i in range(size)
        }
        batch = SparsePenaltyBatch(torch.device("cuda"), True)
        dense = DensePenaltyBatch(size, vocab, config)
        batch.update(requests)
        dense.update(requests)

        def fused_apply(logits, batch=batch):
            return batch.apply(logits)

        original = torch.randn(
            size, vocab, device="cuda", dtype=getattr(torch, args.dtype)
        )
        expected = batch.apply_torch(original.clone())
        torch.testing.assert_close(fused_apply(original.clone()), expected)
        torch.testing.assert_close(dense.apply(original.clone()), expected)
        logits = original.clone()
        paths = {
            "dense_apply": partial(dense.apply, logits),
            "sparse_torch_apply": partial(batch.apply_torch, logits),
            "fused_apply": partial(fused_apply, logits),
            "sparse_update": partial(batch.update, requests),
            "dense_update_and_apply": partial(
                update_and_apply,
                dense,
                requests,
                logits,
            ),
            "sparse_torch_update_and_apply": partial(
                update_and_apply,
                batch,
                requests,
                logits,
                torch_reference=True,
            ),
            "sparse_update_and_apply": partial(
                update_and_apply,
                batch,
                requests,
                logits,
                apply_fn=fused_apply,
            ),
        }
        timings = {name: [] for name in paths}
        for _ in range(args.repeats):
            order = list(paths)
            rng.shuffle(order)
            for name in order:
                timings[name].extend(
                    measure(paths[name], logits, original, args.iterations)
                )
        summaries = {name: summarize(samples) for name, samples in timings.items()}
        row = {
            "batch_size": size,
            "vocab_size": vocab,
            "window": window,
            "unique_updates": batch.num_updates,
            "sparse_device_state_bytes": sum(
                x.numel() * x.element_size() for x in (batch.indices, batch.parameters)
            ),
            "dense_device_state_bytes": sum(
                x.numel() * x.element_size() for x in (dense.counts, dense.seen)
            ),
            "timings": summaries,
            "apply_speedup_vs_dense": summaries["dense_apply"]["median_us"]
            / summaries["fused_apply"]["median_us"],
            "apply_speedup_vs_sparse_torch": summaries["sparse_torch_apply"][
                "median_us"
            ]
            / summaries["fused_apply"]["median_us"],
            "whole_step_speedup_vs_dense": summaries["dense_update_and_apply"][
                "median_us"
            ]
            / summaries["sparse_update_and_apply"]["median_us"],
            "whole_step_speedup_vs_sparse_torch": summaries[
                "sparse_torch_update_and_apply"
            ]["median_us"]
            / summaries["sparse_update_and_apply"]["median_us"],
        }
        report["rows"].append(row)
        print(
            f"batch={size} vocab={vocab} window={window}: apply={row['apply_speedup_vs_dense']:.2f}x whole_step={row['whole_step_speedup_vs_dense']:.2f}x",
            flush=True,
        )
        if args.output:
            args.output.write_text(json.dumps(report, indent=2) + "\n")
    report["completed"] = True
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
