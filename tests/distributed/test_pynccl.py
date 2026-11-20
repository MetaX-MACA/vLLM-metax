# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import time
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import vllm_metax
from vllm_metax.distributed.device_communicators.pynccl_wrapper import MCCLLibrary


def _check_collectives(rank, init_method):
    vllm_metax.register_out_of_tree()
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=2,
        init_method=init_method,
        timeout=timedelta(seconds=30),
    )
    try:
        comm = PyNcclCommunicator(dist.group.WORLD, device=device)
        assert comm.available and not comm.disabled
        assert isinstance(comm.nccl, MCCLLibrary)
        try:
            x = torch.arange(12, device=device, dtype=torch.float32).reshape(3, 4)
            data = x + rank * 20
            result = comm.all_reduce(data)
            torch.testing.assert_close(result, 2 * x + 20, atol=0, rtol=0)

            gathered = torch.empty((6, 4), device=device)
            comm.all_gather(gathered, data)
            torch.testing.assert_close(gathered, torch.cat((x, x + 20)), atol=0, rtol=0)

            uneven = torch.empty((3, 4), device=device)
            comm.all_gatherv(uneven, data[: rank + 1].contiguous(), sizes=[1, 2])
            expected = torch.cat((x[:1], x[:2] + 20))
            torch.testing.assert_close(uneven, expected, atol=0, rtol=0)

            scattered = torch.empty((rank + 1, 4), device=device)
            comm.reduce_scatterv(scattered, data, sizes=[1, 2])
            torch.testing.assert_close(
                scattered,
                (2 * x + 20).split([1, 2])[rank],
                atol=0,
                rtol=0,
            )

            for source in range(2):
                broadcast = data.clone()
                comm.broadcast(broadcast, src=source)
                torch.testing.assert_close(broadcast, x + source * 20, atol=0, rtol=0)
            dist.barrier()
        finally:
            comm.destroy()
    finally:
        dist.destroy_process_group()


# Adapted from vLLM v0.33.0 tests/distributed/test_pynccl.py.
@pytest.mark.distributed
@pytest.mark.requires_maca
def test_two_rank_mccl_collectives(tmp_path):
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two MACA devices")
    context = mp.spawn(
        _check_collectives,
        args=((tmp_path / "rendezvous").as_uri(),),
        nprocs=2,
        join=False,
    )
    deadline = time.monotonic() + 90
    try:
        while not context.join(timeout=max(0, deadline - time.monotonic())):
            if time.monotonic() >= deadline:
                pytest.fail("two-rank MCCL collectives exceeded 90 seconds")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.kill()
            process.join()
