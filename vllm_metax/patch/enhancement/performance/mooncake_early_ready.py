# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#
# -----------------------------------------------------------------------------
# Note: Tell the Mooncake prefill (P) workers "the KV of this request is ready to
#       send" straight from the P scheduler over ZMQ, instead of waiting for the
#       next ``schedule()`` -> next model batch to carry it in the connector
#       metadata.
#
#       Upstream flow: the scheduler process learns that a request finished
#       (``MooncakeConnectorScheduler.request_finished``) and parks it in
#       ``_reqs_need_send``; the information only reaches the workers inside the
#       connector metadata of the NEXT scheduled batch, which a worker can only
#       start after the batch it is currently executing. On a loaded P instance
#       the signal is therefore delayed by about one P step (~140 ms for a
#       2048-token chunk), although the KV itself needs only ~5 ms to transfer.
#
#       Controlled by ``VLLM_METAX_MOONCAKE_EARLY_READY`` (default 1; use 0
#       to disable). The upstream path still runs, so a lost
#       early message only means the old (slower) behaviour.
#
# Affected versions: v0.30.0 (ced6857afa0e)
#
# Remove at: Remove after the upstream Mooncake connector delivers the "ready to
#            send" event to the workers without waiting for the next batch.
# -----------------------------------------------------------------------------
"""Early "ready to send" notification for the vLLM Mooncake producer (P) side."""

import asyncio
import collections
import json
import logging
import time
from typing import Any

import httpx
import zmq

from vllm import envs
from vllm.utils.network_utils import make_zmq_path

from vllm_metax.patch.utils import PATCH_MARKER, patch

logger = logging.getLogger(
    "vllm.metax.mooncake_early_ready"
)  # under the "vllm" logger so that INFO shows up in the server log

_MOD = "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector"


def _check_compatible() -> None:
    """Refuse to patch (before touching anything) if the upstream connector does not look like the one this patch was written for."""
    import importlib

    mod = importlib.import_module(_MOD)
    required = {
        "MooncakeConnectorScheduler": ("__init__", "request_finished"),
        "MooncakeConnectorWorker": ("__init__", "record_send_reqs"),
    }
    for cls_name, methods in required.items():
        cls = getattr(mod, cls_name, None)
        if cls is None or any(not hasattr(cls, m) for m in methods):
            raise ImportError(
                f"{_MOD}.{cls_name} does not match what mooncake_early_ready expects"
            )
    for name in ("SendBlockMeta", "get_mooncake_bootstrap_addr"):
        if not hasattr(mod, name):
            raise ImportError(f"{_MOD}.{name} is missing")


_check_compatible()
_PREFIX = b"PDREADY|"
_SEEN_MAX = 8192  # remembered transfer ids (guard against a message that arrives late)
_EARLY_MAX = 8192  # safety cap for ids waiting for the regular metadata


def _original(replacement: Any, target: str) -> Any:
    return getattr(replacement, PATCH_MARKER)[f"{_MOD}.{target}"]


# --------------------------------------------------------------------------- scheduler side
@patch(_MOD, "MooncakeConnectorScheduler.__init__")
def scheduler_init(self, *args: Any, **kwargs: Any) -> None:
    _original(scheduler_init, "MooncakeConnectorScheduler.__init__")(
        self, *args, **kwargs
    )
    # (vllm_config, engine_id, kv_cache_config): remember the engine id so that only
    # the workers of THIS instance (not other DP engines known to the bootstrap
    # server) are notified.
    self._pd_engine_id = kwargs.get("engine_id", args[1] if len(args) > 1 else None)


@patch(_MOD, "MooncakeConnectorScheduler._pd_notify", allow_missing=True)
def _pd_notify(self, transfer_id: str, p_req_id: str, blocks: Any) -> None:
    if time.monotonic() < getattr(
        self, "_pd_retry_at", 0.0
    ):  # last attempt failed: do not stall the scheduler again yet
        return
    try:
        from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import (
            get_mooncake_bootstrap_addr,
        )

        if not hasattr(self, "_pd_socks"):
            host, port = get_mooncake_bootstrap_addr(self.vllm_config)
            data = httpx.get(
                make_zmq_path("http", host, port) + "/query", timeout=5
            ).json()
            self._pd_ctx = zmq.Context()
            self._pd_socks = []
            # Match MooncakeConnectorWorker.__init__ in the target version;
            # get_mooncake_dp_engine_index no longer exists upstream.
            pc = self.vllm_config.parallel_config
            dp_index = str(
                pc.data_parallel_rank_local
                if pc.local_engines_only
                else pc.data_parallel_index
            )
            for dp_key, dp_entry in data.items():
                # only the workers of THIS engine: same DP index and (when known) same engine id
                if dp_key != dp_index:
                    continue
                if (
                    self._pd_engine_id is not None
                    and dp_entry.get("engine_id") != self._pd_engine_id
                ):
                    continue
                for pp_map in dp_entry["worker_addr"].values():
                    for addr in pp_map.values():
                        sock = self._pd_ctx.socket(zmq.DEALER)
                        sock.setsockopt(zmq.LINGER, 0)
                        sock.connect(addr)
                        self._pd_socks.append(sock)
            expected = pc.tensor_parallel_size * pc.pipeline_parallel_size
            if len(self._pd_socks) != expected:
                raise RuntimeError(
                    f"Expected {expected} P workers, discovered {len(self._pd_socks)}"
                )
            logger.info(
                "Mooncake early-ready channel to %d P workers", len(self._pd_socks)
            )
        payload = (
            _PREFIX
            + json.dumps(
                {
                    "tid": transfer_id,
                    "preq": p_req_id,
                    "blocks": [[int(b) for b in group] for group in blocks],
                }
            ).encode()
        )
        for sock in self._pd_socks:
            sock.send(payload, zmq.NOBLOCK)
    except Exception as e:  # never let the optimisation break a request
        self._pd_retry_at = time.monotonic() + 30.0
        if hasattr(self, "_pd_socks"):
            for sock in self._pd_socks:
                sock.close(linger=0)
            del self._pd_socks
        if hasattr(self, "_pd_ctx"):
            self._pd_ctx.term()
            del self._pd_ctx
        if not getattr(self, "_pd_warned", False):
            self._pd_warned = True
            logger.warning(
                "Mooncake early-ready notification failed, falling back to the regular path: %s",
                e,
            )


@patch(_MOD, "MooncakeConnectorScheduler.request_finished")
def request_finished(self, request: Any, block_ids: Any) -> Any:
    result = _original(request_finished, "MooncakeConnectorScheduler.request_finished")(
        self, request, block_ids
    )
    delay_free_blocks = result[0] if isinstance(result, tuple) else False
    entry = self._reqs_need_send.get(request.request_id) if delay_free_blocks else None
    params = request.kv_transfer_params
    if entry is not None and entry[1] and params and params.get("transfer_id"):
        # the same (sliding-window clipped) block ids the regular path puts in the metadata
        self._pd_notify(params["transfer_id"], request.request_id, entry[1])
    return result


# --------------------------------------------------------------------------- worker side
@patch(_MOD, "MooncakeConnectorWorker.__init__")
def worker_init(self, *args: Any, **kwargs: Any) -> None:
    _original(worker_init, "MooncakeConnectorWorker.__init__")(self, *args, **kwargs)
    queue = getattr(
        self, "sender_worker_queue", None
    )  # only exists on the producer side
    if queue is None:
        return
    self._pd_early = set()
    self._pd_seen = collections.OrderedDict()
    original_put = queue.put

    async def put(item):
        # runs in the sender listener's event loop (same thread as record_send_reqs)
        _identity, payload = item
        if payload.startswith(_PREFIX):
            try:
                self._pd_early_ready(payload[len(_PREFIX) :])
            except Exception:
                # The normal connector metadata remains the fallback even if
                # this optional notification cannot be decoded or applied.
                logger.warning(
                    "Ignoring invalid Mooncake early-ready message", exc_info=True
                )
            return
        await original_put(item)

    queue.put = put


@patch(_MOD, "MooncakeConnectorWorker._pd_early_ready", allow_missing=True)
def _pd_early_ready(self, raw: bytes) -> None:
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import (
        SendBlockMeta,
    )

    msg = json.loads(raw)
    tid = msg["tid"]
    if tid in self._pd_seen:  # the regular metadata already handled this transfer
        return
    if tid not in self._pd_early and len(self._pd_early) >= _EARLY_MAX:
        # Keep existing ids until regular metadata consumes them. Forgetting
        # them could recreate an already-completed transfer in the normal path.
        return
    send_meta = self.reqs_need_send.get(tid)
    if (
        send_meta is None
    ):  # the D side has not asked yet: create the entry like send_kv_to_decode
        send_meta = SendBlockMeta(
            p_req_id=msg["preq"],
            transfer_id=tid,
            local_block_ids=[],
            ready=asyncio.Event(),
        )
        self.reqs_need_send[tid] = send_meta
    send_meta.p_req_id = msg["preq"]
    send_meta.local_block_ids = msg["blocks"]
    send_meta.expire_time = (
        time.perf_counter() + envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT
    )
    self._pd_early.add(tid)
    send_meta.ready.set()


@patch(_MOD, "MooncakeConnectorWorker.record_send_reqs")
async def record_send_reqs(self, metadata: Any) -> None:
    early = getattr(self, "_pd_early", None)
    if early is not None:
        for p_req_id, (tid, block_ids) in list(metadata.reqs_to_send.items()):
            if not block_ids:
                continue
            self._pd_seen[tid] = None
            if len(self._pd_seen) > _SEEN_MAX:
                self._pd_seen.popitem(last=False)
            if (
                tid in early
            ):  # already made ready by the early message: drop the duplicate
                early.discard(tid)
                del metadata.reqs_to_send[p_req_id]
    return await _original(
        record_send_reqs, "MooncakeConnectorWorker.record_send_reqs"
    )(self, metadata)
