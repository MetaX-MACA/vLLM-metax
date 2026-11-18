# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NIXL pull P/D proxy with independent round-robin prefill and decode pools.

The proxy preserves the original prompt, request options and complete producer
handoff metadata. Qwen QSA recurrent state requires the producer's N-1 boundary,
so the frontend does not append a sampled token or recompute decode state.
Single-pair --prefill/--decode commands remain supported. Pools are static:
requests are not automatically retried or replayed after a backend failure.
All backends must serve compatible model/tokenizer and NIXL configurations.
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import logging
import time
import uuid
from collections.abc import Sequence
from typing import Any
from urllib.parse import urlsplit

import aiohttp
from aiohttp import web

logger = logging.getLogger(__name__)
COMPLETION_PATHS = ("/v1/completions", "/v1/chat/completions")


def prefill_request(payload: dict[str, Any]) -> dict[str, Any]:
    """Keep tokenization identical while limiting the producer to one token."""
    if payload.get("n", 1) != 1:
        raise ValueError("PD requires n=1 per request.")
    prompt = payload.get("prompt")
    if (
        isinstance(prompt, list)
        and prompt
        and not all(isinstance(token, int) for token in prompt)
    ):
        raise ValueError("Send batched prompts as separate PD requests.")
    if payload.get("kv_transfer_params"):
        raise ValueError("This frontend accepts one-way PD requests only.")
    result = dict(payload)
    for key in (
        "max_completion_tokens",
        "min_tokens",
        "min_completion_tokens",
        "stream_options",
        "logprobs",
        "top_logprobs",
        "prompt_logprobs",
    ):
        result.pop(key, None)
    result.update(
        max_tokens=1, stream=False, kv_transfer_params={"do_remote_decode": True}
    )
    return result


def decode_request(
    payload: dict[str, Any],
    producer_response: Any,
) -> dict[str, Any]:
    """Forward the original prompt and the complete validated P handoff."""
    if not isinstance(producer_response, dict):
        raise ValueError("Producer response must be a JSON object.")
    metadata = producer_response.get("kv_transfer_params")
    if not isinstance(metadata, dict):
        raise ValueError("Producer did not return NIXL handoff metadata.")
    if (
        metadata.get("do_remote_prefill") is not True
        or metadata.get("do_remote_decode") is not False
    ):
        raise ValueError("Producer did not authorize remote prefill on the consumer.")
    for key in ("remote_engine_id", "remote_request_id", "remote_host"):
        if not isinstance(metadata.get(key), str) or not metadata[key]:
            raise ValueError(f"Producer handoff is missing {key}.")
    for key in ("remote_port", "tp_size", "dcp_size", "pp_size"):
        if type(metadata.get(key)) is not int or metadata[key] <= 0:
            raise ValueError(f"Producer handoff has invalid {key}.")
    if metadata["remote_port"] > 65535 or metadata["pp_size"] != 1:
        raise ValueError("Producer handoff has unsupported port or pipeline topology.")
    if (
        type(metadata.get("remote_num_tokens")) is not int
        or metadata["remote_num_tokens"] < 0
    ):
        raise ValueError("Producer handoff has invalid remote_num_tokens.")
    blocks = metadata.get("remote_block_ids")
    if (
        not isinstance(blocks, list)
        or not blocks
        or not any(blocks)
        or any(
            not isinstance(group, list)
            or any(type(block) is not int or block < 0 for block in group)
            for group in blocks
        )
    ):
        raise ValueError("Producer handoff has invalid remote_block_ids.")
    if metadata.get("transfer_mode") != "pull":
        raise ValueError("This frontend requires the NIXL pull protocol.")
    result = dict(payload)
    result["kv_transfer_params"] = metadata
    return result


# /-------------------- MetaX Modification --------------------\
# Normalize static pools once and reject credential-bearing URLs.
def normalize_urls(value: str | Sequence[str]) -> tuple[str, ...]:
    values = (value,) if isinstance(value, str) else tuple(value)
    if not values:
        raise ValueError("Each pool must contain at least one HTTP base URL.")
    result = []
    for url in values:
        if not isinstance(url, str):
            raise ValueError("Backend URLs must be strings.")
        parsed = urlsplit(url)
        port = parsed.port  # Raises ValueError for malformed/out-of-range ports.
        if (
            parsed.scheme not in ("http", "https")
            or not parsed.hostname
            or any(char.isspace() for char in url)
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or (port is not None and not 1 <= port <= 65535)
        ):
            raise ValueError("Use HTTP base URLs without credentials/query/fragment.")
        result.append(url.rstrip("/"))
    return tuple(result)


# \-------------------- MetaX Modification --------------------/


def create_app(
    prefill_url: str | Sequence[str],
    decode_url: str | Sequence[str],
    *,
    timeout: float = 600,
    connection_limit: int = 256,
    reuse_connections: bool = True,
) -> web.Application:
    # /-------------------- MetaX Modification --------------------\
    prefills, decodes = normalize_urls(prefill_url), normalize_urls(decode_url)
    if timeout <= 0 or connection_limit < 0:
        raise ValueError("timeout must be positive; connection_limit must be >= 0.")
    # next() has no await: selection is atomic within this app's event loop.
    prefill_iterator, decode_iterator = (
        itertools.cycle(prefills),
        itertools.cycle(decodes),
    )
    # \-------------------- MetaX Modification --------------------/
    app = web.Application(client_max_size=16 * 1024 * 1024)
    client_key = web.AppKey("pd_http_client", aiohttp.ClientSession)
    # Each upstream HTTP call has its own budget, including P queue/prefill time.
    client_timeout = aiohttp.ClientTimeout(total=timeout)

    async def client_context(application: web.Application):
        connector = aiohttp.TCPConnector(
            limit=connection_limit,
            force_close=not reuse_connections,
        )
        async with aiohttp.ClientSession(
            connector=connector,
            timeout=client_timeout,
        ) as session:
            application[client_key] = session
            yield

    app.cleanup_ctx.append(client_context)

    async def health(request: web.Request) -> web.Response:
        async def check(url: str) -> int:
            try:
                async with request.app[client_key].get(
                    url + "/health",
                    timeout=aiohttp.ClientTimeout(total=min(timeout, 5)),
                ) as response:
                    await response.read()
                    return response.status
            except (TimeoutError, aiohttp.ClientError):
                return 503

        # /-------------------- MetaX Modification --------------------\
        # Check all configured P/D backends rather than just one pair.
        statuses = await asyncio.gather(*(check(url) for url in prefills + decodes))
        ps, ds = statuses[: len(prefills)], statuses[len(prefills) :]
        p = 200 if all(status == 200 for status in ps) else 503
        d = 200 if all(status == 200 for status in ds) else 503
        body: dict[str, Any] = {"prefill": p, "decode": d}
        if len(prefills) > 1 or len(decodes) > 1:
            body.update(
                prefills=[
                    {"url": url, "status": status}
                    for url, status in zip(prefills, ps, strict=False)
                ],
                decodes=[
                    {"url": url, "status": status}
                    for url, status in zip(decodes, ds, strict=False)
                ],
            )
        # \-------------------- MetaX Modification --------------------/
        return web.json_response(body, status=200 if p == d == 200 else 503)

    async def healthcheck(request: web.Request) -> web.Response:
        # Match the official toy proxy's process-liveness endpoint.
        return web.json_response(
            {
                "status": "ok",
                "prefill_instances": len(prefills),
                "decode_instances": len(decodes),
            }
        )

    def upstream_headers(request: web.Request) -> dict[str, str]:
        return (
            {"Authorization": request.headers["Authorization"]}
            if "Authorization" in request.headers
            else {}
        )

    async def forward_get(request: web.Request, url: str) -> web.Response:
        try:
            async with request.app[client_key].get(
                url,
                headers=upstream_headers(request),
            ) as response:
                return web.Response(
                    body=await response.read(),
                    status=response.status,
                    headers={
                        "Content-Type": response.headers.get(
                            "Content-Type", "application/json"
                        )
                    },
                )
        except TimeoutError:
            return web.json_response(
                {"error": {"message": "Decoder timed out."}}, status=504
            )
        except aiohttp.ClientError:
            return web.json_response(
                {"error": {"message": "Decoder unavailable."}}, status=502
            )

    async def models(request: web.Request) -> web.Response:
        # Discovery must not advance the generation round-robin iterator.
        return await forward_get(request, decodes[0] + "/v1/models")

    async def metrics(request: web.Request) -> web.Response:
        # /-------------------- MetaX Modification --------------------\
        # Do not misrepresent one decoder's metrics as the whole pool's metrics.
        index = request.query.get("decode_index")
        if index is None and len(decodes) > 1:
            return web.json_response(
                {
                    "error": {
                        "message": "Scrape each decoder directly or specify ?decode_index=0..N-1."
                    }
                },
                status=400,
            )
        try:
            selected = 0 if index is None else int(index)
            if not 0 <= selected < len(decodes):
                raise ValueError
        except ValueError:
            return web.json_response(
                {"error": {"message": "Invalid decode_index."}}, status=400
            )
        # \-------------------- MetaX Modification --------------------/
        return await forward_get(request, decodes[selected] + "/metrics")

    async def complete(request: web.Request) -> web.StreamResponse:
        try:
            payload = await request.json()
            if not isinstance(payload, dict):
                raise ValueError("The request must be a JSON object.")
            p_payload = prefill_request(payload)
        except (ValueError, json.JSONDecodeError) as exc:
            return web.json_response({"error": {"message": str(exc)}}, status=400)

        request_id = request.headers.get("X-Request-Id") or uuid.uuid4().hex
        headers = upstream_headers(request)
        headers["X-Request-Id"] = request_id
        session = request.app[client_key]
        started = time.perf_counter()
        output: web.StreamResponse | None = None
        # /-------------------- MetaX Modification --------------------\
        p_url, d_url = next(prefill_iterator), None
        # \-------------------- MetaX Modification --------------------/
        try:
            async with session.post(
                p_url + request.path, json=p_payload, headers=headers
            ) as response:
                raw = await response.read()
                if response.status != 200:
                    return web.Response(
                        body=raw,
                        status=response.status,
                        content_type="application/json",
                    )
                try:
                    d_payload = decode_request(payload, json.loads(raw))
                except (ValueError, json.JSONDecodeError) as exc:
                    return web.json_response(
                        {"error": {"message": str(exc)}}, status=502
                    )
            p_seconds = time.perf_counter() - started
            # /-------------------- MetaX Modification --------------------\
            # Select D only after a successful P handoff, as in the official proxy.
            d_url = next(decode_iterator)
            # \-------------------- MetaX Modification --------------------/
            async with session.post(
                d_url + request.path, json=d_payload, headers=headers
            ) as response:
                out_headers = {
                    "X-Request-Id": request_id,
                    "X-P1D1-Prefill-Seconds": f"{p_seconds:.6f}",
                }
                if payload.get("stream") and response.status == 200:
                    out_headers["Content-Type"] = "text/event-stream"
                    output = web.StreamResponse(status=200, headers=out_headers)
                    await output.prepare(request)
                    async for chunk in response.content.iter_any():
                        await output.write(chunk)
                    await output.write_eof()
                    return output
                return web.Response(
                    body=await response.read(),
                    status=response.status,
                    headers=out_headers,
                    content_type="application/json",
                )
        except TimeoutError:
            if output is not None and output.prepared:
                output.force_close()
                raise
            return web.json_response(
                {"error": {"message": "P/D upstream timed out."}}, status=504
            )
        except aiohttp.ClientError as exc:
            logger.warning("P/D upstream error for %s: %s", request_id, exc)
            if output is not None and output.prepared:
                output.force_close()
                raise
            return web.json_response(
                {"error": {"message": "P/D upstream unavailable."}}, status=502
            )
        finally:
            logger.info(
                "P/D request %s p=%s d=%s completed in %.6fs",
                request_id,
                p_url,
                d_url,
                time.perf_counter() - started,
            )

    app.router.add_get("/health", health)
    app.router.add_get("/healthcheck", healthcheck)
    app.router.add_get("/v1/models", models)
    app.router.add_get("/metrics", metrics)
    for path in COMPLETION_PATHS:
        app.router.add_post(path, complete)
    return app


# /-------------------- MetaX Modification --------------------\
# Accept legacy single URLs and the official multi-node host/port options.
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p_group = parser.add_mutually_exclusive_group(required=True)
    p_group.add_argument("--prefill", nargs="+", help="Producer HTTP base URL(s)")
    p_group.add_argument("--prefiller-hosts", "--prefiller-host", nargs="+")
    parser.add_argument("--prefiller-ports", "--prefiller-port", type=int, nargs="+")
    d_group = parser.add_mutually_exclusive_group(required=True)
    d_group.add_argument("--decode", nargs="+", help="Consumer HTTP base URL(s)")
    d_group.add_argument("--decoder-hosts", "--decoder-host", nargs="+")
    parser.add_argument("--decoder-ports", "--decoder-port", type=int, nargs="+")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=25537)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--connection-limit", type=int, default=256)
    parser.add_argument("--disable-connection-reuse", action="store_true")
    args = parser.parse_args(argv)

    def resolve(urls, hosts, ports, default_port):
        if urls is not None:
            if ports is not None:
                parser.error("Host/port options cannot be mixed with URL options.")
            return normalize_urls(urls)
        ports = [default_port] if ports is None else ports
        if len(hosts) != len(ports):
            parser.error("Each backend host must have a corresponding port.")
        values = []
        for host, port in zip(hosts, ports, strict=False):
            if not 1 <= port <= 65535 or any(c in host for c in "/@?#"):
                parser.error("Invalid backend host or port.")
            host = host.strip("[]")
            if ":" in host:
                host = f"[{host}]"
            values.append(f"http://{host}:{port}")
        return normalize_urls(values)

    try:
        args.prefill_urls = resolve(
            args.prefill, args.prefiller_hosts, args.prefiller_ports, 8100
        )
        args.decode_urls = resolve(
            args.decode, args.decoder_hosts, args.decoder_ports, 8200
        )
    except ValueError as exc:
        parser.error(str(exc))
    if not 1 <= args.port <= 65535 or args.timeout <= 0 or args.connection_limit < 0:
        parser.error("Invalid listen port, timeout, or connection limit.")
    return args


# \-------------------- MetaX Modification --------------------/


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    logger.info(
        "Starting P/D proxy with %d prefill and %d decode endpoints",
        len(args.prefill_urls),
        len(args.decode_urls),
    )
    web.run_app(
        create_app(
            args.prefill_urls,
            args.decode_urls,
            timeout=args.timeout,
            connection_limit=args.connection_limit,
            reuse_connections=not args.disable_connection_reuse,
        ),
        host=args.host,
        port=args.port,
        handler_cancellation=True,
    )


if __name__ == "__main__":
    main()
