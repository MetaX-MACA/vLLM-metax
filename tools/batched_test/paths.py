# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Resolve paths at input boundaries; execution never depends on chdir()."""

from copy import deepcopy
import os
from pathlib import Path
import shlex
from urllib.parse import urlsplit

TOOL_ROOT = Path(__file__).resolve().parent
LEGACY_ROOTS = ("configs/", "chat_template/", "assets/")
PATH_OPTIONS = {
    "--chat-template",
    "--download-dir",
    "--allowed-local-media-path",
    "--quantization-param-path",
    "--config",
    "--dataset-path",
    "--result-dir",
    "--output-dir",
}
REFERENCE_OPTIONS = {"--tokenizer", "--generation-config"}


def is_url(value: str) -> bool:
    return bool(urlsplit(value).scheme)


def resolve_path(value: str | Path, base: Path, *, legacy: bool = True) -> str:
    """Expand user/env references and anchor a local path without following symlinks.

    Legacy tool-root prefixes are explicit aliases. Use ./configs/... to refer
    to a similarly named directory relative to the declaring file instead.
    """
    raw = os.path.expandvars(os.path.expanduser(str(value)))
    path = Path(raw)
    if not path.is_absolute():
        anchor = TOOL_ROOT if legacy and raw.startswith(LEGACY_ROOTS) else base
        path = anchor / path
    return os.path.abspath(path)


def model_reference(value: str, base: Path) -> str:
    """Keep Hub IDs/URIs; explicit or existing local references become absolute."""
    if is_url(value):
        return value
    expanded = os.path.expandvars(os.path.expanduser(value))
    if (
        Path(expanded).is_absolute()
        or value.startswith(("./", "../", "~", "$"))
        or (base / expanded).exists()
    ):
        return resolve_path(value, base, legacy=False)
    return value


def option_value(option: str, value: str, base: Path) -> str:
    if is_url(value):
        return value
    if option == "--chat-template" and (
        "{{" in value or "{%" in value or "{#" in value
    ):
        return value
    if option in PATH_OPTIONS:
        return resolve_path(value, base)
    if option in REFERENCE_OPTIONS and value not in ("auto", "vllm"):
        return model_reference(value, base)
    return value


def normalize_extra_args(raw, base: Path):
    if isinstance(raw, dict):
        return {
            key: option_value(str(key), str(value), base) if value is not None else None
            for key, value in raw.items()
        }
    if isinstance(raw, (list, str)):
        tokens = shlex.split(raw) if isinstance(raw, str) else list(map(str, raw))
        result = []
        index = 0
        while index < len(tokens):
            token = tokens[index]
            option, separator, value = token.partition("=")
            if option in PATH_OPTIONS | REFERENCE_OPTIONS:
                if separator:
                    token = option + "=" + option_value(option, value, base)
                elif index + 1 < len(tokens) and not tokens[index + 1].startswith("--"):
                    result.extend(
                        (token, option_value(option, tokens[index + 1], base))
                    )
                    index += 2
                    continue
            result.append(token)
            index += 1
        return result
    return raw


def normalize_config(raw: dict, base: Path) -> dict:
    """Normalize only documented path fields, preserving arbitrary values/env."""
    result = deepcopy(raw)
    if result.get("model_path"):
        result["model_path"] = model_reference(str(result["model_path"]), base)
    serve = result.get("serve_config")
    if isinstance(serve, dict) and "extra_args" in serve:
        serve["extra_args"] = normalize_extra_args(serve["extra_args"], base)
    bench = result.get("benchmark")
    if isinstance(bench, dict) and bench.get("bench_param"):
        bench["bench_param"] = resolve_path(bench["bench_param"], base)
    ssh = result.get("ssh")
    if isinstance(ssh, dict):
        for key in ("private_key", "keyfile", "ssh_key"):
            if ssh.get(key):
                ssh[key] = resolve_path(ssh[key], base, legacy=False)
    return result
