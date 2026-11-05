# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Content identities for safe result reuse; never import a candidate package."""

from dataclasses import asdict, dataclass
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import sys


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


@dataclass(frozen=True)
class ValidationIdentity:
    source: str
    environment: str
    checkpoint: str
    inputs: str
    reusable: bool


def runtime_identity(source_roots=None):
    """Capture code and selected environment once per scheduler invocation."""
    injected = os.environ.get("METAX_UPGRADE_VALIDATION_IDENTITY_FILE")
    if injected:
        value = json.loads(Path(injected).read_text())
        if not value.get("source") or not value.get("environment"):
            raise ValueError("Upgrade validation identity needs source and environment")
        return value
    root = Path(__file__).resolve().parents[2]
    roots = (
        list(source_roots) if source_roots is not None else [root, root.parent / "vllm"]
    )
    distributions = list(metadata.distributions())
    if source_roots is None:
        for dist in distributions:
            if (dist.metadata.get("Name") or "").lower().replace("_", "-") in {
                "vllm",
                "vllm-metax",
            }:
                roots.append(Path(dist.locate_file("")))
                direct = dist.read_text("direct_url.json")
                if direct:
                    from urllib.parse import unquote, urlsplit

                    url = urlsplit(json.loads(direct).get("url", ""))
                    if url.scheme == "file":
                        roots.append(Path(unquote(url.path)))
    sources = {}
    for checkout in roots:
        checkout = Path(checkout)
        for folder in ("vllm_metax", "vllm", "csrc", "cmake", "requirements"):
            for path in sorted((checkout / folder).rglob("*")):
                if (
                    path.is_file()
                    and "__pycache__" not in path.parts
                    and path.suffix
                    in {".py", ".cu", ".cpp", ".h", ".cuh", ".cmake", ".txt", ".so"}
                ):
                    sources[str(path)] = file_hash(path)
        for name in ("pyproject.toml", "setup.py", "CMakeLists.txt"):
            if (checkout / name).is_file():
                sources[str(checkout / name)] = file_hash(checkout / name)
    packages = sorted(
        (
            dist.metadata.get("Name", ""),
            dist.version,
            str(dist.locate_file("")),
            digest(dist.read_text("RECORD") or ""),
        )
        for dist in distributions
    )
    environment = {
        "python": sys.executable,
        "version": sys.version,
        "prefix": sys.prefix,
        "packages": packages,
        "maca_path": os.environ.get("MACA_PATH"),
        "visible_devices": os.environ.get(
            "MACA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES")
        ),
    }
    return {
        "source": digest(sources),
        "environment": digest(environment),
        "source_files": sources,
    }


def capture(model, spec, runtime=None):
    runtime = runtime if runtime is not None else runtime_identity()
    model_path = Path(model.model_path)
    checkpoint = {}
    inputs = {}
    reusable = model_path.is_dir()
    if reusable:
        # Full content hashes also detect in-place rewrites with preserved mtimes.
        # Capture is explicit planning work, never a dataclass/property side effect.
        for path in sorted(model_path.rglob("*")):
            if path.is_file() and path.suffix in {
                ".safetensors",
                ".bin",
                ".pt",
                ".pth",
                ".ckpt",
                ".gguf",
            }:
                checkpoint[str(path)] = file_hash(path)
            elif path.is_file() and path.suffix in {
                ".json",
                ".jinja",
                ".model",
                ".txt",
                ".tiktoken",
                ".py",
            }:
                inputs[str(path)] = file_hash(path)
        dummy = model.serve.load_format == "dummy"
        reusable = bool(checkpoint) or dummy

    def inspect(value):
        nonlocal reusable
        if isinstance(value, dict):
            for item in value.values():
                inspect(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                inspect(item)
        elif isinstance(value, str):
            if value.startswith(("http://", "https://")):
                reusable = False
            else:
                try:
                    if Path(value).is_file():
                        inputs[str(Path(value).resolve())] = file_hash(value)
                except OSError:
                    pass  # Prompt text is not necessarily a valid filename.

    inspect(asdict(model))
    inspect(asdict(spec))
    return ValidationIdentity(
        str(runtime["source"]),
        str(runtime["environment"]),
        digest(checkpoint),
        digest(inputs),
        reusable,
    )
