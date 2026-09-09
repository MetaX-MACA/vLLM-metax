# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Example Python client for OpenAI Chat Completion using vLLM API server
NOTE: start a supported chat completion model server with `vllm serve`, e.g.
    vllm serve meta-llama/Llama-2-7b-chat-hf
"""

import argparse
import pybase64 as base64
import os
import mimetypes
from typing import Iterable

from openai import OpenAI

try:
    from .paths import TOOL_ROOT, resolve_path
except ImportError:  # Direct `python api_client.py` execution.
    from paths import TOOL_ROOT, resolve_path

# Modify OpenAI's API key and API base to use vLLM's API server.
OPENAI_API_BASE_FORMAT = "http://%s:%d/v1"

system_message = {
    "role": "system",
    "content": "You must answer as concisely as possible. Any extra information is unnecessary.",
}


def encode_base64_content_from_file(content_path: str) -> str:
    """Encode a content retrieved from a remote url to base64 format."""
    with open(content_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def is_remote_url(s: str) -> bool:
    return s.startswith("http://") or s.startswith("https://")


def file_path_to_data_url(path: str) -> str:
    abs_path = resolve_path(path, TOOL_ROOT)

    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Image file not found: {abs_path}")

    mime, _ = mimetypes.guess_type(abs_path)
    if not mime:
        # Safe fallback; vLLM generally accepts common image types.
        mime = "image/png"

    b64 = encode_base64_content_from_file(abs_path)
    return f"data:{mime};base64,{b64}"


class ApiConnection:
    """Shared transport and cancellation; no chat/embedding inheritance."""

    def __init__(self, host="localhost", port=8000, stop_event=None, timeout=600.0):
        # Optional per-task cancellation signal checked before each request.
        self.stop_event = stop_event
        # Seconds per request, or a callback returning the remaining allowed duration.
        self.timeout = timeout
        # Shared OpenAI-compatible HTTP transport; closed by the owning suite executor.
        self.client = OpenAI(
            api_key="EMPTY",
            base_url=OPENAI_API_BASE_FORMAT % (host, port),
            timeout=600.0,
            max_retries=0,
        )
        # Cached served model ID discovered from /models on first use.
        self._model = None

    def check(self):
        if self.stop_event is not None and self.stop_event.is_set():
            raise RuntimeError("Model run cancelled")

    def request_client(self):
        self.check()
        timeout = self.timeout() if callable(self.timeout) else self.timeout
        return self.client.with_options(timeout=timeout, max_retries=0)

    def get_model(self):
        self.check()
        if self._model is None:
            models = self.request_client().models.list()
            self._model = models.data[0].id
        return self._model

    def close(self):
        self.client.close()


class ChatCompletionClient:
    def __init__(
        self, host="localhost", port=8000, stop_event=None, *, connection=None
    ):
        # Shared transport and cancellation policy, usually injected by the suite executor.
        self.connection = connection or ApiConnection(host, port, stop_event)

    def get_model(self):
        return self.connection.get_model()

    def run_text_only(
        self,
        questions: list[str],
        model: str | None = None,
        max_completion_tokens: int = 256,
    ) -> Iterable[str]:
        if model is None:
            model = self.get_model()

        for question in questions:
            self.connection.check()
            messages = [system_message, {"role": "user", "content": question}]
            response = self.connection.request_client().chat.completions.create(
                messages=messages,
                model=model,
                max_completion_tokens=max_completion_tokens,
            )
            yield response.choices[0].message.content or ""

    # Single-image input inference
    def run_single_image(
        self,
        max_completion_tokens: int,
        image_urls: list[str],
        model: str | None = None,
    ) -> Iterable[str]:
        """Yield image responses; transport failures propagate to the task runner."""
        if model is None:
            model = self.get_model()

        for image_url in image_urls:
            self.connection.check()
            if is_remote_url(image_url):
                url_content = image_url
            else:
                url_content = file_path_to_data_url(image_url)

            chat_completion = self.connection.request_client().chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What's in this image?"},
                            {
                                "type": "image_url",
                                "image_url": {"url": url_content},
                            },
                        ],
                    }
                ],
                model=model,
                max_completion_tokens=max_completion_tokens,
            )
            yield chat_completion.choices[0].message.content or ""


class EmbeddingClient:
    """Client for OpenAI-compatible /v1/embeddings (vLLM served with --task embed)."""

    def __init__(
        self, host="localhost", port=8000, stop_event=None, *, connection=None
    ):
        # Shared transport and cancellation policy; independent of chat client inheritance.
        self.connection = connection or ApiConnection(host, port, stop_event)

    def get_model(self):
        return self.connection.get_model()

    def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        """Return one embedding vector per input text, in input order."""
        self.connection.check()
        if model is None:
            model = self.get_model()
        resp = self.connection.request_client().embeddings.create(
            model=model, input=texts
        )
        ordered = sorted(resp.data, key=lambda d: d.index)
        return [d.embedding for d in ordered]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors; 0.0 if either is zero-length."""
    import math

    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host", type=str, default="localhost", help="vLLM API server host"
    )
    parser.add_argument("--port", type=int, default=8000, help="vLLM API server port")
    parser.add_argument("--mode", type=str, default="text-only", help="inference mode")
    args = parser.parse_args()

    client = ChatCompletionClient(host=args.host, port=args.port)
    model = client.get_model()
    print(f"Using model: {model}")

    questions = [
        "Where's the capital of China?",
        "Who's the founder of Apple?",
        "What's the value of gravity in the earth?",
    ]

    iamge_urls = [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
        "assets/images/leaning_tower.png",
        "assets/images/Tom_and_Jerry.png",
    ]

    if args.mode == "text-only":
        for response in client.run_text_only(questions=questions, model=model):
            print("Response:")
            print(response)
            print("-" * 40)
    elif args.mode == "single-image":
        for response in client.run_single_image(
            model=model, max_completion_tokens=256, image_urls=iamge_urls
        ):
            print("Response:")
            print(response)
        print("-" * 40)
