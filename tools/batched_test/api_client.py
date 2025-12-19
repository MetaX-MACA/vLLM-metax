# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Example Python client for OpenAI Chat Completion using vLLM API server
NOTE: start a supported chat completion model server with `vllm serve`, e.g.
    vllm serve meta-llama/Llama-2-7b-chat-hf
"""

import argparse
import base64
import os
import mimetypes
from typing import Iterable

from openai import OpenAI

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
    abs_path = path
    if not os.path.isabs(abs_path):
        abs_path = os.path.join(os.path.dirname(__file__), path)

    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Image file not found: {abs_path}")

    mime, _ = mimetypes.guess_type(abs_path)
    if not mime:
        # Safe fallback; vLLM generally accepts common image types.
        mime = "image/png"

    b64 = encode_base64_content_from_file(abs_path)
    return f"data:{mime};base64,{b64}"


class ChatCompletionClient:
    """Client for OpenAI Chat Completion using vLLM API server"""

    def __init__(self, host: str = "localhost", port: int = 8000):
        self.client = OpenAI(
            api_key="EMPTY",
            base_url=OPENAI_API_BASE_FORMAT % (host, port),
        )

    def get_model(self):
        models = self.client.models.list()
        return models.data[0].id

    def run_text_only(
        self,
        questions: list[str],
        model: str | None = None,
        max_completion_tokens: int = 256,
    ) -> Iterable[str]:
        if model is None:
            model = self.get_model()

        for question in questions:
            messages = [system_message, {"role": "user", "content": question}]
            response = self.client.chat.completions.create(
                messages=messages,
                model=model,
                max_completion_tokens=max_completion_tokens,
            )
            yield response.choices[0].message.content

    # Single-image input inference
    def run_single_image(
        self,
        max_completion_tokens: int,
        image_urls: list[str],
        model: str | None = None,
    ) -> Iterable[str]:
        """
        If allow_remote_urls=False (recommended for offline), remote http(s) URLs
        will be rejected so you don't trigger server-side fetch + 500.
        """
        if model is None:
            model = self.get_model()

        for image_url in image_urls:
            try:
                if is_remote_url(image_url):
                    url_content = image_url
                else:
                    url_content = file_path_to_data_url(image_url)

                chat_completion = self.client.chat.completions.create(
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
                yield chat_completion.choices[0].message.content

            except Exception as e:
                yield f"type={type(e).__name__} msg={e}"


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
        for response in client.run_text_only(
            questions=questions, model=model, stream=False
        ):
            print("Response:")
            print(response.choices[0].message.content)
            print("-" * 40)
    elif args.mode == "single-image":
        for response in client.run_single_image(
            model=model, max_completion_tokens=256, image_urls=iamge_urls
        ):
            print("Response:")
            print(response)
        print("-" * 40)
