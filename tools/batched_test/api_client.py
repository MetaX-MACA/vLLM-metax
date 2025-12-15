# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Example Python client for OpenAI Chat Completion using vLLM API server
NOTE: start a supported chat completion model server with `vllm serve`, e.g.
    vllm serve meta-llama/Llama-2-7b-chat-hf
"""

import argparse
import base64
import os

from openai import OpenAI

# Modify OpenAI's API key and API base to use vLLM's API server.
OPENAI_API_BASE_FORMAT = "http://%s:%d/v1"

system_message = {
    "role": "system",
    "content": "You must answer as concisely as possible. Any extra information is unnecessary.",
}


def encode_base64_content_from_url(content_path: str) -> str:
    """Encode a content retrieved from a remote url to base64 format."""
    with open(content_path, "rb") as image_file:
        result = base64.b64encode(image_file.read()).decode("utf-8")
    return result


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

    def run_text_only(self, questions: list[str]):
        model = self.get_model()

        for question in questions:
            messages = [system_message, {"role": "user", "content": question}]
            response = self.client.chat.completions.create(
                messages=messages,
                model=model,
                max_completion_tokens=256,
            )

            yield response.choices[0].message.content

    # Single-image input inference
    def run_single_image(
        self, model: str, max_completion_tokens: int, image_urls: list[str]
    ):
        ## Use image url in the payload
        for image_url in image_urls:
            if image_url.startswith("http://") or image_url.startswith("https://"):
                url_content = image_url
            else:
                image_url = os.path.join(os.path.dirname(__file__), image_url)
                url_content = "data:image/png;base64," + encode_base64_content_from_url(
                    image_url
                )

            chat_completion_from_url = self.client.chat.completions.create(
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

            yield chat_completion_from_url.choices[0].message.content


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host", type=str, default="localhost", help="vLLM API server host"
    )
    parser.add_argument("--port", type=int, default=8000, help="vLLM API server port")
    args = parser.parse_args()

    client = ChatCompletionClient(host=args.host, port=args.port)
    model = client.get_model()
    print(f"Using model: {model}")

    questions = [
        "Where's the capital of China?",
        "Who's the founder of Apple?",
        "What's the value of gravity in the earth?",
    ]

    for response in client.run_text_only(
        questions=questions, model=model, stream=False
    ):
        print("Response:")
        print(response.choices[0].message.content)
        print("-" * 40)
