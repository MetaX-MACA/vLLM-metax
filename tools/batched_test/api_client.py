# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Example Python client for OpenAI Chat Completion using vLLM API server
NOTE: start a supported chat completion model server with `vllm serve`, e.g.
    vllm serve meta-llama/Llama-2-7b-chat-hf
"""

import argparse

from openai import OpenAI

# Modify OpenAI's API key and API base to use vLLM's API server.
OPENAI_API_BASE_FORMAT = "http://%s:%d/v1"

system_message = {
    "role": "system",
    "content": "You must answer as concisely as possible. Any extra information is unnecessary.",
}


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

    def create_chat_completion(
        self, questions: list[str], model: str, stream: bool = False
    ):
        for question in questions:
            messages = [system_message, {"role": "user", "content": question}]
            yield self.client.chat.completions.create(
                messages=messages,
                model=model,
                stream=stream,
            )


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

    for response in client.create_chat_completion(
        questions=questions, model=model, stream=False
    ):
        print("Response:")
        print(response.choices[0].message.content)
        print("-" * 40)
