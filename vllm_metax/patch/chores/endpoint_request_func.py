# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ---------------------------------------------------------------
# hotfix: https://github.com/vllm-project/vllm/pull/38732
# ---------------------------------------------------------------

import codecs
import json

class StreamedResponseHandler:
    """Handles streaming HTTP responses by accumulating chunks until complete
    messages are available."""

    def __init__(self):
        self.buffer = ""
        # /------------------------  Metax Modification -------------------------\
        self._decoder = codecs.getincrementaldecoder("utf-8")()
        # \------------------------  Metax Modification -------------------------/

    def add_chunk(self, chunk_bytes: bytes) -> list[str]:
        """Add a chunk of bytes to the buffer and return any complete
        messages."""
        # /------------------------  Metax Modification -------------------------\
        chunk_str = self._decoder.decode(chunk_bytes)
        # \------------------------  Metax Modification -------------------------/
        self.buffer += chunk_str

        messages = []

        # Split by double newlines (SSE message separator)
        while "\n\n" in self.buffer:
            message, self.buffer = self.buffer.split("\n\n", 1)
            message = message.strip()
            if message:
                messages.append(message)

        # if self.buffer is not empty, check if it is a complete message
        # by removing data: prefix and check if it is a valid JSON
        if self.buffer.startswith("data: "):
            message_content = self.buffer.removeprefix("data: ").strip()
            if message_content == "[DONE]":
                messages.append(self.buffer.strip())
                self.buffer = ""
            elif message_content:
                try:
                    json.loads(message_content)
                    messages.append(self.buffer.strip())
                    self.buffer = ""
                except json.JSONDecodeError:
                    # Incomplete JSON, wait for more chunks.
                    pass

        return messages

import vllm.benchmarks.lib.endpoint_request_func
vllm.benchmarks.lib.endpoint_request_func.StreamedResponseHandler = StreamedResponseHandler