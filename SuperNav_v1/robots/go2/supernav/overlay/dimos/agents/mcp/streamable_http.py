# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared Streamable-HTTP/SSE helpers for DimOS MCP surfaces."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
import json
from typing import Any

from starlette.responses import StreamingResponse

DEFAULT_SSE_KEEPALIVE_INTERVAL = 20.0


def sse_frame(data: dict[str, Any]) -> str:
    """Format a JSON-RPC message as an SSE ``event: message`` frame."""

    return f"event: message\ndata: {json.dumps(data)}\n\n"


class StreamableHttpSseHub:
    """Fan-out hub for MCP Streamable-HTTP server-to-client SSE channels."""

    def __init__(self, *, keepalive_interval_s: float = DEFAULT_SSE_KEEPALIVE_INTERVAL) -> None:
        self.keepalive_interval_s = keepalive_interval_s
        self._queues: list[asyncio.Queue[dict[str, Any] | None]] = []
        self._event_loop: asyncio.AbstractEventLoop | None = None

    def fan_out(self, msg: dict[str, Any]) -> None:
        """Forward one JSON-RPC notification to every connected SSE client."""

        loop = self._event_loop
        if loop is None:
            return
        for queue in list(self._queues):
            try:
                asyncio.run_coroutine_threadsafe(queue.put(msg), loop)
            except RuntimeError:
                pass

    def close_all(self) -> None:
        """Close every active SSE stream."""

        for queue in list(self._queues):
            try:
                queue.put_nowait(None)
            except Exception:
                pass
        self._queues.clear()
        self._event_loop = None

    async def response(self) -> StreamingResponse:
        """Create a persistent Streamable-HTTP SSE response."""

        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        self._event_loop = asyncio.get_running_loop()
        self._queues.append(queue)

        async def event_generator() -> AsyncGenerator[str, None]:
            try:
                yield ": connected\n\n"
                while True:
                    try:
                        msg = await asyncio.wait_for(queue.get(), timeout=self.keepalive_interval_s)
                    except asyncio.TimeoutError:
                        yield ": keepalive\n\n"
                        continue
                    if msg is None:
                        return
                    yield sse_frame(msg)
            except asyncio.CancelledError:
                pass
            finally:
                try:
                    self._queues.remove(queue)
                except ValueError:
                    pass

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )
