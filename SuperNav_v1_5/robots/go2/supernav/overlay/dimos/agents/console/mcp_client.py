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

"""Console-specific facade over the existing ``McpAdapter``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import requests

from dimos.agents.console.state import ToolInfo
from dimos.agents.mcp.mcp_adapter import DEFAULT_TIMEOUT, McpAdapter, McpError


class McpAdapterProtocol(Protocol):
    """Subset of ``McpAdapter`` used by the console."""

    url: str

    def wait_for_ready(self, timeout: float = 10.0, interval: float = 0.5) -> bool: ...

    def initialize(self) -> dict[str, Any]: ...

    def list_tools(self) -> list[dict[str, Any]]: ...

    def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...


@dataclass(frozen=True)
class McpStatusSnapshot:
    """Result of a readiness probe plus the initial tool listing."""

    ready: bool
    url: str
    tools: tuple[ToolInfo, ...] = ()
    server_info: dict[str, Any] | None = None
    error: str | None = None


class ConsoleMcpClient:
    """Small wrapper that keeps MCP protocol details inside ``McpAdapter``."""

    def __init__(self, adapter: McpAdapterProtocol) -> None:
        self.adapter = adapter

    @classmethod
    def from_url(cls, url: str | None, timeout: int = DEFAULT_TIMEOUT) -> ConsoleMcpClient:
        """Create a console client for an explicit or default MCP URL."""

        return cls(McpAdapter(url=url, timeout=timeout))

    def readiness_snapshot(self, timeout_s: float) -> McpStatusSnapshot:
        """Wait for readiness, initialize, and read the tool surface."""

        try:
            if not self.adapter.wait_for_ready(timeout=timeout_s):
                return McpStatusSnapshot(
                    ready=False,
                    url=self.adapter.url,
                    error=f"MCP server did not become ready within {timeout_s:.1f}s",
                )
            server_info = self.adapter.initialize()
            tools = tuple(_tool_info(tool) for tool in self.adapter.list_tools())
            return McpStatusSnapshot(
                ready=True,
                url=self.adapter.url,
                tools=tools,
                server_info=server_info,
            )
        except requests.Timeout:
            error = f"MCP connection timed out at {self.adapter.url}. Press F5 to retry."
        except requests.ConnectionError:
            error = f"MCP is not reachable at {self.adapter.url}. Press F5 to retry."
        except McpError as exc:
            error = f"MCP refused readiness: {exc}. Check the server/profile, then press F5."
        except requests.RequestException:
            error = f"MCP request failed at {self.adapter.url}. Check the URL, then press F5."
        return McpStatusSnapshot(ready=False, url=self.adapter.url, error=error)

    def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Call an MCP tool through the existing adapter implementation."""

        return self.adapter.call_tool(name, arguments, meta=meta)


def _tool_info(raw: dict[str, Any]) -> ToolInfo:
    name = str(raw.get("name", ""))
    description = str(raw.get("description", ""))
    return ToolInfo(name=name, description=description, raw=raw)
