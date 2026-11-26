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

"""MCP tool catalog discovery for the native DimOS agent."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import time
from typing import Any

import httpx
from langchain_core.tools import StructuredTool

from dimos.agents.harness.tool_exposure import filter_native_agent_tools
from dimos.agents.runtime_policy import AgentRuntimePolicy

McpRequest = Callable[[str, dict[str, Any] | None], dict[str, Any]]
ToolWrapper = Callable[[dict[str, Any]], StructuredTool]


@dataclass(frozen=True)
class NativeToolCatalog:
    """Filtered MCP tool catalog used by the native LangGraph agent."""

    raw_tools: list[dict[str, Any]]
    visible_tools: list[dict[str, Any]]
    langchain_tools: list[StructuredTool]
    registry: dict[str, dict[str, Any]]


def fetch_mcp_tools_list(
    mcp_request: McpRequest,
    *,
    timeout: float = 60.0,
    interval: float = 1.0,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any] | None:
    """Initialize an MCP connection and return ``tools/list`` with retry."""

    deadline = monotonic() + timeout
    while True:
        try:
            mcp_request("initialize", None)
            break
        except (httpx.ConnectError, httpx.RemoteProtocolError):
            if monotonic() >= deadline:
                return None
            sleep(interval)
    return mcp_request("tools/list", None)


def build_native_tool_catalog(
    raw_tools: list[dict[str, Any]],
    *,
    policy: AgentRuntimePolicy,
    wrap_tool: ToolWrapper,
) -> NativeToolCatalog:
    """Build the native-agent visible tool catalog from raw MCP tools."""

    registry = {
        str(tool["name"]): tool for tool in raw_tools if isinstance(tool, dict) and "name" in tool
    }
    visible_tools = filter_native_agent_tools(raw_tools, policy=policy)
    langchain_tools = [wrap_tool(tool) for tool in visible_tools]
    return NativeToolCatalog(
        raw_tools=raw_tools,
        visible_tools=visible_tools,
        langchain_tools=langchain_tools,
        registry=registry,
    )
