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

"""LangChain tool wrappers for the native DimOS MCP agent."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain_core.tools import StructuredTool

from dimos.agents.harness.tool_contracts import (
    ModelToolResult,
    model_visible_tool_result,
)

McpToolCaller = Callable[[str, dict[str, Any]], dict[str, Any]]
SpeechRefusalCallback = Callable[[dict[str, Any]], str | None]


def native_mcp_tool_to_langchain(
    mcp_tool: dict[str, Any],
    *,
    call_mcp_tool: McpToolCaller,
    speech_refusal: SpeechRefusalCallback | None = None,
) -> StructuredTool:
    """Wrap an MCP tool schema as a LangChain tool for the native agent."""

    name = str(mcp_tool["name"])
    description = str(mcp_tool.get("description", ""))
    input_schema = mcp_tool.get("inputSchema", {"type": "object", "properties": {}})

    def call_tool(**kwargs: Any) -> ModelToolResult:
        if name == "speak" and speech_refusal is not None:
            refusal = speech_refusal(kwargs)
            if refusal is not None:
                return refusal

        result = call_mcp_tool(name, kwargs)
        return model_visible_tool_result(name, result)

    return StructuredTool(
        name=name,
        description=description,
        func=call_tool,
        args_schema=input_schema,
    )
