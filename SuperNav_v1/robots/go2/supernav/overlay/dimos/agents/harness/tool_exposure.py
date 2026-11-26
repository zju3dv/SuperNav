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

"""Native-agent tool exposure filtering for MCP tool catalogs."""

from __future__ import annotations

from typing import Any

from dimos.agents.runtime_policy import AgentRuntimePolicy, tool_exposed_by_policy


def native_agent_tool_visible(tool: dict[str, Any], policy: AgentRuntimePolicy) -> bool:
    """Return whether a fetched MCP tool should be exposed to the native agent."""

    name = str(tool.get("name") or "")
    if not name:
        return False
    return tool_exposed_by_policy(name, _tool_uses(tool), policy=policy)


def filter_native_agent_tools(
    tools: list[dict[str, Any]],
    *,
    policy: AgentRuntimePolicy,
) -> list[dict[str, Any]]:
    """Filter an MCP ``tools/list`` result for the native LangGraph client.

    The MCP server remains authoritative. This native-side filter keeps mocked
    or older servers from over-exposing tools in the model prompt.
    """

    return [tool for tool in tools if native_agent_tool_visible(tool, policy)]


def _tool_uses(tool: dict[str, Any]) -> list[str] | tuple[str, ...]:
    meta = tool.get("_meta") if isinstance(tool.get("_meta"), dict) else {}
    uses = meta.get("dimos/uses") if isinstance(meta, dict) else ()
    return uses if isinstance(uses, list | tuple) else ()
