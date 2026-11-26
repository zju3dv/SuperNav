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

"""Native-agent MCP tool dispatch metadata and authority accounting."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from dimos.agents.harness.task_state import TaskRun
from dimos.agents.harness.tool_call_context import NativeToolCallContext

McpRequestCallable = Callable[[str, dict[str, Any] | None], dict[str, Any]]


@dataclass(frozen=True)
class NativeToolDispatchState:
    """Native turn state needed to dispatch one MCP tool call."""

    movement_confirmed: bool = False
    movement_authorization: str | None = None
    user_message: str = ""
    active_task: TaskRun | None = None
    client_id: str = "dimos-agent"
    speech_authorized: bool = False


@dataclass(frozen=True)
class NativeToolDispatchResult:
    """MCP tool-call result plus updated native authority state."""

    result: dict[str, Any]
    context: NativeToolCallContext
    movement_confirmed: bool
    movement_authorization: str | None


def dispatch_native_mcp_tool(
    tool_name: str,
    arguments: dict[str, Any],
    *,
    state: NativeToolDispatchState,
    mcp_request: McpRequestCallable,
) -> NativeToolDispatchResult:
    """Dispatch one MCP tool call with DimOS metadata and authority accounting."""

    context = NativeToolCallContext.create(
        movement_confirmed=state.movement_confirmed,
        movement_authorization=state.movement_authorization,
        user_message=state.user_message,
        task=state.active_task,
        client_id=state.client_id,
        speech_authorized=state.speech_authorized,
    )
    result = mcp_request("tools/call", context.params(tool_name, arguments))
    movement_confirmed = state.movement_confirmed
    movement_authorization = state.movement_authorization
    if context.consumes_movement_authority(tool_name):
        movement_confirmed = False
        movement_authorization = None
    return NativeToolDispatchResult(
        result=result,
        context=context,
        movement_confirmed=movement_confirmed,
        movement_authorization=movement_authorization,
    )
