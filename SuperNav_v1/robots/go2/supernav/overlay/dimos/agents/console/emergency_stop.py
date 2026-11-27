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

"""Emergency-stop routing for the operator console."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dimos.agents.console.mcp_client import ConsoleMcpClient
from dimos.agents.harness.human_input_dispatch import (
    HumanInputDispatcher,
    HumanInputDispatchResult,
)

STOP_MESSAGE = "/stop"
STOP_TOOL_NAME = "stop_movement"


@dataclass(frozen=True)
class EmergencyStopResult:
    """Visible result of the two stop routes used by ESC."""

    dispatcher_attempted: bool = False
    dispatcher_result: HumanInputDispatchResult | None = None
    dispatcher_error: str | None = None
    stop_movement_attempted: bool = False
    stop_movement_result: dict[str, Any] | None = None
    stop_movement_error: str | None = None

    @property
    def preemption_confirmed(self) -> bool:
        """Return whether MCP agent_send confirmed the preemption route."""

        return bool(
            self.dispatcher_result
            and self.dispatcher_result.route == "mcp_agent_send"
            and self.dispatcher_result.authority_preemption_supported
        )

    @property
    def stop_command_sent(self) -> bool:
        """Return whether explicit stop_movement returned without an exception."""

        return self.stop_movement_result is not None and self.stop_movement_error is None

    @property
    def requires_acknowledgement(self) -> bool:
        """Return whether the UI must stay stopped until operator acknowledgement."""

        return True

    @property
    def severity(self) -> str:
        """Return a coarse operator-facing severity label."""

        if self.preemption_confirmed and self.stop_command_sent:
            return "ok"
        if self.preemption_confirmed or self.stop_command_sent:
            return "warning"
        return "critical"


class EmergencyStopService:
    """Send `/stop` through HumanInputDispatcher, then explicit stop_movement."""

    def __init__(
        self,
        mcp_client: ConsoleMcpClient,
        *,
        dispatcher: HumanInputDispatcher | None = None,
    ) -> None:
        self._mcp_client = mcp_client
        self._dispatcher = dispatcher or HumanInputDispatcher(
            _NoopHumanInputPublisher(),
            mcp_client_factory=lambda: _ConsoleAgentSendClient(mcp_client),
        )

    def request_stop(self) -> EmergencyStopResult:
        """Request an ESC stop through both approved routes."""

        dispatcher_result = None
        dispatcher_error = None
        dispatcher_attempted = True
        try:
            dispatcher_result = self._dispatcher.send(STOP_MESSAGE)
        except Exception as exc:
            dispatcher_error = str(exc)

        stop_result = None
        stop_error = None
        stop_attempted = True
        try:
            stop_result = self._mcp_client.call_tool(STOP_TOOL_NAME, {})
        except Exception as exc:
            stop_error = str(exc)

        return EmergencyStopResult(
            dispatcher_attempted=dispatcher_attempted,
            dispatcher_result=dispatcher_result,
            dispatcher_error=dispatcher_error,
            stop_movement_attempted=stop_attempted,
            stop_movement_result=stop_result,
            stop_movement_error=stop_error,
        )


class _ConsoleAgentSendClient:
    """Adapter from ConsoleMcpClient to HumanInputDispatcher's small protocol."""

    def __init__(self, mcp_client: ConsoleMcpClient) -> None:
        self._mcp_client = mcp_client

    def call_tool_text(self, name: str, arguments: dict[str, Any] | None = None) -> str:
        """Call a tool and return first text content for dispatcher compatibility."""

        result = self._mcp_client.call_tool(name, arguments)
        content = result.get("content", [])
        if isinstance(content, list) and content and isinstance(content[0], dict):
            return str(content[0].get("text", ""))
        return ""


class _NoopHumanInputPublisher:
    def publish(self, message: str) -> None:
        """Compatibility fallback publisher; console does not own LCM publishing."""

        return None
