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

"""Plain state objects for the operator console."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from dimos.agents.console.config import ConsoleConfig

if TYPE_CHECKING:
    from dimos.agents.console.agent_runner import AgentRunSnapshot
    from dimos.agents.console.emergency_stop import EmergencyStopResult
    from dimos.agents.console.manual_commands import ManualCommandResult


class UiMode(str, Enum):
    """Top-level console UI mode."""

    STARTUP = "STARTUP"
    MANUAL = "MANUAL"
    AGENT = "AGENT"
    PIXEL_GOAL = "PIXEL_GOAL"
    STOPPED = "STOPPED"


class McpConnectionState(str, Enum):
    """Connection state rendered in the status bar."""

    UNKNOWN = "UNKNOWN"
    READY = "READY"
    UNREACHABLE = "UNREACHABLE"
    ERROR = "ERROR"


@dataclass(frozen=True)
class ToolInfo:
    """Subset of MCP tool metadata that the console needs to display."""

    name: str
    description: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ConsoleState:
    """Snapshot of the operator console state."""

    config: ConsoleConfig
    ui_mode: UiMode = UiMode.STARTUP
    mcp_state: McpConnectionState = McpConnectionState.UNKNOWN
    mcp_url: str | None = None
    run_id: str | None = None
    run_log_dir: str | None = None
    tools: tuple[ToolInfo, ...] = ()
    startup_messages: tuple[str, ...] = ()
    last_error: str | None = None
    robot_connection: str = "unknown"
    battery_percent: int | None = None
    robot_status_error: str | None = None
    pending_start_confirmation: bool = False
    launch_attempted: bool = False
    launch_pid: int | None = None
    last_manual_result: ManualCommandResult | None = None
    manual_history: tuple[ManualCommandResult, ...] = ()
    active_agent_run: AgentRunSnapshot | None = None
    emergency_stop_result: EmergencyStopResult | None = None
    last_pixel_goal_result: str | None = None

    @property
    def tool_names(self) -> tuple[str, ...]:
        """Return available tool names in MCP order."""

        return tuple(tool.name for tool in self.tools)
