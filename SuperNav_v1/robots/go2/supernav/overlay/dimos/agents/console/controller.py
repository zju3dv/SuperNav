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

"""Mode-state controller for the operator console."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
import threading

from dimos.agents.console.agent_runner import (
    AgentRunArtifacts,
    AgentRunSnapshot,
    AgentRunStatus,
    AgentTaskRunner,
)
from dimos.agents.console.config import ConsoleConfig
from dimos.agents.console.emergency_stop import EmergencyStopResult, EmergencyStopService
from dimos.agents.console.manual_commands import ManualCommandResult, ManualCommandService
from dimos.agents.console.mcp_client import ConsoleMcpClient
from dimos.agents.console.startup import StartupService
from dimos.agents.console.state import ConsoleState, McpConnectionState, ToolInfo, UiMode

ManualServiceFactory = Callable[[ConsoleMcpClient, tuple[ToolInfo, ...]], ManualCommandService]
AgentRunnerFactory = Callable[[ConsoleConfig, str, str | None], AgentTaskRunner]
EmergencyStopFactory = Callable[[ConsoleMcpClient], EmergencyStopService]
_MANUAL_HISTORY_LIMIT = 20


class ConsoleController:
    """Own console state transitions without rendering or MCP protocol logic."""

    def __init__(
        self,
        config: ConsoleConfig,
        *,
        startup_service: StartupService | None = None,
        manual_service_factory: ManualServiceFactory = ManualCommandService,
        mcp_client_factory: Callable[[str | None], ConsoleMcpClient] = ConsoleMcpClient.from_url,
        telemetry_client_factory: Callable[[str | None], ConsoleMcpClient] | None = None,
        agent_runner_factory: AgentRunnerFactory | None = None,
        emergency_stop_factory: EmergencyStopFactory = EmergencyStopService,
    ) -> None:
        self.config = config
        self.startup_service = startup_service or StartupService()
        self._manual_service_factory = manual_service_factory
        self._mcp_client_factory = mcp_client_factory
        self._telemetry_client_factory = telemetry_client_factory or (
            lambda url: ConsoleMcpClient.from_url(url, timeout=2)
        )
        self._telemetry_client: ConsoleMcpClient | None = None
        self._agent_runner_factory = agent_runner_factory or _default_agent_runner_factory
        self._emergency_stop_factory = emergency_stop_factory
        self._state_lock = threading.RLock()
        self._manual_service: ManualCommandService | None = None
        self._agent_runner: AgentTaskRunner | None = None
        self._agent_handle = None
        self._emergency_stop_service: EmergencyStopService | None = None
        self._mcp_client: ConsoleMcpClient | None = None
        self.state = ConsoleState(config=config)

    def startup(self, *, confirm_start: Callable[[], bool] | None = None) -> ConsoleState:
        """Run attach/start readiness and update console state."""

        result = self.startup_service.attach_or_start(self.config, confirm_start=confirm_start)
        self.state = replace(
            self.state,
            ui_mode=UiMode.MANUAL if result.ready else UiMode.STARTUP,
            mcp_state=McpConnectionState.READY if result.ready else McpConnectionState.UNREACHABLE,
            mcp_url=result.mcp_url,
            run_id=result.run_id,
            run_log_dir=result.run_log_dir,
            tools=result.tools,
            startup_messages=result.messages,
            last_error=result.error,
            robot_connection="mcp-ready" if result.ready else "unknown",
            battery_percent=None,
            robot_status_error=None,
            pending_start_confirmation=result.pending_confirmation,
            launch_attempted=result.launch_attempted,
            launch_pid=result.launch_pid,
            last_manual_result=None,
            active_agent_run=None,
            emergency_stop_result=None,
            last_pixel_goal_result=None,
        )
        self._manual_service = None
        self._agent_runner = None
        self._agent_handle = None
        self._emergency_stop_service = None
        self._mcp_client = None
        self._telemetry_client = None
        if result.ready:
            mcp_client = self._mcp_client_factory(result.mcp_url)
            self._mcp_client = mcp_client
            self._manual_service = self._manual_service_factory(
                mcp_client,
                result.tools,
            )
            self._agent_runner = self._agent_runner_factory(
                self.config,
                result.mcp_url,
                result.run_log_dir,
            )
            self._emergency_stop_service = self._emergency_stop_factory(mcp_client)
            if "go2_robot_status" in self.state.tool_names:
                self._telemetry_client = self._telemetry_client_factory(result.mcp_url)
        return self.state

    def refresh_robot_status(self) -> ConsoleState:
        """Update telemetry without changing the current task or stop state."""
        client = self._telemetry_client
        if client is None:
            return self.state
        snapshot = client.robot_status_snapshot()
        with self._state_lock:
            # Discard an old response if F5 established a different connection.
            if client is not self._telemetry_client:
                return self.state
            previous = self.state.robot_connection
            connection = "status-ok" if snapshot.available else (
                "stale" if previous in {"status-ok", "stale"} else "unavailable"
            )
            self.state = replace(
                self.state,
                robot_connection=connection,
                battery_percent=snapshot.battery_percent if snapshot.available else None,
                robot_status_error=snapshot.error,
            )
        return self.state

    def confirm_start(self) -> ConsoleState:
        """Confirm an operator-visible DimOS start request."""

        return self.startup(confirm_start=lambda: True)

    def retry_connection(self) -> ConsoleState:
        """Re-run the existing attach-first readiness probe."""

        if self._stop_requires_acknowledgement():
            self.state = replace(
                self.state,
                last_error="acknowledge emergency stop with Enter before reconnecting",
            )
            return self.state
        return self.startup()

    def submit_manual_command(self, text: str) -> ConsoleState:
        """Execute one discrete manual command and store its displayed result."""

        if self._manual_service is None:
            result = ManualCommandResult.error_result(
                "manual commands require an MCP-ready console session"
            )
        else:
            result = self._manual_service.execute(text)
        with self._state_lock:
            current = self.state
            stopped = current.ui_mode is UiMode.STOPPED
            self.state = replace(
                current,
                ui_mode=UiMode.STOPPED if stopped else UiMode.MANUAL,
                mcp_state=result.mcp_state or current.mcp_state,
                robot_connection="unknown" if result.mcp_state else current.robot_connection,
                last_manual_result=result,
                manual_history=(*current.manual_history, result)[-_MANUAL_HISTORY_LIMIT:],
                last_error=current.last_error
                if stopped
                else (result.error if not result.success else None),
            )
        return self.state

    def submit_agent_task(self, text: str) -> ConsoleState:
        """Submit one complete agent task objective in the background."""

        if self._agent_handle is not None and self._agent_handle.snapshot().running:
            self.state = replace(self.state, last_error="agent task already running")
            return self.state
        if self._agent_runner is None:
            snapshot = _failed_agent_snapshot(
                self.config,
                text,
                "agent tasks require an MCP-ready console session",
            )
            self.state = replace(
                self.state,
                ui_mode=UiMode.AGENT,
                active_agent_run=snapshot,
                last_error=snapshot.error,
            )
            return self.state
        try:
            self._agent_handle = self._agent_runner.submit(text)
            snapshot = self._agent_handle.snapshot()
        except Exception as exc:
            snapshot = _failed_agent_snapshot(self.config, text, str(exc))
        self.state = replace(
            self.state,
            ui_mode=UiMode.AGENT,
            active_agent_run=snapshot,
            last_error=snapshot.error,
        )
        return self.state

    def poll_agent_task(self) -> ConsoleState:
        """Refresh active agent task status and tailed log events."""

        if self._agent_handle is None:
            return self.state
        snapshot = self._agent_handle.snapshot()
        self.state = replace(
            self.state,
            active_agent_run=snapshot,
            last_error=snapshot.error,
        )
        return self.state

    def request_emergency_stop(self) -> ConsoleState:
        """Handle ESC by preempting agent input and explicitly stopping movement."""

        agent_stop_requested = False
        if (
            self._agent_handle is not None
            and self._agent_handle.snapshot().running
            and hasattr(self._agent_handle, "request_stop")
        ):
            self._agent_handle.request_stop()
            agent_stop_requested = True
        pixel_goal_stop_result: str | None = None
        pixel_goal_stop_error = False
        pixel_goal_mode = self.state.ui_mode is UiMode.PIXEL_GOAL
        agent_pixel_goal = (
            self.state.ui_mode is UiMode.AGENT and "pixelgoal_stop" in self.state.tool_names
        )
        if pixel_goal_mode or agent_pixel_goal:
            pixel_goal_stop_result, pixel_goal_stop_error = self._call_pixel_goal_tool(
                "pixelgoal_stop"
            )
        if self._emergency_stop_service is None:
            result = EmergencyStopResult(
                dispatcher_attempted=False,
                dispatcher_error="emergency stop requires an MCP-ready console session",
                stop_movement_attempted=False,
                stop_movement_error="emergency stop requires an MCP-ready console session",
            )
        else:
            result = self._emergency_stop_service.request_stop()
        error = None
        if not result.preemption_confirmed and not result.stop_command_sent:
            error = "emergency stop was not confirmed by MCP"
        if pixel_goal_stop_error:
            error = (
                pixel_goal_stop_result if error is None else f"{pixel_goal_stop_result}; {error}"
            )
        with self._state_lock:
            if self._agent_handle is not None and agent_stop_requested:
                self._agent_handle.add_event(
                    "console",
                    "Agent exit requested: sent /stop route plus explicit stop_movement",
                )
                self.state = replace(self.state, active_agent_run=self._agent_handle.snapshot())
            self.state = replace(
                self.state,
                ui_mode=UiMode.STOPPED,
                emergency_stop_result=result,
                last_pixel_goal_result=pixel_goal_stop_result or self.state.last_pixel_goal_result,
                last_error=error,
            )
        return self.state

    def acknowledge_emergency_stop(self) -> ConsoleState:
        """Return to manual mode after the operator acknowledges a stop result."""

        self.state = replace(self.state, ui_mode=UiMode.MANUAL, last_error=None)
        return self.state

    def select_manual_mode(self) -> ConsoleState:
        """Switch to manual mode unless an emergency stop still needs acknowledgement."""

        if self._stop_requires_acknowledgement():
            self.state = replace(
                self.state,
                last_error="acknowledge emergency stop with Enter before resuming",
            )
            return self.state
        self.state = replace(self.state, ui_mode=UiMode.MANUAL, last_error=None)
        return self.state

    def select_agent_mode(self) -> ConsoleState:
        """Switch to agent mode unless an emergency stop still needs acknowledgement."""

        if self._stop_requires_acknowledgement():
            self.state = replace(
                self.state,
                last_error="acknowledge emergency stop with Enter before resuming",
            )
            return self.state
        self.state = replace(self.state, ui_mode=UiMode.AGENT, last_error=None)
        return self.state


    def select_pixel_goal_mode(self) -> ConsoleState:
        """Switch to the four-way Pixel Goal operator mode."""

        if self._stop_requires_acknowledgement():
            self.state = replace(
                self.state,
                last_error="acknowledge emergency stop with Enter before resuming",
            )
            return self.state
        self.state = replace(self.state, ui_mode=UiMode.PIXEL_GOAL, last_error=None)
        return self.state


    def submit_pixel_goal_command(self, text: str) -> ConsoleState:
        """Run one four-way Pixel Goal session command."""

        command = text.strip().casefold()
        if command == "stop":
            return self.request_emergency_stop()
        tool_name = {
            "refresh": "refresh_rerun_images",
            "reset": "reset_pixel_goal_map",
        }.get(command)
        if tool_name is None:
            result = "Unknown Pixel Goal command. Use: refresh, reset, or stop."
            is_error = True
        else:
            result, is_error = self._call_pixel_goal_tool(tool_name)
        with self._state_lock:
            current = self.state
            stopped = current.ui_mode is UiMode.STOPPED
            self.state = replace(
                current,
                ui_mode=UiMode.STOPPED if stopped else UiMode.PIXEL_GOAL,
                last_pixel_goal_result=current.last_pixel_goal_result if stopped else result,
                last_error=current.last_error if stopped else (result if is_error else None),
            )
        return self.state

    def _call_pixel_goal_tool(self, tool_name: str) -> tuple[str, bool]:
        if self._mcp_client is None:
            return "Pixel Goal requires an MCP-ready console session.", True
        if tool_name not in self.state.tool_names:
            return f"Pixel Goal tool unavailable: {tool_name}.", True
        try:
            raw = self._mcp_client.call_tool(tool_name, {})
        except Exception as exc:  # noqa: BLE001 — surface to the operator
            return f"{tool_name} failed: {exc}", True
        return _mcp_result_text(raw), bool(raw.get("isError"))


    def _stop_requires_acknowledgement(self) -> bool:
        result = self.state.emergency_stop_result
        return (
            self.state.ui_mode is UiMode.STOPPED
            and result is not None
            and result.requires_acknowledgement
        )


def _default_agent_runner_factory(
    config: ConsoleConfig,
    mcp_url: str,
    run_log_dir: str | None,
) -> AgentTaskRunner:
    return AgentTaskRunner(config=config, mcp_url=mcp_url, run_log_dir=run_log_dir)


def _mcp_result_text(raw: dict[str, object]) -> str:
    content = raw.get("content")
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                return str(item["text"])
    return "Pixel Goal command accepted."


def _failed_agent_snapshot(
    config: ConsoleConfig,
    task: str,
    error: str,
) -> AgentRunSnapshot:
    root = (config.run_root or Path.cwd() / "runs" / "operator-console").expanduser()
    return AgentRunSnapshot(
        task=task,
        status=AgentRunStatus.FAILED,
        mode=config.resolved_agent_mode.value,
        tool_profile=config.tool_profile,
        run_root=root,
        artifacts=AgentRunArtifacts(run_root=root),
        returncode=-1,
        reason=error,
        ok=False,
        error=error,
        stop_requested=False,
    )
