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

"""Textual shell for ``dimos operate``."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable

from textual import events, work
from textual.app import App, ComposeResult, SystemCommand
from textual.binding import Binding
from textual.containers import Horizontal
from textual.screen import Screen
from textual.widgets import Input, Static

from dimos.agents.console.config import ConsoleConfig
from dimos.agents.console.controller import ConsoleController
from dimos.agents.console.render import (
    format_agent_panel,
    format_context_panel,
    format_emergency_stop_panel,
    format_footer_note,
    format_manual_panel,
    format_manual_pending,
    format_operator_help,
    format_pixel_goal_panel,
    format_startup_panel,
    format_top_bar,
)
from dimos.agents.console.state import UiMode


class OperateApp(App[None]):
    """Operator console TUI."""

    CSS = """
    Screen {
        layout: vertical;
        background: #0a1016;
        color: #e6e1d7;
    }
    #topbar {
        height: 4;
        padding: 0 2;
        background: #111b24;
        border-bottom: tall #304652;
        text-wrap: nowrap;
        text-overflow: ellipsis;
    }
    #topbar.mcp-ready { border-bottom: tall #4fb596; }
    #topbar.mcp-unreachable, #topbar.mcp-error { border-bottom: tall #df746d; }
    #topbar.stopped { background: #291a18; border-bottom: tall #df746d; }
    #workspace {
        height: 1fr;
        padding: 1 2 0 2;
        background: #0a1016;
    }
    #body {
        width: 1fr;
        height: 1fr;
        padding: 1 2;
        background: #111922;
        overflow-y: auto;
        scrollbar-color: #496674;
        scrollbar-background: #111922;
    }
    #body.warning { border-left: thick #d3a354; }
    #body.error, #body.stopped { border-left: thick #df746d; }
    #context {
        width: 34;
        height: 1fr;
        margin-left: 1;
        padding: 1 2;
        background: #0e171e;
        color: #aebdc4;
        border-left: thick #304652;
        overflow-y: auto;
    }
    .compact #workspace { padding: 0 1; }
    .compact #context { display: none; }
    #manual-input {
        height: 3;
        margin: 0 2;
        padding: 0 1;
        background: #16232c;
        border: none;
        border-left: thick #496674;
    }
    #manual-input:focus { border-left: thick #d3a354; }
    #manual-input.busy { border-left: thick #d3a354; }
    #manual-input.stopped { border-left: thick #df746d; }
    .compact #manual-input { margin: 0 1; }
    #footer-note {
        height: 2;
        padding: 0 2;
        background: #0a1016;
        color: #8799a2;
        text-wrap: nowrap;
        text-overflow: ellipsis;
    }
    """

    BINDINGS = [
        Binding("f1", "manual_mode", "Manual"),
        Binding("f2", "agent_mode", "Agent"),
        Binding("f7", "pixel_goal_mode", "Pixel Goal"),
        Binding("f3", "toggle_operator_help", "Help"),
        Binding("ctrl+c", "quit", "Quit", show=False, priority=True),
        Binding("ctrl+q", "quit", "Quit"),
        Binding("f5", "retry_connection", "Retry MCP"),
        Binding("pageup", "scroll_output_up", "", show=False),
        Binding("pagedown", "scroll_output_down", "", show=False),
        Binding("f8", "emergency_stop", "Exit Agent + Stop"),
        Binding("escape", "emergency_stop", "Emergency stop"),
    ]

    def __init__(self, config: ConsoleConfig) -> None:
        super().__init__()
        self._showing_operator_help = False
        self._telemetry_poll_running = False
        self.manual_command_running = False
        self.pixel_goal_command_running = False
        self._manual_command_text = ""
        self.controller = ConsoleController(config)
        self.topbar = Static("", id="topbar", markup=False)
        self.body = Static("Starting operator console...", id="body", markup=False)
        self.command_input = Input(placeholder="manual> forward 0.5", id="manual-input")
        self.context = Static("", id="context", markup=False)
        self.footer_note = Static(
            format_footer_note(self.controller.state), id="footer-note", markup=False
        )

    def get_system_commands(self, screen: Screen) -> Iterable[SystemCommand]:
        """Add operator actions to Textual's native command palette."""

        yield SystemCommand(
            "Show operator cheatsheet",
            "Manual syntax, shortcuts, safety, and reconnect steps",
            self.action_toggle_operator_help,
        )
        yield SystemCommand(
            "Switch to Manual mode", "Enter discrete commands", self.action_manual_mode
        )
        yield SystemCommand(
            "Switch to Agent mode", "Submit one agent objective", self.action_agent_mode
        )
        yield SystemCommand(
            "Switch to four-way Pixel Goal mode",
            "Refresh frozen Rerun images or reset/stop Pixel Goal",
            self.action_pixel_goal_mode,
        )
        yield SystemCommand(
            "Retry MCP connection",
            "Re-run attach readiness or confirm a pending DimOS start",
            self.action_retry_connection,
        )
        yield SystemCommand(
            "Emergency exit Agent + stop",
            "Terminate the Codex worker, then send /stop and stop_movement through MCP",
            self.action_emergency_stop,
        )
        yield SystemCommand(
            "Scroll output to top",
            "Show the start of the current panel",
            self.action_scroll_output_top,
        )
        yield SystemCommand(
            "Jump to latest output",
            "Show the end of the current panel",
            self.action_scroll_output_latest,
        )
        yield from super().get_system_commands(screen)

    def compose(self) -> ComposeResult:
        yield self.topbar
        with Horizontal(id="workspace"):
            yield self.body
            yield self.context
        yield self.command_input
        yield self.footer_note

    async def on_mount(self) -> None:
        self._refresh()
        self._sync_layout(self.size.width)
        await asyncio.to_thread(self.controller.startup)
        self._refresh()
        self.command_input.focus()
        self.set_interval(0.5, self._poll_agent)
        self.set_interval(2.0, self._poll_robot_status)
        self._poll_robot_status()

    def on_resize(self, event: events.Resize) -> None:
        self._sync_layout(event.size.width)

    async def action_emergency_stop(self) -> None:
        self._showing_operator_help = False
        await asyncio.to_thread(self.controller.request_emergency_stop)
        self._refresh()

    async def action_quit(self) -> None:
        active = self.controller.state.active_agent_run
        if (
            self.manual_command_running
            or self.pixel_goal_command_running
            or (active is not None and active.running)
        ):
            await self.action_emergency_stop()
        self.exit()

    def action_manual_mode(self) -> None:
        if self.manual_command_running:
            return
        self._showing_operator_help = False
        self.controller.select_manual_mode()
        self._refresh()

    def action_agent_mode(self) -> None:
        if self.manual_command_running:
            return
        self._showing_operator_help = False
        self.controller.select_agent_mode()
        self._refresh()


    def action_pixel_goal_mode(self) -> None:
        if self.manual_command_running or self.pixel_goal_command_running:
            return
        self._showing_operator_help = False
        self.controller.select_pixel_goal_mode()
        self._refresh()
        if self.controller.state.ui_mode is UiMode.PIXEL_GOAL:
            self._start_pixel_goal_command("refresh")

    def action_toggle_operator_help(self) -> None:
        if self.controller.state.ui_mode is UiMode.STOPPED:
            self._showing_operator_help = False
            self._refresh()
            return
        self._showing_operator_help = not self._showing_operator_help
        self._refresh()

    async def action_retry_connection(self) -> None:
        if self.manual_command_running:
            self.body.update("A manual command is still active. Press ESC before reconnecting.")
            return
        active = self.controller.state.active_agent_run
        if active is not None and active.running:
            self.body.update("Agent task is running. Press ESC before reconnecting MCP.")
            return
        self._showing_operator_help = False
        if self.controller.state.pending_start_confirmation:
            await asyncio.to_thread(self.controller.confirm_start)
        else:
            await asyncio.to_thread(self.controller.retry_connection)
        self._refresh()
        if not self.command_input.disabled:
            self.command_input.focus()

    def action_scroll_output_up(self) -> None:
        self.body.scroll_to(
            y=self.body.scroll_y - self.body.scrollable_content_region.height,
            animate=False,
            force=True,
            immediate=True,
        )

    def action_scroll_output_down(self) -> None:
        self.body.scroll_to(
            y=self.body.scroll_y + self.body.scrollable_content_region.height,
            animate=False,
            force=True,
            immediate=True,
        )

    def action_scroll_output_top(self) -> None:
        self.body.scroll_to(y=0, animate=False, force=True, immediate=True)

    def action_scroll_output_latest(self) -> None:
        self.body.scroll_to(y=self.body.max_scroll_y, animate=False, force=True, immediate=True)

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        self._showing_operator_help = False
        command = event.value.strip()
        self.command_input.value = ""
        if self.controller.state.ui_mode is UiMode.STOPPED:
            self.controller.acknowledge_emergency_stop()
            self._refresh()
            return
        if not command:
            return
        if self.controller.state.ui_mode is UiMode.AGENT:
            active = self.controller.state.active_agent_run
            if active is not None and active.running:
                self.body.update("Agent task already running; wait for completion or press ESC.")
                return
            await asyncio.to_thread(self.controller.submit_agent_task, command)
        elif self.controller.state.ui_mode is UiMode.PIXEL_GOAL:
            if command == "stop":
                await asyncio.to_thread(self.controller.submit_pixel_goal_command, command)
            else:
                self._start_pixel_goal_command(command)
                return
        elif self.controller.state.ui_mode is UiMode.MANUAL:
            if self.manual_command_running:
                return
            self.manual_command_running = True
            self._manual_command_text = command
            self._refresh()
            self._submit_manual_command(command)
            return
        self._refresh()

    @work(exclusive=True, group="manual-command")
    async def _submit_manual_command(self, command: str) -> None:
        try:
            await asyncio.to_thread(self.controller.submit_manual_command, command)
        finally:
            self.manual_command_running = False
            self._manual_command_text = ""
            self._refresh()
            if self.controller.state.ui_mode is not UiMode.STOPPED:
                self.command_input.focus()

    def _start_pixel_goal_command(self, command: str) -> None:
        if self.pixel_goal_command_running:
            return
        self.pixel_goal_command_running = True
        self._refresh()
        self._submit_pixel_goal_command(command)

    @work(exclusive=True, group="pixel-goal-command")
    async def _submit_pixel_goal_command(self, command: str) -> None:
        try:
            await asyncio.to_thread(self.controller.submit_pixel_goal_command, command)
        finally:
            self.pixel_goal_command_running = False
            self._refresh()
            if self.controller.state.ui_mode is not UiMode.STOPPED:
                self.command_input.focus()

    def _poll_agent(self) -> None:
        active = self.controller.state.active_agent_run
        if active is None or not active.running:
            return
        self.controller.poll_agent_task()
        self._refresh()

    def _poll_robot_status(self) -> None:
        if not self._telemetry_poll_running:
            self._telemetry_poll_running = True
            self._read_robot_status()

    @work(group="robot-status")
    async def _read_robot_status(self) -> None:
        try:
            await asyncio.to_thread(self.controller.refresh_robot_status)
            self._refresh()
        finally:
            self._telemetry_poll_running = False

    def _refresh(self) -> None:
        state = self.controller.state
        classes = [f"mcp-{state.mcp_state.value.lower()}"]
        if state.ui_mode is UiMode.STOPPED:
            classes.append("stopped")
        self.topbar.set_classes(classes)
        body_class = ""
        if state.ui_mode is UiMode.STOPPED:
            body_class = "stopped"
        elif state.ui_mode is UiMode.MANUAL and state.last_manual_result is not None:
            status = state.last_manual_result.display_status
            if status in {"NO PROGRESS", "TIMED OUT"}:
                body_class = "warning"
            elif not state.last_manual_result.success:
                body_class = "error"
        self.body.set_classes(body_class)
        input_class = "stopped" if state.ui_mode is UiMode.STOPPED else ""
        if self.manual_command_running and state.ui_mode is not UiMode.STOPPED:
            input_class = "busy"
        self.command_input.set_classes(input_class)
        self.topbar.update(format_top_bar(state))
        self.context.update(format_context_panel(state))
        active = state.active_agent_run
        agent_running = active is not None and active.running
        self.command_input.disabled = state.ui_mode is not UiMode.STOPPED and (
            self.manual_command_running
            or self.pixel_goal_command_running
            or agent_running
            or state.ui_mode is UiMode.STARTUP
        )
        if self._showing_operator_help:
            self.command_input.placeholder = "F3 closes help"
            self.body.update(format_operator_help(state))
        elif state.ui_mode is UiMode.MANUAL:
            if self.manual_command_running:
                self.command_input.placeholder = "command in progress - ESC stops motion"
                self.body.update(format_manual_pending(state, self._manual_command_text))
            else:
                self.command_input.placeholder = "manual> forward 0.5"
                self.body.update(format_manual_panel(state))
        elif state.ui_mode is UiMode.AGENT:
            self.command_input.placeholder = (
                "Agent task running - F8 exits Agent and stops motion"
                if agent_running
                else "agent task> find the door and return"
            )
            self.body.update(format_agent_panel(state))
        elif state.ui_mode is UiMode.PIXEL_GOAL:
            self.command_input.placeholder = (
                "Pixel Goal command in progress - ESC stops motion"
                if self.pixel_goal_command_running
                else "pixel> refresh | reset | stop"
            )
            self.body.update(format_pixel_goal_panel(state))
        elif state.ui_mode is UiMode.STOPPED:
            self.command_input.placeholder = "ESC stop requested"
            self.body.update(format_emergency_stop_panel(state))
        else:
            self.command_input.placeholder = "startup"
            self.body.update(format_startup_panel(state))
        self.footer_note.update(format_footer_note(state, showing_help=self._showing_operator_help))

    def _sync_layout(self, width: int) -> None:
        compact = width < 100
        self.screen.set_class(compact, "compact")
        self.context.display = not compact


def run_console(config: ConsoleConfig) -> None:
    """Run the Textual operator console."""

    OperateApp(config).run()
