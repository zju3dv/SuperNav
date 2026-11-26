"""Telemetry rendering and polling without a robot, movement, or real MCP."""

import asyncio
from dataclasses import replace
import json
import threading
from types import SimpleNamespace

import pytest
import requests

from dimos.agents.console.config import ConsoleConfig
from dimos.agents.console.controller import ConsoleController
from dimos.agents.console.emergency_stop import EmergencyStopResult
from dimos.agents.console.mcp_client import ConsoleMcpClient, RobotStatusSnapshot
from dimos.agents.console.render import format_context_panel, format_top_bar
from dimos.agents.console.startup import StartupResult
from dimos.agents.console.state import ToolInfo, UiMode


def response(soc=63.0, available=True):
    return {"content": [{"type": "text", "text": json.dumps({
        "success": True, "metadata": {"robot_status": {
            "available": available, "battery": {"soc_percent": soc},
        }},
    })}]}


class Adapter:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def call_tool(self, name, arguments=None, **kwargs):
        self.calls.append((name, arguments))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


@pytest.mark.parametrize("soc, expected", [(63.0, 63), (0, 0), (100, 100),
    (None, None), (float("nan"), None), (101, None), (-1, None), (True, None)])
def test_status_reads_lowstate_battery_without_motion(soc, expected):
    adapter = Adapter(response(soc))
    status = ConsoleMcpClient(adapter).robot_status_snapshot()
    assert status.available and status.battery_percent == expected
    assert adapter.calls == [("go2_robot_status", {})]


@pytest.mark.parametrize("result", [response(available=False), {"isError": True},
    {"content": [{"type": "text", "text": "not json"}]},
    {"content": [{"type": "text", "text": '{"success":true,"metadata":null}'}]},
    requests.Timeout("offline")])
def test_unavailable_or_malformed_telemetry_is_not_connected(result):
    status = ConsoleMcpClient(Adapter(result)).robot_status_snapshot()
    assert not status.available and status.battery_percent is None


def controller_with_telemetry(client):
    result = StartupResult(True, "http://127.0.0.1:9990/mcp",
                           tools=(ToolInfo("go2_robot_status"),))
    controller = ConsoleController(
        ConsoleConfig(),
        startup_service=SimpleNamespace(attach_or_start=lambda *a, **kw: result),
        mcp_client_factory=lambda url: ConsoleMcpClient(Adapter({})),
        telemetry_client_factory=lambda url: client,
        agent_runner_factory=lambda *args: object(),
    )
    return controller


def test_console_refreshes_clears_stale_battery_and_recovers():
    adapter = Adapter(response())
    controller = controller_with_telemetry(ConsoleMcpClient(adapter))
    state = controller.startup()
    assert state.robot_connection == "mcp-ready" and state.battery_percent is None
    controller.select_agent_mode()
    state = controller.refresh_robot_status()
    assert state.ui_mode is UiMode.AGENT
    for render in (format_top_bar, format_context_panel):
        assert "STATUS OK" in render(state) and "63%" in render(state)
    adapter.result = requests.Timeout()
    state = controller.refresh_robot_status()
    assert state.robot_connection == "stale" and state.battery_percent is None
    assert "STALE" in format_top_bar(state) and "63%" not in format_top_bar(state)
    adapter.result = response(62)
    state = controller.refresh_robot_status()
    assert state.robot_connection == "status-ok" and state.battery_percent == 62


def test_old_poll_response_cannot_overwrite_reconnected_state():
    controller = controller_with_telemetry(None)
    def poll():
        controller._telemetry_client = None
        controller.state = replace(controller.state, robot_connection="unknown")
        return RobotStatusSnapshot(True, 63)
    controller._telemetry_client = SimpleNamespace(robot_status_snapshot=poll)
    state = controller.refresh_robot_status()
    assert state.robot_connection == "unknown" and state.battery_percent is None


def test_ui_remains_responsive_and_stop_state_survives_slow_poll():
    from dimos.agents.console.app import OperateApp

    entered, release = threading.Event(), threading.Event()
    def poll():
        entered.set()
        release.wait(5)
        return RobotStatusSnapshot(True, 61)
    controller = controller_with_telemetry(SimpleNamespace(robot_status_snapshot=poll))
    async def exercise():
        app = OperateApp(ConsoleConfig())
        app.controller = controller
        async with app.run_test() as pilot:
            try:
                assert await asyncio.to_thread(entered.wait, 2)
                controller._emergency_stop_service = SimpleNamespace(
                    request_stop=lambda: EmergencyStopResult(stop_movement_result={})
                )
                await pilot.press("f2")
                assert controller.state.ui_mode is UiMode.AGENT
                await pilot.press("escape")
                assert controller.state.ui_mode is UiMode.STOPPED
                release.set()
                for _ in range(30):
                    await pilot.pause(0.02)
                    if not app._telemetry_poll_running:
                        break
                assert controller.state.ui_mode is UiMode.STOPPED
                assert controller.state.battery_percent == 61
            finally:
                release.set()
    asyncio.run(exercise())
