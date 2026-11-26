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

"""Pure render helpers for operator-console state."""

from __future__ import annotations

import json

from dimos.agents.console.log_stream import THINKING_PREFIX
from dimos.agents.console.state import ConsoleState, McpConnectionState


def _robot_status_label(state: ConsoleState) -> str:
    return {
        "mcp-ready": "UNVERIFIED", "status-ok": "STATUS OK",
        "unavailable": "UNAVAILABLE", "stale": "STALE",
    }.get(state.robot_connection, "UNKNOWN")


def format_top_bar(state: ConsoleState) -> str:
    """Return a compact operator status header."""

    config = state.config
    battery = "--" if state.battery_percent is None else f"{state.battery_percent}%"
    robot = _robot_status_label(state)
    activity = "IDLE"
    if state.active_agent_run is not None:
        activity = f"AGENT {state.active_agent_run.status.value}"
    elif state.last_manual_result is not None:
        result = state.last_manual_result
        activity = f"LAST {result.tool_name or 'manual'} {result.display_status}"
    stop_label = "ESC  EMERGENCY STOP"
    if state.emergency_stop_result is not None and state.ui_mode.value == "STOPPED":
        confirmation = (
            "PREEMPTION CONFIRMED"
            if state.emergency_stop_result.preemption_confirmed
            else "PREEMPTION UNCONFIRMED"
        )
        stop_label = f"{confirmation} | ENTER  ACKNOWLEDGE"
    return "\n".join(
        [
            (
                f"DIMOS / OPERATE  |  {state.ui_mode.value}  |  MCP {state.mcp_state.value}  |  "
                f"ROBOT {robot}  |  BATTERY {battery}"
            ),
            (f"run: {config.blueprint}  |  stack: {config.stack}  |  control: {config.control}"),
            f"{activity}  |  {stop_label}",
        ]
    )


def format_startup_panel(state: ConsoleState) -> str:
    """Return startup readiness details without importing Textual."""

    lines = [
        "MCP CONNECTION FAILED" if state.mcp_state is McpConnectionState.UNREACHABLE else "Startup"
    ]
    for message in state.startup_messages:
        prefix = "OK" if state.mcp_state is McpConnectionState.READY else "info"
        lines.append(f"[{prefix}] {message}")
    if state.tool_names:
        lines.append(f"[OK] tools visible: {', '.join(state.tool_names)}")
    if state.launch_attempted:
        lines.append(f"[info] DimOS start command launched pid={state.launch_pid or 'unknown'}")
    if state.last_error and state.last_error not in state.startup_messages:
        lines.append(f"[error] {state.last_error}")
    if state.pending_start_confirmation:
        lines.append("[next] Press F5 to confirm the configured DimOS start.")
    else:
        lines.append(f"[next] Check {state.mcp_url or state.config.mcp_url or 'the MCP URL'}.")
        lines.append("[next] Press F5 to retry. Ctrl+P opens operator commands.")
    lines.append("[info] DimOS start/stop actions require explicit operator confirmation")
    return "\n".join(lines)


def format_operator_help(state: ConsoleState) -> str:
    """Render the in-app operator cheatsheet."""

    return "\n".join(
        [
            "Operator cheatsheet",
            "  Manual: forward/back <m> | left/right <deg> | strafe <m>",
            "          strafe left/right <m> | rotate <deg> | stop",
            "  Modes: F1 Manual | F2 Agent | F7 Pixel Goal | F3 Close help",
            "  Pixel Goal: F7 auto-refreshes; commands are refresh, reset, stop",
            "  Safety: F8 Exit Agent + stop | ESC Emergency stop | Enter acknowledges result",
            "  Connection: F5 Retry MCP (or confirm a pending start)",
            "  Console: Ctrl+P Command palette | Ctrl+Q Quit",
            f"  MCP: {state.mcp_url or state.config.mcp_url or 'active-run/default'}",
        ]
    )


def format_footer_note(state: ConsoleState, *, showing_help: bool = False) -> str:
    """Return one contextual next action above Textual's key footer."""

    if state.ui_mode.value == "STOPPED":
        return "ENTER  Acknowledge stop    ESC  Send stop again    CTRL+Q  Quit"
    if showing_help:
        return "F3  Close help    F1  Manual    F2  Agent    F7  Pixel Goal    ESC  Emergency stop"
    run = state.active_agent_run
    if run is not None and run.running:
        return "PGUP/PGDN  Activity    F8  Exit Agent + Stop    ESC  Emergency stop"
    if state.ui_mode.value in ("AGENT", "MANUAL", "PIXEL_GOAL"):
        return "F1  Manual    F2  Agent    F7  Pixel Goal    F3  Help    ESC  Emergency stop"
    return "F5  Retry MCP    F3  Help    CTRL+Q  Quit"


def format_manual_panel(state: ConsoleState) -> str:
    """Return manual-mode help plus the latest command result."""

    if state.last_manual_result is not None:
        lines = [format_manual_result(state.last_manual_result)]
        history = state.manual_history or (state.last_manual_result,)
        lines.extend(["", "RECENT COMMANDS"])
        lines.extend(_format_manual_history_row(result) for result in history[-6:])
        lines.extend(["", "Next: enter another command, or press F3 for command help."])
        return "\n".join(lines)
    return "\n".join(
        [
            "Manual control ready",
            "  Type one command below. Each command is sent once through MCP.",
            "",
            format_manual_help(state),
        ]
    )


def format_manual_pending(state: ConsoleState, command: str) -> str:
    """Render a non-blocking manual command while MCP work is in flight."""

    lines = [
        "MANUAL / COMMAND ACTIVE",
        "",
        "MOVING" if command.casefold() != "stop" else "STOPPING",
        f"  {command}",
        "",
        "Waiting for robot feedback. ESC remains available at all times.",
    ]
    if state.manual_history:
        lines.extend(["", "RECENT COMMANDS"])
        lines.extend(_format_manual_history_row(result) for result in state.manual_history[-4:])
    return "\n".join(lines)


def format_context_panel(state: ConsoleState) -> str:
    """Render stable session, quick-command, and safety context."""

    robot = _robot_status_label(state)
    battery = "--" if state.battery_percent is None else f"{state.battery_percent}%"
    endpoint = state.mcp_url or state.config.mcp_url or "active run"
    return "\n".join(
        [
            "SESSION",
            f"  MCP     {state.mcp_state.value}",
            f"  ROBOT   {robot}",
            f"  BATTERY {battery}",
            f"  STACK   {state.config.stack}",
            f"  CONTROL {state.config.control}",
            f"  MODEL   {state.config.model}",
            "",
            "ENDPOINT",
            f"  {endpoint}",
            "",
            "QUICK MOVE",
            "  forward 0.2",
            "  back 0.2",
            "  left 30",
            "  right 30",
            "  stop",
            "",
            "SAFETY",
            "  F8   Exit Agent + stop motion",
            "  ESC  Emergency stop",
        ]
    )


def format_agent_panel(state: ConsoleState) -> str:
    """Render the active or last agent task run."""

    run = state.active_agent_run
    if run is None:
        return "\n".join(
            [
                "Agent task ready",
                "  Type one complete objective below, then press Enter.",
                "  Progress appears here; the input locks while the task is running.",
                "  F8 immediately exits Agent and requests both motion-stop routes.",
            ]
        )
    lines = ["Agent task", f"  status: {run.status.value}", f"  task: {run.task}"]
    if run.running:
        lines.append("  F8: exit Agent immediately and stop motion")
    if run.stop_requested:
        lines.append("  stop_requested: true")
    has_result = run.returncode is not None or bool(run.reason or run.error)
    if has_result:
        lines.append("Result")
    if run.returncode is not None:
        lines.append(f"  returncode: {run.returncode}")
    if run.reason:
        lines.append(f"  reason: {run.reason}")
    if run.error:
        lines.append(f"  error: {run.error}")
    if run.events:
        thinking = [
            event.text[len(THINKING_PREFIX) :].strip()
            for event in run.events
            if event.text.startswith(THINKING_PREFIX)
        ]
        if thinking:
            lines.append("Thinking (latest)")
            for thought_line in thinking[-1].splitlines()[:8]:
                if thought_line.strip():
                    lines.append(f"  {thought_line}")
        event_limit = 2 if has_result else 3
        lines.append(
            f"Recent activity (latest {min(event_limit, len(run.events))} of {len(run.events)})"
        )
        lines.extend(
            f"  {event.source}: {_one_line(event.text)}" for event in run.events[-event_limit:]
        )
    lines.extend(
        [
            "Run details",
            f"  control: {state.config.control}",
            f"  root: {run.run_root}",
        ]
    )
    if run.artifacts.audit_path is not None:
        lines.append(f"  audit: {run.artifacts.audit_path}")
    if run.artifacts.codex_stdout_path is not None:
        lines.append(f"  stdout: {run.artifacts.codex_stdout_path}")
    if run.artifacts.codex_stderr_path is not None:
        lines.append(f"  stderr: {run.artifacts.codex_stderr_path}")
    return "\n".join(lines)


def format_pixel_goal_panel(state: ConsoleState) -> str:
    """Render the four-way Rerun Pixel Goal controls."""

    lines = [
        "Four-way Pixel Goal",
        "  Click a frozen front/right/back/left image in Rerun to navigate.",
        "  Commands: refresh | reset | stop",
        "  F7 refreshes all four frozen images when this mode opens.",
    ]
    if state.last_pixel_goal_result:
        lines.extend(["", "Last result", f"  {state.last_pixel_goal_result}"])
    return "\n".join(lines)


def format_emergency_stop_panel(state: ConsoleState) -> str:
    """Render visible ESC route/result details."""

    result = state.emergency_stop_result
    if result is None:
        return "Emergency stop\n  no stop request has been made"
    route = (
        result.dispatcher_result.route if result.dispatcher_result is not None else "unavailable"
    )
    route_status = "confirmed" if result.preemption_confirmed else "UNCONFIRMED"
    movement_status = "accepted" if result.stop_command_sent else "FAILED"
    if result.severity == "critical":
        outcome = "CRITICAL: no MCP stop confirmation; verify robot state before resuming"
    elif result.severity == "warning":
        outcome = "WARNING: one stop route failed or was unconfirmed; verify robot state"
    else:
        outcome = "OK: MCP preemption and explicit stop_movement both returned"
    lines = [
        f"Emergency stop - {result.severity.upper()}",
        f"  /stop route: {route_status} via {route}",
        f"  stop_movement: {movement_status}",
        f"  {outcome}",
        "  Press Enter to acknowledge and return to Manual mode.",
        "",
        "Details",
        f"  severity: {result.severity}",
        "  sent: /stop through HumanInputDispatcher",
        f"  dispatcher_attempted: {str(result.dispatcher_attempted).lower()}",
        f"  preemption_confirmed: {str(result.preemption_confirmed).lower()}",
    ]
    if result.dispatcher_result is not None:
        lines.append(f"  route: {result.dispatcher_result.route}")
        lines.append(
            "  authority_preemption_supported: "
            f"{str(result.dispatcher_result.authority_preemption_supported).lower()}"
        )
        if result.dispatcher_result.response:
            lines.append(
                f"  dispatcher_response: {_format_response_text(result.dispatcher_result.response)}"
            )
        if result.dispatcher_result.fallback_reason:
            lines.append(f"  fallback_reason: {result.dispatcher_result.fallback_reason}")
    if result.dispatcher_error:
        lines.append(f"  dispatcher_error: {result.dispatcher_error}")
    lines.append(
        "  second stop command: "
        f"{'stop_movement accepted' if result.stop_command_sent else 'stop_movement failed'}"
    )
    lines.append(f"  stop_movement_attempted: {str(result.stop_movement_attempted).lower()}")
    if result.stop_movement_result is not None:
        lines.append(f"  stop_movement_result: {_format_stop_result(result.stop_movement_result)}")
    if result.stop_movement_error:
        lines.append(f"  stop_movement_error: {result.stop_movement_error}")
    if state.last_pixel_goal_result:
        lines.append(f"  pixelgoal_stop: {state.last_pixel_goal_result}")
    if state.last_error:
        lines.append(f"  last_error: {state.last_error}")
    return "\n".join(lines)


def format_manual_help(state: ConsoleState) -> str:
    """Render the discrete manual command grammar and schema-derived hints."""

    lines = [
        "Manual commands",
        "  forward <m>       drive forward meters",
        "  back <m>          drive backward meters",
        "  left <deg>        turn left degrees",
        "  right <deg>       turn right degrees",
        "  strafe <m>        strafe signed meters (+left)",
        "  strafe left <m>   strafe left meters",
        "  strafe right <m>  strafe right meters",
        "  rotate <deg>      rotate in place",
        "  stop              stop_movement",
        "  bounds shown from MCP tool schemas when available",
    ]
    hints = _tool_schema_hints(state)
    if hints:
        lines.append("Schema hints")
        lines.extend(f"  {hint}" for hint in hints)
    return "\n".join(lines)


def format_manual_result(result: object) -> str:
    """Render a normalized manual command result."""

    tool_name = getattr(result, "tool_name", "")
    display_status = getattr(result, "display_status", "UNKNOWN")
    measured_delta = getattr(result, "measured_delta", None)
    zero_velocity_sent = getattr(result, "zero_velocity_sent", None)
    duration_ms = getattr(result, "duration_ms", None)
    error = getattr(result, "error", None)
    text = getattr(result, "text", "")
    lines = [
        "LATEST COMMAND",
        f"  {_manual_command_text(result)}",
        "",
        f"  {display_status}",
    ]
    if measured_delta is not None:
        measurement = _format_measurement(measured_delta) or _format_mapping(measured_delta)
        lines.append(f"  measured: {measurement}")
    if duration_ms is not None:
        lines.append(f"  duration: {_format_duration(duration_ms)}")
    if zero_velocity_sent is not None:
        stop_status = "stop sent" if zero_velocity_sent else "STOP NOT CONFIRMED"
        lines.append(f"  safety: {stop_status}")
    if error:
        lines.append(f"  note: {error}")
    elif text:
        lines.append(f"  note: {text}")
    if tool_name and tool_name != _manual_command_text(result):
        lines.append(f"  via: {tool_name}")
    return "\n".join(lines)


def _one_line(text: str, *, limit: int = 140) -> str:
    """Collapse a possibly multi-line activity event into one bounded line."""

    flattened = " ".join(str(text).split())
    if len(flattened) <= limit:
        return flattened
    return flattened[:limit] + "..."


def _format_manual_history_row(result: object) -> str:
    fields = [_manual_command_text(result), str(getattr(result, "display_status", "UNKNOWN"))]
    measurement = _format_measurement(getattr(result, "measured_delta", None))
    if measurement:
        fields.append(measurement)
    duration_ms = getattr(result, "duration_ms", None)
    if duration_ms is not None:
        fields.append(_format_duration(duration_ms))
    zero_velocity_sent = getattr(result, "zero_velocity_sent", None)
    if zero_velocity_sent is True:
        fields.append("stop sent")
    elif zero_velocity_sent is False:
        fields.append("STOP NOT CONFIRMED")
    return "  " + "  |  ".join(fields)


def _manual_command_text(result: object) -> str:
    command = getattr(result, "command", None)
    text = getattr(command, "original_text", "") if command is not None else ""
    return str(text or getattr(result, "tool_name", "manual") or "manual")


def _format_measurement(value: object) -> str:
    if not isinstance(value, dict):
        return ""
    pieces = []
    linear = value.get("linear_m")
    if isinstance(linear, int | float) and not isinstance(linear, bool):
        pieces.append(_format_distance(float(linear)))
    elif "forward" in value:
        pieces.append(f"{value['forward']} m")
    yaw = value.get("yaw_deg")
    if isinstance(yaw, int | float) and not isinstance(yaw, bool) and abs(float(yaw)) >= 0.1:
        pieces.append(f"{float(yaw):.1f} deg")
    return " / ".join(pieces)


def _format_distance(value_m: float) -> str:
    if abs(value_m) < 0.01:
        return f"{abs(value_m) * 1000:.1f} mm"
    return f"{abs(value_m):.2f} m"


def _format_duration(value_ms: object) -> str:
    if isinstance(value_ms, int | float) and not isinstance(value_ms, bool):
        return f"{float(value_ms) / 1000:.2f} s"
    return str(value_ms)


def _tool_schema_hints(state: ConsoleState) -> tuple[str, ...]:
    hints = []
    for tool in state.tools:
        if tool.name not in {"drive", "rotate_in_place", "stop_movement"}:
            continue
        schema = tool.raw.get("inputSchema") or tool.raw.get("schema") or {}
        properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
        if not isinstance(properties, dict):
            continue
        for name, value in properties.items():
            if not isinstance(value, dict):
                continue
            pieces = []
            if "minimum" in value:
                pieces.append(f"min={value['minimum']}")
            if "maximum" in value:
                pieces.append(f"max={value['maximum']}")
            bounds = f" ({', '.join(pieces)})" if pieces else ""
            hints.append(f"{tool.name}.{name}{bounds}")
    return tuple(hints)


def _format_mapping(value: object) -> str:
    if not isinstance(value, dict):
        return str(value)
    return " ".join(f"{key}={item}" for key, item in value.items()) or "{}"


def _format_stop_result(value: dict[str, object]) -> str:
    content = value.get("content")
    if isinstance(content, list) and content and isinstance(content[0], dict):
        text = content[0].get("text")
        if text:
            return str(text)
    return _format_mapping(value)


def _format_response_text(value: str) -> str:
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return value
    if isinstance(payload, dict) and payload.get("message"):
        return str(payload["message"])
    return value
