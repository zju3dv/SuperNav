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

"""Discrete manual command parsing and MCP execution for ``dimos operate``."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any

import requests

from dimos.agents.console.mcp_client import ConsoleMcpClient
from dimos.agents.console.state import McpConnectionState, ToolInfo
from dimos.agents.mcp.external_harness_authority import ExternalHarnessMovementAuthority
from dimos.agents.mcp.mcp_adapter import McpError

_MANUAL_MOVEMENT_CLIENT_ID = "operator-console-manual"


@dataclass(frozen=True)
class ManualCommand:
    """A parsed one-shot manual command and its MCP tool call."""

    original_text: str
    tool_name: str
    arguments: dict[str, float] = field(default_factory=dict)
    requested: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ArgumentGuidance:
    """Schema-derived argument hint for display only."""

    name: str
    minimum: float | int | None = None
    maximum: float | int | None = None
    description: str = ""

    def render(self) -> str:
        """Render a compact schema-derived bounds string."""

        bounds = []
        if self.minimum is not None:
            bounds.append(f"min={self.minimum}")
        if self.maximum is not None:
            bounds.append(f"max={self.maximum}")
        suffix = f" ({', '.join(bounds)})" if bounds else ""
        if self.description:
            return f"{self.name}{suffix}: {self.description}"
        return f"{self.name}{suffix}"


@dataclass(frozen=True)
class ManualCommandResult:
    """Normalized result of one manual MCP command."""

    command: ManualCommand | None
    tool_name: str
    arguments: dict[str, float]
    success: bool
    raw_result: dict[str, Any] | None = None
    text: str = ""
    payload: Any | None = None
    status: str | None = None
    measured_delta: Any | None = None
    duration_ms: float | int | None = None
    feedback_watchdog_triggered: bool | None = None
    precision_class: str | None = None
    zero_velocity_sent: bool | None = None
    error: str | None = None
    mcp_state: McpConnectionState | None = None
    schema_guidance: tuple[ArgumentGuidance, ...] = ()

    @classmethod
    def error_result(
        cls,
        message: str,
        *,
        command: ManualCommand | None = None,
        tool_name: str = "",
        arguments: dict[str, float] | None = None,
        mcp_state: McpConnectionState | None = None,
        schema_guidance: tuple[ArgumentGuidance, ...] = (),
    ) -> ManualCommandResult:
        """Create a failed result for parse, availability, or MCP errors."""

        return cls(
            command=command,
            tool_name=tool_name or (command.tool_name if command else ""),
            arguments=arguments
            if arguments is not None
            else (command.arguments if command else {}),
            success=False,
            error=message,
            mcp_state=mcp_state,
            schema_guidance=schema_guidance,
        )

    @property
    def display_status(self) -> str:
        """Return the best short status label for the top bar/result panel."""

        if self.status == "NO_ODOM":
            if self.feedback_watchdog_triggered and self.measured_delta is not None:
                return "NO PROGRESS"
            return "NO ODOM"
        if self.status:
            return {
                "EXECUTION_FAILED": "FAILED",
                "TIMEOUT": "TIMED OUT",
            }.get(self.status, self.status.replace("_", " "))
        if self.error:
            return "ERROR"
        return "OK" if self.success else "FAILED"


class ManualCommandParseError(ValueError):
    """Raised when operator text does not match the manual grammar."""


class ManualCommandService:
    """Parse and execute discrete manual commands through MCP tools."""

    def __init__(self, mcp_client: ConsoleMcpClient, tools: tuple[ToolInfo, ...]) -> None:
        self._mcp_client = mcp_client
        self._tools = {tool.name: tool for tool in tools}
        self._translation_tool = "drive" if "drive" in self._tools else "relative_move"

    def parse(self, text: str) -> ManualCommand:
        """Parse operator text into a typed command object."""

        tokens = text.strip().split()
        if not tokens:
            raise ManualCommandParseError("empty manual command")

        command = tokens[0].casefold()
        if command not in {"forward", "back", "left", "right", "strafe", "rotate", "stop"}:
            raise ManualCommandParseError(
                "unknown manual command. Use forward, back, left, right, strafe, rotate, or stop."
            )
        if command == "stop":
            if len(tokens) != 1:
                raise ManualCommandParseError("usage: stop")
            return ManualCommand(
                original_text=text,
                tool_name="stop_movement",
                arguments={},
                requested={"forward": 0.0, "left": 0.0, "degrees": 0.0},
            )

        if command == "strafe":
            if len(tokens) == 2:
                value = _parse_float(tokens[1])
            elif len(tokens) == 3 and tokens[1].casefold() in {"left", "right"}:
                value = _parse_float(tokens[2])
                if tokens[1].casefold() == "right":
                    value = -value
            else:
                raise ManualCommandParseError("usage: strafe [left|right] <meters>")
            return _drive_command(
                text,
                tool_name=self._translation_tool,
                arguments={"left": value},
                forward=0.0,
                left=value,
            )

        if len(tokens) != 2:
            raise ManualCommandParseError(f"usage: {command} <number>")

        value = _parse_float(tokens[1])
        if command == "forward":
            return _drive_command(
                text,
                tool_name=self._translation_tool,
                arguments={"forward": value},
                forward=value,
                left=0.0,
            )
        if command == "back":
            return _drive_command(
                text,
                tool_name=self._translation_tool,
                arguments={"forward": -value},
                forward=-value,
                left=0.0,
            )
        if command in {"left", "right", "rotate"}:
            degrees = -value if command == "right" else value
            return ManualCommand(
                original_text=text,
                tool_name="rotate_in_place",
                arguments={"degrees": degrees},
                requested={"forward": 0.0, "left": 0.0, "degrees": degrees},
            )

        raise AssertionError(f"unhandled manual command: {command}")

    def execute(self, text: str) -> ManualCommandResult:
        """Parse one operator command, call MCP once, and normalize the result."""

        try:
            command = self.parse(text)
        except ManualCommandParseError as exc:
            return ManualCommandResult.error_result(str(exc))

        guidance = self.schema_guidance(command.tool_name)
        if command.tool_name not in self._tools:
            return ManualCommandResult.error_result(
                f"tool unavailable in this runtime profile: {command.tool_name}",
                command=command,
                schema_guidance=guidance,
            )

        try:
            meta = None
            if command.tool_name != "stop_movement":
                meta = ExternalHarnessMovementAuthority.create(
                    task_goal=f"operator manual command: {command.original_text}",
                    client_id=_MANUAL_MOVEMENT_CLIENT_ID,
                ).meta()
            raw_result = self._mcp_client.call_tool(
                command.tool_name,
                command.arguments,
                meta=meta,
            )
        except (McpError, requests.RequestException) as exc:
            message, mcp_state = _operator_request_error(exc)
            return ManualCommandResult.error_result(
                message,
                command=command,
                mcp_state=mcp_state,
                schema_guidance=guidance,
            )

        return _normalize_result(command, raw_result, guidance)

    def schema_guidance(self, tool_name: str) -> tuple[ArgumentGuidance, ...]:
        """Return schema-derived argument hints without enforcing a new limit layer."""

        tool = self._tools.get(tool_name)
        if tool is None:
            return ()
        schema = tool.raw.get("inputSchema") or tool.raw.get("schema") or {}
        properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
        if not isinstance(properties, dict):
            return ()
        hints = []
        for name, value in properties.items():
            if not isinstance(value, dict):
                continue
            hints.append(
                ArgumentGuidance(
                    name=str(name),
                    minimum=_number_or_none(value.get("minimum")),
                    maximum=_number_or_none(value.get("maximum")),
                    description=str(value.get("description", "")),
                )
            )
        return tuple(hints)


def _drive_command(
    text: str,
    *,
    tool_name: str,
    arguments: dict[str, float],
    forward: float,
    left: float,
) -> ManualCommand:
    return ManualCommand(
        original_text=text,
        tool_name=tool_name,
        arguments=arguments,
        requested={"forward": forward, "left": left, "degrees": 0.0},
    )


def _parse_float(token: str) -> float:
    try:
        return float(token)
    except ValueError as exc:
        raise ManualCommandParseError(f"expected a number, got: {token}") from exc


def _normalize_result(
    command: ManualCommand,
    raw_result: dict[str, Any],
    guidance: tuple[ArgumentGuidance, ...],
) -> ManualCommandResult:
    text = _result_text(raw_result)
    payload = _json_payload(text)
    payload_dict = payload if isinstance(payload, dict) else {}
    metadata = payload_dict.get("metadata")
    details = payload_dict | metadata if isinstance(metadata, dict) else payload_dict
    is_error = bool(raw_result.get("isError"))
    success = not is_error
    if isinstance(payload_dict.get("success"), bool):
        success = bool(payload_dict["success"])
    elif _plain_text_policy_refusal(text) is not None:
        success = False
    error = None
    if not success:
        refusal = _plain_text_policy_refusal(text)
        if refusal is not None:
            error = _refused_message(refusal)
        else:
            error_code = _optional_string(payload_dict.get("error_code"))
            payload_error = _payload_error(payload_dict)
            if error_code in {"NO_ODOM", "NO_PROGRESS", "TIMEOUT", "EXECUTION_FAILED"}:
                error = _movement_failure_message(error_code, details, payload_error)
            elif payload_error:
                error = _refused_message(payload_error)
    return ManualCommandResult(
        command=command,
        tool_name=command.tool_name,
        arguments=command.arguments,
        success=success,
        raw_result=raw_result,
        text=_optional_string(payload_dict.get("message")) or text,
        payload=payload,
        status=_optional_string(details.get("status")),
        measured_delta=details.get("measured_delta"),
        duration_ms=_number_or_none(details.get("duration_ms")),
        feedback_watchdog_triggered=_optional_bool(details.get("feedback_watchdog_triggered")),
        precision_class=_optional_string(details.get("precision_class")),
        zero_velocity_sent=_optional_bool(details.get("zero_velocity_sent")),
        error=error,
        schema_guidance=guidance,
    )


def _operator_request_error(
    error: McpError | requests.RequestException,
) -> tuple[str, McpConnectionState | None]:
    detail = str(error)
    if isinstance(error, requests.Timeout):
        return (
            "Command timed out without a confirmed result. Press ESC to stop, verify robot "
            "state, then press F5 to reconnect.",
            McpConnectionState.ERROR,
        )
    if isinstance(error, requests.ConnectionError) or any(
        marker in detail.casefold()
        for marker in ("http 502", "bad gateway", "service unavailable", "connection refused")
    ):
        return (
            "MCP disconnected before confirming the command. Press ESC to stop, verify robot "
            "state, then press F5 to reconnect.",
            McpConnectionState.UNREACHABLE,
        )
    if "timed out" in detail.casefold() or "timeout" in detail.casefold():
        return (
            "Command timed out without a confirmed result. Press ESC to stop and verify robot "
            "state before retrying.",
            None,
        )
    if isinstance(error, McpError):
        return _refused_message(detail), None
    return (
        "MCP request failed before confirming the command. Press ESC to stop, verify robot "
        "state, then press F5 to reconnect.",
        McpConnectionState.ERROR,
    )


def _refused_message(detail: str) -> str:
    return (
        f"MCP refused the command: {detail}. Check the active scope and command limits, then retry."
    )


def _movement_failure_message(
    error_code: str,
    details: dict[str, Any],
    fallback: str | None,
) -> str:
    if error_code == "NO_PROGRESS":
        return "No measurable body progress; the motion watchdog stopped the command."
    if error_code == "NO_ODOM":
        if _optional_bool(details.get("feedback_watchdog_triggered")):
            return "No measurable body progress; the motion watchdog stopped the command."
        return "No fresh body odometry was available to verify the command."
    if error_code == "TIMEOUT":
        return "The command timed out before reaching its requested target."
    return fallback or "The robot connection could not execute the command."


def _result_text(raw_result: dict[str, Any]) -> str:
    content = raw_result.get("content", [])
    if not isinstance(content, list):
        return ""
    text_blocks = [
        str(block.get("text", ""))
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    return "\n".join(block for block in text_blocks if block)


def _json_payload(text: str) -> Any | None:
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _payload_error(payload: dict[str, Any]) -> str | None:
    error = payload.get("error")
    if isinstance(error, str):
        return error
    if isinstance(error, dict):
        message = error.get("message")
        return str(message if message is not None else error)
    error_code = payload.get("error_code")
    message = payload.get("message")
    if error_code and message:
        return f"{error_code}: {message}"
    if error_code or message:
        return str(error_code or message)
    return None


def _plain_text_policy_refusal(text: str) -> str | None:
    lowered = text.casefold()
    if (
        "runtime movement policy is lease" in lowered
        or "runtime movement policy refused" in lowered
        or ("movement tool '" in lowered and " refused:" in lowered)
    ):
        return text
    return None


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _number_or_none(value: Any) -> float | int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return value
    return None
