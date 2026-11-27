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

"""Trace-audit checks for MCP tool execution events."""

from __future__ import annotations

from typing import Any

from dimos.agents.harness.validation_types import (
    LiveValidationCheck,
    fail_check as _fail,
    pass_check as _pass,
)


def tool_execution_trace_check(events: list[dict[str, Any]]) -> LiveValidationCheck:
    """Validate tool_call_requested/start/result trace linkage."""

    requested = _events_of_type(events, "tool_call_requested")
    starts = _events_of_type(events, "tool_execution_start")
    results = _events_of_type(events, "tool_execution_result")
    if not requested or not starts or not results:
        return _fail(
            "tool_execution",
            "Missing tool_call_requested, tool_execution_start, or tool_execution_result traces.",
            tool_call_requested=len(requested),
            tool_execution_start=len(starts),
            tool_execution_result=len(results),
        )

    requested_ids = _event_ids(requested)
    start_ids = _event_ids(starts)
    result_ids = _event_ids(results)
    missing_start = sorted(result_ids - start_ids)
    missing_result = sorted(start_ids - result_ids)
    missing_request = sorted((start_ids | result_ids) - requested_ids)
    malformed = [
        *_malformed_requested_events(requested),
        *_malformed_start_events(starts),
        *_malformed_result_events(results),
    ]

    if not missing_start and not missing_result and not missing_request and not malformed:
        return _pass(
            "tool_execution",
            "Tool execution traces include request, start, and result for executed calls.",
            tool_call_requested=len(requested),
            tool_execution_start=len(starts),
            tool_execution_result=len(results),
        )

    return _fail(
        "tool_execution",
        "Tool execution traces are incomplete or malformed.",
        missing_start=missing_start,
        missing_result=missing_result,
        missing_request=missing_request,
        malformed=malformed,
    )


def tool_call_terminal_check(events: list[dict[str, Any]]) -> LiveValidationCheck:
    """Validate that every requested tool has an execution or refusal terminal event."""

    requested = _events_of_type(events, "tool_call_requested")
    if not requested:
        return _fail("tool_call_terminal", "No tool_call_requested traces found.")

    requested_ids = _event_ids(requested)
    result_ids = _event_ids(_events_of_type(events, "tool_execution_result"))
    refused = _events_of_type(events, "tool_call_refused")
    refused_ids = _event_ids(refused)
    terminal_ids = result_ids | refused_ids
    missing_terminal = sorted(requested_ids - terminal_ids)
    orphan_terminal = sorted(terminal_ids - requested_ids)
    malformed_refused = _malformed_refused_events(refused)

    if not missing_terminal and not orphan_terminal and not malformed_refused:
        return _pass(
            "tool_call_terminal",
            "Every requested tool has an execution result or refused terminal trace.",
            tool_call_requested=len(requested),
            tool_execution_result=len(result_ids),
            tool_call_refused=len(refused_ids),
        )

    return _fail(
        "tool_call_terminal",
        "Some requested tools lack a terminal execution/refusal trace.",
        missing_terminal=missing_terminal,
        orphan_terminal=orphan_terminal,
        malformed_refused=malformed_refused,
    )


def _events_of_type(events: list[dict[str, Any]], event_type: str) -> list[dict[str, Any]]:
    return [event for event in events if event.get("event_type") == event_type]


def _payload(event: dict[str, Any]) -> dict[str, Any]:
    payload = event.get("payload") or {}
    return payload if isinstance(payload, dict) else {}


def _event_ids(events: list[dict[str, Any]]) -> set[str]:
    ids: set[str] = set()
    for event in events:
        invocation_id = _payload(event).get("invocation_id")
        if isinstance(invocation_id, str):
            ids.add(invocation_id)
    return ids


def _malformed_requested_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _missing_fields(
        "tool_call_requested",
        events,
        ("invocation_id", "tool_name", "client_id", "found", "arguments"),
    )


def _malformed_start_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _missing_fields(
        "tool_execution_start",
        events,
        ("invocation_id", "tool_name", "client_id", "uses", "lifecycle"),
    )


def _malformed_result_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    malformed = _missing_fields(
        "tool_execution_result",
        events,
        (
            "invocation_id",
            "tool_name",
            "client_id",
            "success",
            "rpc_success",
            "duration_ms",
            "result",
        ),
    )
    for index, event in enumerate(events):
        payload = _payload(event)
        if "success" in payload and not isinstance(payload["success"], bool):
            malformed.append(
                {
                    "event_type": "tool_execution_result",
                    "index": index,
                    "missing": ["success:bool"],
                }
            )
        if "rpc_success" in payload and not isinstance(payload["rpc_success"], bool):
            malformed.append(
                {
                    "event_type": "tool_execution_result",
                    "index": index,
                    "missing": ["rpc_success:bool"],
                }
            )
        if "duration_ms" in payload and not isinstance(payload["duration_ms"], int | float):
            malformed.append(
                {
                    "event_type": "tool_execution_result",
                    "index": index,
                    "missing": ["duration_ms:number"],
                }
            )
        if "result" in payload and not isinstance(payload["result"], dict):
            malformed.append(
                {
                    "event_type": "tool_execution_result",
                    "index": index,
                    "missing": ["result:dict"],
                }
            )
        result = payload.get("result")
        if isinstance(result, dict) and result.get("kind") == "SkillResult":
            for field in ("skill_success", "skill_error_code"):
                if field not in payload:
                    malformed.append(
                        {
                            "event_type": "tool_execution_result",
                            "index": index,
                            "missing": [field],
                        }
                    )
            if "skill_success" in payload and payload.get("success") != payload["skill_success"]:
                malformed.append(
                    {
                        "event_type": "tool_execution_result",
                        "index": index,
                        "missing": ["success_matches_skill_success"],
                    }
                )
    return malformed


def _malformed_refused_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _missing_fields(
        "tool_call_refused",
        events,
        ("invocation_id", "tool_name", "client_id", "refusal_source", "reason"),
    )


def _missing_fields(
    event_type: str,
    events: list[dict[str, Any]],
    required: tuple[str, ...],
) -> list[dict[str, Any]]:
    malformed: list[dict[str, Any]] = []
    for index, event in enumerate(events):
        payload = _payload(event)
        missing = [field for field in required if field not in payload]
        if missing:
            malformed.append({"event_type": event_type, "index": index, "missing": missing})
    return malformed
