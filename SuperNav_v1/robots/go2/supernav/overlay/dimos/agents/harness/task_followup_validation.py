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

"""Trace-audit checks for active-task follow-up behavior."""

from __future__ import annotations

from typing import Any

from dimos.agents.harness.validation_types import (
    LiveValidationCheck,
    fail_check as _fail,
    pass_check as _pass,
)

_PENDING_NAV_STATUSES = {"CREATED", "STARTED", "FOLLOWING_PATH"}
_TERMINAL_NAV_STATUSES = {
    "ARRIVAL_NOT_CONFIRMED",
    "ARRIVED",
    "CANCELLED",
    "GOAL_REACHED",
    "NO_ODOM",
    "NO_PATH",
    "NO_SAFE_GOAL",
    "NOT_CONFIGURED",
    "OBSTACLE",
    "RETRY_LIMIT_EXCEEDED",
    "TARGET_NOT_FOUND",
    "TIMEOUT",
    "VISION_UNAVAILABLE",
}
_TERMINAL_TASK_COMMANDS = {"blocked", "cancel", "complete"}


def task_followup_check(events: list[dict[str, Any]]) -> LiveValidationCheck:
    """Validate turn_end follow-up summaries and required navigation follow-up."""

    turn_ends = [
        (index, event)
        for index, event in enumerate(events)
        if event.get("event_type") == "turn_end"
    ]
    if not turn_ends:
        return _fail("task_followup", "No turn_end traces found for follow-up validation.")

    malformed: list[dict[str, Any]] = []
    pending_navigation: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    for event_index, event in turn_ends:
        payload = _payload(event)
        follow_up = payload.get("follow_up")
        if not isinstance(follow_up, dict):
            malformed.append(
                {
                    "event_index": event_index,
                    "turn_id": payload.get("turn_id"),
                    "missing": ["follow_up"],
                }
            )
            continue

        missing = _missing_followup_fields(follow_up)
        if missing:
            malformed.append(
                {
                    "event_index": event_index,
                    "turn_id": payload.get("turn_id"),
                    "missing": missing,
                }
            )
            continue

        if follow_up.get("pending") is not True:
            continue
        nav_status = _status(follow_up.get("navigation_status"))
        if nav_status not in _PENDING_NAV_STATUSES:
            continue
        item = {
            "event_index": event_index,
            "turn_id": payload.get("turn_id"),
            "task_id": follow_up.get("task_id"),
            "navigation_request_id": follow_up.get("navigation_request_id"),
            "navigation_status": nav_status,
        }
        pending_navigation.append(item)
        if not _has_later_followup(events, event_index, follow_up):
            item["diagnosis"] = _unresolved_followup_diagnosis(events, event_index, payload)
            unresolved.append(item)

    if not malformed and not unresolved:
        return _pass(
            "task_followup",
            "Native turn follow-up traces are shaped and pending navigation receives a later status/terminal event.",
            turn_end=len(turn_ends),
            pending_navigation=pending_navigation,
        )

    return _fail(
        "task_followup",
        "Some active-task follow-up traces are malformed or unresolved.",
        malformed=malformed,
        unresolved=unresolved,
        pending_navigation=pending_navigation,
    )


def _payload(event: dict[str, Any]) -> dict[str, Any]:
    payload = event.get("payload") or {}
    return payload if isinstance(payload, dict) else {}


def _unresolved_followup_diagnosis(
    events: list[dict[str, Any]],
    event_index: int,
    turn_payload: dict[str, Any],
) -> dict[str, Any]:
    """Classify why a pending follow-up did not produce an observable next step."""

    turn_id = _string_or_none(turn_payload.get("turn_id"))
    response = _model_response_for_turn(events, turn_id)
    if response is None:
        return {
            "cause": "model_response_missing",
            "turn_id": turn_id,
            "message": (
                "No model_response trace was found for the turn that left the task pending."
            ),
        }

    tool_call_count = _int_or_zero(response.get("tool_call_count"))
    tool_names = _tool_names(response)
    if tool_call_count == 0:
        return {
            "cause": "model_no_tool_calls",
            "turn_id": turn_id,
            "model_tool_call_count": 0,
            "text_previews": _text_previews(response),
            "message": (
                "The model completed while the task was still pending but emitted no tool "
                "calls, so the trace points to a model/harness continuation failure rather "
                "than a missing tool-status update."
            ),
        }

    return {
        "cause": "later_followup_missing",
        "turn_id": turn_id,
        "model_tool_call_count": tool_call_count,
        "model_tool_names": tool_names,
        "later_events_scanned": max(0, len(events) - event_index - 1),
        "message": (
            "The turn used tools and left navigation pending, but no later navigation_status, "
            "terminal navigation update, terminal tool_status, or task-control closure was "
            "observed."
        ),
    }


def _model_response_for_turn(
    events: list[dict[str, Any]],
    turn_id: str | None,
) -> dict[str, Any] | None:
    if turn_id is None:
        return None
    for event in reversed(events):
        if event.get("event_type") != "model_response":
            continue
        payload = _payload(event)
        if payload.get("turn_id") == turn_id:
            return payload
    return None


def _tool_names(model_response: dict[str, Any]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    messages = model_response.get("messages")
    if not isinstance(messages, list):
        return names
    for message in messages:
        if not isinstance(message, dict):
            continue
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            name = call.get("name")
            if not isinstance(name, str) or not name or name in seen:
                continue
            names.append(name)
            seen.add(name)
    return names


def _text_previews(model_response: dict[str, Any]) -> list[str]:
    previews: list[str] = []
    messages = model_response.get("messages")
    if not isinstance(messages, list):
        return previews
    for message in messages:
        if not isinstance(message, dict):
            continue
        preview = message.get("text_preview")
        if isinstance(preview, str) and preview:
            previews.append(preview)
    return previews


def _int_or_zero(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _missing_followup_fields(follow_up: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    if not isinstance(follow_up.get("pending"), bool):
        missing.append("pending:bool")
    if not isinstance(follow_up.get("reason"), str):
        missing.append("reason")
    if follow_up.get("pending") is True:
        for key in ("task_id", "goal", "next_actions"):
            if key not in follow_up:
                missing.append(key)
        if "next_actions" in follow_up and not isinstance(follow_up["next_actions"], list):
            missing.append("next_actions:list")
    return missing


def _has_later_followup(
    events: list[dict[str, Any]],
    event_index: int,
    follow_up: dict[str, Any],
) -> bool:
    task_id = _string_or_none(follow_up.get("task_id"))
    navigation_request_id = _string_or_none(follow_up.get("navigation_request_id"))
    for later in events[event_index + 1 :]:
        event_type = later.get("event_type")
        payload = _payload(later)
        if event_type in {"tool_call_requested", "tool_execution_start", "tool_execution_result"}:
            if payload.get("tool_name") == "navigation_status":
                return True
        if event_type == "navigation_task_updated" and _same_navigation_request(
            payload,
            navigation_request_id,
        ):
            if _status(payload.get("status")) in _TERMINAL_NAV_STATUSES:
                return True
        if event_type == "tool_status" and _tool_status_has_terminal_navigation(
            payload,
            navigation_request_id,
        ):
            return True
        if (
            event_type == "task_control_command"
            and payload.get("command") in _TERMINAL_TASK_COMMANDS
        ):
            return True
        if event_type == "turn_end" and _later_turn_closes_task(payload, task_id):
            return True
    return False


def _same_navigation_request(payload: dict[str, Any], request_id: str | None) -> bool:
    if request_id is None:
        return True
    return payload.get("request_id") == request_id


def _tool_status_has_terminal_navigation(
    payload: dict[str, Any],
    request_id: str | None,
) -> bool:
    status_context = payload.get("status_context")
    if not isinstance(status_context, dict):
        return False
    nav = status_context.get("navigation_task")
    if not isinstance(nav, dict):
        return False
    if request_id is not None and nav.get("request_id") != request_id:
        return False
    return _status(nav.get("status")) in _TERMINAL_NAV_STATUSES


def _later_turn_closes_task(payload: dict[str, Any], task_id: str | None) -> bool:
    follow_up = payload.get("follow_up")
    if not isinstance(follow_up, dict):
        return False
    if follow_up.get("pending") is False:
        if task_id is None:
            return True
        return follow_up.get("task_id") in {None, task_id}
    if task_id is None or follow_up.get("task_id") != task_id:
        return False
    nav_status = _status(follow_up.get("navigation_status"))
    return bool(nav_status and nav_status not in _PENDING_NAV_STATUSES)


def _status(value: Any) -> str:
    return str(value or "").strip().upper()


def _string_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None
