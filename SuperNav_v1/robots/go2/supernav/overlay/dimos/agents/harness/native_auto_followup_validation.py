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

"""Trace-audit checks for native-agent auto follow-up turns."""

from __future__ import annotations

from typing import Any

from dimos.agents.harness.native_auto_followup import (
    DIAGNOSTIC_TOOL_FOLLOWUP_REASON,
    PENDING_NAVIGATION_FOLLOWUP_REASON,
    ZERO_TOOL_FOLLOWUP_REASON,
    context_gathering_tool_names,
)
from dimos.agents.harness.validation_types import (
    LiveValidationCheck,
    fail_check as _fail,
    pass_check as _pass,
)

_AUTO_FOLLOWUP_SOURCE = "dimos_native:auto_followup"


def auto_followup_trace_check(events: list[dict[str, Any]]) -> LiveValidationCheck:
    """Validate pending tasks queue/process bounded auto follow-ups when required."""

    pending_zero_tool = _pending_zero_tool_turns(events)
    pending_navigation_tool = _pending_navigation_tool_turns(events)
    pending_diagnostic_tool = _pending_diagnostic_tool_turns(events)
    pending_followups = [
        *pending_zero_tool,
        *pending_navigation_tool,
        *pending_diagnostic_tool,
    ]
    queued = _queued_followups(events)
    skipped = _skipped_followups(events)
    malformed_queued = [item for item in queued if item.get("malformed")]
    malformed_skipped = [item for item in skipped if item.get("malformed")]
    missing_queued = [
        turn
        for turn in pending_followups
        if _queued_between(events, queued, turn) is None
        and _max_attempt_skip_between(events, skipped, turn) is None
    ]
    unprocessed_queued = [
        item
        for item in queued
        if item.get("task_id") and not _has_later_auto_followup_turn(events, item)
    ]

    if (
        not malformed_queued
        and not malformed_skipped
        and not missing_queued
        and not unprocessed_queued
    ):
        return _pass(
            "auto_followup",
            "Pending task turns queue/process required auto follow-ups or trace max-attempt skips.",
            zero_tool_pending=pending_zero_tool,
            navigation_tool_pending=pending_navigation_tool,
            diagnostic_tool_pending=pending_diagnostic_tool,
            pending_followups=pending_followups,
            queued=queued,
            skipped=skipped,
        )

    return _fail(
        "auto_followup",
        "Some pending task turns did not queue or process required auto follow-ups.",
        zero_tool_pending=pending_zero_tool,
        navigation_tool_pending=pending_navigation_tool,
        diagnostic_tool_pending=pending_diagnostic_tool,
        pending_followups=pending_followups,
        queued=queued,
        skipped=skipped,
        malformed_queued=malformed_queued,
        malformed_skipped=malformed_skipped,
        missing_queued=missing_queued,
        unprocessed_queued=unprocessed_queued,
    )


def _pending_zero_tool_turns(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    response_by_turn = _model_response_by_turn(events)
    turns: list[dict[str, Any]] = []
    for index, event in enumerate(events):
        if event.get("event_type") != "turn_end":
            continue
        payload = _payload(event)
        if payload.get("outcome") != "model_completed":
            continue
        follow_up = payload.get("follow_up")
        if not isinstance(follow_up, dict) or follow_up.get("pending") is not True:
            continue
        task_id = follow_up.get("task_id")
        turn_id = payload.get("turn_id")
        if not isinstance(task_id, str) or not isinstance(turn_id, str):
            continue
        response = response_by_turn.get(turn_id)
        if response is None or _int_or_zero(response["payload"].get("tool_call_count")) != 0:
            continue
        turns.append(
            {
                "event_index": index,
                "turn_id": turn_id,
                "task_id": task_id,
                "model_response_index": response["event_index"],
                "model_tool_call_count": 0,
                "required_reason": ZERO_TOOL_FOLLOWUP_REASON,
                "category": "zero_tool_pending",
            }
        )
    return turns


def _pending_navigation_tool_turns(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    response_by_turn = _model_response_by_turn(events)
    turns: list[dict[str, Any]] = []
    for index, event in enumerate(events):
        if event.get("event_type") != "turn_end":
            continue
        payload = _payload(event)
        if payload.get("outcome") != "model_completed":
            continue
        follow_up = payload.get("follow_up")
        if not _requires_navigation_auto_followup(follow_up):
            continue
        task_id = follow_up.get("task_id") if isinstance(follow_up, dict) else None
        turn_id = payload.get("turn_id")
        if not isinstance(task_id, str) or not isinstance(turn_id, str):
            continue
        response = response_by_turn.get(turn_id)
        if response is None:
            continue
        tool_call_count = _int_or_zero(response["payload"].get("tool_call_count"))
        if tool_call_count <= 0:
            continue
        turns.append(
            {
                "event_index": index,
                "turn_id": turn_id,
                "task_id": task_id,
                "navigation_request_id": follow_up.get("navigation_request_id"),
                "navigation_status": follow_up.get("navigation_status"),
                "model_response_index": response["event_index"],
                "model_tool_call_count": tool_call_count,
                "required_reason": PENDING_NAVIGATION_FOLLOWUP_REASON,
                "category": "navigation_tool_pending",
            }
        )
    return turns


def _pending_diagnostic_tool_turns(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    response_by_turn = _model_response_by_turn(events)
    turns: list[dict[str, Any]] = []
    for index, event in enumerate(events):
        if event.get("event_type") != "turn_end":
            continue
        payload = _payload(event)
        if payload.get("outcome") != "model_completed":
            continue
        follow_up = payload.get("follow_up")
        if not isinstance(follow_up, dict) or follow_up.get("pending") is not True:
            continue
        if _requires_navigation_auto_followup(follow_up):
            continue
        task_id = follow_up.get("task_id")
        turn_id = payload.get("turn_id")
        if not isinstance(task_id, str) or not isinstance(turn_id, str):
            continue
        response = response_by_turn.get(turn_id)
        if response is None:
            continue
        response_payload = response["payload"]
        tool_call_count = _int_or_zero(response_payload.get("tool_call_count"))
        if tool_call_count <= 0:
            continue
        tool_names = context_gathering_tool_names(_model_tool_call_names(response_payload))
        if not tool_names:
            continue
        turns.append(
            {
                "event_index": index,
                "turn_id": turn_id,
                "task_id": task_id,
                "model_response_index": response["event_index"],
                "model_tool_call_count": tool_call_count,
                "model_tool_call_names": tool_names,
                "required_reason": DIAGNOSTIC_TOOL_FOLLOWUP_REASON,
                "category": "diagnostic_tool_pending",
            }
        )
    return turns


def _queued_followups(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    queued: list[dict[str, Any]] = []
    for index, event in enumerate(events):
        if event.get("event_type") != "auto_followup_queued":
            continue
        payload = _payload(event)
        missing = _missing_queued_fields(payload)
        item = {
            "event_index": index,
            "task_id": payload.get("task_id"),
            "reason": payload.get("reason"),
            "attempt": payload.get("attempt"),
            "max_attempts": payload.get("max_attempts"),
            "followup_reason": payload.get("reason"),
        }
        if missing:
            item["malformed"] = {"missing": missing}
        queued.append(item)
    return queued


def _skipped_followups(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    skipped: list[dict[str, Any]] = []
    for index, event in enumerate(events):
        if event.get("event_type") != "auto_followup_skipped":
            continue
        payload = _payload(event)
        missing = _missing_skipped_fields(payload)
        item = {
            "event_index": index,
            "task_id": payload.get("task_id"),
            "reason": payload.get("reason"),
            "attempt": payload.get("attempt"),
            "max_attempts": payload.get("max_attempts"),
            "followup_reason": _followup_reason(payload),
        }
        if missing:
            item["malformed"] = {"missing": missing}
        skipped.append(item)
    return skipped


def _queued_between(
    events: list[dict[str, Any]],
    queued: list[dict[str, Any]],
    turn: dict[str, Any],
) -> dict[str, Any] | None:
    task_id = turn.get("task_id")
    start = int(turn.get("model_response_index") or 0)
    end = _next_turn_start_index(events, int(turn.get("event_index") or 0))
    for item in queued:
        index = int(item["event_index"])
        if (
            item.get("task_id") == task_id
            and _same_followup_reason(item, turn)
            and start < index < end
        ):
            return item
    return None


def _max_attempt_skip_between(
    events: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
    turn: dict[str, Any],
) -> dict[str, Any] | None:
    task_id = turn.get("task_id")
    start = int(turn.get("model_response_index") or 0)
    end = _next_turn_start_index(events, int(turn.get("event_index") or 0))
    for item in skipped:
        index = int(item["event_index"])
        if (
            item.get("task_id") != task_id
            or not _same_followup_reason(item, turn)
            or not (start < index < end)
        ):
            continue
        if item.get("reason") == "max_attempts_reached" and not item.get("malformed"):
            return item
    return None


def _has_later_auto_followup_turn(
    events: list[dict[str, Any]],
    queued: dict[str, Any],
) -> bool:
    task_id = queued.get("task_id")
    if not isinstance(task_id, str):
        return False
    start_index = int(queued.get("event_index") or 0)
    for event in events[start_index + 1 :]:
        if event.get("event_type") != "turn_start":
            continue
        payload = _payload(event)
        if payload.get("message_source") != _AUTO_FOLLOWUP_SOURCE:
            continue
        active_task = payload.get("active_task")
        if not isinstance(active_task, dict):
            continue
        if active_task.get("task_id") == task_id:
            return True
    return False


def _model_response_by_turn(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    responses: dict[str, dict[str, Any]] = {}
    for index, event in enumerate(events):
        if event.get("event_type") != "model_response":
            continue
        payload = _payload(event)
        turn_id = payload.get("turn_id")
        if isinstance(turn_id, str):
            responses[turn_id] = {"event_index": index, "payload": payload}
    return responses


def _next_turn_start_index(events: list[dict[str, Any]], after_index: int) -> int:
    for index, event in enumerate(events[after_index + 1 :], start=after_index + 1):
        if event.get("event_type") == "turn_start":
            return index
    return len(events)


def _missing_queued_fields(payload: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    if not isinstance(payload.get("task_id"), str):
        missing.append("task_id")
    if not isinstance(payload.get("reason"), str):
        missing.append("reason")
    if not isinstance(payload.get("attempt"), int):
        missing.append("attempt:int")
    if not isinstance(payload.get("max_attempts"), int):
        missing.append("max_attempts:int")
    return missing


def _missing_skipped_fields(payload: dict[str, Any]) -> list[str]:
    missing = _missing_queued_fields(payload)
    if payload.get("reason") != "max_attempts_reached":
        missing.append("reason:max_attempts_reached")
    return missing


def _requires_navigation_auto_followup(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    return (
        value.get("pending") is True
        and value.get("reason") == "navigation_status_guidance"
        and isinstance(value.get("navigation_request_id"), str)
        and bool(value.get("navigation_request_id"))
    )


def _model_tool_call_names(payload: dict[str, Any]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    messages = payload.get("messages")
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


def _same_followup_reason(
    item: dict[str, Any],
    turn: dict[str, Any],
) -> bool:
    required = turn.get("required_reason")
    if not isinstance(required, str):
        return True
    followup_reason = item.get("followup_reason")
    if followup_reason is None and required == ZERO_TOOL_FOLLOWUP_REASON:
        return True
    return followup_reason == required


def _followup_reason(payload: dict[str, Any]) -> str | None:
    details = payload.get("details")
    if not isinstance(details, dict):
        return None
    value = details.get("followup_reason")
    return value if isinstance(value, str) else None


def _payload(event: dict[str, Any]) -> dict[str, Any]:
    payload = event.get("payload") or {}
    return payload if isinstance(payload, dict) else {}


def _int_or_zero(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
