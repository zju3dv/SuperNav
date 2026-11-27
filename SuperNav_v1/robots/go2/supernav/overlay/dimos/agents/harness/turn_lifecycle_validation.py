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

"""Trace-audit checks for native-agent turn lifecycle events."""

from __future__ import annotations

from typing import Any

from dimos.agents.harness.validation_types import (
    LiveValidationCheck,
    fail_check as _fail,
    pass_check as _pass,
)


def turn_lifecycle_check(events: list[dict[str, Any]]) -> LiveValidationCheck:
    """Validate turn_start/turn_end pairs and request_assembled linkage."""

    starts = _events_of_type(events, "turn_start")
    ends = _events_of_type(events, "turn_end")
    requests = _events_of_type(events, "request_assembled")
    if not starts or not ends:
        return _fail(
            "turn_lifecycle",
            "Missing turn_start or turn_end trace events.",
            turn_start=len(starts),
            turn_end=len(ends),
        )

    start_ids = _event_ids(starts)
    end_ids = _event_ids(ends)
    missing_end = sorted(start_ids - end_ids)
    missing_start = sorted(end_ids - start_ids)
    request_orphans = _request_orphans(requests, start_ids)
    malformed = _malformed_turn_events(starts, ends)

    if not missing_end and not missing_start and not request_orphans and not malformed:
        return _pass(
            "turn_lifecycle",
            "Every native turn has start/end traces and request_assembled events link to a turn.",
            turns=len(start_ids),
            requests=len(requests),
        )
    return _fail(
        "turn_lifecycle",
        "Native turn lifecycle traces are incomplete or request_assembled is not linked.",
        missing_end=missing_end,
        missing_start=missing_start,
        request_orphans=request_orphans,
        malformed=malformed,
    )


def turn_error_absence_check(events: list[dict[str, Any]]) -> LiveValidationCheck:
    """Fail when any native-agent turn ended with an error outcome."""

    errors: list[dict[str, Any]] = []
    for index, event in enumerate(_events_of_type(events, "turn_end")):
        payload = event.get("payload") or {}
        if not isinstance(payload, dict):
            continue
        if payload.get("outcome") == "error":
            errors.append(
                {
                    "turn_index": index,
                    "turn_id": payload.get("turn_id"),
                    "error": payload.get("error"),
                }
            )

    if not errors:
        return _pass("turn_errors", "No native-agent turn_end error outcomes found.")
    return _fail(
        "turn_errors",
        "At least one native-agent turn ended with outcome=error.",
        errors=errors,
    )


def model_turn_completion_check(events: list[dict[str, Any]]) -> LiveValidationCheck:
    """Require at least one turn to complete a model stream."""

    completed: list[dict[str, Any]] = []
    for index, event in enumerate(_events_of_type(events, "turn_end")):
        payload = event.get("payload") or {}
        if not isinstance(payload, dict):
            continue
        if payload.get("outcome") == "model_completed":
            completed.append(
                {
                    "turn_index": index,
                    "turn_id": payload.get("turn_id"),
                    "model_updates": payload.get("model_updates"),
                }
            )

    if completed:
        return _pass(
            "model_turn_completed",
            f"Found {len(completed)} native-agent turns that completed model streaming.",
            turns=completed,
        )
    return _fail(
        "model_turn_completed",
        "No native-agent turn_end event with outcome=model_completed was found.",
    )


def model_response_trace_check(events: list[dict[str, Any]]) -> LiveValidationCheck:
    """Require every completed native model turn to have a model_response trace."""

    completed_turn_ids: set[str] = set()
    for event in _events_of_type(events, "turn_end"):
        payload = event.get("payload") or {}
        if not isinstance(payload, dict) or payload.get("outcome") != "model_completed":
            continue
        turn_id = payload.get("turn_id")
        if isinstance(turn_id, str):
            completed_turn_ids.add(turn_id)

    responses = _events_of_type(events, "model_response")
    response_turn_ids = _event_ids(responses)
    malformed = _malformed_model_response_events(responses)
    missing_response = sorted(completed_turn_ids - response_turn_ids)
    orphan_response = sorted(response_turn_ids - completed_turn_ids)

    if completed_turn_ids and not missing_response and not malformed:
        return _pass(
            "model_response",
            "Every completed native model turn has a compact model_response trace.",
            completed_turns=sorted(completed_turn_ids),
            responses=len(responses),
            orphan_responses=orphan_response,
        )
    return _fail(
        "model_response",
        "Completed native model turns are missing compact model_response traces.",
        completed_turns=sorted(completed_turn_ids),
        missing_response=missing_response,
        malformed=malformed,
        orphan_responses=orphan_response,
    )


def _events_of_type(events: list[dict[str, Any]], event_type: str) -> list[dict[str, Any]]:
    return [event for event in events if event.get("event_type") == event_type]


def _event_ids(events: list[dict[str, Any]]) -> set[str]:
    ids: set[str] = set()
    for event in events:
        payload = event.get("payload") or {}
        if isinstance(payload, dict) and isinstance(payload.get("turn_id"), str):
            ids.add(payload["turn_id"])
    return ids


def _request_orphans(
    requests: list[dict[str, Any]],
    start_ids: set[str],
) -> list[dict[str, Any]]:
    orphans: list[dict[str, Any]] = []
    for index, event in enumerate(requests):
        payload = event.get("payload") or {}
        turn_id = payload.get("turn_id") if isinstance(payload, dict) else None
        if not isinstance(turn_id, str) or turn_id not in start_ids:
            orphans.append({"request_index": index, "turn_id": turn_id})
    return orphans


def _malformed_turn_events(
    starts: list[dict[str, Any]],
    ends: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    malformed: list[dict[str, Any]] = []
    for event_type, events, required in (
        ("turn_start", starts, ("turn_id", "message_type", "message_source")),
        ("turn_end", ends, ("turn_id", "outcome", "model_updates")),
    ):
        for index, event in enumerate(events):
            payload = event.get("payload") or {}
            if not isinstance(payload, dict):
                malformed.append({"event_type": event_type, "index": index, "missing": ["payload"]})
                continue
            missing = [field for field in required if field not in payload]
            if missing:
                malformed.append({"event_type": event_type, "index": index, "missing": missing})
    return malformed


def _malformed_model_response_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    malformed: list[dict[str, Any]] = []
    required = ("turn_id", "update_count", "tool_call_count", "messages")
    for index, event in enumerate(events):
        payload = event.get("payload") or {}
        if not isinstance(payload, dict):
            malformed.append(
                {"event_type": "model_response", "index": index, "missing": ["payload"]}
            )
            continue
        missing = [field for field in required if field not in payload]
        messages = payload.get("messages")
        if missing:
            malformed.append({"event_type": "model_response", "index": index, "missing": missing})
        elif not isinstance(messages, list):
            malformed.append(
                {
                    "event_type": "model_response",
                    "index": index,
                    "missing": ["messages:list"],
                }
            )
    return malformed
