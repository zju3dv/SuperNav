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

"""Trace-audit checks for request context compaction events."""

from __future__ import annotations

from typing import Any

from dimos.agents.harness.validation_types import (
    LiveValidationCheck,
    fail_check as _fail,
    pass_check as _pass,
)


def context_compaction_trace_check(
    events: list[dict[str, Any]],
    request_events: list[dict[str, Any]],
) -> LiveValidationCheck:
    """Validate that dropped request context has a durable compaction trace."""

    compacted_requests = [
        _compacted_request_summary(index, event)
        for index, event in enumerate(request_events)
        if _dropped_entries(event)
    ]
    if not compacted_requests:
        return _pass("context_compaction", "No request context compaction was needed.")

    compaction_events = _events_of_type(events, "context_compacted")
    malformed = _malformed_compaction_events(compaction_events)
    compaction_by_turn = {
        _turn_id(event): event for event in compaction_events if _turn_id(event) is not None
    }
    missing: list[dict[str, Any]] = []
    for request in compacted_requests:
        turn_id = request.get("turn_id")
        if isinstance(turn_id, str) and turn_id in compaction_by_turn:
            continue
        missing.append(request)

    if not missing and not malformed:
        return _pass(
            "context_compaction",
            "Every compacted request has a matching context_compacted trace.",
            compacted_requests=compacted_requests,
            context_compacted=len(compaction_events),
        )
    return _fail(
        "context_compaction",
        "Some compacted request_assembled events lack matching context_compacted traces.",
        missing=missing,
        malformed=malformed,
        context_compacted=len(compaction_events),
    )


def _compacted_request_summary(index: int, event: dict[str, Any]) -> dict[str, Any]:
    assembly = _assembly_payload(event)
    payload = _payload(event)
    return {
        "request_index": index,
        "turn_id": payload.get("turn_id"),
        "model": payload.get("model"),
        "dropped": assembly.get("dropped") or [],
    }


def _malformed_compaction_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    malformed: list[dict[str, Any]] = []
    for index, event in enumerate(events):
        payload = _payload(event)
        dropped = payload.get("dropped")
        assembly = payload.get("assembly")
        if not isinstance(assembly, dict):
            malformed.append({"index": index, "missing": ["assembly"]})
        if not isinstance(dropped, list) or not dropped:
            malformed.append({"index": index, "missing": ["dropped:list"]})
    return malformed


def _events_of_type(events: list[dict[str, Any]], event_type: str) -> list[dict[str, Any]]:
    return [event for event in events if event.get("event_type") == event_type]


def _payload(event: dict[str, Any]) -> dict[str, Any]:
    payload = event.get("payload") or {}
    return payload if isinstance(payload, dict) else {}


def _assembly_payload(event: dict[str, Any]) -> dict[str, Any]:
    assembly = _payload(event).get("assembly")
    return assembly if isinstance(assembly, dict) else {}


def _dropped_entries(event: dict[str, Any]) -> list[Any]:
    dropped = _assembly_payload(event).get("dropped")
    return dropped if isinstance(dropped, list) else []


def _turn_id(event: dict[str, Any]) -> str | None:
    value = _payload(event).get("turn_id")
    return value if isinstance(value, str) and value else None
