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

"""Trace-audit checks for harness failure escalation events."""

from __future__ import annotations

from typing import Any

from dimos.agents.harness.validation_types import (
    LiveValidationCheck,
    fail_check as _fail,
    pass_check as _pass,
)

_FAILURE_ESCALATION_FIELDS = (
    "request_id",
    "tool_name",
    "query",
    "source",
    "error_code",
    "failure_count",
    "retry_limit",
)


def failure_escalation_check(events: list[dict[str, Any]]) -> LiveValidationCheck:
    """Require at least one well-formed ``failure_escalated`` event."""

    escalations = _events_of_type(events, "failure_escalated")
    if not escalations:
        return _fail("failure_escalation", "No failure_escalated events found.")

    malformed = _malformed_failure_escalations(escalations)
    if malformed:
        return _fail(
            "failure_escalation",
            "Some failure_escalated events lack retry audit fields.",
            malformed=malformed,
        )
    return _pass(
        "failure_escalation",
        f"Found {len(escalations)} structured failure_escalated events.",
        escalations=_failure_escalation_summaries(escalations),
    )


def failure_escalation_shape_check(
    events: list[dict[str, Any]],
) -> LiveValidationCheck | None:
    """Validate ``failure_escalated`` shape when escalation events are present."""

    escalations = _events_of_type(events, "failure_escalated")
    if not escalations:
        return None

    malformed = _malformed_failure_escalations(escalations)
    if malformed:
        return _fail(
            "navigation_escalation_shape",
            "Some failure_escalated events lack retry audit fields.",
            missing_fields=malformed,
        )
    return _pass(
        "navigation_escalation_shape",
        f"Found {len(escalations)} structured navigation failure escalation events.",
        escalations=_failure_escalation_summaries(escalations),
    )


def failure_escalation_absence_check(events: list[dict[str, Any]]) -> LiveValidationCheck:
    """Fail when navigation failure escalation events are present."""

    escalations = _events_of_type(events, "failure_escalated")
    if not escalations:
        return _pass("navigation_escalation_absence", "No navigation failure escalations found.")
    return _fail(
        "navigation_escalation_absence",
        "Navigation failure escalation events were present.",
        escalations=_failure_escalation_summaries(escalations),
    )


def _malformed_failure_escalations(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    malformed: list[dict[str, Any]] = []
    for index, event in enumerate(events):
        payload = _payload(event)
        if not payload:
            malformed.append({"index": index, "missing": ["payload"]})
            continue
        missing = [field for field in _FAILURE_ESCALATION_FIELDS if field not in payload]
        if missing:
            malformed.append({"index": index, "event": payload, "missing": missing})
    return malformed


def _failure_escalation_summaries(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for event in events:
        payload = _payload(event)
        if not payload:
            continue
        summaries.append({field: payload.get(field) for field in _FAILURE_ESCALATION_FIELDS})
    return summaries


def _events_of_type(events: list[dict[str, Any]], event_type: str) -> list[dict[str, Any]]:
    return [event for event in events if event.get("event_type") == event_type]


def _payload(event: dict[str, Any]) -> dict[str, Any]:
    payload = event.get("payload") or {}
    return payload if isinstance(payload, dict) else {}
