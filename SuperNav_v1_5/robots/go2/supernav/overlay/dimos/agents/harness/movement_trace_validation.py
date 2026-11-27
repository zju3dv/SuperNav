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

"""Trace-audit checks for movement authority and direct motion results."""

from __future__ import annotations

from typing import Any

from dimos.agents.harness.validation_types import (
    LiveValidationCheck,
    fail_check as _fail,
    pass_check as _pass,
)
from dimos.agents.runtime_policy import is_movement_tool


def movement_trace_checks(events: list[dict[str, Any]]) -> list[LiveValidationCheck]:
    """Validate movement policy decisions and task-authority traces."""

    movement_policy_events = [
        event
        for event in _events_of_type(events, "tool_policy_decision")
        if _is_movement_policy(event)
    ]
    checks: list[LiveValidationCheck] = []
    if not movement_policy_events:
        checks.append(
            _fail(
                "movement_policy_decisions",
                "No movement tool_policy_decision events found.",
            )
        )
        return checks

    checks.append(
        _pass(
            "movement_policy_decisions",
            f"Found {len(movement_policy_events)} movement policy decisions.",
        )
    )
    checks.append(_movement_policy_shape_check(movement_policy_events))
    checks.append(_movement_authority_check(events))
    return checks


def rotation_result_check(events: list[dict[str, Any]]) -> LiveValidationCheck:
    """Validate rotate_in_place result traces include measured yaw evidence."""

    rotation_events = [
        event
        for event in _events_of_type(events, "movement_skill_result")
        if _payload_tool_name(event) == "rotate_in_place"
    ]
    if not rotation_events:
        return _fail("rotation_result", "No rotate_in_place movement_skill_result events found.")

    acceptable: list[dict[str, Any]] = []
    weak: list[dict[str, Any]] = []
    for event in rotation_events:
        payload = event.get("payload") or {}
        metadata = payload.get("metadata") if isinstance(payload, dict) else None
        if not isinstance(payload, dict) or not isinstance(metadata, dict):
            weak.append({"reason": "missing metadata", "payload": payload})
            continue
        measured = {
            key: metadata.get(key)
            for key in ("status", "measured_cumulative_yaw_deg", "measured_delta")
            if key in metadata
        }
        has_measurement = {
            "status",
            "measured_cumulative_yaw_deg",
            "measured_delta",
        }.issubset(metadata)
        if payload.get("success") is True and has_measurement:
            acceptable.append({"success": True, **measured})
        elif payload.get("success") is False and payload.get("error_code") and has_measurement:
            acceptable.append(
                {"success": False, "error_code": payload.get("error_code"), **measured}
            )
        else:
            weak.append(
                {
                    "success": payload.get("success"),
                    "error_code": payload.get("error_code"),
                    **measured,
                }
            )

    if acceptable:
        return _pass(
            "rotation_result",
            "Found rotate_in_place result with measured yaw evidence.",
            rotations=acceptable,
        )
    return _fail(
        "rotation_result",
        "rotate_in_place traces exist but none include measured success/failure evidence.",
        rotations=weak,
    )


def relative_move_result_check(events: list[dict[str, Any]]) -> LiveValidationCheck:
    """Validate relative_move traces include measured odom and arrival evidence."""

    relative_move_events = [
        event
        for event in _events_of_type(events, "movement_skill_result")
        if _payload_tool_name(event) == "relative_move"
    ]
    if not relative_move_events:
        return _fail("relative_move_result", "No relative_move movement_skill_result events found.")

    acceptable: list[dict[str, Any]] = []
    weak: list[dict[str, Any]] = []
    for event in relative_move_events:
        payload = event.get("payload") or {}
        metadata = payload.get("metadata") if isinstance(payload, dict) else None
        if not isinstance(payload, dict) or not isinstance(metadata, dict):
            weak.append({"reason": "missing metadata", "payload": payload})
            continue

        summary = {
            "success": payload.get("success"),
            "error_code": payload.get("error_code"),
            "duration_ms": payload.get("duration_ms"),
            "status": metadata.get("status"),
            "measured_delta": metadata.get("measured_delta"),
            "arrival_check": metadata.get("arrival_check"),
        }
        has_duration = isinstance(payload.get("duration_ms"), int | float)
        has_measured_delta = isinstance(metadata.get("measured_delta"), dict)
        arrival_check = metadata.get("arrival_check")
        has_arrival_evidence = isinstance(arrival_check, dict) or metadata.get("status") in {
            "NO_ODOM",
            "NO_PATH",
            "TIMEOUT",
            "CANCELLED",
        }
        if has_duration and has_measured_delta and has_arrival_evidence:
            acceptable.append(summary)
        else:
            weak.append(summary)

    if acceptable:
        return _pass(
            "relative_move_result",
            "Found relative_move result with measured odom, arrival, and duration evidence.",
            relative_moves=acceptable,
        )
    return _fail(
        "relative_move_result",
        "relative_move traces exist but none include measured odom, arrival, and duration evidence.",
        relative_moves=weak,
    )


def stop_result_check(events: list[dict[str, Any]]) -> LiveValidationCheck:
    """Validate stop_movement traces prove a zero-velocity command was sent."""

    stop_events = [
        event
        for event in _events_of_type(events, "movement_skill_result")
        if _payload_tool_name(event) == "stop_movement"
    ]
    if not stop_events:
        return _fail("stop_result", "No stop_movement movement_skill_result events found.")

    summaries: list[dict[str, Any]] = []
    for event in stop_events:
        payload = event.get("payload") or {}
        metadata = payload.get("metadata") if isinstance(payload, dict) else None
        if not isinstance(payload, dict) or not isinstance(metadata, dict):
            continue
        summary = {
            "success": payload.get("success"),
            "error_code": payload.get("error_code"),
            "navigation_cancelled": metadata.get("navigation_cancelled"),
            "zero_velocity_sent": metadata.get("zero_velocity_sent"),
        }
        summaries.append(summary)
        if payload.get("success") is True and metadata.get("zero_velocity_sent") is True:
            return _pass(
                "stop_result",
                "Found stop_movement result with zero-velocity command evidence.",
                stops=summaries,
            )
    return _fail(
        "stop_result",
        "stop_movement traces exist but none prove a successful zero-velocity stop.",
        stops=summaries,
    )


def _movement_policy_shape_check(events: list[dict[str, Any]]) -> LiveValidationCheck:
    missing_fields: list[dict[str, Any]] = []
    for event in events:
        payload = event.get("payload") or {}
        missing = [
            key for key in ("tool_name", "policy", "client_id", "allowed") if key not in payload
        ]
        if missing:
            missing_fields.append({"event": payload, "missing": missing})

    if missing_fields:
        return _fail(
            "movement_trace_shape",
            "Some movement policy decisions lack audit fields.",
            missing_fields=missing_fields,
        )
    return _pass(
        "movement_trace_shape",
        "Movement policy decisions include tool, policy, client, and outcome.",
    )


def _movement_authority_check(events: list[dict[str, Any]]) -> LiveValidationCheck:
    lease_events = _events_of_type(events, "movement_lease_created") + _events_of_type(
        events, "movement_lease_consumed"
    )
    if lease_events:
        return _pass(
            "movement_authority",
            f"Found {len(lease_events)} movement authority events.",
        )
    return _fail(
        "movement_authority",
        "No movement lease/task-authority events found for movement run.",
    )


def _events_of_type(events: list[dict[str, Any]], event_type: str) -> list[dict[str, Any]]:
    return [event for event in events if event.get("event_type") == event_type]


def _is_movement_policy(event: dict[str, Any]) -> bool:
    payload = event.get("payload") or {}
    if not isinstance(payload, dict):
        return False
    tool_name = str(payload.get("tool_name") or "")
    uses = payload.get("uses") or []
    return is_movement_tool(tool_name, uses)


def _payload_tool_name(event: dict[str, Any]) -> str:
    payload = event.get("payload") or {}
    if not isinstance(payload, dict):
        return ""
    return str(payload.get("tool_name") or "")
