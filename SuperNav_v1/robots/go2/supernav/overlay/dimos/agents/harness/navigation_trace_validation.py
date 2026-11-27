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

"""Trace-audit checks for navigation task lifecycle events."""

from __future__ import annotations

from typing import Any

from dimos.agents.harness.failure_escalation_validation import failure_escalation_shape_check
from dimos.agents.harness.validation_types import (
    LiveValidationCheck,
    fail_check as _fail,
    pass_check as _pass,
)


def navigation_trace_checks(events: list[dict[str, Any]]) -> list[LiveValidationCheck]:
    """Validate navigation task lifecycle and planner result traces."""

    navigation_events = _events_of_type(events, "navigation_task_updated")
    checks: list[LiveValidationCheck] = []
    if not navigation_events:
        return [
            _fail(
                "navigation_lifecycle",
                "No navigation_task_updated events found.",
            )
        ]

    checks.append(
        _pass(
            "navigation_lifecycle",
            f"Found {len(navigation_events)} navigation task lifecycle events.",
        )
    )
    checks.append(_navigation_trace_shape_check(navigation_events))
    checks.append(_navigation_consistency_check(navigation_events))
    checks.append(_navigation_planner_result_check(navigation_events))
    escalation_shape = failure_escalation_shape_check(events)
    if escalation_shape is not None:
        checks.append(escalation_shape)
    return checks


def _navigation_trace_shape_check(events: list[dict[str, Any]]) -> LiveValidationCheck:
    missing_fields: list[dict[str, Any]] = []
    for event in events:
        payload = event.get("payload") or {}
        if not isinstance(payload, dict):
            missing_fields.append({"event": event, "missing": ["payload"]})
            continue
        task = payload.get("navigation_task")
        missing = [
            key
            for key in ("request_id", "tool_name", "query", "source", "status")
            if key not in payload
        ]
        if not isinstance(task, dict):
            missing.append("navigation_task")
        elif task.get("request_id") != payload.get("request_id"):
            missing.append("navigation_task.request_id")
        if missing:
            missing_fields.append({"event": payload, "missing": missing})

    if missing_fields:
        return _fail(
            "navigation_trace_shape",
            "Some navigation task events lack audit fields.",
            missing_fields=missing_fields,
        )
    return _pass(
        "navigation_trace_shape",
        "Navigation task events include request, tool, source, status, and task payload.",
    )


def _navigation_consistency_check(events: list[dict[str, Any]]) -> LiveValidationCheck:
    updates_by_request: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        payload = event.get("payload") or {}
        if not isinstance(payload, dict):
            continue
        request_id = str(payload.get("request_id") or "")
        if request_id:
            updates_by_request.setdefault(request_id, []).append(payload)

    inconsistent: list[dict[str, Any]] = []
    failure_statuses = {
        "ARRIVAL_NOT_CONFIRMED",
        "CANCELLED",
        "NO_ODOM",
        "NO_PATH",
        "NO_SAFE_GOAL",
        "NOT_CONFIGURED",
        "RETRY_LIMIT_EXCEEDED",
        "TARGET_NOT_FOUND",
        "TIMEOUT",
        "VISION_UNAVAILABLE",
    }
    for request_id, updates in updates_by_request.items():
        seen_terminal_failure = False
        for update in updates:
            status = str(update.get("status") or "")
            if seen_terminal_failure and status == "STARTED":
                inconsistent.append({"request_id": request_id, "status_after_failure": status})
            if status in failure_statuses:
                seen_terminal_failure = True

    if not inconsistent:
        return _pass("navigation_consistency", "No navigation task starts after a failure status.")
    return _fail(
        "navigation_consistency",
        "Navigation task trace shows STARTED after a failure status.",
        inconsistent=inconsistent,
    )


def _navigation_planner_result_check(events: list[dict[str, Any]]) -> LiveValidationCheck:
    planner_events: list[dict[str, Any]] = []
    missing_or_weak: list[dict[str, Any]] = []
    for event in events:
        payload = event.get("payload") or {}
        if not isinstance(payload, dict):
            continue
        status = str(payload.get("status") or "")
        if status in {"CREATED", "TARGET_NOT_FOUND", "RETRY_LIMIT_EXCEEDED"}:
            continue
        task = payload.get("navigation_task")
        planner_goal = payload.get("planner_goal_result")
        if planner_goal is None and isinstance(task, dict):
            planner_goal = task.get("planner_goal_result")
        if planner_goal is None:
            missing_or_weak.append({"status": status, "missing": "planner_goal_result"})
            continue
        if not isinstance(planner_goal, dict):
            missing_or_weak.append({"status": status, "missing": "planner_goal_result.object"})
            continue
        missing = [
            key
            for key in ("accepted", "planner_request_id", "safe_goal_found", "path_found")
            if key not in planner_goal
        ]
        if missing:
            missing_or_weak.append({"status": status, "missing": missing})
            continue
        planner_events.append(
            {
                "status": status,
                "accepted": planner_goal.get("accepted"),
                "error_code": planner_goal.get("error_code"),
                "path_found": planner_goal.get("path_found"),
                "safe_goal_found": planner_goal.get("safe_goal_found"),
            }
        )

    if planner_events and not missing_or_weak:
        return _pass(
            "navigation_planner_result",
            "Navigation task traces include structured planner goal results.",
            planner_results=planner_events,
        )
    if not planner_events and not missing_or_weak:
        return _pass(
            "navigation_planner_result",
            "No planner-submitted navigation events required planner result evidence.",
        )
    return _fail(
        "navigation_planner_result",
        "Some planner-submitted navigation events lack structured planner goal results.",
        missing_or_weak=missing_or_weak,
        planner_results=planner_events,
    )


def _events_of_type(events: list[dict[str, Any]], event_type: str) -> list[dict[str, Any]]:
    return [event for event in events if event.get("event_type") == event_type]
