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

"""Trace-audit checks for typed tool-status isolation."""

from __future__ import annotations

from typing import Any

from dimos.agents.harness.validation_types import (
    LiveValidationCheck,
    fail_check as _fail,
    pass_check as _pass,
)


def tool_status_isolation_check(events: list[dict[str, Any]]) -> LiveValidationCheck:
    """Validate that tool-status events stay out of model request history."""

    status_events = _events_of_type(events, "tool_status")
    status_turns = _tool_status_turn_ids(events)
    if not status_events and not status_turns:
        return _fail(
            "tool_status_isolation",
            "No typed tool_status traces or tool-status turns found.",
        )

    malformed_status = _malformed_status_events(status_events)
    status_turn_violations = _status_turn_violations(events, status_turns)
    if not malformed_status and not status_turn_violations:
        return _pass(
            "tool_status_isolation",
            "Tool-status updates are traced as UI/status events and do not assemble model requests.",
            tool_status_events=len(status_events),
            tool_status_turns=sorted(status_turns),
        )

    return _fail(
        "tool_status_isolation",
        "Some tool-status updates are not isolated from model request history.",
        malformed_status=malformed_status,
        status_turn_violations=status_turn_violations,
    )


def tool_status_context_check(events: list[dict[str, Any]]) -> LiveValidationCheck:
    """Validate that tool-status traces carry UI panel context."""

    status_events = _events_of_type(events, "tool_status")
    if not status_events:
        return _fail(
            "tool_status_context",
            "No tool_status traces found.",
        )

    missing_context: list[dict[str, Any]] = []
    malformed_context: list[dict[str, Any]] = []
    context_count = 0
    navigation_context_count = 0
    for index, event in enumerate(status_events):
        payload = _payload(event)
        context = payload.get("status_context")
        if not isinstance(context, dict):
            missing_context.append(
                {
                    "event_index": index,
                    "tool": payload.get("tool"),
                }
            )
            continue
        context_count += 1
        missing = _missing_status_context_fields(context)
        navigation_task = context.get("navigation_task")
        if isinstance(navigation_task, dict):
            navigation_context_count += 1
            missing.extend(
                f"navigation_task.{field}"
                for field in ("request_id", "status", "source", "error_code")
                if field not in navigation_task
            )
        if missing:
            malformed_context.append(
                {
                    "event_index": index,
                    "tool": payload.get("tool"),
                    "missing": missing,
                }
            )

    missing_navigation = _missing_navigation_status_context(events, navigation_context_count)
    if not missing_context and not malformed_context and not missing_navigation:
        return _pass(
            "tool_status_context",
            "Tool-status traces carry Human CLI panel context for policy, authority, and protection.",
            tool_status_events=len(status_events),
            context_events=context_count,
            navigation_context_events=navigation_context_count,
        )

    return _fail(
        "tool_status_context",
        "Some tool-status traces are missing Human CLI panel context.",
        missing_context=missing_context,
        malformed_context=malformed_context,
        missing_navigation=missing_navigation,
    )


def _events_of_type(events: list[dict[str, Any]], event_type: str) -> list[dict[str, Any]]:
    return [event for event in events if event.get("event_type") == event_type]


def _payload(event: dict[str, Any]) -> dict[str, Any]:
    payload = event.get("payload") or {}
    return payload if isinstance(payload, dict) else {}


def _tool_status_turn_ids(events: list[dict[str, Any]]) -> set[str]:
    turn_ids: set[str] = set()
    for event in _events_of_type(events, "turn_start"):
        payload = _payload(event)
        if payload.get("message_source") != "tool_status":
            continue
        turn_id = payload.get("turn_id")
        if isinstance(turn_id, str):
            turn_ids.add(turn_id)
    return turn_ids


def _malformed_status_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    malformed: list[dict[str, Any]] = []
    required_values = {
        "route": "ui_status_only",
        "model_history_appended": False,
        "external_user_turn": False,
    }
    for index, event in enumerate(events):
        payload = _payload(event)
        mismatches = {
            field: payload.get(field)
            for field, expected in required_values.items()
            if payload.get(field) != expected
        }
        missing = [field for field in ("tool", "event_kind", "text") if field not in payload]
        if mismatches or missing:
            malformed.append(
                {
                    "event_index": index,
                    "missing": missing,
                    "mismatches": mismatches,
                }
            )
    return malformed


def _status_turn_violations(
    events: list[dict[str, Any]],
    status_turn_ids: set[str],
) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    if not status_turn_ids:
        return violations

    assembled_turn_ids = _event_ids(_events_of_type(events, "request_assembled"))
    ended_as_status = {
        turn_id
        for turn_id, outcome in _turn_end_outcomes(events).items()
        if outcome == "tool_status"
    }
    for turn_id in sorted(status_turn_ids):
        if turn_id in assembled_turn_ids:
            violations.append(
                {
                    "turn_id": turn_id,
                    "reason": "tool_status_turn_assembled_model_request",
                }
            )
        if turn_id not in ended_as_status:
            violations.append(
                {
                    "turn_id": turn_id,
                    "reason": "tool_status_turn_did_not_end_as_status",
                }
            )
    return violations


def _event_ids(events: list[dict[str, Any]]) -> set[str]:
    ids: set[str] = set()
    for event in events:
        turn_id = _payload(event).get("turn_id")
        if isinstance(turn_id, str):
            ids.add(turn_id)
    return ids


def _turn_end_outcomes(events: list[dict[str, Any]]) -> dict[str, str]:
    outcomes: dict[str, str] = {}
    for event in _events_of_type(events, "turn_end"):
        payload = _payload(event)
        turn_id = payload.get("turn_id")
        outcome = payload.get("outcome")
        if isinstance(turn_id, str) and isinstance(outcome, str):
            outcomes[turn_id] = outcome
    return outcomes


def _missing_status_context_fields(context: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    policy = context.get("policy")
    if not isinstance(policy, dict):
        missing.append("policy")
    else:
        missing.extend(
            f"policy.{field}"
            for field in ("profile", "movement_mode", "tool_scope")
            if field not in policy
        )

    authority = context.get("movement_authority")
    if not isinstance(authority, dict):
        missing.append("movement_authority")
    else:
        missing.extend(
            f"movement_authority.{field}"
            for field in ("client_id", "authorization", "confirmed")
            if field not in authority
        )

    protection = context.get("protection")
    if not isinstance(protection, dict):
        missing.append("protection")
    else:
        missing.extend(
            f"protection.{field}"
            for field in ("blocker", "collision_obstacle_protection", "unitree_protection")
            if field not in protection
        )
    return missing


def _missing_navigation_status_context(
    events: list[dict[str, Any]],
    navigation_context_count: int,
) -> list[dict[str, Any]]:
    navigation_events = _events_of_type(events, "navigation_task_updated")
    if not navigation_events or navigation_context_count:
        return []
    return [
        {
            "reason": "navigation_task_updated_without_tool_status_context",
            "navigation_events": len(navigation_events),
        }
    ]
