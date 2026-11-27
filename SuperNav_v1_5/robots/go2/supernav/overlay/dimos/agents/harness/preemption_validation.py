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

"""Validation checks for movement-authority preemption traces."""

from __future__ import annotations

from typing import Any

from dimos.agents.harness.validation_types import (
    LiveValidationCheck,
    fail_check,
    pass_check,
)


def preemption_checks(
    events: list[dict[str, Any]],
    *,
    preemption_max_ms: float | None,
    expected_preempted_by: list[str] | None,
) -> list[LiveValidationCheck]:
    """Validate preemptive movement-stop trace shape, latency, and reason."""

    preemptions = [event for event in events if event.get("event_type") == "movement_preempted"]
    if not preemptions:
        checks = [fail_check("preemption", "No movement_preempted events found.")]
        terminal_check = _task_terminal_preemption_check(events, preemptions)
        if terminal_check is not None:
            checks.append(terminal_check)
        return checks

    checks = [pass_check("preemption", f"Found {len(preemptions)} movement_preempted events.")]
    checks.append(_preemption_shape_check(preemptions))
    checks.append(_post_preemption_movement_check(events))
    terminal_check = _task_terminal_preemption_check(events, preemptions)
    if terminal_check is not None:
        checks.append(terminal_check)
    if preemption_max_ms is not None:
        checks.append(_preemption_latency_check(preemptions, preemption_max_ms))
    if expected_preempted_by:
        checks.append(_preemption_reason_check(preemptions, expected_preempted_by))
    return checks


def _task_terminal_preemption_check(
    events: list[dict[str, Any]],
    preemptions: list[dict[str, Any]],
) -> LiveValidationCheck | None:
    requests = [
        event for event in events if event.get("event_type") == "task_terminal_preemption_requested"
    ]
    unavailable = [
        event
        for event in events
        if event.get("event_type") == "task_terminal_preemption_unavailable"
    ]
    if not requests and not unavailable:
        return None

    if unavailable:
        return fail_check(
            "task_terminal_preemption",
            "Model task-terminal preemption request could not reach the MCP server.",
            unavailable=[event.get("payload") or {} for event in unavailable],
        )

    missing: list[dict[str, Any]] = []
    matched: list[dict[str, Any]] = []
    for request_index, event in enumerate(requests):
        payload = event.get("payload") or {}
        if not isinstance(payload, dict):
            missing.append({"request_index": request_index, "reason": "missing_payload"})
            continue
        expected_count = _safe_int(payload.get("preempted_count"))
        if expected_count <= 0:
            matched.append(
                {
                    "request_index": request_index,
                    "preempted_by": payload.get("preempted_by"),
                    "preempted_count": expected_count,
                    "matching_preemptions": 0,
                }
            )
            continue

        matches = _matching_task_terminal_preemptions(payload, preemptions)
        if len(matches) >= expected_count:
            matched.append(
                {
                    "request_index": request_index,
                    "preempted_by": payload.get("preempted_by"),
                    "preempted_count": expected_count,
                    "matching_preemptions": len(matches),
                }
            )
            continue
        missing.append(
            {
                "request_index": request_index,
                "preempted_by": payload.get("preempted_by"),
                "client_ids": payload.get("client_ids"),
                "preempted_count": expected_count,
                "matching_preemptions": len(matches),
            }
        )

    if not missing:
        return pass_check(
            "task_terminal_preemption",
            "Model task-terminal preemption requests have matching movement_preempted traces.",
            matched=matched,
        )
    return fail_check(
        "task_terminal_preemption",
        "Some model task-terminal preemption requests lack matching movement_preempted traces.",
        matched=matched,
        missing=missing,
    )


def _matching_task_terminal_preemptions(
    request: dict[str, Any],
    preemptions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    preempted_by = request.get("preempted_by")
    raw_client_ids = request.get("client_ids")
    client_ids = (
        {str(client_id) for client_id in raw_client_ids if client_id}
        if isinstance(raw_client_ids, list)
        else set()
    )
    matches: list[dict[str, Any]] = []
    for event_index, event in enumerate(preemptions):
        payload = event.get("payload") or {}
        if not isinstance(payload, dict):
            continue
        if payload.get("preempted_by") != preempted_by:
            continue
        if client_ids and str(payload.get("client_id")) not in client_ids:
            continue
        matches.append(
            {
                "event_index": event_index,
                "client_id": payload.get("client_id"),
                "lease_id": payload.get("lease_id"),
                "preempted_by": payload.get("preempted_by"),
            }
        )
    return matches


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _preemption_reason_check(
    events: list[dict[str, Any]],
    expected_reasons: list[str],
) -> LiveValidationCheck:
    expected = {str(reason) for reason in expected_reasons if str(reason)}
    found: list[dict[str, Any]] = []
    missing = set(expected)
    for index, event in enumerate(events):
        payload = event.get("payload") or {}
        reason = payload.get("preempted_by") if isinstance(payload, dict) else None
        if reason in expected:
            missing.discard(str(reason))
            found.append({"event_index": index, "preempted_by": reason})

    if not missing:
        return pass_check(
            "preemption_reason",
            f"Found expected preemption reason(s): {', '.join(sorted(expected))}.",
            expected=sorted(expected),
            found=found,
        )
    return fail_check(
        "preemption_reason",
        f"Missing expected preemption reason(s): {', '.join(sorted(missing))}.",
        expected=sorted(expected),
        found=found,
        missing=sorted(missing),
    )


def _preemption_shape_check(events: list[dict[str, Any]]) -> LiveValidationCheck:
    missing_fields: list[dict[str, Any]] = []
    for event in events:
        payload = event.get("payload") or {}
        if not isinstance(payload, dict):
            missing_fields.append({"event": event, "missing": ["payload"]})
            continue
        missing = [
            key
            for key in (
                "preempted_by",
                "attempted_stop_tools",
                "succeeded_stop_tools",
                "stop_results",
                "duration_ms",
                "client_id",
                "lease_id",
                "epoch",
            )
            if key not in payload
        ]
        if missing:
            missing_fields.append({"event": payload, "missing": missing})
            continue
        attempted = payload.get("attempted_stop_tools")
        succeeded = payload.get("succeeded_stop_tools")
        if not isinstance(attempted, list) or not attempted:
            missing_fields.append({"event": payload, "missing": ["attempted_stop_tools[]"]})
        if not isinstance(succeeded, list) or not succeeded:
            missing_fields.append({"event": payload, "missing": ["succeeded_stop_tools[]"]})
        stop_results = payload.get("stop_results")
        if not isinstance(stop_results, list):
            missing_fields.append({"event": payload, "missing": ["stop_results[]"]})
            continue
        for stop_result in stop_results:
            if not isinstance(stop_result, dict):
                missing_fields.append({"event": payload, "missing": ["stop_results[].object"]})
                continue
            stop_missing = [
                key for key in ("tool_name", "success", "duration_ms") if key not in stop_result
            ]
            if stop_missing:
                missing_fields.append({"event": stop_result, "missing": stop_missing})

    if not missing_fields:
        return pass_check(
            "preemption_shape",
            "Preemption traces include stop tools, results, and measured duration.",
        )
    return fail_check(
        "preemption_shape",
        "Some movement_preempted events lack stop/preemption audit fields.",
        missing_fields=missing_fields,
    )


def _post_preemption_movement_check(events: list[dict[str, Any]]) -> LiveValidationCheck:
    """Fail if a preempted movement ticket is consumed or succeeds after stop."""

    scoped_preemptions = _scoped_preemptions(events)
    violations: list[dict[str, Any]] = []
    for preemption in scoped_preemptions:
        for event_index, event in enumerate(events[preemption["event_index"] + 1 :]):
            actual_index = preemption["event_index"] + 1 + event_index
            violation = _post_preemption_violation(preemption, event, actual_index)
            if violation is not None:
                violations.append(violation)

    if not violations:
        return pass_check(
            "post_preemption_movement",
            "No preempted movement ticket was consumed or reported successful after preemption.",
            preemptions_checked=len(scoped_preemptions),
        )
    return fail_check(
        "post_preemption_movement",
        "A movement ticket continued after its authority was preempted.",
        violations=violations,
    )


def _scoped_preemptions(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scoped: list[dict[str, Any]] = []
    for index, event in enumerate(events):
        if event.get("event_type") != "movement_preempted":
            continue
        payload = event.get("payload") or {}
        if not isinstance(payload, dict):
            continue
        client_id = payload.get("client_id")
        lease_id = payload.get("lease_id")
        epoch = payload.get("epoch")
        if client_id is None or lease_id is None or epoch is None:
            continue
        scoped.append(
            {
                "event_index": index,
                "client_id": client_id,
                "lease_id": lease_id,
                "epoch": epoch,
                "preempted_by": payload.get("preempted_by"),
            }
        )
    return scoped


def _post_preemption_violation(
    preemption: dict[str, Any],
    event: dict[str, Any],
    event_index: int,
) -> dict[str, Any] | None:
    event_type = str(event.get("event_type") or "")
    payload = event.get("payload") or {}
    if not isinstance(payload, dict):
        return None

    if event_type == "movement_lease_consumed" and _same_scope(preemption, payload):
        return {
            "event_index": event_index,
            "event_type": event_type,
            "reason": "preempted_lease_consumed_again",
            "client_id": payload.get("client_id"),
            "lease_id": payload.get("lease_id"),
            "epoch": payload.get("epoch"),
            "tool_name": payload.get("tool_name"),
        }

    if event_type in {"tool_execution_start", "tool_execution_result"}:
        ticket = payload.get("movement_ticket")
        if not isinstance(ticket, dict) or not _same_scope(preemption, ticket):
            return None
        if event_type == "tool_execution_start":
            return {
                "event_index": event_index,
                "event_type": event_type,
                "reason": "preempted_ticket_started_execution",
                "client_id": ticket.get("client_id"),
                "lease_id": ticket.get("lease_id"),
                "epoch": ticket.get("epoch"),
                "tool_name": ticket.get("tool_name"),
            }
        if payload.get("success") is True:
            return {
                "event_index": event_index,
                "event_type": event_type,
                "reason": "preempted_ticket_reported_success",
                "client_id": ticket.get("client_id"),
                "lease_id": ticket.get("lease_id"),
                "epoch": ticket.get("epoch"),
                "tool_name": ticket.get("tool_name"),
            }
    return None


def _same_scope(preemption: dict[str, Any], payload: dict[str, Any]) -> bool:
    return (
        payload.get("client_id") == preemption.get("client_id")
        and payload.get("lease_id") == preemption.get("lease_id")
        and payload.get("epoch") == preemption.get("epoch")
    )


def _preemption_latency_check(
    events: list[dict[str, Any]],
    max_ms: float,
) -> LiveValidationCheck:
    over_budget: list[dict[str, Any]] = []
    for event in events:
        payload = event.get("payload") or {}
        if not isinstance(payload, dict):
            over_budget.append({"duration_ms": None, "max_ms": max_ms})
            continue
        duration_ms = payload.get("duration_ms")
        try:
            if float(duration_ms) > max_ms:
                over_budget.append({"duration_ms": duration_ms, "max_ms": max_ms})
        except (TypeError, ValueError):
            over_budget.append({"duration_ms": duration_ms, "max_ms": max_ms})

    if not over_budget:
        return pass_check(
            "preemption_latency",
            f"All movement_preempted events completed within {max_ms:g} ms.",
        )
    return fail_check(
        "preemption_latency",
        f"At least one movement_preempted event exceeded {max_ms:g} ms.",
        over_budget=over_budget,
    )
