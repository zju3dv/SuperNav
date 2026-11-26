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

"""Trace-audit checks for native harness task state and control commands."""

from __future__ import annotations

from typing import Any

from dimos.agents.harness.context_trace_validation import assembly_payload
from dimos.agents.harness.validation_types import (
    LiveValidationCheck,
    fail_check as _fail,
    pass_check as _pass,
)


def task_state_check(
    events: list[dict[str, Any]],
    request_events: list[dict[str, Any]],
) -> LiveValidationCheck:
    """Validate active task state is both updated and pinned into requests."""

    updates = _events_of_type(events, "task_state_updated")
    pinned_requests: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for index, event in enumerate(request_events):
        payload = event.get("payload") or {}
        task_state = payload.get("task_state") if isinstance(payload, dict) else None
        active_task = task_state.get("active_task") if isinstance(task_state, dict) else None
        assembly = assembly_payload(event)
        if isinstance(active_task, dict) and assembly.get("pinned_messages"):
            pinned_requests.append(
                {
                    "request_index": index,
                    "task_id": active_task.get("task_id"),
                    "goal": active_task.get("goal"),
                    "pinned_messages": assembly.get("pinned_messages"),
                }
            )
        else:
            missing.append(
                {
                    "request_index": index,
                    "has_task_state": isinstance(task_state, dict),
                    "pinned_messages": assembly.get("pinned_messages"),
                }
            )

    if updates and pinned_requests:
        return _pass(
            "task_state",
            "Found task_state_updated events and pinned task context in request assembly.",
            updates=len(updates),
            pinned_requests=pinned_requests,
        )
    return _fail(
        "task_state",
        "No pinned active task state found in request_assembled events.",
        updates=len(updates),
        missing=missing,
    )


def task_control_check(events: list[dict[str, Any]]) -> LiveValidationCheck:
    """Validate local task-control commands are visible in trace logs."""

    controls = _events_of_type(events, "task_control_command")
    if controls:
        commands: list[dict[str, Any]] = []
        for event in controls:
            payload = event.get("payload") or {}
            if not isinstance(payload, dict):
                continue
            commands.append(
                {
                    "command": payload.get("command"),
                    "route": payload.get("route"),
                    "active_task": payload.get("active_task"),
                }
            )
        return _pass(
            "task_control",
            f"Found {len(controls)} task-control command traces.",
            commands=commands,
        )
    return _fail(
        "task_control",
        "No task_control_command trace events found.",
    )


def _events_of_type(events: list[dict[str, Any]], event_type: str) -> list[dict[str, Any]]:
    return [event for event in events if event.get("event_type") == event_type]
