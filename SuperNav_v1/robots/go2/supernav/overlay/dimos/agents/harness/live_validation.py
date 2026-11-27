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

"""Trace-based live validation for Go2 harness experiments."""

from __future__ import annotations

from typing import Any

from dimos.agents.harness.context_compaction_validation import context_compaction_trace_check
from dimos.agents.harness.context_trace_validation import (
    context_budget_check,
    context_budget_exhaustion_trace_check,
    context_image_count_check,
    context_image_token_budget_check,
    context_token_growth_check,
    request_assembly_check,
    visual_evidence_check,
)
from dimos.agents.harness.failure_escalation_validation import (
    failure_escalation_absence_check,
    failure_escalation_check,
)
from dimos.agents.harness.movement_trace_validation import (
    movement_trace_checks,
    relative_move_result_check,
    rotation_result_check,
    stop_result_check,
)
from dimos.agents.harness.native_auto_followup_validation import auto_followup_trace_check
from dimos.agents.harness.navigation_trace_validation import navigation_trace_checks
from dimos.agents.harness.preemption_validation import preemption_checks
from dimos.agents.harness.request_contract_validation import request_contract_check
from dimos.agents.harness.request_scope_validation import request_tool_scope_check
from dimos.agents.harness.session_lifecycle_validation import session_lifecycle_check
from dimos.agents.harness.task_followup_validation import task_followup_check
from dimos.agents.harness.task_trace_validation import (
    task_control_check,
    task_state_check,
)
from dimos.agents.harness.tool_execution_validation import (
    tool_call_terminal_check,
    tool_execution_trace_check,
)
from dimos.agents.harness.tool_status_validation import (
    tool_status_context_check,
    tool_status_isolation_check,
)
from dimos.agents.harness.trace_payload_validation import raw_trace_payload_absence_check
from dimos.agents.harness.turn_lifecycle_validation import (
    model_response_trace_check,
    model_turn_completion_check,
    turn_error_absence_check,
    turn_lifecycle_check,
)
from dimos.agents.harness.validation_types import (
    LiveValidationCheck,
    LiveValidationReport,
    fail_check as _fail,
    pass_check as _pass,
)


def validate_live_trace_events(
    events: list[dict[str, Any]],
    *,
    require_visual: bool = False,
    require_session_lifecycle: bool = False,
    require_turn_lifecycle: bool = False,
    require_model_turn: bool = False,
    require_model_response: bool = False,
    forbid_turn_errors: bool = False,
    require_movement: bool = False,
    require_navigation: bool = False,
    require_request_contract: bool = False,
    require_request_tool_scope: bool = False,
    require_task_state: bool = False,
    require_task_control: bool = False,
    require_task_followup: bool = False,
    require_auto_followup: bool = False,
    require_tool_status_isolated: bool = False,
    require_tool_status_context: bool = False,
    require_tool_execution: bool = False,
    require_tool_call_terminal: bool = False,
    forbid_raw_trace_payloads: bool = False,
    require_context_compaction: bool = False,
    require_failure_escalation: bool = False,
    require_relative_move: bool = False,
    require_rotation: bool = False,
    require_stop: bool = False,
    require_preemption: bool = False,
    preemption_max_ms: float | None = None,
    expected_preempted_by: list[str] | None = None,
    max_context_token_growth: int | None = None,
    max_context_images: int | None = None,
    max_context_image_tokens: int | None = None,
    forbid_navigation_escalation: bool = False,
) -> LiveValidationReport:
    """Validate high-level PRD invariants from harness trace events."""

    checks: list[LiveValidationCheck] = []
    checks.append(_events_present_check(events))
    if require_session_lifecycle:
        checks.append(session_lifecycle_check(events))
    if require_turn_lifecycle:
        checks.append(turn_lifecycle_check(events))
    if require_model_turn:
        checks.append(model_turn_completion_check(events))
    if require_model_response:
        checks.append(model_response_trace_check(events))
    if forbid_turn_errors:
        checks.append(turn_error_absence_check(events))

    request_events = _events_of_type(events, "request_assembled")
    checks.append(request_assembly_check(request_events))
    if require_request_contract:
        checks.append(request_contract_check(request_events))
    if require_request_tool_scope:
        checks.append(request_tool_scope_check(request_events))
    checks.append(context_budget_check(request_events))
    checks.append(context_budget_exhaustion_trace_check(events, request_events))
    if max_context_token_growth is not None:
        checks.append(context_token_growth_check(request_events, max_context_token_growth))
    if max_context_images is not None:
        checks.append(context_image_count_check(request_events, max_context_images))
    if max_context_image_tokens is not None:
        checks.append(context_image_token_budget_check(request_events, max_context_image_tokens))
    if require_context_compaction:
        checks.append(context_compaction_trace_check(events, request_events))

    if require_visual:
        checks.append(visual_evidence_check(request_events))
    if require_task_state:
        checks.append(task_state_check(events, request_events))
    if require_task_control:
        checks.append(task_control_check(events))
    if require_task_followup:
        checks.append(task_followup_check(events))
    if require_auto_followup:
        checks.append(auto_followup_trace_check(events))
    if require_tool_status_isolated:
        checks.append(tool_status_isolation_check(events))
    if require_tool_status_context:
        checks.append(tool_status_context_check(events))
    if require_tool_execution:
        checks.append(tool_execution_trace_check(events))
    if require_tool_call_terminal:
        checks.append(tool_call_terminal_check(events))
    if forbid_raw_trace_payloads:
        checks.append(raw_trace_payload_absence_check(events))
    if require_failure_escalation:
        checks.append(failure_escalation_check(events))
    if require_movement:
        checks.extend(movement_trace_checks(events))
    if require_navigation:
        checks.extend(navigation_trace_checks(events))
    if require_relative_move:
        checks.append(relative_move_result_check(events))
    if require_rotation:
        checks.append(rotation_result_check(events))
    if require_stop:
        checks.append(stop_result_check(events))
    if forbid_navigation_escalation:
        checks.append(failure_escalation_absence_check(events))
    if require_preemption or preemption_max_ms is not None or expected_preempted_by:
        checks.extend(
            preemption_checks(
                events,
                preemption_max_ms=preemption_max_ms,
                expected_preempted_by=expected_preempted_by,
            )
        )

    return LiveValidationReport(checks=checks, event_count=len(events))


def _events_present_check(events: list[dict[str, Any]]) -> LiveValidationCheck:
    if events:
        return _pass("events_present", f"Found {len(events)} harness trace events.")
    return _fail("events_present", "No harness trace events found in the selected log window.")


def _events_of_type(events: list[dict[str, Any]], event_type: str) -> list[dict[str, Any]]:
    return [event for event in events if event.get("event_type") == event_type]
