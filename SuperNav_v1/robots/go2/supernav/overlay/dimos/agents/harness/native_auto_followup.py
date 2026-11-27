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

"""Bounded native-agent follow-up prompts for stalled active tasks."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import HumanMessage

from dimos.agents.harness.native_internal_message import native_internal_message
from dimos.agents.harness.task_followup import task_followup_summary

DEFAULT_MAX_AUTO_FOLLOWUPS_PER_TASK = 1
DEFAULT_MAX_NAVIGATION_STATUS_FOLLOWUPS_PER_TASK = 3
PENDING_NAVIGATION_FOLLOWUP_REASON = "pending_navigation_requires_status_followup"
ZERO_TOOL_FOLLOWUP_REASON = "pending_task_model_no_tool_calls"
DIAGNOSTIC_TOOL_FOLLOWUP_REASON = "pending_task_after_diagnostic_tool_calls"
CONTEXT_GATHERING_TOOL_NAMES = frozenset(
    {
        "last_motion_diagnostics",
        "motion_safety_status",
        "observe",
        "read_image",
        "server_status",
        "visual_find",
        "look_out_for",
    }
)


@dataclass(frozen=True)
class NativeAutoFollowupState:
    """Per-task auto-follow-up counters carried by the native runtime state."""

    attempts: tuple[tuple[str, int], ...] = ()

    def count_for(self, task_id: str, *, reason: str | None = None) -> int:
        """Return how many auto follow-ups were already queued for a task."""

        key = _attempt_key(task_id, reason)
        for existing_id, count in self.attempts:
            if existing_id == key:
                return count
        return 0

    def increment(
        self,
        task_id: str,
        *,
        reason: str | None = None,
    ) -> NativeAutoFollowupState:
        """Return a state with one more follow-up attempt for ``task_id``."""

        key = _attempt_key(task_id, reason)
        updated: list[tuple[str, int]] = []
        found = False
        for existing_id, count in self.attempts:
            if existing_id == key:
                updated.append((existing_id, count + 1))
                found = True
            else:
                updated.append((existing_id, count))
        if not found:
            updated.append((key, 1))
        return NativeAutoFollowupState(attempts=tuple(updated[-20:]))


@dataclass(frozen=True)
class NativeAutoFollowupDecision:
    """Decision for whether to queue an internal native-agent follow-up turn."""

    should_enqueue: bool
    reason: str
    state: NativeAutoFollowupState
    message: HumanMessage | None = None
    task_id: str | None = None
    attempt: int = 0
    max_attempts: int = DEFAULT_MAX_AUTO_FOLLOWUPS_PER_TASK
    details: dict[str, Any] = field(default_factory=dict)

    def trace_payload(self) -> dict[str, Any]:
        """Return trace-safe metadata for an auto-follow-up decision."""

        return {
            "task_id": self.task_id,
            "reason": self.reason,
            "attempt": self.attempt,
            "max_attempts": self.max_attempts,
            "details": self.details,
        }


def decide_native_auto_followup(
    *,
    task_snapshot: dict[str, Any],
    model_update_count: int,
    model_tool_call_count: int,
    model_tool_call_names: list[str] | None = None,
    queue_empty: bool,
    state: NativeAutoFollowupState,
    max_attempts: int = DEFAULT_MAX_AUTO_FOLLOWUPS_PER_TASK,
    max_navigation_attempts: int = DEFAULT_MAX_NAVIGATION_STATUS_FOLLOWUPS_PER_TASK,
) -> NativeAutoFollowupDecision:
    """Queue bounded internal continuations for pending active tasks.

    Generic "the model said something but did not use tools" stalls get one
    nudge. Navigation status polling gets a separate small budget because a
    physical navigation task often needs more than one status check to reach a
    terminal state.
    """

    active_task = _active_task(task_snapshot)
    if not queue_empty:
        return _no_followup("queue_not_empty", state, active_task)
    if active_task is None:
        return _no_followup("no_active_task", state, active_task)
    if str(active_task.get("status") or "") != "active":
        return _no_followup("task_not_active", state, active_task)
    if model_update_count <= 0:
        return _no_followup("no_model_updates", state, active_task)

    task_id = str(active_task.get("task_id") or "")
    if not task_id:
        return _no_followup("missing_task_id", state, active_task)

    follow_up = task_followup_summary(active_task)
    if model_tool_call_count > 0:
        if _needs_navigation_status_followup(follow_up):
            return _queue_followup(
                active_task=active_task,
                follow_up=follow_up,
                state=state,
                task_id=task_id,
                reason=PENDING_NAVIGATION_FOLLOWUP_REASON,
                max_attempts=max_navigation_attempts,
                model_update_count=model_update_count,
                model_tool_call_count=model_tool_call_count,
                prompt_cause=(
                    "your previous turn used tool calls, but navigation is still non-terminal"
                ),
            )
        context_tools = context_gathering_tool_names(model_tool_call_names or [])
        if context_tools:
            return _queue_followup(
                active_task=active_task,
                follow_up=follow_up,
                state=state,
                task_id=task_id,
                reason=DIAGNOSTIC_TOOL_FOLLOWUP_REASON,
                max_attempts=max_attempts,
                model_update_count=model_update_count,
                model_tool_call_count=model_tool_call_count,
                prompt_cause=(
                    "your previous turn only used observation/diagnostic tool calls: "
                    f"{', '.join(context_tools)}"
                ),
            )
        return _no_followup("model_used_tools", state, active_task)

    return _queue_followup(
        active_task=active_task,
        follow_up=follow_up,
        state=state,
        task_id=task_id,
        reason=ZERO_TOOL_FOLLOWUP_REASON,
        max_attempts=max_attempts,
        model_update_count=model_update_count,
        model_tool_call_count=model_tool_call_count,
        prompt_cause="your previous turn emitted no tool calls",
    )


def _queue_followup(
    *,
    active_task: dict[str, Any],
    follow_up: dict[str, Any],
    state: NativeAutoFollowupState,
    task_id: str,
    reason: str,
    max_attempts: int,
    model_update_count: int,
    model_tool_call_count: int,
    prompt_cause: str,
) -> NativeAutoFollowupDecision:
    max_attempts = max(0, int(max_attempts))
    attempt_reason = _attempt_reason(reason)
    attempt = state.count_for(task_id, reason=attempt_reason) + 1
    if attempt > max_attempts:
        return NativeAutoFollowupDecision(
            should_enqueue=False,
            reason="max_attempts_reached",
            state=state,
            task_id=task_id,
            attempt=attempt - 1,
            max_attempts=max_attempts,
            details={"followup_reason": reason},
        )

    updated_state = state.increment(task_id, reason=attempt_reason)
    return NativeAutoFollowupDecision(
        should_enqueue=True,
        reason=reason,
        state=updated_state,
        message=_followup_message(active_task, follow_up=follow_up, prompt_cause=prompt_cause),
        task_id=task_id,
        attempt=attempt,
        max_attempts=max_attempts,
        details={
            "goal": active_task.get("goal"),
            "follow_up_reason": follow_up.get("reason"),
            "navigation_request_id": follow_up.get("navigation_request_id"),
            "navigation_status": follow_up.get("navigation_status"),
            "model_update_count": model_update_count,
            "model_tool_call_count": model_tool_call_count,
        },
    )


def _followup_message(
    active_task: dict[str, Any],
    *,
    follow_up: dict[str, Any],
    prompt_cause: str,
) -> HumanMessage:
    next_actions = follow_up.get("next_actions")
    action_lines = ""
    if isinstance(next_actions, list) and next_actions:
        action_lines = "\n".join(f"- {action}" for action in next_actions)
    else:
        action_lines = "- continue with the best available tool, or mark the task done/blocked"

    return native_internal_message(
        (
            "# DIMOS AUTO FOLLOW-UP\n"
            f"The active task is still pending, and {prompt_cause}.\n"
            f"- task_id: {active_task.get('task_id')}\n"
            f"- goal: {active_task.get('goal')}\n"
            "- Continue now by using the appropriate tool(s), or explicitly explain why the "
            "task is done or blocked.\n"
            "- If the task is complete, end your final answer with exactly "
            "`DIMOS_TASK_DONE: <brief reason>`.\n"
            "- If the task is blocked, end your final answer with exactly "
            "`DIMOS_TASK_BLOCKED: <brief reason>`.\n"
            "- Do not ask the user for another movement confirmation while server-side task "
            "authority remains active and the action stays inside this goal.\n"
            "# Suggested next actions\n"
            f"{action_lines}"
        ),
        source="auto_followup",
    )


def _needs_navigation_status_followup(follow_up: dict[str, Any]) -> bool:
    return (
        follow_up.get("pending") is True
        and follow_up.get("reason") == "navigation_status_guidance"
        and bool(follow_up.get("navigation_request_id"))
    )


def context_gathering_tool_names(tool_names: Iterable[str]) -> list[str]:
    """Return ordered unique context-gathering tools when all calls are safe to continue."""

    names: list[str] = []
    seen: set[str] = set()
    for tool_name in tool_names:
        normalized = str(tool_name).strip()
        if not normalized:
            continue
        if normalized not in CONTEXT_GATHERING_TOOL_NAMES:
            return []
        if normalized in seen:
            continue
        names.append(normalized)
        seen.add(normalized)
    return names


def _attempt_key(task_id: str, reason: str | None) -> str:
    if not reason:
        return task_id
    return f"{task_id}::{reason}"


def _attempt_reason(reason: str) -> str | None:
    if reason == PENDING_NAVIGATION_FOLLOWUP_REASON:
        return reason
    return None


def _no_followup(
    reason: str,
    state: NativeAutoFollowupState,
    active_task: dict[str, Any] | None,
) -> NativeAutoFollowupDecision:
    task_id = active_task.get("task_id") if isinstance(active_task, dict) else None
    return NativeAutoFollowupDecision(
        should_enqueue=False,
        reason=reason,
        state=state,
        task_id=str(task_id) if task_id else None,
    )


def _active_task(task_snapshot: dict[str, Any]) -> dict[str, Any] | None:
    active_task = task_snapshot.get("active_task")
    return active_task if isinstance(active_task, dict) else None
