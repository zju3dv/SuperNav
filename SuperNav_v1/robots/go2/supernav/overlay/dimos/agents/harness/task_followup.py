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

"""Trace-safe follow-up summaries for active native-agent tasks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dimos.agents.harness.navigation_status_guidance import navigation_task_guidance


def task_followup_summary(active_task: Mapping[str, Any] | None) -> dict[str, Any]:
    """Summarize whether a native-agent task still needs follow-up work."""

    if not isinstance(active_task, Mapping):
        return {"pending": False, "reason": "no_active_task"}

    status = str(active_task.get("status") or "")
    if status != "active":
        return {
            "pending": False,
            "reason": f"task_{status or 'not_active'}",
            "task_id": active_task.get("task_id"),
        }

    tool_status = _mapping(active_task.get("tool_status"))
    status_context = _mapping(tool_status.get("status_context"))
    navigation_task = _mapping(status_context.get("navigation_task"))
    guidance = navigation_task_guidance(navigation_task)

    next_actions: list[str] = []
    reason = "active_task"
    if guidance:
        reason = "navigation_status_guidance"
        next_actions.append(guidance)
    elif tool_status:
        reason = "latest_tool_status"
        next_actions.append("continue from the latest tool status or mark the task done/blocked")
    else:
        next_actions.append("continue the task, mark it done, or mark it blocked")

    return {
        "pending": True,
        "reason": reason,
        "task_id": active_task.get("task_id"),
        "goal": active_task.get("goal"),
        "movement_authorization": active_task.get("movement_authorization"),
        "latest_tool": tool_status.get("tool_name") if tool_status else None,
        "navigation_request_id": navigation_task.get("request_id") if navigation_task else None,
        "navigation_status": navigation_task.get("status") if navigation_task else None,
        "next_actions": next_actions,
    }


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}
