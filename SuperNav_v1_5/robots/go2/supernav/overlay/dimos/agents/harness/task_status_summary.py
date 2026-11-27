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

"""Human-readable task status summaries for native DimOS harness control."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dimos.agents.harness.artifact_ledger import artifact_placeholder_text
from dimos.agents.harness.task_state import TaskRun
from dimos.agents.harness.tool_status_display import tool_status_context_lines


def format_task_status(task: TaskRun | None) -> str:
    """Return the local `/status` response for the current task state."""

    if task is None:
        return "No active DimOS task."
    if task.status != "active":
        return format_terminal_task(task, prefix="Last DimOS task")

    lines = [
        "Active DimOS task:",
        f"- task_id: {task.task_id}",
        f"- status: {task.status}",
        f"- origin: {task.origin}",
        f"- goal: {task.goal}",
        f"- movement_authorization: {task.movement_authorization or 'none'}",
        f"- turn_count: {task.turn_count}",
    ]
    lines.extend(_visual_artifact_lines(task))
    lines.extend(_tool_status_lines(task.tool_status))
    return "\n".join(lines)


def format_terminal_task(task: TaskRun, *, prefix: str | None = None) -> str:
    """Return a local response for a completed/cancelled/blocked task."""

    label = prefix or f"{task.status.capitalize()} DimOS task"
    lines = [
        f"{label}:",
        f"- task_id: {task.task_id}",
        f"- status: {task.status}",
        f"- goal: {task.goal}",
    ]
    if task.terminal_reason:
        lines.append(f"- reason: {task.terminal_reason}")
    lines.extend(_visual_artifact_lines(task))
    lines.extend(_tool_status_lines(task.tool_status))
    return "\n".join(lines)


def _visual_artifact_lines(task: TaskRun) -> list[str]:
    if not task.visual_artifact_ids:
        return []
    lines = ["- pinned_visual_artifacts:"]
    lines.extend(
        f"  - {artifact_placeholder_text(artifact_id, kind='image')}"
        for artifact_id in task.visual_artifact_ids
    )
    return lines


def _tool_status_lines(status: Mapping[str, Any] | None) -> list[str]:
    if not isinstance(status, Mapping) or not status:
        return []
    lines = [
        "- latest_tool_status:",
        f"  - tool: {status.get('tool_name') or 'tool'}",
        f"  - event: {status.get('event_kind') or 'message'}",
    ]
    progress = status.get("progress")
    if progress is not None:
        lines.append(f"  - progress: {progress}")
    message = status.get("message")
    if message:
        lines.append(f"  - message: {message}")

    context_lines = tool_status_context_lines(status.get("status_context"))
    if context_lines:
        lines.append("  - compact_status_context:")
        lines.extend(f"    - {line}" for line in context_lines)
    return lines
