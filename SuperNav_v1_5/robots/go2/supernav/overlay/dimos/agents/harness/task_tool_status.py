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

"""Compact task-pinned summaries for asynchronous tool status."""

from __future__ import annotations

from typing import Any

from dimos.agents.harness.display_content import content_to_display_text, redact_display_content
from dimos.agents.harness.tool_status import ToolStatusMessage
from dimos.agents.harness.tool_status_display import tool_status_context_lines

_MAX_TOOL_STATUS_CHARS = 500


def task_tool_status_snapshot(status: ToolStatusMessage) -> dict[str, Any]:
    """Return a trace/model-safe snapshot of one asynchronous tool status."""

    return {
        "tool_name": status.tool_name,
        "event_kind": status.event_kind,
        "progress_token": status.progress_token,
        "progress": status.progress,
        "message": _truncate_status_text(content_to_display_text(status.content)),
        "status_context": _compact_status_context(status.status_context),
    }


def task_tool_status_context_text(status: dict[str, Any] | None) -> str:
    """Render one pinned tool-status snapshot into model-visible task context."""

    if not status:
        return ""
    lines = [
        "# DIMOS TASK TOOL STATUS",
        "Latest asynchronous tool status for this active task. It is not a new human request.",
        f"- tool: {status.get('tool_name') or 'tool'}",
        f"- event: {status.get('event_kind') or 'message'}",
    ]
    progress = status.get("progress")
    if progress is not None:
        lines.append(f"- progress: {progress}")
    message = status.get("message")
    if message:
        lines.append(f"- message: {message}")
    context_lines = tool_status_context_lines(status.get("status_context"))
    if context_lines:
        lines.append("- compact status context:")
        lines.extend(f"  - {line}" for line in context_lines)
    return "\n".join(lines) + "\n"


def _compact_status_context(status_context: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(status_context, dict):
        return None
    compact = {
        key: status_context[key]
        for key in ("policy", "movement_authority", "protection", "navigation_task")
        if key in status_context
    }
    sanitized = redact_display_content(compact)
    return sanitized if isinstance(sanitized, dict) and sanitized else None


def _truncate_status_text(text: str) -> str:
    compact = " ".join(text.split())
    if len(compact) <= _MAX_TOOL_STATUS_CHARS:
        return compact
    return f"{compact[: _MAX_TOOL_STATUS_CHARS - 3]}..."
