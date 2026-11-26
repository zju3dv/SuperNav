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

"""Strict model-emitted task terminal directives for the native agent."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal

from langchain_core.messages.base import BaseMessage

TaskTerminalStatus = Literal["completed", "blocked"]
TaskTerminalPreemptionReason = Literal["task_completed", "task_blocked"]

_DONE_PREFIX = "DIMOS_TASK_DONE:"
_BLOCKED_PREFIX = "DIMOS_TASK_BLOCKED:"


@dataclass(frozen=True)
class NativeTaskTerminalDirective:
    """A strict terminal directive emitted by the model as the final text line."""

    status: TaskTerminalStatus
    reason: str
    raw_line: str
    message_index: int

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


def terminal_directive_from_message(
    message: BaseMessage,
    *,
    message_index: int,
) -> NativeTaskTerminalDirective | None:
    """Return a terminal directive from one AI message, if present."""

    if str(getattr(message, "type", "")) != "ai":
        return None
    return terminal_directive_from_text(_content_text(message.content), message_index=message_index)


def terminal_directive_from_text(
    text: str,
    *,
    message_index: int = 0,
) -> NativeTaskTerminalDirective | None:
    """Parse only the final non-empty line as a task terminal directive."""

    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    if not lines:
        return None

    final_line = lines[-1]
    normalized = final_line.upper()
    if normalized.startswith(_DONE_PREFIX):
        reason = final_line[len(_DONE_PREFIX) :].strip()
        return NativeTaskTerminalDirective(
            status="completed",
            reason=reason or "model declared the task complete",
            raw_line=final_line,
            message_index=message_index,
        )
    if normalized.startswith(_BLOCKED_PREFIX):
        reason = final_line[len(_BLOCKED_PREFIX) :].strip()
        return NativeTaskTerminalDirective(
            status="blocked",
            reason=reason or "model declared the task blocked",
            raw_line=final_line,
            message_index=message_index,
        )
    return None


def terminal_preemption_reason(status: TaskTerminalStatus) -> TaskTerminalPreemptionReason:
    """Return the movement-authority preemption reason for a terminal task state."""

    if status == "completed":
        return "task_completed"
    return "task_blocked"


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, Mapping) and item.get("type") in {"text", "input_text"}:
                parts.append(str(item.get("text") or ""))
        return "\n".join(parts)
    return str(content)
