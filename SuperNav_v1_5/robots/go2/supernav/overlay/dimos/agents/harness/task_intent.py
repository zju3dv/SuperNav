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

"""Task-control intent parsing for the native DimOS harness."""

from __future__ import annotations

from typing import Literal

TaskCommand = Literal["goal", "cancel", "status", "continue", "complete", "blocked"]

_NATURAL_LANGUAGE_CONTINUE_EXACT = frozenset(
    {
        "carry on",
        "continue",
        "continue the task",
        "go on",
        "keep going",
        "resume",
        "resume task",
        "接着",
        "接着做",
        "接着执行",
        "继续",
        "继续吧",
        "继续任务",
        "继续执行",
        "不要停",
        "别停",
    }
)
_NATURAL_LANGUAGE_CONTINUE_PREFIXES = (
    "continue ",
    "keep going ",
    "resume ",
    "接着",
    "继续",
    "不要停",
    "别停",
)


def parse_task_command(text: str) -> tuple[TaskCommand | None, str]:
    """Parse slash-style task control commands from user text."""

    stripped = text.strip()
    if not stripped.startswith("/"):
        return None, stripped
    command, _, rest = stripped.partition(" ")
    normalized = command.lower()
    if normalized in {"/goal", "/task"}:
        return "goal", rest.strip()
    if normalized in {"/cancel", "/stop-goal", "/stop_task", "/stop-task"}:
        return "cancel", rest.strip()
    if normalized in {"/status", "/goal-status", "/task-status"}:
        return "status", rest.strip()
    if normalized in {"/continue", "/resume-task", "/continue-task"}:
        return "continue", rest.strip()
    if normalized in {"/done", "/complete", "/completed", "/finish", "/task-done"}:
        return "complete", rest.strip()
    if normalized in {"/blocked", "/block", "/task-blocked"}:
        return "blocked", rest.strip()
    return None, stripped


def is_natural_language_continue(text: str) -> bool:
    """Return whether ordinary text asks to continue the active task."""

    normalized = _normalize(text)
    if normalized in _NATURAL_LANGUAGE_CONTINUE_EXACT:
        return True
    return any(normalized.startswith(prefix) for prefix in _NATURAL_LANGUAGE_CONTINUE_PREFIXES)


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())
