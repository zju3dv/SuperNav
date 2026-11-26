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

"""Native task-control command routing for the DimOS fallback agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from dimos.agents.harness.task_state import TaskRun, TaskStateUpdate
from dimos.agents.harness.task_status_summary import format_task_status, format_terminal_task

TaskControlRoute = Literal["model", "local_response"]


@dataclass(frozen=True)
class TaskControlDecision:
    """How the native agent should handle a task-control command."""

    route: TaskControlRoute
    response_text: str | None = None
    model_text: str | None = None


def decide_task_control(update: TaskStateUpdate) -> TaskControlDecision | None:
    """Return a control decision for slash commands, or None for normal turns."""

    if update.command is None:
        return None
    if update.command == "status":
        return TaskControlDecision(
            route="local_response",
            response_text=format_task_status(update.task),
        )
    if update.command == "cancel":
        return TaskControlDecision(
            route="local_response",
            response_text=(
                update.message if update.task is None else format_terminal_task(update.task)
            ),
        )
    if update.command in {"complete", "blocked"}:
        return TaskControlDecision(
            route="local_response",
            response_text=(
                update.message if update.task is None else format_terminal_task(update.task)
            ),
        )
    if update.command == "goal" and update.message == "No goal text provided.":
        return TaskControlDecision(
            route="local_response",
            response_text="No goal text provided. Use `/goal <task>` to start an explicit task.",
        )
    if update.command == "continue":
        if update.task is None:
            return TaskControlDecision(
                route="local_response",
                response_text=(
                    "No active task to continue. Start one with `/goal <task>` or a normal "
                    "natural-language request."
                ),
            )
        return TaskControlDecision(
            route="model",
            model_text=_format_continue_prompt(update.task),
        )
    return None


def _format_continue_prompt(task: TaskRun) -> str:
    return (
        "Continue the active DimOS task. Use available status, observation, diagnostic, "
        "and movement tools as needed until the task is complete, blocked, interrupted, "
        "or outside the active runtime policy.\n"
        "If the task is complete, end your final answer with exactly "
        "`DIMOS_TASK_DONE: <brief reason>`. If the task is blocked, end with exactly "
        "`DIMOS_TASK_BLOCKED: <brief reason>`.\n"
        f"Task id: {task.task_id}\n"
        f"Goal: {task.goal}"
    )
