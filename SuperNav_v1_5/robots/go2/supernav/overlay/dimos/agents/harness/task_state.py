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

"""Minimal task-state tracking for native DimOS harness turns.

External harnesses can own richer task planning. The native LangGraph client only
needs a small, explicit task envelope so current user intent, movement authority,
and `/goal` style control commands are visible to request assembly and traces.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import time
from typing import Any, Literal

from langchain_core.messages import SystemMessage
from langchain_core.messages.base import BaseMessage

from dimos.agents.harness.artifact_ledger import artifact_placeholder_text, is_valid_artifact_id
from dimos.agents.harness.task_intent import (
    TaskCommand,
    is_natural_language_continue,
    parse_task_command,
)
from dimos.agents.harness.task_tool_status import (
    task_tool_status_context_text,
    task_tool_status_snapshot,
)
from dimos.agents.harness.tool_status import ToolStatusMessage

TaskStatus = Literal["active", "cancelled", "completed", "blocked"]
TaskOrigin = Literal["implicit_user_message", "goal_command"]


@dataclass(frozen=True)
class TaskRun:
    """Current native-agent task envelope."""

    task_id: str
    goal: str
    status: TaskStatus
    origin: TaskOrigin
    created_at: float
    updated_at: float
    movement_authorization: str | None = None
    turn_count: int = 1
    terminal_reason: str | None = None
    visual_artifact_ids: tuple[str, ...] = ()
    tool_status: dict[str, Any] | None = None

    def model_dump(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["visual_artifact_ids"] = list(self.visual_artifact_ids)
        return payload


@dataclass(frozen=True)
class TaskStateUpdate:
    """Result of applying a user message to task state."""

    command: TaskCommand | None
    task: TaskRun | None
    message: str

    def model_dump(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "task": self.task.model_dump() if self.task is not None else None,
            "message": self.message,
        }


class TaskStateManager:
    """Track the native agent's active task without owning long-horizon planning."""

    def __init__(self, *, max_visual_artifacts: int = 3) -> None:
        self._active: TaskRun | None = None
        self._last_terminal: TaskRun | None = None
        self._next_id = 1
        self._max_visual_artifacts = max(0, int(max_visual_artifacts))

    def update_from_user_message(
        self,
        text: str,
        *,
        movement_authorization: str | None = None,
        now: float | None = None,
    ) -> TaskStateUpdate:
        """Apply a user message or `/goal` command to the active task."""

        now = now if now is not None else time.time()
        command, payload = parse_task_command(text)
        if command == "cancel":
            cancelled = self._finish("cancelled", payload or None, now)
            return TaskStateUpdate(
                command=command,
                task=cancelled,
                message="Active task cancelled." if cancelled else "No active task to cancel.",
            )
        if command == "status":
            return TaskStateUpdate(
                command=command,
                task=self._active or self._last_terminal,
                message="Task status requested.",
            )
        if command == "continue":
            if self._active is None:
                return TaskStateUpdate(
                    command=command,
                    task=None,
                    message="No active task to continue.",
                )
            task = self._refresh_active(
                movement_authorization=self._active.movement_authorization,
                now=now,
            )
            return TaskStateUpdate(
                command=command,
                task=task,
                message="Task continuation requested.",
            )
        if command == "complete":
            completed = self._finish("completed", payload or None, now)
            return TaskStateUpdate(
                command=command,
                task=completed,
                message="Active task completed." if completed else "No active task to complete.",
            )
        if command == "blocked":
            blocked = self._finish("blocked", payload or None, now)
            return TaskStateUpdate(
                command=command,
                task=blocked,
                message="Active task blocked." if blocked else "No active task to mark blocked.",
            )
        if command == "goal":
            if not payload:
                return TaskStateUpdate(
                    command=command,
                    task=self._active,
                    message="No goal text provided.",
                )
            task = self._start(
                payload,
                origin="goal_command",
                movement_authorization=movement_authorization,
                now=now,
                reuse_active=False,
            )
            return TaskStateUpdate(command=command, task=task, message="Goal task started.")

        normalized = " ".join(text.split())
        if not normalized:
            return TaskStateUpdate(command=None, task=self._active, message="Empty message.")
        if is_natural_language_continue(normalized):
            if self._active is None:
                return TaskStateUpdate(
                    command="continue",
                    task=None,
                    message="No active task to continue.",
                )
            task = self._refresh_active(
                movement_authorization=self._active.movement_authorization,
                now=now,
            )
            return TaskStateUpdate(
                command="continue",
                task=task,
                message="Task continuation requested.",
            )
        if movement_authorization == "explicit_confirmation" and self._active is not None:
            task = self._refresh_active(
                movement_authorization=movement_authorization,
                now=now,
            )
            return TaskStateUpdate(
                command=None,
                task=task,
                message="Active task explicitly confirmed.",
            )
        task = self._start(
            normalized,
            origin="implicit_user_message",
            movement_authorization=movement_authorization,
            now=now,
        )
        return TaskStateUpdate(command=None, task=task, message="Implicit task started.")

    def active_task(self) -> TaskRun | None:
        """Return the currently active task, if any."""

        return self._active

    def pin_visual_artifacts(
        self,
        artifact_ids: list[str],
        *,
        now: float | None = None,
    ) -> TaskRun | None:
        """Mark image artifacts as task-critical visual context for future turns."""

        task = self._active
        if task is None or task.status != "active" or self._max_visual_artifacts <= 0:
            return task

        pinned = list(task.visual_artifact_ids)
        changed = False
        for artifact_id in artifact_ids:
            if not is_valid_artifact_id(artifact_id):
                continue
            if artifact_id in pinned:
                pinned.remove(artifact_id)
            pinned.append(artifact_id)
            changed = True
        if not changed:
            return task

        pinned = pinned[-self._max_visual_artifacts :]
        updated = TaskRun(
            **{
                **task.model_dump(),
                "updated_at": now if now is not None else time.time(),
                "visual_artifact_ids": tuple(pinned),
            }
        )
        self._active = updated
        return updated

    def pin_visual_artifact(
        self,
        artifact_id: str,
        *,
        now: float | None = None,
    ) -> TaskRun | None:
        """Mark one image artifact as task-critical visual context."""

        return self.pin_visual_artifacts([artifact_id], now=now)

    def pin_tool_status(
        self,
        status: ToolStatusMessage,
        *,
        now: float | None = None,
    ) -> TaskRun | None:
        """Pin the latest task-relevant tool status into active task context."""

        task = self._active
        if task is None or task.status != "active":
            return task

        updated = TaskRun(
            **{
                **task.model_dump(),
                "updated_at": now if now is not None else time.time(),
                "visual_artifact_ids": tuple(task.visual_artifact_ids),
                "tool_status": task_tool_status_snapshot(status),
            }
        )
        self._active = updated
        return updated

    def finish_from_model(
        self,
        status: Literal["completed", "blocked"],
        reason: str | None,
        *,
        now: float | None = None,
    ) -> TaskStateUpdate:
        """Apply a strict model-emitted terminal directive to the active task."""

        now = now if now is not None else time.time()
        command: TaskCommand = "complete" if status == "completed" else "blocked"
        finished = self._finish(status, reason or None, now)
        if status == "completed":
            message = (
                "Model marked active task completed." if finished else "No active task to complete."
            )
        else:
            message = (
                "Model marked active task blocked."
                if finished
                else "No active task to mark blocked."
            )
        return TaskStateUpdate(command=command, task=finished, message=message)

    def context_messages(self) -> list[BaseMessage]:
        """Return model-visible pinned task context."""

        task = self._active
        if task is None or task.status != "active":
            return []
        movement = task.movement_authorization or "none"
        visual_context = ""
        if task.visual_artifact_ids:
            refs = "\n".join(
                f"- {artifact_placeholder_text(artifact_id, kind='image')} "
                "task-critical visual evidence"
                for artifact_id in task.visual_artifact_ids
            )
            visual_context = (
                "# DIMOS TASK VISUAL CONTEXT\n"
                "These artifacts are pinned because they remain relevant to the active task. "
                "Use the re-injected images, not the placeholder text, when answering.\n"
                f"{refs}\n"
            )
        tool_status_context = task_tool_status_context_text(task.tool_status)
        return [
            SystemMessage(
                content=(
                    "# DIMOS TASK STATE\n"
                    f"- task_id: {task.task_id}\n"
                    f"- status: {task.status}\n"
                    f"- origin: {task.origin}\n"
                    f"- goal: {task.goal}\n"
                    f"- movement_authorization: {movement}\n"
                    f"- turn_count: {task.turn_count}\n"
                    "- Continue using available tools until the task is complete, blocked, "
                    "interrupted, or outside the active runtime policy.\n"
                    "- Do not ask for another movement confirmation while server-side task "
                    "authority remains active and the requested action stays inside this goal.\n"
                    "- When the active task is complete, end your final answer with exactly "
                    "`DIMOS_TASK_DONE: <brief reason>`.\n"
                    "- When the active task is blocked, end your final answer with exactly "
                    "`DIMOS_TASK_BLOCKED: <brief reason>`.\n"
                    f"{visual_context}"
                    f"{tool_status_context}"
                )
            )
        ]

    def snapshot(self) -> dict[str, Any]:
        """Return serializable task state for traces/status."""

        return {
            "active_task": self._active.model_dump() if self._active is not None else None,
            "last_terminal_task": (
                self._last_terminal.model_dump() if self._last_terminal is not None else None
            ),
        }

    def _start(
        self,
        goal: str,
        *,
        origin: TaskOrigin,
        movement_authorization: str | None,
        now: float,
        reuse_active: bool = True,
    ) -> TaskRun:
        existing = self._active
        should_reuse = reuse_active and existing is not None and existing.status == "active"
        task_id = existing.task_id if should_reuse and existing is not None else self._new_id()
        created_at = existing.created_at if should_reuse and existing is not None else now
        turn_count = (existing.turn_count + 1) if should_reuse and existing is not None else 1
        task = TaskRun(
            task_id=task_id,
            goal=goal,
            status="active",
            origin=origin,
            created_at=created_at,
            updated_at=now,
            movement_authorization=movement_authorization,
            turn_count=turn_count,
            visual_artifact_ids=(
                existing.visual_artifact_ids if should_reuse and existing is not None else ()
            ),
            tool_status=existing.tool_status if should_reuse and existing is not None else None,
        )
        self._active = task
        return task

    def _refresh_active(
        self,
        *,
        movement_authorization: str | None,
        now: float,
    ) -> TaskRun:
        assert self._active is not None
        task = TaskRun(
            **{
                **self._active.model_dump(),
                "updated_at": now,
                "movement_authorization": movement_authorization,
                "turn_count": self._active.turn_count + 1,
                "visual_artifact_ids": tuple(self._active.visual_artifact_ids),
                "tool_status": self._active.tool_status,
            }
        )
        self._active = task
        return task

    def _finish(
        self,
        status: Literal["cancelled", "completed", "blocked"],
        reason: str | None,
        now: float,
    ) -> TaskRun | None:
        if self._active is None:
            return None
        task = TaskRun(
            **{
                **self._active.model_dump(),
                "status": status,
                "updated_at": now,
                "terminal_reason": reason,
                "visual_artifact_ids": tuple(self._active.visual_artifact_ids),
                "tool_status": self._active.tool_status,
            }
        )
        self._active = None
        self._last_terminal = task
        return task

    def _new_id(self) -> str:
        task_id = f"task_{self._next_id:06d}"
        self._next_id += 1
        return task_id
