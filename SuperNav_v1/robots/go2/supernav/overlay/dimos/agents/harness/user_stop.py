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

"""Shared parsing for user input that should preempt movement authority."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

AuthorityPreemptionReason = Literal["user_stop", "task_completed", "task_blocked"]


@dataclass(frozen=True)
class AuthorityPreemptionIntent:
    """Movement-authority preemption requested by a human input message."""

    should_preempt: bool
    preempted_by: AuthorityPreemptionReason | None = None


def movement_authority_preemption_intent(message: str) -> AuthorityPreemptionIntent:
    """Return movement-authority preemption intent for a human input message."""

    normalized = " ".join(str(message or "").strip().lower().split())
    if normalized in {
        "/cancel",
        "/stop",
        "stop",
        "cancel",
        "停止",
        "取消",
    } or normalized.startswith(("/cancel ", "/stop ")):
        return AuthorityPreemptionIntent(should_preempt=True, preempted_by="user_stop")
    if normalized in {"/done", "/complete", "/completed", "/finish", "/task-done"} or (
        normalized.startswith(("/done ", "/complete ", "/completed ", "/finish ", "/task-done "))
    ):
        return AuthorityPreemptionIntent(should_preempt=True, preempted_by="task_completed")
    if normalized in {"/blocked", "/block", "/task-blocked"} or normalized.startswith(
        ("/blocked ", "/block ", "/task-blocked ")
    ):
        return AuthorityPreemptionIntent(should_preempt=True, preempted_by="task_blocked")
    return AuthorityPreemptionIntent(should_preempt=False)


def is_user_stop_message(message: str) -> bool:
    """Return whether a human input is a direct stop/cancel request."""

    return movement_authority_preemption_intent(message).preempted_by == "user_stop"
