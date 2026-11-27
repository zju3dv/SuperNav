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

"""Actionable guidance for pinned navigation status context."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def navigation_task_guidance(navigation_task: Mapping[str, Any] | None) -> str | None:
    """Return a compact next-step hint for one navigation task snapshot."""

    if not isinstance(navigation_task, Mapping):
        return None

    status = _normalize_status(navigation_task.get("status"))
    if not status:
        return None

    request_id = _request_id(navigation_task)
    status_call = (
        f'navigation_status(request_id="{request_id}")' if request_id else "navigation_status()"
    )

    if status == "RETRY_LIMIT_EXCEEDED" or navigation_task.get("escalated") is True:
        return "retry limit reached; stop retrying and ask the user for a new target or help"
    if status in {"CREATED", "STARTED", "FOLLOWING_PATH"}:
        return (
            f"call {status_call} before claiming arrival, failure, or completion; "
            "do not infer final outcome from a STARTED status"
        )
    if status in {"ARRIVED", "GOAL_REACHED"}:
        return (
            "navigation reports arrival; if the task requires visual confirmation, "
            "observe or visual_find before making the final claim"
        )
    if status in {"OBSTACLE", "LOCAL_OBSTACLE"}:
        return (
            "call motion_safety_status() before retrying; consider rotate/observe/replan "
            "or ask the user instead of repeating the same path"
        )
    if status in {"NO_PATH", "NO_SAFE_GOAL"}:
        return (
            "do not claim navigation started; ask for a new target or choose a different "
            "evidence source before retrying"
        )
    if status in {"NO_ODOM", "TIMEOUT"}:
        return (
            "call motion_safety_status() or last_motion_diagnostics() before retrying; "
            "report the blocker if it persists"
        )
    if status in {"VISION_UNAVAILABLE", "NOT_CONFIGURED"}:
        return (
            "visual navigation is unavailable; do not fall back to semantic memory unless "
            "the user allowed semantic navigation"
        )
    if status == "TARGET_NOT_FOUND":
        return (
            "target was not found from the requested evidence source; ask for clarification "
            "or use a different explicit navigation tool"
        )
    if status == "CANCELLED":
        return "navigation is cancelled; do not resume motion without a new task or user request"
    if status == "ARRIVAL_NOT_CONFIRMED":
        return "arrival was not confirmed; inspect status/diagnostics before claiming success"

    error_code = _normalize_status(navigation_task.get("error_code"))
    if error_code:
        return (
            "navigation has an error code; inspect motion_safety_status() or "
            "navigation_status() before choosing the next action"
        )
    return None


def _normalize_status(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().upper()


def _request_id(navigation_task: Mapping[str, Any]) -> str:
    value = navigation_task.get("request_id")
    if not isinstance(value, str):
        return ""
    value = value.strip()
    return value if value.startswith("nav_") else ""
