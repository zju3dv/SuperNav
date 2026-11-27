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

"""Helpers for logging structured skill results as harness trace events."""

from __future__ import annotations

from typing import Any, TypeVar

from dimos.agents.harness.display_content import redact_display_content
from dimos.agents.harness.trace import emit_trace_event
from dimos.agents.skill_result import SkillResult

E = TypeVar("E", bound=str)


def skill_result_payload(
    tool_name: str,
    result: SkillResult[Any],
    **context: Any,
) -> dict[str, Any]:
    """Build a trace payload from the existing ``SkillResult`` contract."""

    payload = {str(key): redact_display_content(value) for key, value in context.items()}
    payload.update(
        {
            "tool_name": tool_name,
            "success": result.success,
            "error_code": result.error_code,
            "message": redact_display_content(result.message),
            "duration_ms": round(result.duration_ms, 1),
        }
    )
    if result.metadata:
        payload["metadata"] = redact_display_content(result.metadata)
    return payload


def emit_skill_result_trace(
    logger: Any,
    event_type: str,
    tool_name: str,
    result: SkillResult[E],
    **context: Any,
) -> SkillResult[E]:
    """Emit a harness trace event and return ``result`` for fluent skill code."""

    emit_trace_event(logger, event_type, **skill_result_payload(tool_name, result, **context))
    return result
