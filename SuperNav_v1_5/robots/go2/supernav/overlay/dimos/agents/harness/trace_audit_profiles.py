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

"""Named trace-audit profiles for repeatable Go2 validation gates."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TraceAuditProfile:
    """A named bundle of trace-audit options."""

    name: str
    description: str
    options: Mapping[str, Any]

    def model_dump(self) -> dict[str, Any]:
        """Return a JSON-serializable profile summary."""

        return {
            "name": self.name,
            "description": self.description,
            "options": dict(self.options),
        }


_GO2_PRELIVE_OPTIONS: dict[str, Any] = {
    "require_visual": True,
    "require_session_lifecycle": True,
    "require_turn_lifecycle": True,
    "require_model_turn": True,
    "require_model_response": True,
    "forbid_turn_errors": True,
    "require_request_contract": True,
    "require_request_tool_scope": True,
    "require_task_state": True,
    "require_task_followup": True,
    "require_auto_followup": True,
    "require_tool_status_isolated": True,
    "require_tool_status_context": True,
    "require_tool_execution": True,
    "require_tool_call_terminal": True,
    "forbid_raw_trace_payloads": True,
    "require_context_compaction": True,
    "max_context_token_growth": 8192,
    "max_context_images": 4,
    "max_context_image_tokens": 8192,
}


TRACE_AUDIT_PROFILES: dict[str, TraceAuditProfile] = {
    "go2-prelive": TraceAuditProfile(
        name="go2-prelive",
        description=(
            "Go2 native-harness readiness before a risky motion drill: session/turn "
            "lifecycle, request contract, task state, visual evidence, context bounds, "
            "compaction traceability, tool execution, status context, and raw-payload "
            "absence."
        ),
        options=_GO2_PRELIVE_OPTIONS,
    ),
    "go2-live-motion": TraceAuditProfile(
        name="go2-live-motion",
        description=(
            "Go2 live motion acceptance drill; includes go2-prelive gates plus "
            "relative movement, navigation, rotation, stop, and preemption evidence."
        ),
        options={
            **_GO2_PRELIVE_OPTIONS,
            "require_movement": True,
            "require_navigation": True,
            "require_relative_move": True,
            "require_rotation": True,
            "require_stop": True,
            "require_preemption": True,
            "preemption_max_ms": 500.0,
        },
    ),
    "go2-navigation-success": TraceAuditProfile(
        name="go2-navigation-success",
        description=(
            "Go2 navigation drill for path-found/local-obstacle runs; requires "
            "navigation lifecycle evidence and forbids retry-limit escalation."
        ),
        options={
            **_GO2_PRELIVE_OPTIONS,
            "require_navigation": True,
            "forbid_navigation_escalation": True,
        },
    ),
    "go2-navigation-escalation": TraceAuditProfile(
        name="go2-navigation-escalation",
        description=(
            "Go2 navigation no-path/retry-limit drill; requires navigation lifecycle "
            "evidence plus a structured failure_escalated trace."
        ),
        options={
            **_GO2_PRELIVE_OPTIONS,
            "require_navigation": True,
            "require_failure_escalation": True,
        },
    ),
}


def available_trace_audit_profiles() -> tuple[TraceAuditProfile, ...]:
    """Return profiles in CLI display order."""

    return tuple(TRACE_AUDIT_PROFILES[name] for name in sorted(TRACE_AUDIT_PROFILES))


def trace_audit_profile(name: str) -> TraceAuditProfile:
    """Resolve a named trace-audit profile."""

    try:
        return TRACE_AUDIT_PROFILES[name]
    except KeyError as exc:
        available = ", ".join(sorted(TRACE_AUDIT_PROFILES))
        raise ValueError(f"Unknown trace-audit profile '{name}'. Available: {available}") from exc


def merge_trace_audit_options(
    profile_name: str,
    explicit_options: Mapping[str, Any],
) -> tuple[dict[str, Any], TraceAuditProfile | None]:
    """Merge a profile into CLI options.

    Profiles only add requirements. Boolean CLI flags can add more requirements,
    numeric/list options can override profile defaults when explicitly provided.
    """

    profile = trace_audit_profile(profile_name) if profile_name else None
    merged = dict(profile.options) if profile is not None else {}
    for key, value in explicit_options.items():
        merged[key] = _merge_option(merged.get(key), value)
    return merged, profile


def _merge_option(profile_value: Any, explicit_value: Any) -> Any:
    if isinstance(explicit_value, bool):
        return bool(profile_value) or explicit_value
    if isinstance(explicit_value, list):
        if not explicit_value:
            return profile_value if profile_value is not None else []
        profile_items = profile_value if isinstance(profile_value, list) else []
        return [*profile_items, *[item for item in explicit_value if item not in profile_items]]
    if explicit_value is None:
        return profile_value
    return explicit_value
