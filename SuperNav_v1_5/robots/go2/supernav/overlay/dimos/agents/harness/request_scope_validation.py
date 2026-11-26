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

"""Trace-audit checks for request-time tool-scope exposure."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from dimos.agents.harness.validation_types import (
    LiveValidationCheck,
    fail_check as _fail,
    pass_check as _pass,
)
from dimos.agents.runtime_policy import (
    AgentRuntimePolicy,
    resolve_agent_runtime_policy,
    tool_exposure_refusal,
)

_POLICY_FIELDS = (
    "profile",
    "speak_mode",
    "movement_mode",
    "dangerous_skills",
    "tool_scope",
    "vlm_provider",
    "spatial_memory_mode",
    "tool_policy_admin_override",
)


def request_tool_scope_check(events: list[dict[str, Any]]) -> LiveValidationCheck:
    """Validate exposed native tools against the request's runtime policy."""

    if not events:
        return _fail(
            "request_tool_scope",
            "No request_assembled events found; exposed tool scope is not auditable.",
        )

    invalid_policy: list[dict[str, Any]] = []
    mismatched_details: list[dict[str, Any]] = []
    forbidden_visible: list[dict[str, Any]] = []
    checked_tools = 0
    for request_index, event in enumerate(events):
        payload = event.get("payload") or {}
        if not isinstance(payload, dict):
            invalid_policy.append(
                {"request_index": request_index, "reason": "payload_missing_or_not_object"}
            )
            continue

        policy = _policy_from_payload(payload.get("policy"))
        if policy is None:
            invalid_policy.append(
                {
                    "request_index": request_index,
                    "policy": payload.get("policy"),
                    "reason": "policy_missing_or_invalid",
                }
            )
            continue

        tools, mismatch = _tool_details_from_payload(payload)
        if mismatch is not None:
            mismatched_details.append({"request_index": request_index, **mismatch})
        for tool in tools:
            checked_tools += 1
            refusal = tool_exposure_refusal(
                str(tool.get("name") or ""),
                tool.get("uses", []),
                policy=policy,
            )
            if refusal is not None:
                forbidden_visible.append(
                    {
                        "request_index": request_index,
                        "tool_name": tool.get("name"),
                        "uses": tool.get("uses", []),
                        "tool_scope": policy.tool_scope,
                        "refusal": refusal,
                    }
                )

    if not invalid_policy and not mismatched_details and not forbidden_visible:
        return _pass(
            "request_tool_scope",
            "Every request_assembled event exposes tools allowed by its runtime policy.",
            requests=len(events),
            checked_tools=checked_tools,
        )
    return _fail(
        "request_tool_scope",
        "Some request_assembled events expose tools that conflict with runtime policy.",
        invalid_policy=invalid_policy,
        mismatched_details=mismatched_details,
        forbidden_visible=forbidden_visible,
    )


def _policy_from_payload(payload: Any) -> AgentRuntimePolicy | None:
    if not isinstance(payload, dict):
        return None
    if any(field not in payload for field in _POLICY_FIELDS):
        return None
    try:
        return resolve_agent_runtime_policy(
            SimpleNamespace(
                agent_profile=payload["profile"],
                agent_speak_mode=payload["speak_mode"],
                agent_movement_mode=payload["movement_mode"],
                agent_dangerous_skills=payload["dangerous_skills"],
                agent_tool_scope=payload["tool_scope"],
                vlm_provider=payload["vlm_provider"],
                spatial_memory_mode=payload["spatial_memory_mode"],
                agent_tool_policy_admin_override=payload["tool_policy_admin_override"],
            )
        )
    except ValueError:
        return None


def _tool_details_from_payload(
    payload: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    names = _tool_names(payload.get("tools_exposed"))
    details = _normalize_tool_details(payload.get("tools_exposed_details"))
    if not details:
        return [{"name": name, "uses": []} for name in names], None

    detail_names = sorted(str(tool["name"]) for tool in details)
    mismatch = None
    if names and names != detail_names:
        mismatch = {
            "tools_exposed": names,
            "tools_exposed_details": detail_names,
        }
    return details, mismatch


def _tool_names(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return sorted(str(item) for item in value if isinstance(item, str) and item)


def _normalize_tool_details(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    tools: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "")
        if not name:
            continue
        uses = item.get("uses", [])
        tools.append(
            {
                "name": name,
                "uses": sorted(str(use) for use in uses) if isinstance(uses, list | tuple) else [],
            }
        )
    return sorted(tools, key=lambda item: item["name"])
