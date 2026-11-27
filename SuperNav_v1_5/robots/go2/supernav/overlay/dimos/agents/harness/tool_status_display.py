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

"""Human-facing summaries for typed tool-status context."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dimos.agents.harness.navigation_status_guidance import navigation_task_guidance


def tool_status_context_lines(context: Mapping[str, Any] | None) -> list[str]:
    """Render compact status lines for the Human CLI tool panel."""

    if not isinstance(context, Mapping):
        return []
    lines: list[str] = []
    policy = _mapping(context.get("policy"))
    if policy:
        lines.append(
            "policy "
            f"profile={_value(policy.get('profile'))} "
            f"scope={_value(policy.get('tool_scope'))} "
            f"movement={_value(policy.get('movement_mode'))}"
            f"{' admin_override' if policy.get('admin_override') else ''}"
        )

    movement = _mapping(context.get("movement_authority"))
    if movement:
        remaining = _mapping(movement.get("remaining"))
        budget = ""
        if remaining:
            budget = (
                " remaining="
                f"{_value(remaining.get('tool_calls'))} calls/"
                f"{_short_float(remaining.get('linear_m'))}m/"
                f"{_short_float(remaining.get('rotation_deg'))}deg"
            )
        task_label = movement.get("task_id") or movement.get("task_goal")
        lines.append(
            "authority "
            f"client={_value(movement.get('client_id'))} "
            f"task={_value(task_label)} "
            f"auth={_value(movement.get('authorization'))} "
            f"lease={_value(movement.get('lease_id'))}"
            f"{budget}"
        )

    protection = _mapping(context.get("protection"))
    if protection:
        collision = _mapping(protection.get("collision_obstacle_protection"))
        unitree = _mapping(protection.get("unitree_protection"))
        battery = _mapping(protection.get("battery_status"))
        lines.append(
            "protection "
            f"blocker={_value(protection.get('blocker'))} "
            f"collision={_protection_summary(collision)} "
            f"unitree={_protection_summary(unitree)} "
            f"battery={_battery_summary(battery)}"
        )

    nav = _mapping(context.get("navigation_task"))
    if nav:
        lines.append(
            "nav "
            f"request={_value(nav.get('request_id'))} "
            f"status={_value(nav.get('status'))} "
            f"source={_value(nav.get('source'))} "
            f"error={_value(nav.get('error_code'))}"
        )
        guidance = navigation_task_guidance(nav)
        if guidance:
            lines.append(f"nav_guidance {guidance}")

    return lines


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _protection_summary(value: Mapping[str, Any]) -> str:
    state = _value(value.get("state"))
    labels: list[str] = []
    if value.get("available") is False:
        labels.append("unavailable")
    if value.get("needs_live_sdk_probe"):
        labels.append("sdk_probe")
    if not labels:
        return state
    return f"{state}/{'/'.join(labels)}"


def _battery_summary(value: Mapping[str, Any]) -> str:
    state = _value(value.get("state"))
    soc = value.get("soc_percent")
    try:
        return f"{state}({float(soc):.0f}%)"
    except (TypeError, ValueError):
        return state


def _value(value: Any) -> str:
    if value is None or value == "":
        return "-"
    text = str(value).replace("\n", " ")
    return text[:48] + "..." if len(text) > 51 else text


def _short_float(value: Any) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "-"
