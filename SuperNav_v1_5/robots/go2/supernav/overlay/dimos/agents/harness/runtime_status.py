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

"""Runtime status payloads for MCP clients and external harnesses."""

from __future__ import annotations

from typing import Any

from dimos.agents.harness.go2_harness_profiles import go2_harness_profile_for_policy
from dimos.agents.runtime_policy import (
    AgentRuntimePolicy,
    tool_exposure_refusal,
)
from dimos.navigation.motion_diagnostics import build_motion_diagnostics


def build_mcp_runtime_status(
    *,
    pid: int,
    skills: list[Any],
    policy: AgentRuntimePolicy,
    cap_registry: Any,
    movement_leases: Any,
    runtime_config: Any | None = None,
    disabled_modules: tuple[str, ...] | list[str] = (),
) -> dict[str, Any]:
    """Build a JSON-serializable MCP runtime status payload."""

    modules = list(dict.fromkeys(str(skill.class_name) for skill in skills))
    exposed_skills: list[str] = []
    hidden_skills: list[dict[str, str]] = []
    for skill in skills:
        tool_name = str(skill.func_name)
        uses = tuple(getattr(skill, "uses", ()) or ())
        required_modules = tuple(getattr(skill, "required_modules", ()) or ())
        refusal = tool_exposure_refusal(
            tool_name,
            uses,
            policy=policy,
            required_modules=required_modules,
            disabled_modules=disabled_modules,
        )
        if refusal is None:
            exposed_skills.append(tool_name)
        else:
            hidden_skills.append({"name": tool_name, "reason": refusal})

    authority_snapshot = _safe_snapshot(movement_leases)
    go2_profile = go2_harness_profile_for_policy(policy)
    return {
        "pid": pid,
        "modules": modules,
        "skills": [str(skill.func_name) for skill in skills],
        "exposed_skills": exposed_skills,
        "hidden_skills": hidden_skills,
        "disabled_modules": list(disabled_modules),
        "agent_runtime_policy": policy.model_dump(),
        "go2_harness_profile": go2_profile.model_dump(),
        "capability_holders": _safe_snapshot(cap_registry),
        "movement_leases": authority_snapshot,
        "active_movement_authorities": authority_snapshot,
        "robot_protection_status": _robot_protection_status(runtime_config),
    }


def _safe_snapshot(obj: Any) -> dict[str, Any]:
    snapshot = getattr(obj, "snapshot", None)
    if not callable(snapshot):
        return {}
    payload = snapshot()
    return payload if isinstance(payload, dict) else {}


def _robot_protection_status(runtime_config: Any | None) -> dict[str, Any]:
    obstacle_avoidance = (
        getattr(runtime_config, "obstacle_avoidance", None) if runtime_config is not None else None
    )
    payload = build_motion_diagnostics(obstacle_avoidance_config=obstacle_avoidance)
    payload["obstacle_avoidance_config"] = obstacle_avoidance
    return payload
