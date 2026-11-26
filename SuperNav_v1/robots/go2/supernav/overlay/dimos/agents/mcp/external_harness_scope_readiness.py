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

"""Tool-scope readiness checks for supervised external harness bring-up."""

from __future__ import annotations

from typing import Any

from dimos.agents.mcp.external_harness_types import (
    HarnessValidationCheck,
    fail_check,
    pass_check,
)
from dimos.agents.runtime_policy import AgentRuntimePolicy, is_task_authority_mode

_READONLY_REQUIRED_TOOLS = frozenset(
    {
        "server_status",
        "read_image",
        "observe",
        "go2_robot_status",
        "motion_safety_status",
        "last_motion_diagnostics",
        "robot_protection_status",
    }
)
_READONLY_FORBIDDEN_TOOLS = frozenset(
    {
        "agent_send",
        "begin_exploration",
        "execute_sport_command",
        "follow_person",
        "navigate_to_semantic",
        "navigate_to_tag",
        "navigate_with_text",
        "relative_move",
        "rotate_in_place",
        "speak",
        "start_patrol",
        "tag_location",
        "visual_approach",
    }
)
_VISION_LIDAR_LOCAL_MOTION_TOOLS = frozenset({"relative_move", "rotate_in_place"})
_VISION_LIDAR_REQUIRED_DIAGNOSTIC_TOOLS = frozenset(
    {
        "motion_safety_status",
        "last_motion_diagnostics",
        "robot_protection_status",
        "go2_robot_status",
        "server_status",
    }
)
_VISION_LIDAR_FORBIDDEN_TOOLS = frozenset(
    {
        "begin_exploration",
        "follow_person",
        "navigate_to_semantic",
        "navigate_to_tag",
        "navigate_with_text",
        "start_patrol",
        "tag_location",
    }
)


def scope_readiness_check(
    tools: list[dict[str, Any]], policy: AgentRuntimePolicy
) -> HarnessValidationCheck | None:
    """Return a scope-specific external harness readiness check when applicable."""

    if policy.tool_scope == "readonly":
        return _readonly_readiness_check(tools, policy)
    if policy.tool_scope == "vision_lidar_only":
        return _vision_lidar_only_readiness_check(tools, policy)
    return None


def _readonly_readiness_check(
    tools: list[dict[str, Any]], policy: AgentRuntimePolicy
) -> HarnessValidationCheck:
    tool_names = _tool_names(tools)
    visible_required = sorted(tool_names & _READONLY_REQUIRED_TOOLS)
    missing_required = sorted(_READONLY_REQUIRED_TOOLS - tool_names)
    forbidden_visible = sorted(tool_names & _READONLY_FORBIDDEN_TOOLS)
    missing: list[str] = []
    if missing_required:
        missing.extend(missing_required)
    if policy.movement_mode != "deny":
        missing.append("deny_movement_mode")
    if forbidden_visible:
        return fail_check(
            "readonly_readiness",
            "readonly exposes movement, speech, agent-input, or write-capable tools.",
            forbidden_tools=forbidden_visible,
            visible_required=visible_required,
            missing=missing,
            movement_mode=policy.movement_mode,
        )
    if missing:
        return fail_check(
            "readonly_readiness",
            "readonly is missing required inspection/diagnostic tools or deny movement mode.",
            missing=missing,
            visible_required=visible_required,
            movement_mode=policy.movement_mode,
        )
    return pass_check(
        "readonly_readiness",
        "readonly exposes inspection and diagnostics while denying movement authority.",
        visible_required=visible_required,
        movement_mode=policy.movement_mode,
    )


def _vision_lidar_only_readiness_check(
    tools: list[dict[str, Any]], policy: AgentRuntimePolicy
) -> HarnessValidationCheck:
    tool_names = _tool_names(tools)
    visible_local_motion = sorted(tool_names & _VISION_LIDAR_LOCAL_MOTION_TOOLS)
    missing_local_motion = sorted(_VISION_LIDAR_LOCAL_MOTION_TOOLS - tool_names)
    forbidden_visible = sorted(tool_names & _VISION_LIDAR_FORBIDDEN_TOOLS)
    diagnostics_visible = sorted(tool_names & _VISION_LIDAR_REQUIRED_DIAGNOSTIC_TOOLS)
    missing_diagnostics = sorted(_VISION_LIDAR_REQUIRED_DIAGNOSTIC_TOOLS - tool_names)
    missing: list[str] = []
    if missing_local_motion:
        missing.extend(missing_local_motion)
    if missing_diagnostics:
        missing.extend(missing_diagnostics)
    if not is_task_authority_mode(policy.movement_mode):
        missing.append("task_authority_movement_mode")
    if forbidden_visible:
        return fail_check(
            "vision_lidar_only_readiness",
            "vision_lidar_only exposes prebuilt map/frontier/semantic tools or spatial-memory writes.",
            forbidden_tools=forbidden_visible,
            visible_local_motion=visible_local_motion,
            diagnostics_visible=diagnostics_visible,
            missing=missing,
            movement_mode=policy.movement_mode,
        )
    if missing:
        return fail_check(
            "vision_lidar_only_readiness",
            "vision_lidar_only is missing required supervised low-speed readiness signals.",
            missing=missing,
            visible_local_motion=visible_local_motion,
            diagnostics_visible=diagnostics_visible,
            movement_mode=policy.movement_mode,
        )
    return pass_check(
        "vision_lidar_only_readiness",
        "vision_lidar_only exposes local motion and diagnostics while hiding prebuilt navigation.",
        visible_local_motion=visible_local_motion,
        diagnostics_visible=diagnostics_visible,
        movement_mode=policy.movement_mode,
    )


def _tool_names(tools: list[dict[str, Any]]) -> set[str]:
    return {str(tool.get("name") or "") for tool in tools}
