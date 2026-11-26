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

"""Named Go2 harness tool-profile labels for audits and prompts.

Profiles are labels over the existing runtime policy scopes. They do not create
another permission system; MCP server enforcement still comes from
``runtime_policy`` and the authority proxy envelope.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

GO2_STACK_DISABLED_MODULES: dict[str, tuple[str, ...]] = {
    "full": (),
    "mapping": (
        "wavefront-frontier-explorer",
        "patrolling-module",
        "websocket-vis-module",
    ),
    "motion": (
        "voxel-grid-mapper",
        "cost-mapper",
        "wavefront-frontier-explorer",
        "patrolling-module",
        "websocket-vis-module",
    ),
}

_STACK_ALIASES = {
    "full": "full",
    "full-stack": "full",
    "mapping": "mapping",
    "navigation-lite": "mapping",
    "motion": "motion",
    "motion-only": "motion",
}

_CONTROL_TOOL_PROFILES = {
    "inspect": "readonly_inspection",
    "direct": "vision_odom_direct_motion",
    "local-planner": "vision_odom_local_motion",
    "frontier": "full_navigation_allowed",
}

_CONTROL_ALIASES = {
    "inspect": "inspect",
    "readonly-inspection": "inspect",
    "direct": "direct",
    "vision-odom-direct-motion": "direct",
    "local-planner": "local-planner",
    "vision-odom-local-motion": "local-planner",
    "frontier": "frontier",
    "full-navigation-allowed": "frontier",
}

_CONTROL_REQUIRED_MODULES = {
    "inspect": (),
    "direct": (),
    "local-planner": ("voxel-grid-mapper", "cost-mapper"),
    "frontier": (
        "voxel-grid-mapper",
        "cost-mapper",
        "wavefront-frontier-explorer",
        "patrolling-module",
    ),
}


@dataclass(frozen=True)
class Go2HarnessToolProfile:
    """Audit label for a Go2 experiment tool surface."""

    name: str
    tool_scope: str
    description: str
    allowed_strategy: str
    disallowed_strategy: str
    default_authority_movement_tools: tuple[str, ...]
    benchmark_label: str

    def model_dump(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["default_authority_movement_tools"] = list(self.default_authority_movement_tools)
        payload["control"] = _control_for_tool_profile(self.name)
        return payload


@dataclass(frozen=True)
class Go2ProfileAxes:
    """Resolved public Go2 stack/control axes over the legacy runtime mechanisms."""

    stack: str
    control: str
    edge_profile: str | None
    tool_profile: str
    tool_scope: str
    disabled_modules: tuple[str, ...]
    allowed_movement_tools: tuple[str, ...]
    effective_tools: tuple[str, ...]


GO2_HARNESS_TOOL_PROFILES: dict[str, Go2HarnessToolProfile] = {
    "readonly_inspection": Go2HarnessToolProfile(
        name="readonly_inspection",
        tool_scope="readonly",
        description="Status, diagnostics, and vision only; no robot movement.",
        allowed_strategy="observe/status/diagnostics/read_image only",
        disallowed_strategy="all movement, speech, navigation, frontier, semantic-memory writes",
        default_authority_movement_tools=(),
        benchmark_label="readonly",
    ),
    "vision_odom_local_motion": Go2HarnessToolProfile(
        name="vision_odom_local_motion",
        tool_scope="vision_lidar_only",
        description=(
            "Current visual/lidar evidence plus bounded local odometry movement; "
            "no prebuilt map, frontier, patrol, tags, or semantic navigation."
        ),
        allowed_strategy="observe/status/drive/rotate_in_place/relative_move/stop/safe posture",
        disallowed_strategy="frontier, map-backed navigation, semantic memory, tags, patrol, speech",
        default_authority_movement_tools=(
            "drive",
            "relative_move",
            "rotate_in_place",
            "pixelgoal_navigate",
            "full_lie_down",
            "lie_down",
        ),
        benchmark_label="vision_odom_local_motion",
    ),
    "vision_odom_direct_motion": Go2HarnessToolProfile(
        name="vision_odom_direct_motion",
        tool_scope="vision_lidar_only",
        description=(
            "Current visual/lidar evidence plus direct drive/rotation movement; "
            "no planner-backed relative_move."
        ),
        allowed_strategy="observe/status/drive/rotate_in_place/stop/safe posture",
        disallowed_strategy="relative_move, frontier, map-backed navigation, semantic memory, tags, patrol, speech",
        default_authority_movement_tools=(
            "drive",
            "rotate_in_place",
            "pixelgoal_navigate",
            "full_lie_down",
            "lie_down",
        ),
        benchmark_label="vision_odom_direct_motion",
    ),
    "no_spatial_memory": Go2HarnessToolProfile(
        name="no_spatial_memory",
        tool_scope="semantic_memory_off",
        description="Navigation may use current evidence, but semantic/tagged spatial memory is off.",
        allowed_strategy="non-semantic current-state navigation and local motion",
        disallowed_strategy="navigate_to_semantic, navigate_to_tag, navigate_with_text semantic fallback, tag_location",
        default_authority_movement_tools=(
            "drive",
            "relative_move",
            "rotate_in_place",
            "pixelgoal_navigate",
            "full_lie_down",
            "lie_down",
        ),
        benchmark_label="no_spatial_memory",
    ),
    "no_map_local_motion": Go2HarnessToolProfile(
        name="no_map_local_motion",
        tool_scope="no_map",
        description="Current vision/lidar and local motion only; map-backed navigation is off.",
        allowed_strategy="observe/status/local motion without map-backed navigation or spatial-memory writes",
        disallowed_strategy="map-backed navigation, tags, frontier, patrol, and semantic navigation",
        default_authority_movement_tools=(
            "drive",
            "relative_move",
            "rotate_in_place",
            "pixelgoal_navigate",
            "full_lie_down",
            "lie_down",
        ),
        benchmark_label="no_map_local_motion",
    ),
    "mapping_allowed": Go2HarnessToolProfile(
        name="mapping_allowed",
        tool_scope="no_frontier",
        description="Mapping/local navigation may be used, but autonomous frontier exploration is off.",
        allowed_strategy="map-backed navigation and local motion when explicitly selected",
        disallowed_strategy="frontier exploration unless a full-navigation profile is selected",
        default_authority_movement_tools=(
            "drive",
            "relative_move",
            "rotate_in_place",
            "pixelgoal_navigate",
            "full_lie_down",
            "lie_down",
        ),
        benchmark_label="mapping_allowed_no_frontier",
    ),
    "full_navigation_allowed": Go2HarnessToolProfile(
        name="full_navigation_allowed",
        tool_scope="full",
        description="Full supervised DimOS navigation surface; use only for explicit comparison runs.",
        allowed_strategy="all supervised navigation, mapping, frontier, semantic, and local motion tools",
        disallowed_strategy="none beyond runtime safety, dangerous-skill, battery, and authority gates",
        default_authority_movement_tools=(
            "drive",
            "relative_move",
            "rotate_in_place",
            "pixelgoal_navigate",
            "begin_exploration",
            "start_patrol",
            "full_lie_down",
            "lie_down",
        ),
        benchmark_label="full_navigation_allowed",
    ),
    "posture_shutdown": Go2HarnessToolProfile(
        name="posture_shutdown",
        tool_scope="vision_lidar_only",
        description="One-shot safe shutdown posture profile; no locomotion authority.",
        allowed_strategy="stop/status plus full_lie_down or lie_down only",
        disallowed_strategy="relative_move, rotate_in_place, exploration, navigation, patrol, speech",
        default_authority_movement_tools=("full_lie_down", "lie_down"),
        benchmark_label="posture_shutdown_only",
    ),
    "posture_startup": Go2HarnessToolProfile(
        name="posture_startup",
        tool_scope="vision_lidar_only",
        description="One-shot startup posture profile; no locomotion authority.",
        allowed_strategy="status plus stand_up and balance_stand only",
        disallowed_strategy="relative_move, rotate_in_place, exploration, navigation, patrol, speech",
        default_authority_movement_tools=("balance_stand", "stand_up"),
        benchmark_label="posture_startup_only",
    ),
}


def resolve_go2_profile_axes(
    stack: str = "full",
    control: str = "local-planner",
) -> Go2ProfileAxes:
    """Resolve and validate the public two-axis Go2 profile model."""

    resolved_stack = _resolve_alias(stack, _STACK_ALIASES, "stack")
    resolved_control = _resolve_alias(control, _CONTROL_ALIASES, "control")
    tool_profile = go2_harness_tool_profile(_CONTROL_TOOL_PROFILES[resolved_control])
    disabled_modules = GO2_STACK_DISABLED_MODULES[resolved_stack]
    missing_modules = tuple(
        module
        for module in _CONTROL_REQUIRED_MODULES[resolved_control]
        if module in disabled_modules
    )
    stack_tools = _stack_movement_tools(resolved_stack)
    missing_tools = tuple(
        tool for tool in tool_profile.default_authority_movement_tools if tool not in stack_tools
    )
    if missing_modules or missing_tools:
        details = []
        if missing_modules:
            details.append(f"missing modules: {', '.join(missing_modules)}")
        if missing_tools:
            details.append(f"unavailable tools: {', '.join(missing_tools)}")
        raise ValueError(
            f"Incompatible --stack/--control: stack={resolved_stack!r}, "
            f"control={resolved_control!r}; " + "; ".join(details)
        )

    allowed = tuple(
        tool for tool in tool_profile.default_authority_movement_tools if tool in stack_tools
    )
    return Go2ProfileAxes(
        stack=resolved_stack,
        control=resolved_control,
        edge_profile={"full": None, "mapping": "navigation-lite", "motion": "motion-only"}[
            resolved_stack
        ],
        tool_profile=tool_profile.name,
        tool_scope=tool_profile.tool_scope,
        disabled_modules=disabled_modules,
        allowed_movement_tools=allowed,
        effective_tools=tuple(sorted({"stop_movement", *allowed})),
    )


def go2_control_tool_profile(control: str) -> Go2HarnessToolProfile:
    """Return the legacy harness profile derived from a public control value or alias."""

    resolved = _resolve_alias(control, _CONTROL_ALIASES, "control")
    return go2_harness_tool_profile(_CONTROL_TOOL_PROFILES[resolved])


def _resolve_alias(value: str, aliases: dict[str, str], axis: str) -> str:
    normalized = str(value or "").strip().lower().replace("_", "-")
    try:
        return aliases[normalized]
    except KeyError as exc:
        available = ", ".join(sorted(set(aliases.values())))
        raise ValueError(f"Unknown Go2 {axis} {value!r}. Available: {available}") from exc


def _stack_movement_tools(stack: str) -> tuple[str, ...]:
    control = {"full": "frontier", "mapping": "local-planner", "motion": "direct"}[stack]
    profile = go2_harness_tool_profile(_CONTROL_TOOL_PROFILES[control])
    return profile.default_authority_movement_tools


def _control_for_tool_profile(tool_profile: str) -> str | None:
    return next(
        (control for control, profile in _CONTROL_TOOL_PROFILES.items() if profile == tool_profile),
        None,
    )


_SCOPE_DEFAULT_PROFILE = {
    "readonly": "readonly_inspection",
    "vision_lidar_only": "vision_odom_local_motion",
    "semantic_memory_off": "no_spatial_memory",
    "no_map": "no_map_local_motion",
    "no_frontier": "mapping_allowed",
    "full": "full_navigation_allowed",
}


def go2_harness_tool_profile(name: str) -> Go2HarnessToolProfile:
    """Resolve a named Go2 harness profile."""

    normalized = str(name or "").strip().lower().replace("-", "_")
    try:
        return GO2_HARNESS_TOOL_PROFILES[normalized]
    except KeyError as exc:
        available = ", ".join(sorted(GO2_HARNESS_TOOL_PROFILES))
        raise ValueError(
            f"Unknown Go2 harness tool profile '{name}'. Available: {available}"
        ) from exc


def default_go2_harness_tool_profile_for_scope(tool_scope: str) -> Go2HarnessToolProfile:
    """Return the canonical audit label for an existing runtime tool scope."""

    normalized_scope = str(tool_scope or "full").strip().lower().replace("-", "_")
    return go2_harness_tool_profile(
        _SCOPE_DEFAULT_PROFILE.get(normalized_scope, "full_navigation_allowed")
    )


def go2_harness_profile_for_policy(policy: Any) -> Go2HarnessToolProfile:
    """Return the audit label implied by a resolved ``AgentRuntimePolicy``."""

    control = getattr(policy, "control", None)
    if control:
        return go2_control_tool_profile(control)
    return default_go2_harness_tool_profile_for_scope(getattr(policy, "tool_scope", "full"))


def go2_harness_profile_prompt_lines(profile: Go2HarnessToolProfile) -> list[str]:
    """Short prompt lines that explain an active profile without rewriting policy."""

    return [
        f"Go2 harness tool profile: {profile.name} (runtime tool_scope={profile.tool_scope}).",
        f"Allowed strategy: {profile.allowed_strategy}.",
        f"Disallowed strategy: {profile.disallowed_strategy}.",
        "The DimOS MCP server and authority proxy enforce the actual policy; treat this profile as an audit label and operating boundary.",
    ]
