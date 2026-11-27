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

from collections.abc import Iterable
from dataclasses import asdict, dataclass
import re
from typing import Any, Literal

from dimos.utils.logging_config import setup_logger

logger = setup_logger()

AgentProfile = Literal["dev_text_only", "supervised_live", "demo_voice", "autonomous_lab"]
SpeakMode = Literal["off", "on_request", "status_only", "always"]
MovementMode = Literal["deny", "confirm", "lease", "task_authority", "allow_safe", "allow_all"]
DangerousSkillsMode = Literal["deny", "confirm", "allow"]
ToolScope = Literal[
    "full",
    "no_frontier",
    "no_map",
    "semantic_memory_off",
    "vision_lidar_only",
    "readonly",
]
VlmProvider = Literal[
    "disabled",
    "moondream-local",
    "openai-compatible",
    "qwen-dashscope",
    "moondream-hosted",
]
SpatialMemoryMode = Literal["ephemeral", "session", "persistent"]

_SPEAK_MODES = ("off", "on_request", "status_only", "always")
_MOVEMENT_MODES = (
    "deny",
    "confirm",
    "lease",
    "task_authority",
    "allow_safe",
    "allow_all",
)
_DANGEROUS_SKILLS_MODES = ("deny", "confirm", "allow")
_TOOL_SCOPES = (
    "full",
    "no_frontier",
    "no_map",
    "semantic_memory_off",
    "vision_lidar_only",
    "readonly",
)
_VLM_PROVIDERS = (
    "disabled",
    "moondream-local",
    "openai-compatible",
    "qwen-dashscope",
    "moondream-hosted",
)
_SPATIAL_MEMORY_MODES = ("ephemeral", "session", "persistent")

_MOVEMENT_TOOL_NAMES = frozenset(
    {
        "begin_exploration",
        "balance_stand",
        "execute_sport_command",
        "follow_person",
        "full_lie_down",
        "lie_down",
        "navigate_to_semantic",
        "navigate_to_tag",
        "navigate_with_text",
        "relative_move",
        "return_to_task_start",
        "rotate_in_place",
        "drive",
        "stand_up",
        "start_patrol",
        "visual_approach",
    }
)
_VLM_TOOL_NAMES = frozenset(
    {
        "look_out_for",
        "visual_approach",
        "visual_find",
    }
)
_FRONTIER_TOOL_NAMES = frozenset({"begin_exploration"})
_MAP_BACKED_TOOL_NAMES = frozenset(
    {
        "begin_exploration",
        "navigate_to_semantic",
        "navigate_to_tag",
        "navigate_with_text",
        "return_to_task_start",
        "start_patrol",
        "tag_location",
    }
)
_SEMANTIC_MEMORY_TOOL_NAMES = frozenset(
    {
        "navigate_to_semantic",
        "navigate_to_tag",
        "navigate_with_text",
        "tag_location",
    }
)
_VISION_LIDAR_ONLY_BLOCKED_TOOLS = frozenset(
    {
        "begin_exploration",
        "follow_person",
        "navigate_to_semantic",
        "navigate_to_tag",
        "navigate_with_text",
        "return_to_task_start",
        "start_patrol",
        "tag_location",
    }
)
_READONLY_BLOCKED_TOOLS = frozenset(
    {
        "agent_send",
        "begin_exploration",
        "balance_stand",
        "execute_sport_command",
        "follow_person",
        "full_lie_down",
        "lie_down",
        "navigate_to_semantic",
        "navigate_to_tag",
        "navigate_with_text",
        "relative_move",
        "return_to_task_start",
        "rotate_in_place",
        "speak",
        "stand_up",
        "start_patrol",
        "tag_location",
        "visual_approach",
    }
)
_STOP_TOOL_NAMES = frozenset(
    {
        "end_exploration",
        "stop_following",
        "stop_looking_out",
        "stop_movement",
        "stop_navigation",
        "stop_patrol",
    }
)
_DANGEROUS_SPORT_COMMANDS = frozenset(
    {
        "Backflip",
        "Bound",
        "CrossStep",
        "Dance1",
        "Dance2",
        "FingerHeart",
        "FrontFlip",
        "FrontJump",
        "FrontPounce",
        "Handstand",
        "LeftFlip",
        "MoonWalk",
        "OnesidedStep",
        "RightFlip",
        "Wallow",
    }
)


@dataclass(frozen=True)
class AgentRuntimePolicy:
    """Resolved policy for agent-facing robot runtime behavior."""

    profile: AgentProfile
    speak_mode: SpeakMode
    movement_mode: MovementMode
    dangerous_skills: DangerousSkillsMode
    tool_scope: ToolScope
    vlm_provider: VlmProvider
    spatial_memory_mode: SpatialMemoryMode
    tool_policy_admin_override: bool
    control: str | None = None
    allowed_movement_tools: tuple[str, ...] | None = None

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _ProfileDefaults:
    speak_mode: SpeakMode
    movement_mode: MovementMode
    dangerous_skills: DangerousSkillsMode
    tool_scope: ToolScope
    vlm_provider: VlmProvider
    spatial_memory_mode: SpatialMemoryMode


_PROFILE_DEFAULTS: dict[AgentProfile, _ProfileDefaults] = {
    "dev_text_only": _ProfileDefaults(
        speak_mode="off",
        movement_mode="deny",
        dangerous_skills="deny",
        tool_scope="readonly",
        vlm_provider="disabled",
        spatial_memory_mode="ephemeral",
    ),
    "supervised_live": _ProfileDefaults(
        speak_mode="on_request",
        movement_mode="task_authority",
        dangerous_skills="deny",
        tool_scope="full",
        vlm_provider="disabled",
        spatial_memory_mode="session",
    ),
    "demo_voice": _ProfileDefaults(
        speak_mode="status_only",
        movement_mode="confirm",
        dangerous_skills="deny",
        tool_scope="full",
        vlm_provider="openai-compatible",
        spatial_memory_mode="session",
    ),
    "autonomous_lab": _ProfileDefaults(
        speak_mode="status_only",
        movement_mode="allow_safe",
        dangerous_skills="confirm",
        tool_scope="full",
        vlm_provider="openai-compatible",
        spatial_memory_mode="persistent",
    ),
}


def _normalize(value: str) -> str:
    return value.strip().lower().replace("_", "-")


def _normalize_underscore(value: str) -> str:
    return value.strip().lower().replace("-", "_")


def _choice(value: str | None, choices: tuple[str, ...], name: str) -> str | None:
    if value is None:
        return None
    normalized = _normalize(value)
    if normalized not in choices:
        raise ValueError(f"{name} must be one of {choices}, got {value!r}")
    return normalized


def _choice_underscore(value: str | None, choices: tuple[str, ...], name: str) -> str | None:
    if value is None:
        return None
    normalized = _normalize_underscore(value)
    if normalized not in choices:
        raise ValueError(f"{name} must be one of {choices}, got {value!r}")
    return normalized


def resolve_agent_runtime_policy(config: Any | None = None) -> AgentRuntimePolicy:
    """Resolve profile defaults plus explicit overrides into one policy object."""

    if config is None:
        from dimos.core.global_config import global_config

        config = global_config

    profile = _choice_underscore(
        getattr(config, "agent_profile", None) or "supervised_live",
        tuple(_PROFILE_DEFAULTS.keys()),
        "agent_profile",
    )
    assert profile is not None
    defaults = _PROFILE_DEFAULTS[profile]  # type: ignore[index]

    speak_mode = _choice_underscore(
        getattr(config, "agent_speak_mode", None), _SPEAK_MODES, "agent_speak_mode"
    )
    movement_mode = _choice_underscore(
        getattr(config, "agent_movement_mode", None), _MOVEMENT_MODES, "agent_movement_mode"
    )
    dangerous_skills = _choice_underscore(
        getattr(config, "agent_dangerous_skills", None),
        _DANGEROUS_SKILLS_MODES,
        "agent_dangerous_skills",
    )
    control = None
    allowed_movement_tools = None
    configured_control = getattr(config, "agent_control", None)
    if configured_control:
        from dimos.agents.harness.go2_harness_profiles import resolve_go2_profile_axes

        axes = resolve_go2_profile_axes("full", configured_control)
        control = axes.control
        tool_scope = axes.tool_scope
        allowed_movement_tools = axes.allowed_movement_tools
    else:
        tool_scope = _choice_underscore(
            getattr(config, "agent_tool_scope", None), _TOOL_SCOPES, "agent_tool_scope"
        )
    vlm_provider = _choice(getattr(config, "vlm_provider", None), _VLM_PROVIDERS, "vlm_provider")
    spatial_memory_mode = _choice_underscore(
        getattr(config, "spatial_memory_mode", None),
        _SPATIAL_MEMORY_MODES,
        "spatial_memory_mode",
    )

    return AgentRuntimePolicy(
        profile=profile,  # type: ignore[arg-type]
        speak_mode=(speak_mode or defaults.speak_mode),  # type: ignore[arg-type]
        movement_mode=(movement_mode or defaults.movement_mode),  # type: ignore[arg-type]
        dangerous_skills=(dangerous_skills or defaults.dangerous_skills),  # type: ignore[arg-type]
        tool_scope=(tool_scope or defaults.tool_scope),  # type: ignore[arg-type]
        vlm_provider=(vlm_provider or defaults.vlm_provider),  # type: ignore[arg-type]
        spatial_memory_mode=(spatial_memory_mode or defaults.spatial_memory_mode),  # type: ignore[arg-type]
        tool_policy_admin_override=bool(getattr(config, "agent_tool_policy_admin_override", False)),
        control=control,
        allowed_movement_tools=allowed_movement_tools,
    )


def is_stop_tool(tool_name: str) -> bool:
    """Return whether a tool is explicitly allowed as a stop/cancel action."""

    return tool_name in _STOP_TOOL_NAMES


def is_movement_tool(tool_name: str, uses: list[str] | tuple[str, ...] = ()) -> bool:
    """Return whether a tool can initiate robot movement."""

    return tool_name in _MOVEMENT_TOOL_NAMES or "movement" in uses


def is_task_authority_mode(movement_mode: MovementMode) -> bool:
    """Return whether movement is governed by a bounded task authority."""

    return movement_mode in {"lease", "task_authority"}


def is_dangerous_tool(tool_name: str, args: dict[str, Any]) -> bool:
    """Return whether a movement tool is too risky for normal supervised use."""

    if tool_name != "execute_sport_command":
        return False
    return str(args.get("command_name") or "") in _DANGEROUS_SPORT_COMMANDS


def is_vlm_tool(tool_name: str) -> bool:
    """Return whether a tool requires the configured VLM path."""

    return tool_name in _VLM_TOOL_NAMES


def tool_exposed_by_policy(
    tool_name: str,
    uses: list[str] | tuple[str, ...] = (),
    *,
    policy: AgentRuntimePolicy,
    required_modules: Iterable[str] = (),
    disabled_modules: Iterable[str] = (),
) -> bool:
    """Return whether a tool should be visible in MCP ``tools/list``."""

    return (
        tool_exposure_refusal(
            tool_name,
            uses,
            policy=policy,
            required_modules=required_modules,
            disabled_modules=disabled_modules,
        )
        is None
    )


def tool_exposure_refusal(
    tool_name: str,
    uses: list[str] | tuple[str, ...] = (),
    *,
    policy: AgentRuntimePolicy,
    required_modules: Iterable[str] = (),
    disabled_modules: Iterable[str] = (),
) -> str | None:
    """Return why a tool should be hidden from ``tools/list``, if any.

    Exposure policy intentionally does not hide movement tools just because
    movement needs confirmation or task authority; the model still needs the
    schema to call them after authority is granted. It does hide tools that are
    categorically unavailable in the current runtime profile or task scope.
    """

    dependency_refusal = _disabled_dependency_refusal(required_modules, disabled_modules)
    if dependency_refusal is not None:
        return f"Tool '{tool_name}' requires disabled module(s): {dependency_refusal}."
    if policy.tool_policy_admin_override:
        return None
    scope_refusal = tool_scope_refusal(tool_name, {}, uses, policy=policy)
    if scope_refusal is not None:
        return scope_refusal
    if tool_name == "speak" and policy.speak_mode == "off":
        return "Runtime speech policy is off."
    if is_vlm_tool(tool_name) and policy.vlm_provider == "disabled":
        return "Runtime VLM provider is disabled."
    return None


def _disabled_dependency_refusal(
    required_modules: Iterable[str],
    disabled_modules: Iterable[str],
) -> str | None:
    disabled_by_key = {
        _module_dependency_key(name): str(name)
        for name in disabled_modules
        if _module_dependency_key(name)
    }
    if not disabled_by_key:
        return None

    missing = [
        str(name)
        for name in required_modules
        if (key := _module_dependency_key(name)) and key in disabled_by_key
    ]
    if not missing:
        return None
    return ", ".join(missing)


def _module_dependency_key(name: object) -> str:
    text = str(name or "").strip()
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    return re.sub(r"[^a-z0-9]", "", text.casefold())


def tool_scope_refusal(
    tool_name: str,
    args: dict[str, Any],
    uses: list[str] | tuple[str, ...],
    *,
    policy: AgentRuntimePolicy,
) -> str | None:
    """Return a refusal when the active task/tool scope excludes a tool."""

    if is_stop_tool(tool_name):
        return None

    if (
        policy.allowed_movement_tools is not None
        and is_movement_tool(tool_name, uses)
        and tool_name not in policy.allowed_movement_tools
    ):
        return f"Runtime control '{policy.control}' does not allow movement tool '{tool_name}'."

    if policy.tool_scope == "full":
        return None

    reason = _scope_refusal_reason(tool_name, args, uses, policy.tool_scope)
    if reason is None:
        return None
    return (
        f"Runtime tool scope is {policy.tool_scope}. Tool '{tool_name}' was not executed: {reason}"
    )


def _scope_refusal_reason(
    tool_name: str,
    args: dict[str, Any],
    uses: list[str] | tuple[str, ...],
    tool_scope: ToolScope,
) -> str | None:
    if tool_scope == "no_frontier" and tool_name in _FRONTIER_TOOL_NAMES:
        return "frontier exploration is out of scope."
    if tool_scope == "no_map" and tool_name in _MAP_BACKED_TOOL_NAMES:
        return "map-backed navigation, exploration, or spatial-memory writes are out of scope."
    if tool_scope == "semantic_memory_off" and tool_name in _SEMANTIC_MEMORY_TOOL_NAMES:
        return "semantic or tagged spatial memory is out of scope."
    if tool_scope == "vision_lidar_only":
        if tool_name in _VISION_LIDAR_ONLY_BLOCKED_TOOLS:
            return "prebuilt map/frontier/semantic navigation is out of scope."
        if is_movement_tool(tool_name, uses):
            return _local_motion_scope_refusal(tool_name, args)
    if tool_scope == "readonly":
        if tool_name in _READONLY_BLOCKED_TOOLS or is_movement_tool(tool_name, uses):
            return "readonly scope permits inspection/status tools only."
    return None


def _local_motion_scope_refusal(tool_name: str, args: dict[str, Any]) -> str | None:
    if tool_name in {"relative_move", "drive"}:
        linear = (_float_arg(args, "forward") ** 2 + _float_arg(args, "left") ** 2) ** 0.5
        yaw_deg = abs(_float_arg(args, "degrees"))
        if linear > 1.0 or yaw_deg > 90.0:
            return "vision_lidar_only allows only local relative/drive moves up to 1m and 90deg."
    if tool_name == "rotate_in_place" and abs(_float_arg(args, "degrees")) > 90.0:
        return "vision_lidar_only allows only local rotations up to 90deg."
    return None


def _float_arg(args: dict[str, Any], key: str) -> float:
    value = args.get(key, 0.0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def movement_policy_refusal(
    tool_name: str,
    args: dict[str, Any],
    uses: list[str] | tuple[str, ...],
    *,
    policy: AgentRuntimePolicy,
    confirmed: bool,
    dangerous_confirmed: bool | None = None,
) -> str | None:
    """Return a refusal message when runtime policy blocks an agent tool call."""

    if is_stop_tool(tool_name) or not is_movement_tool(tool_name, uses):
        return None

    dangerous_confirmed = confirmed if dangerous_confirmed is None else dangerous_confirmed
    if is_dangerous_tool(tool_name, args):
        if policy.dangerous_skills == "deny":
            return (
                f"Runtime policy denied dangerous skill '{tool_name}' with args {args}. "
                "Use a non-dynamic posture or movement command instead."
            )
        if policy.dangerous_skills == "confirm" and not dangerous_confirmed:
            return (
                f"Dangerous skill '{tool_name}' requires explicit user confirmation. "
                "Explain the risk and wait for the user to confirm before retrying."
            )

    if policy.movement_mode == "deny":
        return (
            f"Runtime movement policy is deny. Tool '{tool_name}' was not executed. "
            "Stop/cancel tools remain available."
        )

    if policy.movement_mode == "confirm" and not confirmed:
        return (
            f"Runtime movement policy requires explicit user confirmation before "
            f"running '{tool_name}' with args {args}. Ask the user to confirm, then retry."
        )

    return None


def tool_policy_refusal(
    tool_name: str,
    args: dict[str, Any],
    uses: list[str] | tuple[str, ...],
    *,
    policy: AgentRuntimePolicy,
    confirmed: bool,
    dangerous_confirmed: bool | None = None,
) -> str | None:
    """Return a refusal when runtime policy blocks any MCP tool dispatch.

    This is the server-side authority. It applies to every MCP client by default
    so clients that do not know DimOS-specific metadata cannot bypass robot
    safety policy accidentally.
    """

    if policy.tool_policy_admin_override:
        logger.warning(
            "Runtime tool policy admin override is enabled; allowing tool dispatch",
            tool=tool_name,
        )
        return None

    scope_refusal = tool_scope_refusal(tool_name, args, uses, policy=policy)
    if scope_refusal is not None:
        return scope_refusal

    if tool_name == "speak" and policy.speak_mode == "off":
        return "Runtime speech policy is off. Tool 'speak' was not executed."

    if is_vlm_tool(tool_name) and policy.vlm_provider == "disabled":
        return (
            f"Runtime VLM provider is disabled. Tool '{tool_name}' was not executed. "
            "Use non-visual status tools or enable a VLM provider first."
        )

    return movement_policy_refusal(
        tool_name,
        args,
        uses,
        policy=policy,
        confirmed=confirmed,
        dangerous_confirmed=dangerous_confirmed,
    )
